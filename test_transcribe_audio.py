#!/usr/bin/env python3
"""
Unit tests for transcribe_audio.py - Word-level bracket timestamps only
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
from pathlib import Path
import time
import json

# Add the current directory to the path so we can import transcribe_audio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcribe_audio import (
    SRTFormatter, 
    process_transcription_to_srt, 
    join_srt_files_with_overlap, 
    has_timestamps,
    srt_to_json,
    convert_srt_to_json,
    MistralAudioTranscriber
)


class TestSRTFormatter(unittest.TestCase):
    """Test cases for SRTFormatter class"""
    
    def test_parse_time_basic(self):
        """Test basic time parsing functionality"""
        # Test standard format with comma
        self.assertAlmostEqual(SRTFormatter.parse_time("00:01:23,456"), 83.456, places=3)
        
        # Test with dot separator
        self.assertAlmostEqual(SRTFormatter.parse_time("00:01:23.456"), 83.456, places=3)
        
        # Test zero time
        self.assertAlmostEqual(SRTFormatter.parse_time("00:00:00,000"), 0.0, places=3)
        
        # Test hours
        self.assertAlmostEqual(SRTFormatter.parse_time("01:30:45,123"), 5445.123, places=3)
    
    def test_parse_time_edge_cases(self):
        """Test edge cases in time parsing"""
        # Test without milliseconds
        self.assertAlmostEqual(SRTFormatter.parse_time("00:01:23"), 83.0, places=3)
        
        # Test with whitespace
        self.assertAlmostEqual(SRTFormatter.parse_time("  00:01:23,456  "), 83.456, places=3)
    
    def test_format_time_roundtrip(self):
        """Test that format_time and parse_time are consistent"""
        test_times = [0.0, 83.456, 3661.789, 7323.123]
        
        for time_val in test_times:
            formatted = SRTFormatter.format_time(time_val)
            parsed = SRTFormatter.parse_time(formatted)
            self.assertAlmostEqual(time_val, parsed, places=2)


class TestSRTMerging(unittest.TestCase):
    """Test cases for SRT file merging and overlap handling"""
    
    def setUp(self):
        """Set up temporary directory for test files"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_srt_file(self, filename: str, entries: list) -> str:
        """Helper to create test SRT files"""
        filepath = os.path.join(self.temp_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, (start, end, text) in enumerate(entries, 1):
                start_formatted = SRTFormatter.format_time(start)
                end_formatted = SRTFormatter.format_time(end)
                f.write(f"{i}\n{start_formatted} --> {end_formatted}\n{text}\n\n")
        return filepath
    
    def test_single_segment_no_overlap(self):
        """Test merging a single segment with no overlap"""
        # Create test SRT file
        srt_path = self.create_test_srt_file("segment_001.srt", [
            (0.0, 2.0, "Hello"),
            (2.5, 4.5, "world"),
            (5.0, 7.0, "test")
        ])
        
        srt_files = {0: srt_path}
        segment_info = {
            0: {
                'start_time': 0.0,
                'overlap_start': 0.0
            }
        }
        
        result = join_srt_files_with_overlap(srt_files, segment_info, keep_segments=False)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['text'], "Hello")
        self.assertAlmostEqual(result[0]['start_time'], 0.0, places=1)
        self.assertEqual(result[1]['text'], "world")
        self.assertAlmostEqual(result[1]['start_time'], 2.5, places=1)
        self.assertEqual(result[2]['text'], "test")
        self.assertAlmostEqual(result[2]['start_time'], 5.0, places=1)
    
    def test_two_segments_with_overlap(self):
        """Test merging two segments with overlap"""
        # Create first segment SRT
        srt1_path = self.create_test_srt_file("segment_001.srt", [
            (0.0, 2.0, "First"),
            (2.5, 4.5, "segment"),
            (5.0, 7.0, "content")
        ])
        
        # Create second segment SRT (with content that should be filtered due to overlap)
        srt2_path = self.create_test_srt_file("segment_002.srt", [
            (0.0, 2.0, "overlapped"),  # This should be filtered out (0 + 60 = 60 <= 65)
            (2.5, 4.5, "content"),     # This should be filtered out (2.5 + 60 = 62.5 <= 65)
            (6.0, 8.0, "new"),         # This should be kept (6.0 + 60 = 66 > 65)
            (8.5, 10.5, "content")     # This should be kept (8.5 + 60 = 68.5 > 65)
        ])
        
        srt_files = {0: srt1_path, 1: srt2_path}
        segment_info = {
            0: {
                'start_time': 0.0,
                'overlap_start': 0.0
            },
            1: {
                'start_time': 60.0,  # Second segment starts at 60s
                'overlap_start': 5.0  # 5s overlap (overlap ends at 60 + 5 = 65s)
            }
        }
        
        result = join_srt_files_with_overlap(srt_files, segment_info, keep_segments=False)
        
        # Should have 3 entries from first segment + 2 entries from second segment (after overlap filtering)
        self.assertEqual(len(result), 5)
        
        # Check first segment entries (no time shift)
        self.assertEqual(result[0]['text'], "First")
        self.assertAlmostEqual(result[0]['start_time'], 0.0, places=1)
        self.assertEqual(result[1]['text'], "segment")
        self.assertAlmostEqual(result[1]['start_time'], 2.5, places=1)
        self.assertEqual(result[2]['text'], "content")
        self.assertAlmostEqual(result[2]['start_time'], 5.0, places=1)
        
        # Check second segment entries (with time shift and overlap filtering)
        # Only entries starting at 6.0s and 8.5s should remain (after 5s overlap filter)
        # They should be shifted by 60s (segment start time)
        self.assertEqual(result[3]['text'], "new")
        self.assertAlmostEqual(result[3]['start_time'], 66.0, places=1)  # 6.0 + 60.0
        self.assertEqual(result[4]['text'], "content")
        self.assertAlmostEqual(result[4]['start_time'], 68.5, places=1)  # 8.5 + 60.0
    
    def test_overlap_boundary_conditions(self):
        """Test overlap filtering at exact boundaries"""
        # Create segment with entries exactly at overlap boundary
        srt_path = self.create_test_srt_file("segment_002.srt", [
            (29.5, 31.0, "just_before"),  # Should be filtered (29.5 + 60 = 89.5 <= 90)
            (30.0, 32.0, "exactly_at"),   # Should be filtered (30.0 + 60 = 90.0 <= 90)
            (30.5, 32.5, "just_after")    # Should be kept (30.5 + 60 = 90.5 > 90)
        ])
        
        srt_files = {1: srt_path}
        segment_info = {
            1: {
                'start_time': 60.0,
                'overlap_start': 30.0  # Overlap ends at 60 + 30 = 90s
            }
        }
        
        result = join_srt_files_with_overlap(srt_files, segment_info, keep_segments=False)
        
        # Only the "just_after" entry should remain
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], "just_after")
        self.assertAlmostEqual(result[0]['start_time'], 90.5, places=1)
    
    def test_multiple_segments_complex_overlap(self):
        """Test merging multiple segments with complex overlap scenarios"""
        # Create three segments
        srt1_path = self.create_test_srt_file("segment_001.srt", [
            (0.0, 2.0, "Segment1_Entry1"),
            (3.0, 5.0, "Segment1_Entry2")
        ])
        
        srt2_path = self.create_test_srt_file("segment_002.srt", [
            (0.0, 2.0, "Overlap_Content"),  # Should be filtered
            (4.0, 6.0, "Segment2_Entry1"),  # Should be kept
            (7.0, 9.0, "Segment2_Entry2")   # Should be kept
        ])
        
        srt3_path = self.create_test_srt_file("segment_003.srt", [
            (0.0, 2.0, "More_Overlap"),     # Should be filtered
            (3.0, 5.0, "Segment3_Entry1"),  # Should be kept
            (6.0, 8.0, "Segment3_Entry2")   # Should be kept
        ])
        
        srt_files = {0: srt1_path, 1: srt2_path, 2: srt3_path}
        segment_info = {
            0: {'start_time': 0.0, 'overlap_start': 0.0},
            1: {'start_time': 30.0, 'overlap_start': 3.0},   # 3s overlap
            2: {'start_time': 60.0, 'overlap_start': 2.5}    # 2.5s overlap
        }
        
        result = join_srt_files_with_overlap(srt_files, segment_info, keep_segments=False)
        
        # Should have: 2 from segment1 + 2 from segment2 + 2 from segment3 = 6 entries
        self.assertEqual(len(result), 6)
        
        # Check proper indexing
        for i, entry in enumerate(result):
            self.assertEqual(entry['index'], i + 1)
        
        # Check that timestamps are properly shifted
        # Segment 2 entries should start at 30 + original_time (after overlap filter)
        # Segment 3 entries should start at 60 + original_time (after overlap filter)
        segment2_entries = [e for e in result if 'Segment2' in e['text']]
        self.assertEqual(len(segment2_entries), 2)
        self.assertAlmostEqual(segment2_entries[0]['start_time'], 34.0, places=1)  # 4.0 + 30.0
        
        segment3_entries = [e for e in result if 'Segment3' in e['text']]
        self.assertEqual(len(segment3_entries), 2)
        self.assertAlmostEqual(segment3_entries[0]['start_time'], 63.0, places=1)  # 3.0 + 60.0
    
    def test_empty_srt_file_handling(self):
        """Test handling of empty or malformed SRT files"""
        # Create empty SRT file
        empty_srt_path = os.path.join(self.temp_dir, "empty.srt")
        with open(empty_srt_path, 'w', encoding='utf-8') as f:
            f.write("")
        
        # Create normal SRT file
        normal_srt_path = self.create_test_srt_file("normal.srt", [
            (0.0, 2.0, "Normal content")
        ])
        
        srt_files = {0: empty_srt_path, 1: normal_srt_path}
        segment_info = {
            0: {'start_time': 0.0, 'overlap_start': 0.0},
            1: {'start_time': 30.0, 'overlap_start': 0.0}
        }
        
        result = join_srt_files_with_overlap(srt_files, segment_info, keep_segments=False)
        
        # Should only have the entry from the normal file
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], "Normal content")
        self.assertAlmostEqual(result[0]['start_time'], 30.0, places=1)
    
    def test_keep_segments_flag(self):
        """Test that segment files are preserved when keep_segments=True"""
        srt_path = self.create_test_srt_file("test_segment.srt", [
            (0.0, 2.0, "Test content")
        ])
        
        srt_files = {0: srt_path}
        segment_info = {0: {'start_time': 0.0, 'overlap_start': 0.0}}
        
        # Test with keep_segments=True
        result = join_srt_files_with_overlap(srt_files, segment_info, keep_segments=True)
        
        # File should still exist
        self.assertTrue(os.path.exists(srt_path))
        self.assertEqual(len(result), 1)
        
        # Test with keep_segments=False
        result = join_srt_files_with_overlap(srt_files, segment_info, keep_segments=False)
        
        # File should be removed
        self.assertFalse(os.path.exists(srt_path))


class TestProcessTranscriptionToSrt(unittest.TestCase):
    """Test cases for process_transcription_to_srt function - bracket timestamps only"""

    def test_word_level_timestamps_bracket_format(self):
        """Test word-level timestamps in [HH:MM.SS] format"""
        transcription = "[00:01.23] Hello [00:02.45] world [00:03.67] test"
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['text'], 'Hello')
        
        # With non-overlapping logic: 
        # Word 1 (Hello): middle=1.23, start=halfway between start and 1.23, end=halfway between 1.23 and 2.45
        # For first word, start = 1.23 - (1.84-1.23) = 1.23 - 0.61 = 0.62
        # End = (1.23 + 2.45) / 2 = 1.84
        self.assertAlmostEqual(result[0]['start_time'], 0.62, places=2)
        self.assertAlmostEqual(result[0]['end_time'], 1.84, places=2)
        
        self.assertEqual(result[1]['text'], 'world')
        # Word 2 (world): middle=2.45, start=(1.23+2.45)/2=1.84, end=(2.45+3.67)/2=3.06
        self.assertAlmostEqual(result[1]['start_time'], 1.84, places=2)
        self.assertAlmostEqual(result[1]['end_time'], 3.06, places=2)
        
        self.assertEqual(result[2]['text'], 'test')
        # Word 3 (test): middle=3.67, start=(2.45+3.67)/2=3.06, end=3.67+(3.67-2.45)/2=4.28
        self.assertAlmostEqual(result[2]['start_time'], 3.06, places=2)
        self.assertAlmostEqual(result[2]['end_time'], 4.28, places=2)

    def test_bracket_format_with_offset(self):
        """Test bracket format with segment offset"""
        transcription = "[00:01.00] Hello [00:02.00] world"
        segment_offset = 10.0
        
        result = process_transcription_to_srt(transcription, segment_offset_seconds=segment_offset)
        
        self.assertEqual(len(result), 2)
        # With non-overlapping logic and offset:
        # Word 1 (Hello): middle=11.0 (1.0+10.0), end=(11.0+12.0)/2=11.5
        # For first word, start = 11.0 - (11.5-11.0) = 10.5
        self.assertAlmostEqual(result[0]['start_time'], 10.5, places=1)
        self.assertAlmostEqual(result[0]['end_time'], 11.5, places=1)
        
        # Word 2 (world): middle=12.0, start=(11.0+12.0)/2=11.5, end=12.0+(12.0-11.0)/2=12.5
        self.assertAlmostEqual(result[1]['start_time'], 11.5, places=1)
        self.assertAlmostEqual(result[1]['end_time'], 12.5, places=1)

    def test_bracket_format_end_times(self):
        """Test that end times are calculated correctly for bracket format"""
        transcription = "[00:01.00] Hello [00:02.00] world [00:03.00] test"
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 3)
        # With non-overlapping logic:
        # Word 1: middle=1.0, end=(1.0+2.0)/2=1.5
        self.assertAlmostEqual(result[0]['end_time'], 1.5, places=1)
        
        # Word 2: middle=2.0, end=(2.0+3.0)/2=2.5
        self.assertAlmostEqual(result[1]['end_time'], 2.5, places=1)
        
        # Word 3: middle=3.0, end=3.0+(3.0-2.0)/2=3.5
        self.assertAlmostEqual(result[2]['end_time'], 3.5, places=1)

    def test_bracket_format_single_word(self):
        """Test single word with bracket timestamp"""
        transcription = "[00:01.23] Hello"
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'Hello')
        # Single word: middle=1.23, start=1.23-0.25=0.98, end=1.23+0.25=1.48 (0.5s total duration)
        self.assertAlmostEqual(result[0]['start_time'], 0.98, places=2)
        self.assertAlmostEqual(result[0]['end_time'], 1.48, places=2)

    def test_bracket_format_with_punctuation(self):
        """Test bracket format with punctuation"""
        transcription = "[00:01.00] Hello, [00:02.00] world! [00:03.00] How [00:04.00] are [00:05.00] you?"
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]['text'], 'Hello,')
        self.assertEqual(result[1]['text'], 'world!')
        self.assertEqual(result[4]['text'], 'you?')

    def test_no_timestamps_fallback(self):
        """Test fallback to sentence-based timing when no timestamps found"""
        transcription = "This is a test without any timestamps."
        
        result = process_transcription_to_srt(transcription)
        
        # Should create sentence-based entries
        self.assertGreater(len(result), 0)
        # All entries should have reasonable timing
        for entry in result:
            self.assertGreater(entry['end_time'], entry['start_time'])

    def test_empty_transcription(self):
        """Test handling of empty transcription"""
        transcription = ""
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 0)

    def test_whitespace_only_transcription(self):
        """Test handling of whitespace-only transcription"""
        transcription = "   \n\n   \t  "
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 0)

    def test_bracket_format_with_minutes(self):
        """Test bracket format with minutes"""
        transcription = "[01:23.45] Hello [02:34.56] world"
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 2)
        # With non-overlapping logic:
        # Word 1 (Hello): middle=83.45, end=(83.45+154.56)/2=119.005
        # For first word, start = 83.45 - (119.005-83.45) = 47.895
        self.assertAlmostEqual(result[0]['start_time'], 47.895, places=2)
        self.assertAlmostEqual(result[0]['end_time'], 119.005, places=2)
        
        # Word 2 (world): middle=154.56, start=(83.45+154.56)/2=119.005
        # end=154.56+(154.56-83.45)/2=190.115
        self.assertAlmostEqual(result[1]['start_time'], 119.005, places=2)
        self.assertAlmostEqual(result[1]['end_time'], 190.115, places=2)


class TestTimestampDetection(unittest.TestCase):
    """Test cases for timestamp detection functionality"""
    
    def test_has_timestamps_mistral_format(self):
        """Test detection of Mistral timestamp format"""
        # Test single timestamp
        transcription_with_timestamps = "Hello [ 0m1s122ms ] world [ 0m2s456ms ]"
        self.assertTrue(has_timestamps(transcription_with_timestamps))
        
        # Test range format
        transcription_with_range = "[ 0m0s694ms - 0m3s34ms ] This is a test"
        self.assertTrue(has_timestamps(transcription_with_range))
        
        # Test without timestamps
        transcription_without_timestamps = "Hello world this is just text"
        self.assertFalse(has_timestamps(transcription_without_timestamps))
    
    def test_has_timestamps_bracket_format(self):
        """Test detection of bracket timestamp format"""
        # Test bracket format
        transcription_with_brackets = "[00:01.23] Hello [00:02.45] world"
        self.assertTrue(has_timestamps(transcription_with_brackets))
        
        # Test simple timestamp format
        transcription_with_simple = "0:01.23 Hello 0:02.45 world"
        self.assertTrue(has_timestamps(transcription_with_simple))
        
        # Test mixed content
        transcription_mixed = "Some text [00:01.23] with timestamps"
        self.assertTrue(has_timestamps(transcription_mixed))
    
    def test_has_timestamps_edge_cases(self):
        """Test edge cases for timestamp detection"""
        # Empty string
        self.assertFalse(has_timestamps(""))
        
        # Whitespace only
        self.assertFalse(has_timestamps("   \n\t  "))
        
        # False positives (should not match)
        false_positive = "This has [brackets] but no timestamps"
        self.assertFalse(has_timestamps(false_positive))
        
        # Malformed timestamps
        malformed = "[ invalid timestamp format ]"
        self.assertFalse(has_timestamps(malformed))


class TestRetryLogic(unittest.TestCase):
    """Test cases for retry logic in transcription"""
    
    def setUp(self):
        """Set up test environment"""
        self.transcriber = MistralAudioTranscriber("test_api_key")
    
    @patch('transcribe_audio.MistralAudioTranscriber.upload_audio_file')
    @patch('transcribe_audio.MistralAudioTranscriber.get_signed_url')
    @patch('transcribe_audio.MistralAudioTranscriber.transcribe_audio')
    @patch('transcribe_audio.time.sleep')  # Mock sleep to speed up tests
    def test_retry_success_on_second_attempt(self, mock_sleep, mock_transcribe, mock_signed_url, mock_upload):
        """Test successful retry on second attempt"""
        # Setup mocks
        mock_upload.return_value = "test_file_id"
        mock_signed_url.return_value = "test_signed_url"
        
        # First call returns transcription without timestamps, second call with timestamps
        mock_transcribe.side_effect = [
            "Just plain text without any timestamps",  # First attempt fails
            "Hello [ 0m1s122ms ] world [ 0m2s456ms ]"   # Second attempt succeeds
        ]
        
        # Call transcribe_segment
        result = self.transcriber.transcribe_segment("test_audio.mp3", max_retries=3)
        
        # Verify result
        self.assertEqual(result, "Hello [ 0m1s122ms ] world [ 0m2s456ms ]")
        self.assertEqual(mock_transcribe.call_count, 2)
        mock_sleep.assert_called_once_with(2)  # Should sleep once between attempts
    
    @patch('transcribe_audio.MistralAudioTranscriber.upload_audio_file')
    @patch('transcribe_audio.MistralAudioTranscriber.get_signed_url')
    @patch('transcribe_audio.MistralAudioTranscriber.transcribe_audio')
    @patch('transcribe_audio.time.sleep')
    def test_retry_all_attempts_fail(self, mock_sleep, mock_transcribe, mock_signed_url, mock_upload):
        """Test behavior when all retry attempts fail to produce timestamps"""
        # Setup mocks
        mock_upload.return_value = "test_file_id"
        mock_signed_url.return_value = "test_signed_url"
        
        # All attempts return transcription without timestamps
        mock_transcribe.return_value = "Just plain text without any timestamps"
        
        # Call transcribe_segment with 3 retries
        result = self.transcriber.transcribe_segment("test_audio.mp3", max_retries=3)
        
        # Verify result - should return the transcription even without timestamps
        self.assertEqual(result, "Just plain text without any timestamps")
        self.assertEqual(mock_transcribe.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)  # Should sleep between attempts
    
    @patch('transcribe_audio.MistralAudioTranscriber.upload_audio_file')
    @patch('transcribe_audio.MistralAudioTranscriber.get_signed_url')
    @patch('transcribe_audio.MistralAudioTranscriber.transcribe_audio')
    def test_retry_success_on_first_attempt(self, mock_transcribe, mock_signed_url, mock_upload):
        """Test no retry needed when first attempt succeeds"""
        # Setup mocks
        mock_upload.return_value = "test_file_id"
        mock_signed_url.return_value = "test_signed_url"
        
        # First attempt succeeds with timestamps
        mock_transcribe.return_value = "Hello [ 0m1s122ms ] world [ 0m2s456ms ]"
        
        # Call transcribe_segment
        result = self.transcriber.transcribe_segment("test_audio.mp3", max_retries=3)
        
        # Verify result
        self.assertEqual(result, "Hello [ 0m1s122ms ] world [ 0m2s456ms ]")
        self.assertEqual(mock_transcribe.call_count, 1)  # Should only call once
    
    @patch('transcribe_audio.MistralAudioTranscriber.upload_audio_file')
    @patch('transcribe_audio.MistralAudioTranscriber.get_signed_url')
    @patch('transcribe_audio.MistralAudioTranscriber.transcribe_audio')
    @patch('transcribe_audio.time.sleep')
    def test_retry_with_api_errors(self, mock_sleep, mock_transcribe, mock_signed_url, mock_upload):
        """Test retry behavior when API errors occur"""
        # Setup mocks
        mock_upload.return_value = "test_file_id"
        mock_signed_url.return_value = "test_signed_url"
        
        # First two attempts raise exceptions, third succeeds
        mock_transcribe.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            "Hello [ 0m1s122ms ] world [ 0m2s456ms ]"
        ]
        
        # Call transcribe_segment
        result = self.transcriber.transcribe_segment("test_audio.mp3", max_retries=3)
        
        # Verify result
        self.assertEqual(result, "Hello [ 0m1s122ms ] world [ 0m2s456ms ]")
        self.assertEqual(mock_transcribe.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)  # Should sleep after each error
        
        # Verify sleep was called with 5 seconds for errors
        mock_sleep.assert_any_call(5)
    
    @patch('transcribe_audio.MistralAudioTranscriber.upload_audio_file')
    @patch('transcribe_audio.MistralAudioTranscriber.get_signed_url')
    @patch('transcribe_audio.MistralAudioTranscriber.transcribe_audio')
    @patch('transcribe_audio.time.sleep')
    def test_retry_all_attempts_error(self, mock_sleep, mock_transcribe, mock_signed_url, mock_upload):
        """Test behavior when all retry attempts result in errors"""
        # Setup mocks
        mock_upload.return_value = "test_file_id"
        mock_signed_url.return_value = "test_signed_url"
        
        # All attempts raise exceptions
        mock_transcribe.side_effect = Exception("Persistent API Error")
        
        # Call transcribe_segment - should raise exception
        with self.assertRaises(Exception) as context:
            self.transcriber.transcribe_segment("test_audio.mp3", max_retries=3)
        
        self.assertEqual(str(context.exception), "Persistent API Error")
        self.assertEqual(mock_transcribe.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
    
    @patch('transcribe_audio.MistralAudioTranscriber.upload_audio_file')
    @patch('transcribe_audio.MistralAudioTranscriber.get_signed_url')
    @patch('transcribe_audio.MistralAudioTranscriber.transcribe_audio')
    def test_retry_with_single_attempt(self, mock_transcribe, mock_signed_url, mock_upload):
        """Test behavior with max_retries=1"""
        # Setup mocks
        mock_upload.return_value = "test_file_id"
        mock_signed_url.return_value = "test_signed_url"
        
        # Single attempt without timestamps
        mock_transcribe.return_value = "Just plain text without any timestamps"
        
        # Call transcribe_segment with only 1 retry
        result = self.transcriber.transcribe_segment("test_audio.mp3", max_retries=1)
        
        # Verify result
        self.assertEqual(result, "Just plain text without any timestamps")
        self.assertEqual(mock_transcribe.call_count, 1)
    
    @patch('transcribe_audio.MistralAudioTranscriber.upload_audio_file')
    @patch('transcribe_audio.MistralAudioTranscriber.get_signed_url')
    @patch('transcribe_audio.MistralAudioTranscriber.transcribe_audio')
    @patch('transcribe_audio.time.sleep')
    def test_retry_mixed_failures_and_success(self, mock_sleep, mock_transcribe, mock_signed_url, mock_upload):
        """Test mixed scenario with errors and missing timestamps before success"""
        # Setup mocks
        mock_upload.return_value = "test_file_id"
        mock_signed_url.return_value = "test_signed_url"
        
        # Mixed failures: error, no timestamps, success
        mock_transcribe.side_effect = [
            Exception("API Error"),
            "Plain text without timestamps",
            "Hello [ 0m1s122ms ] world [ 0m2s456ms ]"
        ]
        
        # Call transcribe_segment
        result = self.transcriber.transcribe_segment("test_audio.mp3", max_retries=3)
        
        # Verify result
        self.assertEqual(result, "Hello [ 0m1s122ms ] world [ 0m2s456ms ]")
        self.assertEqual(mock_transcribe.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)
        
        # Verify different sleep times for error vs missing timestamps
        mock_sleep.assert_any_call(5)  # Error sleep
        mock_sleep.assert_any_call(2)  # Missing timestamp sleep


class TestSRTToJSON(unittest.TestCase):
    """Test SRT to JSON conversion functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_srt_file(self, content: str, filename: str = "test.srt") -> str:
        """Helper to create test SRT files"""
        srt_path = os.path.join(self.temp_dir, filename)
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return srt_path
    
    def test_word_level_srt_to_json(self):
        """Test conversion of word-level SRT to JSON format"""
        srt_content = """1
00:00:05,000 --> 00:00:07,240
Forrettskjøl!

2
00:00:07,240 --> 00:00:07,660
Du

3
00:00:07,660 --> 00:00:08,240
erier

4
00:00:08,240 --> 00:00:08,640
ass!

5
00:00:08,640 --> 00:00:09,420
Det

6
00:00:09,420 --> 00:00:09,640
var

7
00:00:09,640 --> 00:00:09,880
helt

8
00:00:09,880 --> 00:00:10,260
rått

9
00:00:10,260 --> 00:00:10,740
ass!

"""
        srt_path = self.create_test_srt_file(srt_content)
        result = srt_to_json(srt_path)
        
        # Check structure
        self.assertIn('text', result)
        self.assertIn('segments', result)
        
        # Check full text
        expected_text = "Forrettskjøl! Du erier ass! Det var helt rått ass!"
        self.assertEqual(result['text'], expected_text)
        
        # Check segments
        self.assertEqual(len(result['segments']), 1)  # Should group into one segment
        segment = result['segments'][0]
        
        self.assertEqual(segment['text'], expected_text)
        self.assertEqual(segment['start'], 5.0)
        self.assertEqual(segment['end'], 10.74)
        self.assertEqual(segment['id'], 0)
        
        # Check words
        self.assertEqual(len(segment['words']), 9)
        first_word = segment['words'][0]
        self.assertEqual(first_word['text'], 'Forrettskjøl!')
        self.assertEqual(first_word['start'], 5.0)
        self.assertEqual(first_word['end'], 7.24)
    
    def test_sentence_level_srt_to_json(self):
        """Test conversion of sentence-level SRT to JSON format"""
        srt_content = """1
00:00:05,000 --> 00:00:10,740
Forrettskjøl! Du erier ass! Det var helt rått ass!

2
00:00:10,740 --> 00:00:14,080
Nå har vi bare en time å sove før

3
00:00:14,080 --> 00:00:15,920
vi må reise. Ja ja, det er alt.

"""
        srt_path = self.create_test_srt_file(srt_content)
        result = srt_to_json(srt_path)
        
        # Check structure
        self.assertIn('text', result)
        self.assertIn('segments', result)
        
        # Check segments
        self.assertEqual(len(result['segments']), 3)
        
        # Check first segment
        segment1 = result['segments'][0]
        self.assertEqual(segment1['text'], 'Forrettskjøl! Du erier ass! Det var helt rått ass!')
        self.assertEqual(segment1['start'], 5.0)
        self.assertEqual(segment1['end'], 10.74)
        self.assertEqual(len(segment1['words']), 9)  # Should split into words
        
        # Check word timing estimation - use more lenient checks
        first_word = segment1['words'][0]
        self.assertEqual(first_word['text'], 'Forrettskjøl!')
        self.assertAlmostEqual(first_word['start'], 5.0, places=1)
        
        second_word = segment1['words'][1]
        self.assertEqual(second_word['text'], 'Du')
        # Check that second word starts after or at the same time as first word ends
        self.assertGreaterEqual(second_word['start'], first_word['end'])
    
    def test_empty_srt_file(self):
        """Test handling of empty SRT file"""
        srt_path = self.create_test_srt_file("")
        result = srt_to_json(srt_path)
        
        self.assertEqual(result['text'], "")
        self.assertEqual(result['segments'], [])
    
    def test_malformed_srt_entries(self):
        """Test handling of malformed SRT entries"""
        srt_content = """1
00:00:05,000 --> 00:00:07,240
Valid entry

invalid entry without timing

3
invalid timing format
Another text

4
00:00:10,000 --> 00:00:12,000
Another valid entry

"""
        srt_path = self.create_test_srt_file(srt_content)
        result = srt_to_json(srt_path)
        
        # Should only process valid entries
        self.assertEqual(len(result['segments']), 2)
        self.assertIn('Valid entry', result['text'])
        self.assertIn('Another valid entry', result['text'])
    
    def test_convert_srt_to_json_function(self):
        """Test the standalone convert_srt_to_json function"""
        srt_content = """1
00:00:00,000 --> 00:00:02,000
Test content

"""
        srt_path = self.create_test_srt_file(srt_content)
        json_path = os.path.join(self.temp_dir, "output.json")
        
        # Test successful conversion
        result = convert_srt_to_json(srt_path, json_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(json_path))
        
        # Verify JSON content
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        self.assertEqual(json_data['text'], 'Test content')
        self.assertEqual(len(json_data['segments']), 1)
    
    def test_file_not_found_error(self):
        """Test error handling for non-existent SRT file"""
        non_existent_path = os.path.join(self.temp_dir, "nonexistent.srt")
        
        with self.assertRaises(FileNotFoundError):
            srt_to_json(non_existent_path)
    
    def test_unicode_content(self):
        """Test handling of Unicode content in SRT files"""
        srt_content = """1
00:00:00,000 --> 00:00:02,000
Forrettskjøl! Café naïve résumé

2
00:00:02,000 --> 00:00:04,000
测试中文内容 🎵

"""
        srt_path = self.create_test_srt_file(srt_content)
        result = srt_to_json(srt_path)
        
        self.assertIn('Forrettskjøl!', result['text'])
        self.assertIn('Café', result['text'])
        self.assertIn('测试中文内容', result['text'])
        self.assertIn('🎵', result['text'])
    
    def test_segment_gap_detection(self):
        """Test detection of gaps between segments for word-level grouping"""
        # Create word-level SRT with a gap
        srt_content = """1
00:00:00,000 --> 00:00:01,000
First

2
00:00:01,000 --> 00:00:02,000
word

3
00:00:05,000 --> 00:00:06,000
Second

4
00:00:06,000 --> 00:00:07,000
segment

"""
        srt_path = self.create_test_srt_file(srt_content)
        result = srt_to_json(srt_path)
        
        # Should create two segments due to the gap
        self.assertEqual(len(result['segments']), 2)
        
        segment1 = result['segments'][0]
        self.assertEqual(segment1['text'], 'First word')
        self.assertEqual(len(segment1['words']), 2)
        
        segment2 = result['segments'][1]
        self.assertEqual(segment2['text'], 'Second segment')
        self.assertEqual(len(segment2['words']), 2)


if __name__ == '__main__':
    unittest.main()
