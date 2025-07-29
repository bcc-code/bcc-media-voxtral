#!/usr/bin/env python3
"""
Unit tests for transcribe_audio.py - Word-level bracket timestamps only
"""

import unittest
from unittest.mock import patch
import sys
import os

# Add the current directory to the path so we can import transcribe_audio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcribe_audio import process_transcription_to_srt


class TestProcessTranscriptionToSrt(unittest.TestCase):
    """Test cases for process_transcription_to_srt function - bracket timestamps only"""

    def test_word_level_timestamps_bracket_format(self):
        """Test word-level timestamps in [HH:MM.SS] format"""
        transcription = "[00:01.23] Hello [00:02.45] world [00:03.67] test"
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['text'], 'Hello')
        self.assertAlmostEqual(result[0]['start_time'], 1.23, places=2)
        self.assertEqual(result[1]['text'], 'world')
        self.assertAlmostEqual(result[1]['start_time'], 2.45, places=2)
        self.assertEqual(result[2]['text'], 'test')
        self.assertAlmostEqual(result[2]['start_time'], 3.67, places=2)

    def test_bracket_format_with_offset(self):
        """Test bracket format with segment offset"""
        transcription = "[00:01.00] Hello [00:02.00] world"
        segment_offset = 10.0
        
        result = process_transcription_to_srt(transcription, segment_offset_seconds=segment_offset)
        
        self.assertEqual(len(result), 2)
        # First word should start at offset + original time
        self.assertAlmostEqual(result[0]['start_time'], 11.0, places=1)
        self.assertAlmostEqual(result[1]['start_time'], 12.0, places=1)

    def test_bracket_format_end_times(self):
        """Test that end times are calculated correctly for bracket format"""
        transcription = "[00:01.00] Hello [00:02.00] world [00:03.00] test"
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 3)
        # End time should be next word's timestamp
        self.assertAlmostEqual(result[0]['end_time'], 2.0, places=1)
        self.assertAlmostEqual(result[1]['end_time'], 3.0, places=1)
        # Last word gets 0.5 seconds added
        self.assertAlmostEqual(result[2]['end_time'], 3.5, places=1)

    def test_bracket_format_single_word(self):
        """Test bracket format with single word"""
        transcription = "[00:01.23] Hello"
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'Hello')
        self.assertAlmostEqual(result[0]['start_time'], 1.23, places=2)
        # Single word should get 0.5 seconds duration
        self.assertAlmostEqual(result[0]['end_time'], 1.73, places=2)

    def test_bracket_format_with_punctuation(self):
        """Test bracket format preserves punctuation in words"""
        transcription = "[00:01.00] Hello, [00:02.00] world! [00:03.00] How [00:04.00] are [00:05.00] you?"
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]['text'], 'Hello,')
        self.assertEqual(result[1]['text'], 'world!')
        self.assertEqual(result[2]['text'], 'How')
        self.assertEqual(result[3]['text'], 'are')
        self.assertEqual(result[4]['text'], 'you?')

    def test_empty_transcription(self):
        """Test handling of empty transcription"""
        result = process_transcription_to_srt("")
        self.assertEqual(len(result), 0)

    def test_whitespace_only_transcription(self):
        """Test handling of whitespace-only transcription"""
        result = process_transcription_to_srt("   \n\t  ")
        self.assertEqual(len(result), 0)

    def test_no_timestamps_found(self):
        """Test fallback when no bracket timestamps are found"""
        transcription = "Hello world test"
        
        result = process_transcription_to_srt(transcription)
        
        # Should fall back to sentence-based timing
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'Hello world test')

    def test_malformed_timestamps_ignored(self):
        """Test that malformed timestamps are ignored and fallback is used"""
        transcription = "Hello [invalid] world [also:invalid] test."
        
        result = process_transcription_to_srt(transcription)
        
        # Should fall back to sentence-based timing
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'Hello [invalid] world [also:invalid] test')

    def test_return_format_consistency(self):
        """Test that return format is consistent for bracket timestamps"""
        transcription = "[00:01.23] Hello [00:02.45] world"
        
        result = process_transcription_to_srt(transcription)
        
        for entry in result:
            # Check required fields for word-level timestamps
            self.assertIn('index', entry)
            self.assertIn('start_time', entry)
            self.assertIn('end_time', entry)
            self.assertIn('text', entry)
            
            # Check data types
            self.assertIsInstance(entry['index'], int)
            self.assertIsInstance(entry['start_time'], (int, float))
            self.assertIsInstance(entry['end_time'], (int, float))
            self.assertIsInstance(entry['text'], str)
            
            # Check logical constraints
            self.assertGreater(entry['index'], 0)
            self.assertGreaterEqual(entry['end_time'], entry['start_time'])
            self.assertGreater(len(entry['text'].strip()), 0)

    @patch('builtins.print')
    def test_debug_output_suppressed_in_tests(self, mock_print):
        """Test that debug output doesn't interfere with tests"""
        transcription = "[00:01.23] Hello [00:02.45] world"
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 2)
        # Debug prints should have been called
        self.assertTrue(mock_print.called)

    def test_minutes_and_seconds_format(self):
        """Test bracket format with minutes and seconds"""
        transcription = "[01:23.45] Hello [02:34.56] world"
        
        result = process_transcription_to_srt(transcription)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], 'Hello')
        self.assertAlmostEqual(result[0]['start_time'], 83.45, places=2)  # 1*60 + 23.45
        self.assertEqual(result[1]['text'], 'world')
        self.assertAlmostEqual(result[1]['start_time'], 154.56, places=2)  # 2*60 + 34.56


if __name__ == '__main__':
    unittest.main()
