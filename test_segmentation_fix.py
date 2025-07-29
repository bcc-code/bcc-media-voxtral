#!/usr/bin/env python3
"""
Test script to verify the segmentation fix for file size and overlap issues.
"""

import tempfile
import os
import numpy as np
from pedalboard.io import AudioFile
from audio import AudioSplitter

def create_test_audio(duration_seconds, sample_rate=44100, output_path=None):
    """Create a test audio file with specified duration"""
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.mp3')
    
    # Generate a simple sine wave
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate))
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    audio_data = audio_data.reshape(1, -1)  # Make it mono
    
    with AudioFile(output_path, 'w', sample_rate, 1) as f:
        f.write(audio_data)
    
    return output_path

def test_segmentation():
    """Test the segmentation logic"""
    print("Testing segmentation fix...")
    
    # Create a test audio file that would be large (simulate ~60MB file)
    # We'll create a shorter file but the logic should work the same
    test_duration = 3600  # 1 hour
    temp_audio = create_test_audio(test_duration)
    
    try:
        # Get file size
        file_size_mb = os.path.getsize(temp_audio) / (1024 * 1024)
        print(f"Test audio file: {file_size_mb:.2f} MB, {test_duration} seconds")
        
        # Test with AudioSplitter
        splitter = AudioSplitter(max_duration_minutes=25, overlap_seconds=30)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            segments = splitter.split_audio(temp_audio, temp_dir)
            
            print(f"\nSegmentation results:")
            print(f"Number of segments: {len(segments)}")
            
            # Assert we have segments
            assert len(segments) > 1, "Should create multiple segments for large file"
            
            for i, segment in enumerate(segments):
                segment_size_mb = os.path.getsize(segment['path']) / (1024 * 1024)
                print(f"Segment {i+1}: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s, "
                      f"Duration: {segment['duration']:.1f}s, Size: {segment_size_mb:.2f}MB")
                
                # Assert segment is under 20MB
                assert segment_size_mb <= 20.0, f"Segment {i+1} exceeds 20MB limit: {segment_size_mb:.2f}MB"
            
            # Check overlap
            print(f"\nOverlap verification:")
            for i in range(len(segments) - 1):
                current_end = segments[i]['end_time']
                next_start = segments[i+1]['start_time']
                overlap = current_end - next_start
                print(f"Segments {i+1}-{i+2}: overlap = {overlap:.1f}s")
                
                # Assert overlap is approximately 30s (allow 5s tolerance)
                assert 25 <= overlap <= 35, f"Unexpected overlap: {overlap:.1f}s (expected ~30s)"
            
            print(f"\nAll assertions passed! Segmentation is working correctly.")
    
    finally:
        # Clean up
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

if __name__ == "__main__":
    test_segmentation()
