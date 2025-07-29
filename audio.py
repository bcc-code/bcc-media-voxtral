import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from pedalboard.io import AudioFile
import logging


class AudioConverter:
    """Handles audio format conversion and channel processing"""

    @staticmethod
    def get_supported_formats():
        """Return list of supported audio/video formats"""
        return ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma',
                '.mp4', '.avi', '.mov', '.mkv', '.webm', '.3gp']

    @staticmethod
    def needs_conversion(file_path: str) -> bool:
        """Check if file needs conversion to MP3"""
        return Path(file_path).suffix.lower() != '.mp3'

    @staticmethod
    def convert_to_mp3(input_path: str, output_path: str) -> str:
        """Convert audio file to MP3 with optimized settings"""
        logging.info(f"Converting {Path(input_path).name} to MP3 with optimized settings...")

        try:
            with AudioFile(input_path) as f:
                # Read the audio data
                audio = f.read(f.frames)
                original_sample_rate = f.samplerate
                num_channels = audio.shape[0]

                logging.info(f"Original: {num_channels} channels, {original_sample_rate}Hz")

                # Handle channel processing
                if num_channels == 1:
                    # Already mono
                    processed_audio = audio
                else:
                    # Stereo to mono: average the two channels
                    processed_audio = np.mean(audio[:2], axis=0, keepdims=True)

                # Write as MP3 with optimized settings
                with AudioFile(output_path, 'w', original_sample_rate, 1) as out_f:
                    out_f.write(processed_audio)

                # Check output file size
                output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logging.info(f"Converted to: 1 channel (mono), {original_sample_rate}Hz, {output_size_mb:.2f}MB")
                return output_path

        except Exception as e:
            raise Exception(f"Error converting audio file: {e}")


class AudioSplitter:
    """Handles audio file splitting using pedalboard"""

    def __init__(self, max_duration_minutes: int = 25, overlap_seconds: int = 30):
        self.max_duration_seconds = max_duration_minutes * 60
        self.overlap_seconds = overlap_seconds

    def split_audio(self, input_path: str, temp_dir: str) -> List[Dict[str, Any]]:
        """Split audio file into segments if needed"""
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        logging.info(f"Input file size: {file_size_mb:.2f} MB")

        # Force split if file is too large (>20MB) regardless of duration
        if file_size_mb > 20:
            logging.warning(
                f"File is {file_size_mb:.2f} MB, which exceeds safe limit. Will split regardless of duration.")
            force_split = True
        else:
            force_split = False

        # Get audio duration
        try:
            with AudioFile(input_path) as f:
                audio = f.read(f.frames)
                total_duration = f.frames / f.samplerate
                sample_rate = f.samplerate

            logging.info(f"Total audio duration: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")

            if not force_split and total_duration <= self.max_duration_seconds:
                # File is short enough, return as single segment
                return [{
                    'path': input_path,
                    'start_time': 0,
                    'end_time': total_duration,
                    'duration': total_duration
                }]

            # Calculate segment parameters
            if force_split and total_duration <= self.max_duration_seconds:
                # Short but large file - split based on file size
                mb_per_second = file_size_mb / total_duration
                target_segments = max(2, int(file_size_mb / 15))  # Target ~15MB per segment
                effective_segment_duration = total_duration / target_segments
                logging.info(
                    f"Splitting short but large file into {target_segments} segments of ~{effective_segment_duration:.1f}s each")
            else:
                # Long file - split based on duration
                segment_count = int(np.ceil(total_duration / self.max_duration_seconds))
                effective_segment_duration = total_duration / segment_count

                # For large files, also consider file size
                if file_size_mb > 10:
                    mb_per_second = file_size_mb / total_duration
                    logging.info(
                        f"Calculated segment duration: {effective_segment_duration:.1f}s (based on {mb_per_second:.3f} MB/s)")

            segment_count = int(np.ceil(total_duration / effective_segment_duration))
            logging.info(f"Splitting into {segment_count} segments with {self.overlap_seconds}s overlap")

            segments = []

            for i in range(segment_count):
                # Calculate start and end times
                start_time = i * effective_segment_duration
                end_time = min(start_time + effective_segment_duration + self.overlap_seconds, total_duration)

                # Adjust for actual audio boundaries
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)

                segment_audio = audio[:, start_sample:end_sample]

                # Handle channel conversion to mono (same logic as AudioConverter)
                num_channels = segment_audio.shape[0]
                if num_channels == 1:
                    # Already mono
                    processed_segment_audio = segment_audio
                elif num_channels == 2:
                    # Stereo to mono: average the two channels
                    processed_segment_audio = np.mean(segment_audio[:2], axis=0, keepdims=True)
                else:
                    # Multiple channels: use first two and downmix to mono
                    processed_segment_audio = np.mean(segment_audio[:2], axis=0, keepdims=True)

                segment_filename = f"segment_{i + 1:03d}.mp3"
                segment_path = os.path.join(temp_dir, segment_filename)

                # Keep original sample rate for better quality
                final_sample_rate = sample_rate

                with AudioFile(segment_path, 'w', final_sample_rate, 1) as f:
                    f.write(processed_segment_audio)

                # Check segment file size
                segment_size_mb = os.path.getsize(segment_path) / (1024 * 1024)
                logging.info(f"Created segment {i + 1}/{segment_count}: {segment_filename} "
                             f"({start_time:.1f}s - {end_time:.1f}s, {segment_size_mb:.2f}MB)")

                segments.append({
                    'path': segment_path,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })

        except Exception as e:
            raise Exception(f"Error splitting audio file: {e}")

        return segments