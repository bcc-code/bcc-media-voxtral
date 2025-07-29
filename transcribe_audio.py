#!/usr/bin/env python3
"""
Audio Transcription Script using Mistral AI API
Transcribes audio files and outputs in SRT format
Handles files longer than 25 minutes by splitting them into segments
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import re

# Optional: pedalboard for audio manipulation
try:
    from pedalboard import Pedalboard, Reverb, Compressor, Resample
    from pedalboard.io import AudioFile
    import numpy as np
except ImportError:
    raise ImportError("pedalboard is required for audio preprocessing")


class SRTFormatter:
    """Handles SRT subtitle formatting"""
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    @staticmethod
    def create_srt_entry(index: int, start_time: float, end_time: float, text: str) -> str:
        """Create a single SRT entry"""
        start_formatted = SRTFormatter.format_time(start_time)
        end_formatted = SRTFormatter.format_time(end_time)
        return f"{index}\n{start_formatted} --> {end_formatted}\n{text}\n\n"


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
        """Convert audio/video file to MP3 with mono downmix and optimized for smaller file size"""
        print(f"Converting {Path(input_path).name} to MP3 with optimized settings...")
        
        try:
            with AudioFile(input_path) as f:
                # Read the audio data
                audio = f.read(f.frames)
                original_sample_rate = f.samplerate
                num_channels = audio.shape[0]
                
                print(f"Original: {num_channels} channels, {original_sample_rate}Hz")
                
                # Handle channel processing
                if num_channels == 1:
                    # Already mono
                    processed_audio = audio
                elif num_channels == 2:
                    # Stereo to mono: average the two channels
                    processed_audio = np.mean(audio[:2], axis=0, keepdims=True)
                else:
                    # Multiple channels: use first two and downmix to mono
                    processed_audio = np.mean(audio[:2], axis=0, keepdims=True)
                    print(f"Downmixed from {num_channels} channels to mono using first 2 channels")
                
                # Keep original sample rate for better quality
                final_sample_rate = original_sample_rate
                print(f"Preserving original quality: {original_sample_rate}Hz")
                
                # Write as MP3 with optimized settings
                with AudioFile(output_path, 'w', final_sample_rate, 1) as out_f:
                    out_f.write(processed_audio)
                
                # Check output file size
                output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"Converted to: 1 channel (mono), {final_sample_rate}Hz, {output_size_mb:.2f}MB")
                return output_path
                
        except Exception as e:
            raise Exception(f"Failed to convert audio file: {str(e)}")


class AudioSplitter:
    """Handles audio file splitting using pedalboard"""
    
    def __init__(self, max_duration_minutes: int = 25, overlap_seconds: int = 30):
        self.max_duration_seconds = max_duration_minutes * 60
        self.overlap_seconds = overlap_seconds
        
    def split_audio(self, input_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """Split audio file into segments with overlap"""
        # Check file size first
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        print(f"Input file size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 18:  # Leave some margin below 20MB limit
            print(f"File is {file_size_mb:.2f} MB, which exceeds safe limit. Will split regardless of duration.")
            force_split = True
        else:
            force_split = False
        
        segments = []
        
        with AudioFile(input_path) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate
            total_duration = len(audio[0]) / sample_rate
            
            print(f"Total audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
            
            if total_duration <= self.max_duration_seconds and not force_split:
                # No need to split
                return [{
                    'path': input_path,
                    'start_time': 0,
                    'duration': total_duration,
                    'overlap_start': 0,
                    'overlap_end': 0
                }]
            
            # Calculate segments with overlap
            if force_split and total_duration <= self.max_duration_seconds:
                # File is short but large - split into smaller time segments
                target_segments = max(2, int(file_size_mb / 15))  # Aim for ~15MB segments
                effective_segment_duration = total_duration / target_segments
                print(f"Splitting short but large file into {target_segments} segments of ~{effective_segment_duration:.1f}s each")
            else:
                # Calculate segment duration based on file size to ensure <20MB segments
                # Estimate: file_size_mb / total_duration gives MB per second
                mb_per_second = file_size_mb / total_duration
                target_mb_per_segment = 18  # Target 18MB to leave safety margin
                max_seconds_per_segment = target_mb_per_segment / mb_per_second
                
                # Don't exceed the original max duration, but ensure we stay under size limit
                effective_segment_duration = min(self.max_duration_seconds - self.overlap_seconds, max_seconds_per_segment - self.overlap_seconds)
                
                print(f"Calculated segment duration: {effective_segment_duration:.1f}s (based on {mb_per_second:.3f} MB/s)")
            
            segment_count = int(np.ceil((total_duration - self.overlap_seconds) / effective_segment_duration))
            print(f"Splitting into {segment_count} segments with {self.overlap_seconds}s overlap")
            
            for i in range(segment_count):
                # Calculate start and end times
                start_time = i * effective_segment_duration
                end_time = min(start_time + effective_segment_duration + self.overlap_seconds, total_duration)
                
                # Adjust for actual audio boundaries
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                segment_audio = audio[:, start_sample:end_sample]
                
                segment_filename = f"segment_{i+1:03d}.mp3"
                segment_path = os.path.join(output_dir, segment_filename)
                
                # Keep original sample rate for better quality
                final_sample_rate = sample_rate
                
                with AudioFile(segment_path, 'w', final_sample_rate, 1) as f:
                    f.write(segment_audio)
                
                # Check segment file size
                segment_size_mb = os.path.getsize(segment_path) / (1024 * 1024)
                print(f"Created segment {i+1}/{segment_count}: {segment_filename} "
                      f"({start_time:.1f}s - {end_time:.1f}s, {segment_size_mb:.2f}MB)")
                
                # Calculate overlap information for timestamp adjustment
                overlap_start = self.overlap_seconds if i > 0 else 0
                overlap_end = self.overlap_seconds if i < segment_count - 1 else 0
                
                segments.append({
                    'path': segment_path,
                    'start_time': start_time,
                    'duration': end_time - start_time,
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'segment_index': i
                })
        
        return segments


class MistralAudioTranscriber:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    
    def upload_audio_file(self, audio_path: str) -> str:
        """Upload audio file to Mistral API and return file ID"""
        print(f"Uploading audio file: {os.path.basename(audio_path)}")
        
        url = f"{self.base_url}/files"
        
        with open(audio_path, 'rb') as audio_file:
            files = {
                'file': audio_file,
                'purpose': (None, 'audio')
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.post(url, headers=headers, files=files)
            
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                print(f"HTTP Error: {e}")
                print(f"Response status code: {response.status_code}")
                print(f"Response headers: {response.headers}")
                print(f"Response body: {response.text}")
                raise
            
            result = response.json()
            file_id = result.get('id')
            print(f"File uploaded successfully. ID: {file_id}")
            return file_id
    
    def get_signed_url(self, file_id: str, expiry_hours: int = 24) -> str:
        """Get signed URL for the uploaded file"""
        print(f"Getting signed URL for file ID: {file_id}")
        
        url = f"{self.base_url}/files/{file_id}/url?expiry={expiry_hours}"
        
        response = requests.get(url, headers=self.headers)
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response body: {response.text}")
            raise
        
        result = response.json()
        signed_url = result.get('url')
        print("Signed URL retrieved successfully")
        return signed_url
    
    def transcribe_audio(self, signed_url: str, custom_prompt: str = "Transcribe the attached audio. Keep the transcript as exact as possible. Provide word level timestamps.") -> str:
        """Send transcription request to Mistral API"""
        print("Sending transcription request...")
        
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": "voxtral-small-2507",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": signed_url,
                                "format": "mp3"
                            }
                        },
                        {
                            "type": "text",
                            "text": custom_prompt
                        }
                    ]
                }
            ]
        }
        
        headers = {
            **self.headers,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response body: {response.text}")
            raise
        
        result = response.json()
        transcription = result['choices'][0]['message']['content']
        print("Transcription completed successfully")
        
        # Debug: Show raw transcription output
        print("\n" + "="*50)
        print("DEBUG: Raw transcription from model:")
        print("="*50)
        print(transcription[:500] + "..." if len(transcription) > 500 else transcription)
        print("="*50 + "\n")
        
        return transcription
    
    def transcribe_segment(self, audio_path: str, language: str = None) -> str:
        """Complete transcription workflow for a single audio segment"""
        file_id = self.upload_audio_file(audio_path)
        signed_url = self.get_signed_url(file_id)
        
        if language:
            custom_prompt = f"Transcribe the attached audio. Keep the transcript as exact as possible. Assume {language} language. Provide word level timestamps."
        else:
            custom_prompt = "Transcribe the attached audio. Keep the transcript as exact as possible. Provide word level timestamps."
        
        transcription = self.transcribe_audio(signed_url, custom_prompt)
        return transcription


def process_transcription_to_srt(transcription: str, segment_offset_seconds: float = 0, word_level: bool = False) -> List[Dict]:
    """
    Process transcription text and convert to SRT entries
    Handles both word-level timestamps and fallback sentence-based timing
    """
    entries = []
    

    # Check if the transcription contains word-level timestamps
    timestamp_patterns = [
        r'(\S+?)(?:,)?\s*\[\s*(\d+m\d+s\d+ms)\s*\]',  # word, [ 0m1s122ms ] or word [ 0m1s122ms ] - word-level format
        r'\[\s*(\d+m\d+s\d+ms)\s*-\s*(\d+m\d+s\d+ms)\s*\]\s*(.+?)(?=\[|$)',  # Mistral format: [ 0m0s694ms - 0m3s34ms ] text
        r'\[(\d{1,2}:\d{2}\.\d{2})\]\s*(\S+)',  # [00:01.23] word
    ]
    
    word_timestamps = []
    
    for pattern in timestamp_patterns:
        matches = re.findall(pattern, transcription, re.DOTALL)
        print(f"DEBUG: Testing pattern: {pattern}")
        print(f"DEBUG: Found {len(matches)} matches")
        if matches and len(matches) > 0:
            print(f"DEBUG: First few matches: {matches[:3]}")
        
        if matches:
            print(f"Found {len(matches)} timestamped segments using pattern: {pattern}")
            
            if pattern.startswith(r'(\S+?)(?:,)?\s*\[\s*(\d+m\d+s\d+ms)'):  # Word-level format: word, [ 0m1s122ms ] or word [ 0m1s122ms ]
                for match in matches:
                    word, time_str = match
                    
                    # Convert Mistral timestamp format to seconds
                    def parse_mistral_time(time_str):
                        # Parse format like "0m3s34ms" or "1m23s456ms"
                        import re
                        time_match = re.match(r'(\d+)m(\d+)s(\d+)ms', time_str)
                        if time_match:
                            minutes, seconds, milliseconds = map(int, time_match.groups())
                            return minutes * 60 + seconds + milliseconds / 1000.0
                        return 0
                    
                    timestamp_seconds = parse_mistral_time(time_str)
                    
                    word_timestamps.append({
                        'word': word.strip(),
                        'timestamp': segment_offset_seconds + timestamp_seconds
                    })
            elif 'm' in pattern and 's' in pattern:  # Mistral format
                for match in matches:
                    start_time_str, end_time_str, text = match
                    
                    # Convert Mistral timestamp format to seconds
                    def parse_mistral_time(time_str):
                        # Parse format like "0m3s34ms" or "1m23s456ms"
                        time_match = re.match(r'(\d+)m(\d+)s(\d+)ms', time_str)
                        if time_match:
                            minutes, seconds, milliseconds = map(int, time_match.groups())
                            return minutes * 60 + seconds + milliseconds / 1000.0
                        return 0
                    
                    start_seconds = parse_mistral_time(start_time_str)
                    end_seconds = parse_mistral_time(end_time_str)
                    
                    word_timestamps.append({
                        'text': text.strip(),
                        'start_timestamp': segment_offset_seconds + start_seconds,
                        'end_timestamp': segment_offset_seconds + end_seconds
                    })
            else:
                # Handle other timestamp formats (existing logic)
                for match in matches:
                    if len(match) == 2:
                        if ':' in match[0]:  # timestamp first
                            time_str, word = match
                        else:  # word first
                            word, time_str = match
                        
                        # Convert timestamp to seconds
                        try:
                            if '.' in time_str:
                                minutes, seconds = time_str.split(':')
                                total_seconds = int(minutes) * 60 + float(seconds)
                            else:
                                total_seconds = float(time_str)
                            
                            word_timestamps.append({
                                'word': word.strip(),
                                'timestamp': segment_offset_seconds + total_seconds
                            })
                        except ValueError:
                            continue
            break  # Use first pattern that matches
    
    if word_timestamps:
        # Process timestamped segments from Mistral
        if 'start_timestamp' in word_timestamps[0]:  # Mistral format with segments
            print(f"Processing {len(word_timestamps)} timestamped segments from Mistral")
            
            for i, segment_data in enumerate(word_timestamps):
                text = segment_data['text']
                start_time = segment_data['start_timestamp']
                end_time = segment_data['end_timestamp']
                
                # Always split segment text into words and create individual entries
                words = text.split()
                segment_duration = end_time - start_time
                time_per_word = segment_duration / len(words) if words else 1.0
                
                for j, word in enumerate(words):
                    word_start = start_time + (j * time_per_word)
                    word_end = start_time + ((j + 1) * time_per_word)
                    
                    entries.append({
                        'index': len(entries) + 1,
                        'start_time': word_start,
                        'end_time': word_end,
                        'text': word
                    })
        
        else:
            # Handle other timestamp formats (existing word-level logic)
            for i, word_data in enumerate(word_timestamps):
                word = word_data['word']
                timestamp = word_data['timestamp']
                
                # Calculate end time (use next word's timestamp or estimate)
                if i < len(word_timestamps) - 1:
                    end_time = word_timestamps[i + 1]['timestamp']
                else:
                    end_time = timestamp + 0.5  # Add 0.5 seconds for last word
                
                # Create individual SRT entry for each word
                entries.append({
                    'index': i + 1,
                    'start_time': timestamp,
                    'end_time': end_time,
                    'text': word
                })
    
    else:
        # Fallback to sentence-based timing if no word-level timestamps found
        print("No word-level timestamps found, using sentence-based timing")
        
        if word_level:
            # Create word-level entries by estimating timing
            print("Creating word-level SRT entries with estimated timing")
            
            # Split transcription into words
            words = re.findall(r'\S+', transcription.strip())
            
            # Estimate timing: assume average speaking rate of 150 words per minute
            words_per_second = 150 / 60  # ~2.5 words per second
            seconds_per_word = 1 / words_per_second  # ~0.4 seconds per word
            
            for i, word in enumerate(words):
                start_time = segment_offset_seconds + (i * seconds_per_word)
                end_time = segment_offset_seconds + ((i + 1) * seconds_per_word)
                
                entries.append({
                    'index': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': word
                })
        else:
            # Split transcription into sentences/phrases
            sentences = re.split(r'[.!?]+', transcription.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Estimate timing (this is basic - 3 seconds per sentence)
            for i, sentence in enumerate(sentences):
                start_time = segment_offset_seconds + (i * 3)
                end_time = segment_offset_seconds + ((i + 1) * 3)
                
                entries.append({
                    'index': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': sentence
                })
    
    return entries


def main():
    parser = argparse.ArgumentParser(description='Transcribe audio files using Mistral AI API')
    
    # Get supported formats for help text
    supported_formats = AudioConverter.get_supported_formats()
    formats_text = ', '.join(supported_formats)
    
    parser.add_argument('audio_file', 
                       help=f'Path to the audio/video file to transcribe. Supported formats: {formats_text}')
    parser.add_argument('-o', '--output', help='Output SRT file path', default='transcription.srt')
    parser.add_argument('--api-key', help='Mistral API key (or set MISTRAL_API_KEY env var)')
    parser.add_argument('--keep-segments', action='store_true', help='Keep audio segments after processing')
    parser.add_argument('--keep-converted', action='store_true', help='Keep converted MP3 file after processing')
    parser.add_argument('--language', help='Expected language of the audio (e.g., "Norwegian", "German"). If not specified, uses generic original language prompt.')
    parser.add_argument('--word-level', action='store_true', help='Generate word-level SRT entries (one entry per word)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('MISTRAL_API_KEY')
    if not api_key:
        print("Error: Mistral API key not provided. Use --api-key or set MISTRAL_API_KEY environment variable.")
        sys.exit(1)
    
    # Validate input file
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Check if file format is supported
    file_ext = Path(args.audio_file).suffix.lower()
    if file_ext not in AudioConverter.get_supported_formats():
        print(f"Error: Unsupported file format '{file_ext}'. Supported formats: {formats_text}")
        sys.exit(1)
    
    print(f"Starting transcription of: {args.audio_file}")
    
    # Initialize components
    transcriber = MistralAudioTranscriber(api_key)
    splitter = AudioSplitter(max_duration_minutes=25, overlap_seconds=30)
    converter = AudioConverter()
    
    # Create temporary directory for segments
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Convert input file to MP3 if necessary
            converted_path = None
            if converter.needs_conversion(args.audio_file):
                converted_path = os.path.join(temp_dir, 'converted.mp3')
                converter.convert_to_mp3(args.audio_file, converted_path)
                input_path = converted_path
            else:
                input_path = args.audio_file
            
            # Split audio if necessary
            segments = splitter.split_audio(input_path, temp_dir)
            
            all_srt_entries = []
            
            # Process each segment
            for i, segment in enumerate(segments):
                print(f"\nProcessing segment {i+1}/{len(segments)}")
                
                # Transcribe segment
                transcription = transcriber.transcribe_segment(segment['path'], args.language)
                
                # For overlapping segments, we need to calculate the effective start time
                # First segment: use actual start time (0)
                # Subsequent segments: use start time + overlap to skip the overlapping part
                if i == 0:
                    effective_start_time = segment['start_time']
                else:
                    effective_start_time = segment['start_time'] + segment['overlap_start']
                
                # Convert to SRT entries using the effective start time
                srt_entries = process_transcription_to_srt(transcription, effective_start_time, args.word_level)
                
                # For segments with overlap, we need to filter out entries that fall within the overlap region
                if segment['overlap_start'] > 0:
                    # Remove entries that fall within the overlap at the beginning
                    overlap_end_time = segment['start_time'] + segment['overlap_start']
                    srt_entries = [entry for entry in srt_entries if entry['start_time'] >= overlap_end_time]
                
                # Adjust indices for global numbering
                for entry in srt_entries:
                    entry['index'] = len(all_srt_entries) + 1
                    all_srt_entries.append(entry)

            # Generate SRT file
            with open(args.output, 'w', encoding='utf-8') as f:
                for entry in all_srt_entries:
                    srt_text = SRTFormatter.create_srt_entry(
                        entry['index'],
                        entry['start_time'],
                        entry['end_time'],
                        entry['text']
                    )
                    f.write(srt_text)
            
            # Handle converted file cleanup
            if converted_path and args.keep_converted:
                # Move converted file to output directory
                final_converted_path = os.path.join(
                    os.path.dirname(args.output), 
                    f"{Path(args.audio_file).stem}_converted.mp3"
                )
                import shutil
                shutil.copy2(converted_path, final_converted_path)
                print(f"Converted MP3 saved to: {final_converted_path}")
            
            print(f"\nTranscription completed successfully!")
            print(f"Output saved to: {args.output}")
            print(f"Total entries: {len(all_srt_entries)}")
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
