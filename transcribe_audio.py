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
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from pedalboard.io import AudioFile
import time
import json


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
    
    @staticmethod
    def parse_time(time_str: str) -> float:
        """Parse SRT time format (HH:MM:SS,mmm) to seconds"""
        time_str = time_str.strip()
        # Handle both comma and dot as decimal separator
        time_str = time_str.replace(',', '.')
        
        # Split into time and milliseconds parts
        if '.' in time_str:
            time_part, ms_part = time_str.split('.')
            milliseconds = float('0.' + ms_part)
        else:
            time_part = time_str
            milliseconds = 0.0
        
        # Parse HH:MM:SS
        parts = time_part.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds
        return total_seconds


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
                
                segment_filename = f"segment_{i+1:03d}.mp3"
                segment_path = os.path.join(output_dir, segment_filename)
                
                # Keep original sample rate for better quality
                final_sample_rate = sample_rate
                
                with AudioFile(segment_path, 'w', final_sample_rate, 1) as f:
                    f.write(processed_segment_audio)
                
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
    
    def transcribe_audio(self, signed_url: str, custom_prompt: str = "Transcribe the attached audio. Keep the transcript as exact as possible. Provide word level timestamps.", store_results: bool = False, results_dir: str = None, segment_name: str = None) -> str:
        """Send transcription request to Mistral API"""
        print("Sending transcription request...")

        custom_prompt = custom_prompt + " Here is an example of how the output should look like: `Word1 [ 0m0s694ms ] Word2 [ 0m1s164ms ] word3 [ 0m1s314ms ]`"
        
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
        
        # Store API results if requested
        if store_results and results_dir and segment_name:
            try:
                # Store raw JSON response
                json_filename = f"{segment_name}_api_response.json"
                json_path = os.path.join(results_dir, json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Stored API response: {json_path}")
                
                # Store transcription text
                txt_filename = f"{segment_name}_transcription.txt"
                txt_path = os.path.join(results_dir, txt_filename)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                print(f"Stored transcription: {txt_path}")
                
                # Store request payload for reference
                req_filename = f"{segment_name}_request.json"
                req_path = os.path.join(results_dir, req_filename)
                with open(req_path, 'w', encoding='utf-8') as f:
                    # Remove the signed URL from payload for security
                    safe_payload = payload.copy()
                    safe_payload['messages'][0]['content'][0]['input_audio']['data'] = "[SIGNED_URL_REDACTED]"
                    json.dump(safe_payload, f, indent=2, ensure_ascii=False)
                print(f"Stored request payload: {req_path}")
                
            except Exception as e:
                print(f"Warning: Failed to store API results: {e}")
        
        # Debug: Show raw transcription output
        print("\n" + "="*50)
        print("DEBUG: Raw transcription from model:")
        print("="*50)
        print(transcription[:500] + "..." if len(transcription) > 500 else transcription)
        print("="*50 + "\n")
        
        return transcription

    def transcribe_segment(self, audio_path: str, language: str = None, store_results: bool = False, results_dir: str = None, segment_name: str = None, max_retries: int = 3) -> str:
        """Complete transcription workflow for a single audio segment with retry logic for missing timestamps"""
        print(f"Processing audio file: {audio_path}")
        
        # Upload audio file
        file_id = self.upload_audio_file(audio_path)
        print(f"File uploaded with ID: {file_id}")
        
        # Get signed URL
        signed_url = self.get_signed_url(file_id)
        print(f"Got signed URL for file")
        
        # Prepare custom prompt
        prompt = "Transcribe the attached audio. Keep the transcript as exact as possible. Provide word level timestamps."
        if language:
            prompt += f" The audio is in {language}."
        
        # Retry logic for missing timestamps
        for attempt in range(max_retries):
            try:
                print(f"Transcription attempt {attempt + 1}/{max_retries}")
                
                # Get transcription
                transcription = self.transcribe_audio(
                    signed_url, 
                    prompt, 
                    store_results, 
                    results_dir, 
                    f"{segment_name}_attempt_{attempt + 1}" if segment_name else None
                )
                
                # Check if timestamps are present
                if has_timestamps(transcription):
                    print(f"✓ Timestamps detected in transcription (attempt {attempt + 1})")
                    return transcription
                else:
                    print(f"⚠ No timestamps detected in transcription (attempt {attempt + 1})")
                    
                    if attempt < max_retries - 1:
                        print(f"Retrying in 2 seconds...")
                        time.sleep(2)  # Brief delay before retry
                    else:
                        print(f"⚠ All {max_retries} attempts failed to produce timestamps. Using transcription without timestamps.")
                        return transcription
                        
            except Exception as e:
                print(f"Error in transcription attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)  # Longer delay after error
                else:
                    print(f"All {max_retries} attempts failed.")
                    raise
        
        return transcription  # Fallback (should not reach here)


def has_timestamps(transcription: str) -> bool:
    """
    Detect if transcription contains timestamps.
    Returns True if timestamps are found, False otherwise.
    """
    timestamp_patterns = [
        r'\[\s*\d+m\d+s\d+ms\s*\]',  # Mistral format: [ 0m1s122ms ]
        r'\[\s*\d+m\d+s\d+ms\s*-\s*\d+m\d+s\d+ms\s*\]',  # Mistral range format: [ 0m0s694ms - 0m3s34ms ]
        r'\[\d{1,2}:\d{2}\.\d{2}\]',  # [00:01.23] format
        r'\d+:\d{2}\.\d{2}',  # Simple timestamp format
    ]
    
    for pattern in timestamp_patterns:
        if re.search(pattern, transcription):
            return True
    
    return False


def process_transcription_to_srt(transcription: str, segment_offset_seconds: float = 0, word_level: bool = False) -> List[Dict]:
    """
    Process transcription text and convert to SRT entries
    Handles both word-level timestamps and fallback sentence-based timing
    
    NOTE: Word timestamps are treated as indicating the MIDDLE of each word,
    not the start or end time. Start and end times are calculated based on
    this midpoint positioning.
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
                    # Calculate word timing with timestamp representing the middle of each word
                    word_center = start_time + (j + 0.5) * time_per_word
                    half_word_duration = time_per_word / 2
                    
                    word_start = word_center - half_word_duration
                    word_end = word_center + half_word_duration
                    
                    # Ensure start time is not negative
                    word_start = max(0, word_start)
                    
                    entries.append({
                        'index': len(entries) + 1,
                        'start_time': word_start,
                        'end_time': word_end,
                        'text': word
                    })
        
        else:
            # Handle other timestamp formats (existing word-level logic)
            # NOTE: Timestamps from API indicate the MIDDLE of each word
            for i, word_data in enumerate(word_timestamps):
                word = word_data['word']
                middle_timestamp = word_data['timestamp']
                
                # Calculate start and end times to avoid overlaps
                if i < len(word_timestamps) - 1:
                    next_middle = word_timestamps[i + 1]['timestamp']
                    # End time is halfway between this word's middle and next word's middle
                    end_time = (middle_timestamp + next_middle) / 2
                else:
                    # For last word, estimate duration based on previous word or use default
                    if i > 0:
                        prev_middle = word_timestamps[i - 1]['timestamp']
                        word_duration = middle_timestamp - prev_middle
                        end_time = middle_timestamp + word_duration / 2
                    else:
                        # Single word case - use reasonable default
                        end_time = middle_timestamp + 0.25  # 0.5s total duration
                
                # Start time is calculated similarly for previous boundary
                if i > 0:
                    prev_middle = word_timestamps[i - 1]['timestamp']
                    # Start time is halfway between previous word's middle and this word's middle
                    start_time = (prev_middle + middle_timestamp) / 2
                else:
                    # For first word, calculate start based on end time and reasonable duration
                    if i < len(word_timestamps) - 1:
                        word_duration = end_time - middle_timestamp
                        start_time = middle_timestamp - word_duration
                    else:
                        # Single word case
                        start_time = middle_timestamp - 0.25  # 0.5s total duration
                
                # Ensure start time is not negative
                start_time = max(0, start_time)
                
                # Create individual SRT entry for each word
                entries.append({
                    'index': i + 1,
                    'start_time': start_time,
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
                word_center = segment_offset_seconds + (i + 0.5) * seconds_per_word
                half_word_duration = seconds_per_word / 2
                
                start_time = word_center - half_word_duration
                end_time = word_center + half_word_duration
                
                # Ensure start time is not negative
                start_time = max(0, start_time)
                
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


def join_srt_files_with_overlap(srt_files: Dict[int, str], segment_info: Dict[int, Dict], keep_segments: bool) -> List[Dict]:
    """Join SRT files from segments with overlap handling"""
    all_srt_entries: List[Dict] = []
    
    # Process segments in order
    for i in sorted(srt_files.keys()):
        srt_path = srt_files[i]
        segment = segment_info[i]
        
        print(f"Processing SRT file for segment {i+1}: {srt_path}")
        
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        
        if not srt_content:
            continue
            
        # Parse SRT entries
        srt_blocks = srt_content.split('\n\n')
        for block in srt_blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
                
            try:
                # Parse SRT block
                index = int(lines[0])
                timestamp_line = lines[1]
                text = '\n'.join(lines[2:])
                
                # Parse timestamps
                start_str, end_str = timestamp_line.split(' --> ')
                start_time = SRTFormatter.parse_time(start_str)
                end_time = SRTFormatter.parse_time(end_str)
                
                # Apply segment time offset to get global timestamps
                global_start_time = start_time + segment['start_time']
                global_end_time = end_time + segment['start_time']
                
                # Handle overlap filtering
                if segment['overlap_start'] > 0:
                    # Skip entries that fall within the overlap region at the beginning of this segment
                    overlap_end_time = segment['start_time'] + segment['overlap_start']
                    if global_start_time <= overlap_end_time:
                        print(f"  Skipping overlapped entry: {global_start_time:.2f}s <= {overlap_end_time:.2f}s")
                        continue
                
                # Add to final results with global indexing
                all_srt_entries.append({
                    'index': len(all_srt_entries) + 1,
                    'start_time': global_start_time,
                    'end_time': global_end_time,
                    'text': text
                })
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse SRT block: {e}")
                continue
    
    print(f"Joined {len(all_srt_entries)} total SRT entries from {len(srt_files)} segments")
    
    # Clean up temporary SRT files if not keeping segments
    if not keep_segments:
        for srt_path in srt_files.values():
            try:
                os.remove(srt_path)
                print(f"Removed temporary SRT file: {srt_path}")
            except OSError as e:
                print(f"Warning: Failed to remove {srt_path}: {e}")
    else:
        print(f"Keeping segment SRT files (--keep-segments enabled)")
    
    return all_srt_entries


def srt_to_json(srt_file_path: str) -> Dict[str, Any]:
    """
    Convert SRT file to JSON format with segments and word-level timing.
    
    Args:
        srt_file_path: Path to the SRT file
        
    Returns:
        Dictionary in the specified JSON format with text, segments, and word timing
    """
    if not os.path.exists(srt_file_path):
        raise FileNotFoundError(f"SRT file not found: {srt_file_path}")
    
    with open(srt_file_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    
    # Parse SRT entries
    srt_entries = []
    entries = srt_content.strip().split('\n\n')
    
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) >= 3:
            try:
                # Parse SRT block
                index = int(lines[0])
                time_line = lines[1]
                text = '\n'.join(lines[2:])
                
                # Parse timing
                start_str, end_str = time_line.split(' --> ')
                start_time = SRTFormatter.parse_time(start_str)
                end_time = SRTFormatter.parse_time(end_str)
                
                srt_entries.append({
                    'index': index,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                })
            except (ValueError, IndexError):
                continue
    
    if not srt_entries:
        return {"text": "", "segments": []}
    
    # Combine all text
    full_text = ' '.join(entry['text'] for entry in srt_entries)
    
    # Group entries into segments based on timing gaps or sentence boundaries
    segments = []
    current_segment = {
        'text': '',
        'start': None,
        'end': None,
        'words': [],
        'id': 0
    }
    
    # If entries are word-level (short text), group them into meaningful segments
    # Otherwise treat each entry as a segment
    if srt_entries and len(srt_entries[0]['text'].split()) <= 2:
        # Word-level entries - group into segments
        segment_gap_threshold = 2.0  # seconds gap to start new segment
        
        for i, entry in enumerate(srt_entries):
            # Check if we should start a new segment
            if (current_segment['start'] is not None and 
                entry['start_time'] - current_segment['end'] > segment_gap_threshold):
                # Finalize current segment
                if current_segment['words']:
                    segments.append(current_segment.copy())
                    current_segment = {
                        'text': '',
                        'start': None,
                        'end': None,
                        'words': [],
                        'id': len(segments)
                    }
            
            # Add word to current segment
            if current_segment['start'] is None:
                current_segment['start'] = entry['start_time']
            
            current_segment['end'] = entry['end_time']
            current_segment['text'] += (' ' if current_segment['text'] else '') + entry['text']
            current_segment['words'].append({
                'text': entry['text'],
                'start': entry['start_time'],
                'end': entry['end_time']
            })
        
        # Add final segment
        if current_segment['words']:
            segments.append(current_segment)
    
    else:
        # Sentence/phrase-level entries - each entry becomes a segment
        for i, entry in enumerate(srt_entries):
            # Split text into words and estimate word timing
            words = entry['text'].split()
            if not words:
                continue
                
            segment_duration = entry['end_time'] - entry['start_time']
            time_per_word = segment_duration / len(words) if words else 0
            
            word_list = []
            for j, word in enumerate(words):
                word_start = entry['start_time'] + j * time_per_word
                word_end = entry['start_time'] + (j + 1) * time_per_word
                word_list.append({
                    'text': word,
                    'start': round(word_start, 3),  # Use more precision to avoid timing conflicts
                    'end': round(word_end, 3)
                })
            
            segments.append({
                'text': entry['text'],
                'start': entry['start_time'],
                'end': entry['end_time'],
                'words': word_list,
                'id': i
            })
    
    return {
        'text': full_text,
        'segments': segments
    }


def convert_srt_to_json(srt_file_path: str, output_path: str) -> bool:
    """
    Convert SRT file to JSON format with segments and word-level timing.
    
    Args:
        srt_file_path: Path to the SRT file
        output_path: Path to the output JSON file
        
    Returns:
        True if conversion is successful, False otherwise
    """
    try:
        json_data = srt_to_json(srt_file_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"SRT file converted to JSON successfully!")
        print(f"Input: {srt_file_path}")
        print(f"Output: {output_path}")
        return True
    except Exception as e:
        print(f"Error converting SRT to JSON: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Transcribe audio files using Mistral AI API')
    
    parser.add_argument('input_file', help='Input audio file path or SRT file path (for --convert-srt mode)')
    parser.add_argument('--output', '-o', help='Output SRT file path (default: input_file.srt)')
    parser.add_argument('--language', help='Language code for transcription (optional)')
    parser.add_argument('--custom-prompt', help='Custom prompt for transcription')
    parser.add_argument('--parallel-workers', type=int, default=3, help='Number of parallel workers for API requests (default: 3)')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retries for missing timestamps (default: 3)')
    parser.add_argument('--store-api-results', action='store_true', help='Store API call results as files for debugging')
    parser.add_argument('--keep-segments', action='store_true', help='Keep individual segment SRT files for debugging')
    parser.add_argument('--to-json', action='store_true', help='Convert SRT output to JSON format with segments and word timing')
    parser.add_argument('--convert-srt', action='store_true', help='Convert existing SRT file to JSON format (no transcription)')
    
    args = parser.parse_args()
    
    # Handle SRT to JSON conversion mode
    if args.convert_srt:
        if not args.input_file.endswith('.srt'):
            print("Error: --convert-srt mode requires an SRT file as input")
            return False
        
        if not os.path.exists(args.input_file):
            print(f"Error: SRT file not found: {args.input_file}")
            return False
        
        json_output_path = args.output if args.output else args.input_file.replace('.srt', '.json')
        return convert_srt_to_json(args.input_file, json_output_path)
    
    # Get API key
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        print("Error: Mistral API key not provided. Use MISTRAL_API_KEY environment variable.")
        sys.exit(1)
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Audio file not found: {args.input_file}")
        sys.exit(1)
    
    # Check if file format is supported
    file_ext = Path(args.input_file).suffix.lower()
    supported_formats = AudioConverter.get_supported_formats()
    if file_ext not in supported_formats:
        print(f"Error: Unsupported file format '{file_ext}'. Supported formats: {', '.join(supported_formats)}")
        sys.exit(1)
    
    print(f"Starting transcription of: {args.input_file}")
    
    # Initialize components
    transcriber = MistralAudioTranscriber(api_key)
    splitter = AudioSplitter(max_duration_minutes=1, overlap_seconds=30)

    converter = AudioConverter()
    
    # Create temporary directory for segments
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create API results directory if storing results
            api_results_dir = None
            if args.store_api_results:
                # Create results directory next to output file
                output_dir = os.path.dirname(os.path.abspath(args.output))
                base_name = Path(args.input_file).stem
                api_results_dir = os.path.join(output_dir, f"{base_name}_api_results")
                os.makedirs(api_results_dir, exist_ok=True)
                print(f"API results will be stored in: {api_results_dir}")
            
            # Convert input file to MP3 if necessary
            converted_path = None
            if converter.needs_conversion(args.input_file):
                converted_path = os.path.join(temp_dir, 'converted.mp3')
                converter.convert_to_mp3(args.input_file, converted_path)
                input_path = converted_path
            else:
                input_path = args.input_file
            
            # Split audio if necessary
            segments = splitter.split_audio(input_path, temp_dir)
            
            # Process segments in parallel while maintaining order
            def process_segment(segment_info):
                """Process a single segment and return results with segment info"""
                i, segment = segment_info
                print(f"Starting transcription for segment {i+1}/{len(segments)}")
                
                # Transcribe segment
                transcription = transcriber.transcribe_segment(
                    segment['path'], 
                    args.language, 
                    args.store_api_results, 
                    api_results_dir, 
                    f"segment_{i+1:03d}",
                    args.max_retries
                )
                
                # Convert to SRT entries with NO time offset (segment-local timestamps)
                srt_entries = process_transcription_to_srt(transcription, 0, False)
                
                # Generate individual SRT file for this segment
                segment_srt_filename = f"segment_{i+1:03d}.srt"
                segment_srt_path = os.path.join(temp_dir, segment_srt_filename)
                
                with open(segment_srt_path, 'w', encoding='utf-8') as f:
                    for j, entry in enumerate(srt_entries):
                        srt_text = SRTFormatter.create_srt_entry(
                            j + 1,  # Local indexing for this segment
                            entry['start_time'],
                            entry['end_time'],
                            entry['text']
                        )
                        f.write(srt_text)
                
                print(f"Completed transcription for segment {i+1}/{len(segments)} -> {segment_srt_filename}")
                return i, segment_srt_path, segment
            
            # Submit all segment processing tasks
            with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
                # Create futures with segment index and data
                future_to_index = {}
                for i, segment in enumerate(segments):
                    future = executor.submit(process_segment, (i, segment))
                    future_to_index[future] = i
                
                # Collect results maintaining order
                segment_srt_files = {}
                segment_info = {}
                for future in as_completed(future_to_index):
                    segment_index, srt_path, segment = future.result()
                    segment_srt_files[segment_index] = srt_path
                    segment_info[segment_index] = segment
                
                # Join SRT files with overlap handling
                all_srt_entries = join_srt_files_with_overlap(segment_srt_files, segment_info, args.keep_segments)
            
            # Generate SRT file
            output_path = args.output
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in all_srt_entries:
                    srt_text = SRTFormatter.create_srt_entry(
                        entry['index'],
                        entry['start_time'],
                        entry['end_time'],
                        entry['text']
                    )
                    f.write(srt_text)
            
            # Convert to JSON if requested
            if args.to_json:
                json_output_path = output_path.replace('.srt', '.json')
                try:
                    json_data = srt_to_json(output_path)
                    with open(json_output_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)
                    print(f"JSON output saved to: {json_output_path}")
                except Exception as e:
                    print(f"Error converting to JSON: {e}")
            
            # Handle converted file cleanup
            if converted_path and args.keep_converted:
                # Move converted file to output directory
                final_converted_path = os.path.join(
                    os.path.dirname(args.output), 
                    f"{Path(args.input_file).stem}_converted.mp3"
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
