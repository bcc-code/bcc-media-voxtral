#!/usr/bin/env python3
"""
Audio Transcription Script using Mistral AI API
Transcribes audio files and outputs in SRT format
Handles files longer than 25 minutes by splitting them into segments
"""

import os
import requests
from typing import Dict, Any, List
import re
import time
import json
import logging
from srt import SRTFormatter

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
        logging.info(f"Uploading audio file: {os.path.basename(audio_path)}")
        
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
                logging.error(f"HTTP Error: {e}")
                logging.error(f"Response status code: {response.status_code}")
                logging.error(f"Response headers: {response.headers}")
                logging.error(f"Response body: {response.text}")
                raise
            
            result = response.json()
            file_id = result.get('id')
            logging.info(f"File uploaded successfully. ID: {file_id}")
            return file_id
    
    def get_signed_url(self, file_id: str, expiry_hours: int = 24) -> str:
        """Get signed URL for the uploaded file"""
        logging.info(f"Getting signed URL for file ID: {file_id}")
        
        url = f"{self.base_url}/files/{file_id}/url?expiry={expiry_hours}"
        
        response = requests.get(url, headers=self.headers)
        
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP Error: {e}")
            logging.error(f"Response status code: {response.status_code}")
            logging.error(f"Response headers: {response.headers}")
            logging.error(f"Response body: {response.text}")
            raise
        
        result = response.json()
        signed_url = result.get('url')
        logging.info("Signed URL retrieved successfully")
        return signed_url
    
    def transcribe_audio(self, signed_url: str, custom_prompt: str = "Transcribe the attached audio. Keep the transcript as exact as possible. Provide word level timestamps.", store_results: bool = False, results_dir: str = None, segment_name: str = None) -> str:
        """Send transcription request to Mistral API"""
        logging.info("Sending transcription request...")

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
            logging.error(f"HTTP Error: {e}")
            logging.error(f"Response status code: {response.status_code}")
            logging.error(f"Response headers: {response.headers}")
            logging.error(f"Response body: {response.text}")
            raise
        
        result = response.json()
        transcription = result['choices'][0]['message']['content']
        logging.info("Transcription completed successfully")
        
        # Store API results if requested
        if store_results and results_dir and segment_name:
            try:
                # Store raw JSON response
                json_filename = f"{segment_name}_api_response.json"
                json_path = os.path.join(results_dir, json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logging.info(f"Stored API response: {json_path}")
                
                # Store transcription text
                txt_filename = f"{segment_name}_transcription.txt"
                txt_path = os.path.join(results_dir, txt_filename)
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                logging.info(f"Stored transcription: {txt_path}")
                
                # Store request payload for reference
                req_filename = f"{segment_name}_request.json"
                req_path = os.path.join(results_dir, req_filename)
                with open(req_path, 'w', encoding='utf-8') as f:
                    # Remove the signed URL from payload for security
                    safe_payload = payload.copy()
                    safe_payload['messages'][0]['content'][0]['input_audio']['data'] = "[SIGNED_URL_REDACTED]"
                    json.dump(safe_payload, f, indent=2, ensure_ascii=False)
                logging.info(f"Stored request payload: {req_path}")
                
            except Exception as e:
                logging.warning(f"Warning: Failed to store API results: {e}")
        
        return transcription

    def transcribe_segment(self, audio_path: str, language: str = None, store_results: bool = False, results_dir: str = None, segment_name: str = None, max_retries: int = 3) -> str:
        """Complete transcription workflow for a single audio segment with retry logic for missing timestamps"""
        logging.info(f"Processing audio file: {audio_path}")
        
        # Upload audio file
        file_id = self.upload_audio_file(audio_path)
        logging.info(f"File uploaded with ID: {file_id}")
        
        # Get signed URL
        signed_url = self.get_signed_url(file_id)
        logging.info(f"Got signed URL for file")
        
        # Prepare custom prompt
        prompt = "Transcribe the attached audio. Keep the transcript as exact as possible. Provide word level timestamps."
        if language:
            prompt += f" The audio is in {language}."
        
        # Retry logic for missing timestamps
        for attempt in range(max_retries):
            try:
                logging.info(f"Transcription attempt {attempt + 1}/{max_retries}")
                
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
                    logging.info(f"✓ Timestamps detected in transcription (attempt {attempt + 1})")
                    return transcription
                else:
                    logging.warning(f"⚠ No timestamps detected in transcription (attempt {attempt + 1})")
                    
                    if attempt < max_retries - 1:
                        logging.info(f"Retrying in 2 seconds...")
                        time.sleep(2)  # Brief delay before retry
                    else:
                        logging.warning(f"⚠ All {max_retries} attempts failed to produce timestamps. Using transcription without timestamps.")
                        return transcription
                        
            except Exception as e:
                logging.error(f"Error in transcription attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in 5 seconds...")
                    time.sleep(5)  # Longer delay after error
                else:
                    logging.error(f"All {max_retries} attempts failed.")
                    raise
        
        return ""


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
        logging.info(f"DEBUG: Testing pattern: {pattern}")
        logging.info(f"DEBUG: Found {len(matches)} matches")
        if matches and len(matches) > 0:
            logging.info(f"DEBUG: First few matches: {matches[:3]}")
        
        if matches:
            logging.info(f"Found {len(matches)} timestamped segments using pattern: {pattern}")
            
            if pattern.startswith(r'(\S+?)(?:,)?\s*\[\s*(\d+m\d+s\d+ms)'):  # Word-level format: word, [ 0m1s122ms ] or word [ 0m1s122ms ]
                for match in matches:
                    word, time_str = match
                    
                    # Convert Mistral timestamp format to seconds
                    def parse_mistral_time(time_str):
                        # Parse format like "0m3s34ms" or "1m23s456ms"
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
            logging.info(f"Processing {len(word_timestamps)} timestamped segments from Mistral")
            
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
        logging.info("No word-level timestamps found, using sentence-based timing")
        
        if word_level:
            # Create word-level entries by estimating timing
            logging.info("Creating word-level SRT entries with estimated timing")
            
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
        
        logging.info(f"Processing SRT file for segment {i+1}: {srt_path}")
        
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
                        logging.info(f"  Skipping overlapped entry: {global_start_time:.2f}s <= {overlap_end_time:.2f}s")
                        continue
                
                # Add to final results with global indexing
                all_srt_entries.append({
                    'index': len(all_srt_entries) + 1,
                    'start_time': global_start_time,
                    'end_time': global_end_time,
                    'text': text
                })
                
            except (ValueError, IndexError) as e:
                logging.warning(f"Warning: Failed to parse SRT block: {e}")
                continue
    
    logging.info(f"Joined {len(all_srt_entries)} total SRT entries from {len(srt_files)} segments")
    
    # Clean up temporary SRT files if not keeping segments
    if not keep_segments:
        for srt_path in srt_files.values():
            try:
                os.remove(srt_path)
                logging.info(f"Removed temporary SRT file: {srt_path}")
            except OSError as e:
                logging.warning(f"Warning: Failed to remove {srt_path}: {e}")
    else:
        logging.info(f"Keeping segment SRT files (--keep-segments enabled)")
    
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
        logging.info(f"SRT file converted to JSON successfully!")
        logging.info(f"Input: {srt_file_path}")
        logging.info(f"Output: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error converting SRT to JSON: {e}")
        return False
