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
                    raise Exception(f"Failed to transcribe audio after {max_retries} attempts.")
        
        return ""


timestamp_pattern = r'\[\s*\d+m\d+s\d+ms\s*(?:-\s*\d+m\d+s\d+ms\s*)?\]'


def has_timestamps(transcription: str) -> bool:
    """
    Detect if transcription contains timestamps in Mistral format.
    Supports:
    - Mistral format: [ 0m1s122ms ]
    - Range format: [ 0m0s694ms - 0m3s34ms ]
    """
    if not transcription or not transcription.strip():
        return False
    
    # Mistral timestamp format (word-level and range)
    return re.search(timestamp_pattern, transcription) is not None


def process_transcription_to_srt(transcription: str, segment_offset_seconds: float = 0, word_level: bool = False) -> List[Dict]:
    """
    Process transcription text and convert to SRT entries
    Handles both word-level timestamps and fallback sentence-based timing
    
    NOTE: Word timestamps are treated as indicating the MIDDLE of each word,
    not the start or end time. Start and end times are calculated based on
    this midpoint positioning.
    """
    entries = []
    
    word_timestamps = []

    pattern = timestamp_pattern
    matches = re.findall(pattern, transcription, re.DOTALL)
    logging.info(f"DEBUG: Testing pattern: {pattern}")
    logging.info(f"DEBUG: Found {len(matches)} matches")
    if matches and len(matches) > 0:
        logging.info(f"DEBUG: First few matches: {matches[:3]}")

    if not matches:
        return entries

    logging.info(f"Found {len(matches)} timestamped segments using pattern: {pattern}")

    for match in matches:
        word, time_str = match

        timestamp_seconds = parse_mistral_time(time_str)

        word_timestamps.append({
            'word': word.strip(),
            'timestamp': segment_offset_seconds + timestamp_seconds
        })

    if not word_timestamps:
        return entries

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

# Convert Mistral timestamp format to seconds
def parse_mistral_time(time_str):
    # Parse format like "0m3s34ms" or "1m23s456ms"
    time_match = re.match(r'(\d+)m(\d+)s(\d+)ms', time_str)
    if time_match:
        minutes, seconds, milliseconds = map(int, time_match.groups())
        return minutes * 60 + seconds + milliseconds / 1000.0
    return 0
