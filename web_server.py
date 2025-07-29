#!/usr/bin/env python3
"""
Web API Server for Audio Transcription
Provides REST API endpoints for job-based transcription processing
"""

import json
import logging
import os
import queue
import requests
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from flask import Flask, request, jsonify
from pathlib import Path
from typing import Dict, List, Optional, Any
from audio import AudioConverter, AudioSplitter

# Import our transcription modules
from transcribe_audio import (
    MistralAudioTranscriber,
    process_transcription_to_srt,
    join_srt_files_with_overlap,
    srt_to_json,
    SRTFormatter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class JobStatus(Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class TranscriptionJob:
    id: str
    path: str
    language: str
    format: str = "json"
    callback: Optional[str] = None
    output_path: Optional[str] = None
    model: str = "voxtral-small-latest"
    priority: int = 1000
    progress: int = 0
    status: JobStatus = JobStatus.QUEUED
    result: str = ""
    duration: float = 0.0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        # Convert datetime objects to ISO format
        for field in ['created_at', 'started_at', 'completed_at']:
            if data[field]:
                data[field] = data[field].isoformat()
        # Rename duration field to match API spec
        data['Duration'] = data.pop('duration')
        return data

class JobQueue:
    """Thread-safe job queue with priority support"""
    
    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._jobs: Dict[str, TranscriptionJob] = {}
        self._lock = threading.Lock()
        self._completed_jobs: List[str] = []  # Keep track of last 10 completed jobs
        
    def add_job(self, job: TranscriptionJob) -> None:
        """Add job to queue with priority"""
        with self._lock:
            self._jobs[job.id] = job
            # Priority queue uses lower numbers for higher priority
            # API spec: 500-9999 are high priority, 1-499 are low priority
            if 500 <= job.priority <= 9999:
                priority = -job.priority  # High priority (negative for higher precedence)
            else:
                priority = job.priority if 1 <= job.priority <= 499 else 1000
            
            self._queue.put((priority, time.time(), job.id))
    
    def get_next_job(self) -> Optional[TranscriptionJob]:
        """Get next job from queue (blocking)"""
        try:
            _, _, job_id = self._queue.get(timeout=1.0)
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status == JobStatus.QUEUED:
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.now()
                    return job
        except queue.Empty:
            pass
        return None
    
    def get_job(self, job_id: str) -> Optional[TranscriptionJob]:
        """Get job by ID"""
        with self._lock:
            return self._jobs.get(job_id)
    
    def update_job(self, job_id: str, **kwargs) -> bool:
        """Update job fields"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                
                # Track completed jobs
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    job.completed_at = datetime.now()
                    if job_id not in self._completed_jobs:
                        self._completed_jobs.append(job_id)
                        # Keep only last 10 completed jobs
                        if len(self._completed_jobs) > 10:
                            old_job_id = self._completed_jobs.pop(0)
                            if old_job_id in self._jobs:
                                del self._jobs[old_job_id]
                
                return True
        return False
    
    def delete_job(self, job_id: str) -> bool:
        """Delete job if it's queued"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == JobStatus.QUEUED:
                del self._jobs[job_id]
                return True
        return False
    
    def get_all_jobs(self) -> List[TranscriptionJob]:
        """Get all jobs"""
        with self._lock:
            return list(self._jobs.values())
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self._lock:
            stats = {
                "Queued": sum(1 for job in self._jobs.values() if job.status == JobStatus.QUEUED),
                "Running": sum(1 for job in self._jobs.values() if job.status == JobStatus.RUNNING),
                "Processed": len(self._completed_jobs)
            }
        return stats

class TranscriptionWorker:
    """Background worker for processing transcription jobs"""
    
    def __init__(self, job_queue: JobQueue, api_key: str):
        self.job_queue = job_queue
        self.api_key = api_key
        self.transcriber = MistralAudioTranscriber(api_key)
        self.converter = AudioConverter()
        self.splitter = AudioSplitter()
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the worker thread"""
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        logger.info("Transcription worker started")
    
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Transcription worker stopped")
    
    def _worker_loop(self):
        """Main worker loop"""
        while self.running:
            job = self.job_queue.get_next_job()
            if job:
                logger.info(f"Processing job {job.id}: {job.path}")
                try:
                    self._process_job(job)
                except Exception as e:
                    logger.error(f"Error processing job {job.id}: {e}")
                    self.job_queue.update_job(
                        job.id,
                        status=JobStatus.FAILED,
                        error_message=str(e)
                    )
                    self._send_callback(job)
    
    def _process_job(self, job: TranscriptionJob):
        """Process a single transcription job"""
        start_time = time.time()
        
        try:
            # Validate input file
            if not os.path.exists(job.path):
                raise FileNotFoundError(f"Input file not found: {job.path}")
            
            # Check file format
            file_ext = Path(job.path).suffix.lower()
            supported_formats = self.converter.get_supported_formats()
            if file_ext not in supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Set up output directory
            if job.output_path:
                output_dir = Path(job.output_path)
                output_dir.mkdir(parents=True, exist_ok=True)
                base_name = Path(job.path).stem
            else:
                output_dir = Path(job.path).parent
                base_name = Path(job.path).stem
            
            # Process transcription
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert to MP3 if needed
                if self.converter.needs_conversion(job.path):
                    converted_path = os.path.join(temp_dir, 'converted.mp3')
                    self.converter.convert_to_mp3(job.path, converted_path)
                    input_path = converted_path
                else:
                    input_path = job.path
                
                # Split audio if necessary
                segments = self.splitter.split_audio(input_path, temp_dir)
                logger.info(f"Audio split into {len(segments)} segments")
                
                # Process segments in parallel with 5 workers by default
                segment_srt_files = {}
                segment_info = {}
                
                def process_segment(segment_data):
                    """Process a single segment and return the results"""
                    i, segment = segment_data
                    logger.info(f"Processing segment {i+1}/{len(segments)}")
                    
                    # Transcribe segment
                    transcription = self.transcriber.transcribe_segment(
                        segment['path'],
                        language=job.language if job.language != "auto" else None,
                        max_retries=3
                    )
                    
                    # Convert to SRT entries
                    srt_entries = process_transcription_to_srt(transcription, 0, False)
                    
                    # Generate SRT file for segment
                    segment_srt_filename = f"segment_{i+1:03d}.srt"
                    segment_srt_path = os.path.join(temp_dir, segment_srt_filename)
                    
                    with open(segment_srt_path, 'w', encoding='utf-8') as f:
                        for entry in srt_entries:
                            srt_text = SRTFormatter.create_srt_entry(
                                entry['index'],
                                entry['start_time'],
                                entry['end_time'],
                                entry['text']
                            )
                            f.write(srt_text)
                    
                    return i, segment_srt_path, segment
                
                # Use ThreadPoolExecutor for parallel processing

                with ThreadPoolExecutor(max_workers=5) as executor:
                    # Submit all segment processing tasks
                    future_to_index = {}
                    for i, segment in enumerate(segments):
                        future = executor.submit(process_segment, (i, segment))
                        future_to_index[future] = i
                    
                    # Collect results as they complete
                    results = {}
                    for future in as_completed(future_to_index):
                        try:
                            i, segment_srt_path, segment = future.result()
                            results[i] = (segment_srt_path, segment)
                            logger.info(f"Completed segment {i+1}/{len(segments)}")
                        except Exception as e:
                            logger.error(f"Error processing segment {future_to_index[future]+1}: {e}")
                            raise e
                    
                    # Store results in order
                    for i in sorted(results.keys()):
                        segment_srt_path, segment = results[i]
                        segment_srt_files[i] = segment_srt_path
                        segment_info[i] = segment
                
                # Join SRT files
                all_srt_entries = join_srt_files_with_overlap(
                    segment_srt_files, segment_info, False
                )
                
                # Generate outputs based on format
                outputs = self._generate_outputs(
                    all_srt_entries, job.format, output_dir, base_name
                )
                
                # Set result based on format
                if job.format == "json":
                    # Generate SRT first, then convert to JSON
                    srt_path = outputs.get('srt')
                    if srt_path:
                        json_data = srt_to_json(srt_path)
                        job.result = json.dumps(json_data, ensure_ascii=False)
                elif job.format == "srt":
                    job.result = f"Output saved to {outputs.get(job.format, 'file')}"
                
                # Update job status
                duration = time.time() - start_time
                self.job_queue.update_job(
                    job.id,
                    status=JobStatus.COMPLETED,
                    duration=duration,
                    progress=100
                )
                
                logger.info(f"Job {job.id} completed in {duration:.2f}s")
                
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            self.job_queue.update_job(
                job.id,
                status=JobStatus.FAILED,
                error_message=str(e)
            )
        
        # Send callback if specified
        self._send_callback(job)
    
    def _generate_outputs(self, srt_entries: List[Dict], format_spec: str, 
                         output_dir: Path, base_name: str) -> Dict[str, str]:
        """Generate output files based on format specification"""
        outputs = {}
        formats = [format_spec] if format_spec != "all" else ["json", "srt"]
        
        for fmt in formats:
            if fmt == "json":
                # First generate SRT, then convert to JSON
                srt_path = output_dir / f"{base_name}.srt"
                with open(srt_path, 'w', encoding='utf-8') as f:
                    for entry in srt_entries:
                        srt_text = SRTFormatter.create_srt_entry(
                            entry['index'],
                            entry['start_time'],
                            entry['end_time'],
                            entry['text']
                        )
                        f.write(srt_text)
                
                # Convert to JSON
                json_data = srt_to_json(str(srt_path))
                output_path = output_dir / f"{base_name}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                outputs['json'] = str(output_path)
            
            elif fmt == "srt":
                output_path = output_dir / f"{base_name}.srt"
                with open(output_path, 'w', encoding='utf-8') as f:
                    for entry in srt_entries:
                        srt_text = SRTFormatter.create_srt_entry(
                            entry['index'],
                            entry['start_time'],
                            entry['end_time'],
                            entry['text']
                        )
                        f.write(srt_text)
                outputs['srt'] = str(output_path)
        
        return outputs
    
    def _send_callback(self, job: TranscriptionJob):
        """Send callback notification if specified"""
        if not job.callback:
            return
        
        try:
            callback_data = job.to_dict()
            response = requests.post(
                job.callback,
                json=callback_data,
                timeout=30
            )
            logger.info(f"Callback sent for job {job.id}: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to send callback for job {job.id}: {e}")

# Flask application
app = Flask(__name__)
job_queue = JobQueue()
worker = None

# Valid values from API spec
VALID_LANGUAGES = {
    'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca', 'cs', 'cy', 
    'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 
    'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 
    'ko', 'la', 'lb', 'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 
    'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 
    'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 
    'tr', 'tt', 'uk', 'ur', 'uz', 'vi', 'yi', 'yo', 'zh'
}

VALID_FORMATS = {'json', 'srt'}

VALID_MODELS = {'voxtral-small-2507', 'voxtral-small-latest'}

@app.route('/transcription/job', methods=['POST'])
def create_job():
    """Create a new transcription job"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate required fields
        if 'path' not in data:
            return jsonify({"error": "Missing required field: path"}), 400
        if 'language' not in data:
            return jsonify({"error": "Missing required field: language"}), 400
        
        # Validate field values
        if data['language'] not in VALID_LANGUAGES:
            return jsonify({"error": f"Invalid language: {data['language']}"}), 400
        
        format_val = data.get('format', 'json')
        if format_val not in VALID_FORMATS:
            return jsonify({"error": f"Invalid format: {format_val}"}), 400
        
        model_val = data.get('model', 'voxtral-small-latest')
        if model_val not in VALID_MODELS:
            return jsonify({"error": f"Invalid model: {model_val}"}), 400
        
        # Create job
        job = TranscriptionJob(
            id=str(uuid.uuid4()),
            path=data['path'],
            language=data['language'],
            format=format_val,
            callback=data.get('callback'),
            output_path=data.get('output_path'),
            model=model_val,
            priority=data.get('priority', 1000)
        )
        
        # Add to queue
        job_queue.add_job(job)
        
        # Return job info
        response_data = job.to_dict()
        # Remove internal fields from response
        for field in ['created_at', 'started_at', 'completed_at', 'error_message']:
            response_data.pop(field, None)
        
        return jsonify(response_data), 201
        
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/transcription/job/<job_id>', methods=['GET'])
def get_job(job_id: str):
    """Get job status and result"""
    job = job_queue.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    response_data = job.to_dict()
    # Remove internal fields from response
    for field in ['created_at', 'started_at', 'completed_at', 'error_message']:
        response_data.pop(field, None)
    
    return jsonify(response_data)

@app.route('/transcription/job/<job_id>', methods=['DELETE'])
def delete_job(job_id: str):
    """Delete a queued job"""
    if job_queue.delete_job(job_id):
        return jsonify({"message": "Job deleted successfully"})
    else:
        job = job_queue.get_job(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        else:
            return jsonify({"error": "Can only delete queued jobs"}), 400

@app.route('/transcription/jobs', methods=['GET'])
def get_all_jobs():
    """Get all jobs"""
    jobs = job_queue.get_all_jobs()
    response_data = []
    for job in jobs:
        job_data = job.to_dict()
        # Remove internal fields from response
        for field in ['created_at', 'started_at', 'completed_at', 'error_message']:
            job_data.pop(field, None)
        response_data.append(job_data)
    
    return jsonify(response_data)

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get queue statistics"""
    return jsonify(job_queue.get_stats())

def main():
    """Main entry point"""
    global worker
    
    # Get API key
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        logger.error("Error: MISTRAL_API_KEY environment variable not set")
        sys.exit(1)
    
    # Start worker
    worker = TranscriptionWorker(job_queue, api_key)
    worker.start()
    
    # Start Flask app
    try:
        app.run(host='0.0.0.0', port=8888, debug=False)
    finally:
        if worker:
            worker.stop()

if __name__ == '__main__':
    main()
