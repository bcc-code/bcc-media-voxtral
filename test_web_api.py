#!/usr/bin/env python3
"""
Test script for the web API server
"""

import unittest
import json
import time
import tempfile
import os
import threading
from unittest.mock import patch, MagicMock

# Import the web server components
import web_server
from web_server import (
    app, TranscriptionJob, JobStatus, JobQueue, 
    TranscriptionWorker, VALID_LANGUAGES, VALID_FORMATS, VALID_MODELS
)

class TestWebAPI(unittest.TestCase):
    """Test the web API endpoints"""
    
    def setUp(self):
        """Set up test fixtures"""
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.temp_dir = tempfile.mkdtemp()
        
        # Reset the global job queue for each test
        web_server.job_queue = JobQueue()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_create_job_valid(self):
        """Test creating a valid transcription job"""
        job_data = {
            "path": "/test/audio.mp3",
            "language": "en",
            "format": "srt",
            "callback": "http://example.com/callback",
            "output_path": "/test/output/",
            "model": "voxtral-small-2507",
            "priority": 500
        }
        
        response = self.client.post('/transcription/job', 
                                  data=json.dumps(job_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 201)
        data = json.loads(response.data)
        
        # Check response structure
        self.assertIn('id', data)
        self.assertEqual(data['path'], job_data['path'])
        self.assertEqual(data['language'], job_data['language'])
        self.assertEqual(data['format'], job_data['format'])
        self.assertEqual(data['status'], 'QUEUED')
        self.assertEqual(data['progress'], 0)
        self.assertEqual(data['priority'], 500)
    
    def test_create_job_missing_required_fields(self):
        """Test creating job with missing required fields"""
        # Missing path
        job_data = {"language": "en"}
        response = self.client.post('/transcription/job',
                                  data=json.dumps(job_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Missing language
        job_data = {"path": "/test/audio.mp3"}
        response = self.client.post('/transcription/job',
                                  data=json.dumps(job_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
    
    def test_create_job_invalid_values(self):
        """Test creating job with invalid field values"""
        # Invalid language
        job_data = {
            "path": "/test/audio.mp3",
            "language": "invalid_lang"
        }
        response = self.client.post('/transcription/job',
                                  data=json.dumps(job_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Invalid format
        job_data = {
            "path": "/test/audio.mp3",
            "language": "en",
            "format": "txt"  # No longer valid
        }
        response = self.client.post('/transcription/job',
                                  data=json.dumps(job_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Invalid model
        job_data = {
            "path": "/test/audio.mp3",
            "language": "en",
            "model": "large-v2"  # No longer valid
        }
        response = self.client.post('/transcription/job',
                                  data=json.dumps(job_data),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
    
    def test_get_job_existing(self):
        """Test getting an existing job"""
        # Create a job first
        job_data = {
            "path": "/test/audio.mp3",
            "language": "en"
        }
        response = self.client.post('/transcription/job',
                                  data=json.dumps(job_data),
                                  content_type='application/json')
        job_id = json.loads(response.data)['id']
        
        # Get the job
        response = self.client.get(f'/transcription/job/{job_id}')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['id'], job_id)
        self.assertEqual(data['path'], job_data['path'])
        self.assertEqual(data['language'], job_data['language'])
    
    def test_get_job_nonexistent(self):
        """Test getting a non-existent job"""
        response = self.client.get('/transcription/job/nonexistent-id')
        self.assertEqual(response.status_code, 404)
    
    def test_delete_job_queued(self):
        """Test deleting a queued job"""
        # Create a job
        job_data = {
            "path": "/test/audio.mp3",
            "language": "en"
        }
        response = self.client.post('/transcription/job',
                                  data=json.dumps(job_data),
                                  content_type='application/json')
        job_id = json.loads(response.data)['id']
        
        # Delete the job
        response = self.client.delete(f'/transcription/job/{job_id}')
        self.assertEqual(response.status_code, 200)
        
        # Verify job is deleted
        response = self.client.get(f'/transcription/job/{job_id}')
        self.assertEqual(response.status_code, 404)
    
    def test_delete_job_nonexistent(self):
        """Test deleting a non-existent job"""
        response = self.client.delete('/transcription/job/nonexistent-id')
        self.assertEqual(response.status_code, 404)
    
    def test_get_all_jobs(self):
        """Test getting all jobs"""
        # Reset job queue to ensure clean state
        web_server.job_queue = JobQueue()
        
        # Create multiple jobs
        job_data_1 = {"path": "/test/audio1.mp3", "language": "en"}
        job_data_2 = {"path": "/test/audio2.mp3", "language": "de"}
        
        response1 = self.client.post('/transcription/job',
                                   data=json.dumps(job_data_1),
                                   content_type='application/json')
        response2 = self.client.post('/transcription/job',
                                   data=json.dumps(job_data_2),
                                   content_type='application/json')
        
        # Get all jobs
        response = self.client.get('/transcription/jobs')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(len(data), 2)
        
        # Check job data
        paths = [job['path'] for job in data]
        self.assertIn('/test/audio1.mp3', paths)
        self.assertIn('/test/audio2.mp3', paths)
    
    def test_get_stats(self):
        """Test getting queue statistics"""
        # Reset job queue to ensure clean state
        web_server.job_queue = JobQueue()
        
        response = self.client.get('/stats')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('Queued', data)
        self.assertIn('Running', data)
        self.assertIn('Processed', data)
        
        # Should reflect current queue state (empty after reset)
        self.assertEqual(data['Queued'], 0)
        self.assertEqual(data['Running'], 0)
        self.assertEqual(data['Processed'], 0)


class TestJobQueue(unittest.TestCase):
    """Test the job queue functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.queue = JobQueue()
    
    def test_add_and_get_job(self):
        """Test adding and retrieving jobs"""
        job = TranscriptionJob(
            id="test-id",
            path="/test/audio.mp3",
            language="en",
            priority=1000
        )
        
        self.queue.add_job(job)
        retrieved_job = self.queue.get_job("test-id")
        
        self.assertIsNotNone(retrieved_job)
        self.assertEqual(retrieved_job.id, "test-id")
        self.assertEqual(retrieved_job.path, "/test/audio.mp3")
        self.assertEqual(retrieved_job.status, JobStatus.QUEUED)
    
    def test_priority_ordering(self):
        """Test that jobs are processed in priority order"""
        # Add low priority job
        low_priority_job = TranscriptionJob(
            id="low-priority",
            path="/test/audio1.mp3",
            language="en",
            priority=100  # Low priority
        )
        
        # Add high priority job
        high_priority_job = TranscriptionJob(
            id="high-priority",
            path="/test/audio2.mp3",
            language="en",
            priority=800  # High priority
        )
        
        # Add jobs in reverse priority order
        self.queue.add_job(low_priority_job)
        self.queue.add_job(high_priority_job)
        
        # High priority job should come first
        first_job = self.queue.get_next_job()
        self.assertEqual(first_job.id, "high-priority")
        
        second_job = self.queue.get_next_job()
        self.assertEqual(second_job.id, "low-priority")
    
    def test_update_job(self):
        """Test updating job status"""
        job = TranscriptionJob(
            id="test-id",
            path="/test/audio.mp3",
            language="en"
        )
        
        self.queue.add_job(job)
        
        # Update job status
        success = self.queue.update_job("test-id", status=JobStatus.RUNNING, progress=50)
        self.assertTrue(success)
        
        # Verify update
        updated_job = self.queue.get_job("test-id")
        self.assertEqual(updated_job.status, JobStatus.RUNNING)
        self.assertEqual(updated_job.progress, 50)
    
    def test_delete_queued_job(self):
        """Test deleting a queued job"""
        job = TranscriptionJob(
            id="test-id",
            path="/test/audio.mp3",
            language="en"
        )
        
        self.queue.add_job(job)
        
        # Delete job
        success = self.queue.delete_job("test-id")
        self.assertTrue(success)
        
        # Verify deletion
        retrieved_job = self.queue.get_job("test-id")
        self.assertIsNone(retrieved_job)
    
    def test_cannot_delete_running_job(self):
        """Test that running jobs cannot be deleted"""
        job = TranscriptionJob(
            id="test-id",
            path="/test/audio.mp3",
            language="en"
        )
        
        self.queue.add_job(job)
        self.queue.update_job("test-id", status=JobStatus.RUNNING)
        
        # Try to delete running job
        success = self.queue.delete_job("test-id")
        self.assertFalse(success)
        
        # Verify job still exists
        retrieved_job = self.queue.get_job("test-id")
        self.assertIsNotNone(retrieved_job)
    
    def test_get_stats(self):
        """Test getting queue statistics"""
        # Add jobs with different statuses
        queued_job = TranscriptionJob(id="queued", path="/test/1.mp3", language="en")
        running_job = TranscriptionJob(id="running", path="/test/2.mp3", language="en")
        completed_job = TranscriptionJob(id="completed", path="/test/3.mp3", language="en")
        
        self.queue.add_job(queued_job)
        self.queue.add_job(running_job)
        self.queue.add_job(completed_job)
        
        # Update statuses
        self.queue.update_job("running", status=JobStatus.RUNNING)
        self.queue.update_job("completed", status=JobStatus.COMPLETED)
        
        stats = self.queue.get_stats()
        self.assertEqual(stats['Queued'], 1)
        self.assertEqual(stats['Running'], 1)
        self.assertEqual(stats['Processed'], 1)


class TestTranscriptionJob(unittest.TestCase):
    """Test the TranscriptionJob dataclass"""
    
    def test_job_creation(self):
        """Test creating a transcription job"""
        job = TranscriptionJob(
            id="test-id",
            path="/test/audio.mp3",
            language="en",
            format="srt",
            priority=500
        )
        
        self.assertEqual(job.id, "test-id")
        self.assertEqual(job.path, "/test/audio.mp3")
        self.assertEqual(job.language, "en")
        self.assertEqual(job.format, "srt")
        self.assertEqual(job.priority, 500)
        self.assertEqual(job.status, JobStatus.QUEUED)
        self.assertEqual(job.progress, 0)
        self.assertIsNotNone(job.created_at)
    
    def test_job_to_dict(self):
        """Test converting job to dictionary"""
        job = TranscriptionJob(
            id="test-id",
            path="/test/audio.mp3",
            language="en"
        )
        
        job_dict = job.to_dict()
        
        self.assertEqual(job_dict['id'], "test-id")
        self.assertEqual(job_dict['path'], "/test/audio.mp3")
        self.assertEqual(job_dict['language'], "en")
        self.assertEqual(job_dict['status'], "QUEUED")
        self.assertIn('Duration', job_dict)  # Renamed from duration


if __name__ == '__main__':
    unittest.main()
