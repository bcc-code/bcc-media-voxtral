# Vox Audio Transcription

Audio transcription using Mistral AI's Voxtral API. Supports long audio files and provides a REST API for job-based processing.

## Features

- Transcribes audio/video files using Mistral AI
- Handles long files by splitting into segments
- REST API with job queue
- Outputs SRT subtitles or JSON format
- Docker support

## Installation

### Using UV
```bash
git clone <repository>
uv sync
```

### Using Docker
```bash
docker build -t transcription-api .
docker run -d -p 8888:8888 -e MISTRAL_API_KEY=your_key transcription-api
```
## Web API Usage

Start the server:
```bash
MISTRAL_API_KEY=your_key uv run python web_server.py
```

### API Endpoints

#### Create Transcription Job
```http
POST /transcription/job
Content-Type: application/json

{
  "path": "/path/to/audio.mp3",
  "language": "en",
  "format": "json",
  "priority": 1000
}
```

Response:
```json
{
  "job_id": "uuid-string",
  "status": "QUEUED"
}
```

#### Get Job Status
```http
GET /transcription/job/{job_id}
```

Response (completed):
```json
{
  "job_id": "uuid-string",
  "status": "COMPLETED",
  "result": {
    "text": "Full transcription...",
    "segments": [
      {
        "text": "Segment text",
        "start": 0.0,
        "end": 5.2,
        "words": [...]
      }
    ]
  }
}
```

#### List All Jobs
```http
GET /transcription/jobs
```

#### Get Queue Stats
```http
GET /stats
```

### Configuration

- **Supported formats**: `json`, `srt`
- **Supported models**: `voxtral-small-2507`, `voxtral-small-latest`
- **Supported languages**: All 67 Mistral AI language codes
- **Priority**: 1-499 (low), 500-9999 (high)

## Requirements

- Python 3.9+
- Mistral AI API key
- FFmpeg (for audio processing)
