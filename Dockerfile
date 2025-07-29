# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml .
COPY uv.lock .

# Install Python dependencies using UV
RUN uv sync --frozen

# Copy application files
COPY transcribe_audio.py .
COPY web_server.py .

# Create directories for temporary files and output
RUN mkdir -p /app/temp /app/output

# Expose the port the app runs on
EXPOSE 8888

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/stats || exit 1

# Run the web server using UV
CMD ["uv", "run", "web_server.py"]
