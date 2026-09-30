# SkinSight Deployment Guide

This guide covers deploying SkinSight to production environments.

## Prerequisites

- Python 3.9 or higher
- OpenRouter API key (for educational cards and chat features)
- Hugging Face model will be downloaded automatically at runtime

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required for educational cards and chat
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Override server host (default: 0.0.0.0)
SKINSIGHT_HOST=0.0.0.0

# Optional: Override server port (default: 8001)
SKINSIGHT_PORT=8001

# Optional: Override Hugging Face model ID (default: PrernaUtane/skin-lesion-densenet121)
HF_MODEL_ID=PrernaUtane/skin-lesion-densenet121
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your OpenRouter API key
5. Start the server:
   ```bash
   python run.py
   ```

The application will be available at `http://localhost:8001`

## Production Deployment

### Recommended Setup

SkinSight is a FastAPI application that can be deployed using:

1. **Uvicorn directly** (simple, for single-instance deployments)
2. **Gunicorn with Uvicorn workers** (for production with multiple workers)
3. **Docker** (for containerized deployments)
4. **Cloud platforms** (Render, Railway, AWS, GCP, etc.)

### Production Start Command

Using Gunicorn with Uvicorn workers (recommended for production):

```bash
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8001 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

Or using Uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 4
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for model cache
RUN mkdir -p /root/.cache/huggingface

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "run.py"]
```

Build and run:

```bash
docker build -t skinsight .
docker run -p 8001:8001 --env-file .env skinsight
```

### Cloud Platform Notes

- **Render/Railway**: Set environment variables in the platform dashboard
- **AWS/GCP**: Ensure sufficient memory for model loading (~2GB minimum)
- **Model weights**: Will be downloaded to `~/.cache/huggingface` on first run
- **API key**: Must be set as environment variable, never committed to code

## Performance Considerations

- **First request**: Will be slower as the Hugging Face model downloads and loads
- **Memory**: DenseNet121 model requires ~500MB RAM
- **GPU**: Not required but can be enabled if available (automatic detection)
- **Concurrent requests**: Use multiple workers for production

## Security Notes

- `.env` file must never be committed to version control
- OpenRouter API key is server-side only, never exposed to browser
- Uploaded images are processed locally and not sent to external APIs
- CORS is enabled for development; restrict origins in production if needed

## Monitoring

The application exposes FastAPI's automatic health check at `/api/status`.

## Troubleshooting

- **Model download fails**: Check internet connectivity and Hugging Face access
- **OpenRouter errors**: Verify API key in `.env` file
- **Memory errors**: Increase available RAM or use a larger instance
- **Port conflicts**: Change `SKINSIGHT_PORT` in `.env`

## Support

For issues or questions, refer to the main README.md or open an issue on GitHub.
