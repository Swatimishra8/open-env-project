# ── OpenEnv Email Triage Environment ─────────────────────────────────────────
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY env/ ./env/
COPY data/ ./data/
COPY app.py .
COPY openenv.yaml .
COPY inference.py .
COPY config.py .
COPY openenv_validate.py .
COPY openenv .

# Make openenv command available in PATH
RUN chmod +x openenv && ln -s /app/openenv /usr/local/bin/openenv

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Expose port
EXPOSE 7860

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
