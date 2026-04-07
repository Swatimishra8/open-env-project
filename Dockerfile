# ── Build stage ──────────────────────────────────────────────────────────────
# Use a more stable base image with explicit tag
FROM python:3.11.10-slim-bookworm

# Set environment variables for better Python behavior in containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps with retry logic for robustness
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies with retry and timeout
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --timeout 120 --retries 5 --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy application code
COPY env/ ./env/
COPY data/ ./data/
COPY server/ ./server/
COPY app.py .
COPY config.py .
COPY openenv.yaml .
COPY inference.py .
COPY __init__.py .
COPY README.md .

# ── Runtime ───────────────────────────────────────────────────────────────────

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start FastAPI server
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
