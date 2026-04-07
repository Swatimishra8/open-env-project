# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency manifest first for layer caching
COPY requirements.txt .

# Install Python dependencies (same set as local venv; no torch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY env/ ./env/
COPY data/ ./data/
COPY app.py .
COPY config.py .
COPY openenv.yaml .
COPY inference.py .
COPY README.md .

# ── Runtime ───────────────────────────────────────────────────────────────────

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
#     CMD curl -f http://localhost:7860/health || exit 1

# Start FastAPI server
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
