# ══════════════════════════════════════════════════════════════
#  DigiKul-v0 — Dockerfile for Hugging Face Spaces
# ══════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    DIGIKUL_TASK=medium

# Create non-root user (HF Spaces runs as uid 1000)
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy application code
COPY models.py /app/models.py
COPY __init__.py /app/__init__.py
COPY client.py /app/client.py
COPY server/ /app/server/
COPY openenv.yaml /app/openenv.yaml

# Switch to non-root user
USER appuser

# Expose default HF Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
