# High-performance, lightweight Python image for fast builds and minimal attack surface
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install curl for docker healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Leverage Docker layer caching by copying requirements first
COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application source code, data, and configs
COPY src/ /app/src/
COPY data/ /app/data/
COPY params.yaml /app/params.yaml

EXPOSE 8000

# Start FastAPI serving via Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
