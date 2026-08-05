# Macromancer — single-container image running the FastAPI backend (8000) and
# the Streamlit frontend (8501) under supervisord.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Frontend talks to the backend inside the same container.
    MACROMANCER_API_URL=http://localhost:8000

# System deps:
#   libomp-dev        -> OpenMP runtime required by XGBoost
#   tesseract-ocr     -> OCR for restaurant receipts (Phase 5)
#   gcc/build-essential -> build any wheels that need compiling
#   curl              -> container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        build-essential \
        libomp-dev \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "supervisor>=4.2"

# Copy the rest of the project (see .dockerignore for exclusions).
COPY . .

# Backend API + Streamlit UI.
EXPOSE 8000 8501

# Healthcheck hits the backend's liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["supervisord", "-c", "/app/supervisord.conf"]
