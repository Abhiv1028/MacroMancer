# Macromancer — single-container image running the FastAPI backend (8000) and
# the Streamlit frontend (8501) under supervisord.
#
# Works as a plain image (`docker run`) and on Hugging Face Spaces (Docker SDK,
# which routes public traffic to app_port 8501 = the Streamlit UI). Runs as a
# non-root user (uid 1000), as HF requires.
FROM python:3.11-slim

# Runtime system deps (NO build toolchain — all Python deps have manylinux wheels):
#   libgomp1     -> GNU OpenMP runtime that XGBoost's Linux wheel links against
#                   (NOT libomp-dev, which is LLVM OpenMP — that was the bug)
#   tesseract-ocr-> OCR for restaurant receipts (Phase 5)
#   curl         -> container HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (Hugging Face Spaces runs containers as uid 1000).
RUN useradd -m -u 1000 user
# Writable, mountable dir for the SQLite DB + RL weights (persisted via a volume
# locally; ephemeral on HF free tier, which is fine — the demo re-seeds).
RUN mkdir -p /data && chown -R user:user /data

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # The UI talks to the backend inside the same container.
    MACROMANCER_API_URL=http://localhost:8000 \
    # Mutable state lives under /data (four slashes => absolute path).
    DATABASE_URL=sqlite:////data/macromentor.db \
    RL_BANDIT_PATH=/data/rl_bandit_weights.json \
    # Seed a demo dataset so the live URL is never blank.
    SEED_DEMO=true

USER user
WORKDIR /home/user/app

# Install Python deps first for better layer caching.
COPY --chown=user:user requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt \
    && pip install --user --no-cache-dir "supervisor>=4.2"

# Copy the rest of the project (see .dockerignore for exclusions).
COPY --chown=user:user . .

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["supervisord", "-c", "/home/user/app/supervisord.conf"]
