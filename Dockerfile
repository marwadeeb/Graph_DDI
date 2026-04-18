# ─────────────────────────────────────────────────────────────────────────────
# DDI Checker — HuggingFace Spaces Dockerfile
#
# HuggingFace Spaces requirements:
#   • Container must listen on port 7860 (mapped automatically by HF)
#   • The image is run as a non-root user; data must live in /app
#   • Secrets (API keys) are injected as Space Secrets → environment variables
#
# Large binary assets (FAISS index, GNN weights) should be committed with
# Git LFS **or** downloaded at build time using the RUN curl/wget block below.
#
# Build locally:
#   docker build -t ddi-checker .
#   docker run -p 7860:7860 ddi-checker
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# System deps — libgomp is required by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer independently of code.
COPY requirements.txt .

# Install CPU-only PyTorch (much smaller image; HuggingFace Spaces free tier
# has no GPU). Override the default torch index so we don't pull CUDA wheels.
RUN pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric CPU wheels
RUN pip install --no-cache-dir \
        torch_geometric \
        pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.2.2+cpu.html || \
    pip install --no-cache-dir torch_geometric

# Install the rest of the project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Pre-download the sentence-transformer model so the image is self-contained.
# Cached in the HuggingFace model cache inside the image (no network at runtime).
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
EOF

# ── HuggingFace Spaces runs on port 7860 ─────────────────────────────────────
EXPOSE 7860

# Set env vars expected by the app
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_NO_TF=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    FLASK_ENV=production

# Gunicorn: single worker (avoids double-loading the 2.5 GB FAISS index),
# generous timeout for the first request that triggers background FAISS load.
CMD ["python", "-m", "gunicorn", \
     "--workers", "1", \
     "--threads", "4", \
     "--bind", "0.0.0.0:7860", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app:app"]
