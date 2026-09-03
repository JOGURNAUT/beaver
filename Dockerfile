# Beaver — container image for Azure Container Apps
#
# Two things matter here, both about image size and cold start:
#
#  1. torch is NOT in requirements.txt — sentence-transformers pulls it in
#     transitively, and the default PyPI wheel is the CUDA build (~4 GB).
#     Installing the CPU-only wheel FIRST means pip sees torch already
#     satisfied later, and the image lands around 1 GB instead.
#
#  2. all-MiniLM-L6-v2 (~80 MB) is downloaded on first use. Baking it into
#     the image at build time keeps cold starts fast and makes the container
#     work even if Hugging Face is unreachable.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# CPU-only torch, before anything can drag in the CUDA build
RUN pip install --no-cache-dir torch \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the embedding model into the image
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

# SQLite lives here. Mount an Azure Files share at /app/data to make it
# survive restarts (see deploy-azure.sh, step 5).
ENV DB_PATH=/app/data/research.db
RUN mkdir -p /app/data

EXPOSE 8501

# enableCORS / enableXsrfProtection off because Container Apps terminates
# TLS at its own ingress and proxies through to us.
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
