# ==========================================
# Stage 1: Build the frontend (Vite React app)
# ==========================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy frontend configuration files
COPY frontend/package*.json ./
COPY frontend/tsconfig*.json ./
COPY frontend/vite.config.ts ./

# Install dependencies
RUN npm ci

# Copy frontend source code and build
COPY frontend/src ./src
COPY frontend/public ./public
COPY frontend/index.html ./
RUN npm run build

# ==========================================
# Stage 2: Create the runtime backend container
# ==========================================
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies needed for audio processing and model downloading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgomp1 \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create models directory and download Piper TTS voices
RUN mkdir -p backend/models && \
    curl -L -f -o backend/models/en_US-ryan-medium.onnx \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx" && \
    curl -L -f -o backend/models/en_US-ryan-medium.onnx.json \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json" && \
    curl -L -f -o backend/models/vi_VN-vais1000-medium.onnx \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx" && \
    curl -L -f -o backend/models/vi_VN-vais1000-medium.onnx.json \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json"

# Copy python dependencies and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-cache Silero VAD model. vad.py loads it via torch.hub.load() at runtime,
# which would download from GitHub on every cold start — fragile under GitHub
# rate-limiting from cloud IPs (can crash-loop the container). Bake it into the
# image's torch hub cache so startup never needs the network for VAD.
RUN python -c "import torch; torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True); print('silero-vad cached')"

# Copy frontend static build from Node stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Copy backend files
COPY backend ./backend

# Environment variables configuration
ENV PYTHONUNBUFFERED=1

# Expose port (standard FastAPI port, Railway will bind to this dynamically using PORT env var)
EXPOSE 8000

# Run uvicorn server, binding to Railway's $PORT at runtime
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
