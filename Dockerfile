# Multi-stage Docker build for Medical Inventory Detection API
# Optimized for production deployment with GPU support

# Stage 1: Base Python environment
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    wget \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Stage 2: Dependencies
FROM base as dependencies

WORKDIR /tmp

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir -r requirements.txt

# Stage 3: Development environment
FROM dependencies as development

WORKDIR /app

# Copy application code
COPY . .

# Install development dependencies
COPY requirements-dev.txt .
RUN pip3 install --no-cache-dir -r requirements-dev.txt

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8001

# Default command for development
CMD ["python3", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Production build
FROM dependencies as production-build

WORKDIR /app

# Copy only necessary files
COPY api_server.py .
COPY medical_yolo.py .
COPY dataset_manager.py .
COPY evaluation_framework.py .
COPY training_pipeline.py .
COPY websocket_server.py .
COPY model_comparison.py .
COPY annotation_tools.py .
COPY data_quality_checker.py .
COPY yolo_app.py .
COPY app.py .

# Copy models directory if it exists
COPY models/ ./models/ 2>/dev/null || true

# Copy static files
COPY static/ ./static/ 2>/dev/null || true
COPY templates/ ./templates/ 2>/dev/null || true

# Create necessary directories
RUN mkdir -p models checkpoints logs data uploads temp

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Stage 5: Production runtime
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODEL_PATH=/app/models \
    LOG_LEVEL=INFO \
    WORKERS=4

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy Python packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application from build stage
COPY --from=production-build /app /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8001

# Production command
CMD ["python3", "-m", "gunicorn", "api_server:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300"]

# Stage 6: GPU-optimized production
FROM production as gpu-production

# Install additional CUDA runtime libraries
RUN apt-get update && apt-get install -y \
    cuda-runtime-11-8 \
    && rm -rf /var/lib/apt/lists/*

# Set GPU-specific environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Override command for GPU deployment
CMD ["python3", "-m", "gunicorn", "api_server:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "600"]