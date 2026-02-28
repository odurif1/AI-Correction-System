# Stage 1: Python builder
FROM python:3.12-slim AS python-builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install system-wide (not --user)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Node builder for Next.js
FROM node:20-alpine AS frontend-builder

WORKDIR /web

# Copy package files
COPY web/package*.json ./
RUN npm ci

# Copy web source and build
COPY web/ ./
RUN npm run build

# Stage 3: Runtime
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage (installed system-wide)
COPY --from=python-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY src/ /app/

# Copy Next.js build artifacts from frontend-builder
COPY --from=frontend-builder /web/.next /app/web/.next
COPY --from=frontend-builder /web/public /app/web/public
COPY --from=frontend-builder /web/package.json /app/web/

# Ensure Python can find installed packages
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run with gunicorn and uvicorn workers for production
CMD ["gunicorn", "api.app:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
