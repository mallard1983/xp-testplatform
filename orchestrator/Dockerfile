# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build
# Output goes to orchestrator/static/ (vite.config.js outDir), but at build time
# we copy it explicitly in Stage 2

# ── Stage 2: Python runtime ────────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator/ .

# Copy built frontend static files from Stage 1
COPY --from=frontend-builder /frontend/dist/ ./static/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]
