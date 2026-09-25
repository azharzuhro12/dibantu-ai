# DibantuAI — local development / portfolio image (Step 9 / Step 11).
# Lightweight Python base; dependencies come from requirements.txt.

FROM python:3.12-slim

# Do not write .pyc files; stream logs unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so code changes do not bust the dep layer.
# CPU-only torch comes from the PyTorch CPU index BEFORE requirements so
# sentence-transformers finds it satisfied and pip never pulls the CUDA
# build (~2 GB of nvidia-* wheels).
COPY requirements.txt ./
RUN pip install --no-cache-dir --retries 5 --timeout 120 torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

# Application code, Alembic migrations, and the container entrypoint.
# .env is excluded via .dockerignore — secrets reach the container
# through environment variables (compose env_file), never a baked file.
COPY app ./app
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Demo knowledge documents (source-controlled) + an empty home for the
# RAG vector store, which lives in the rag_data volume at runtime —
# generated embeddings are never baked into the image or committed.
COPY data/knowledge ./data/knowledge
RUN mkdir -p /app/data/rag

# Run as an unprivileged user.
RUN useradd --create-home appuser \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
