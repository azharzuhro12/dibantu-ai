#!/bin/sh
# DibantuAI container entrypoint (Step 11).
#
# Applies database migrations, seeds the business data once (idempotent
# — a restart never duplicates rows or overwrites real orders), then
# starts the API server.

set -e

echo "[entrypoint] applying database migrations (alembic)..."
alembic upgrade head

echo "[entrypoint] seeding business data if empty..."
python -c "from app.db.seed import main; main()"

echo "[entrypoint] starting uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
