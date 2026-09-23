"""DibantuAI application entrypoint (FastAPI)."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router as api_router
from app.models.schemas import HealthResponse

#: Browser origins allowed to call this API: the Next.js frontend dev
#: server (3000) plus its first fallback port (3002 -- `next dev`
#: selects it automatically when 3000 is taken by another project),
#: on both localhost and 127.0.0.1. Override with CORS_ALLOW_ORIGINS
#: (comma-separated).
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:3002,"
    "http://127.0.0.1:3002"
)

app = FastAPI(
    title="DibantuAI",
    description="AI Agent for Business Operations.",
    version=__version__,
)

# The web frontend (http://localhost:3000) calls this API directly from
# the browser; without CORS the browser blocks the /api/chat preflight.
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Router for additional endpoints (Step 2+).
app.include_router(api_router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Liveness probe for the service."""
    return HealthResponse(version=__version__)
