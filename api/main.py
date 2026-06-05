"""
FastAPI application — exposes the pipeline programmatically.

Shares the same core pipeline code with Chainlit (Module 19).
Neither FastAPI nor Chainlit own the pipeline — both are consumers.

Run: uvicorn api.main:app --port 8000 --reload

Endpoints:
    POST /analyze/url       → full pipeline from EDGAR URL
    POST /analyze/upload    → full pipeline from PDF upload
    POST /ingest/url        → ingest URL into Qdrant (no pipeline)
    POST /ingest/upload     → ingest PDF into Qdrant (no pipeline)
    GET  /ingest/status     → Qdrant collection info
    POST /export/markdown   → export as Markdown
    POST /export/pdf        → export as PDF download
    POST /export/docx       → export as DOCX download
    POST /export/json       → export as raw JSON
    POST /agents/metrics    → test Metrics Agent
    POST /agents/risk       → test Risk Agent
    POST /agents/news       → test News Agent
    POST /agents/synthesis  → test Synthesis Agent
    GET  /health            → health check + provider status
"""

import logging

from fastapi import FastAPI

from api.models import HealthResponse
from api.routers import analyze, ingest, export, agents
from config import settings

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Financial Research Analyst API",
    description=(
        "Multi-agent financial research pipeline. "
        "Analyses SEC filings, extracts metrics, assesses risks, "
        "monitors news sentiment, and generates investment reports."
    ),
    version="0.1.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc",    # ReDoc
)


# ── Register routers ────────────────────────────────────────────────────────
app.include_router(analyze.router)
app.include_router(ingest.router)
app.include_router(export.router)
app.include_router(agents.router)


# ── Health check ─────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check — verifies API keys and Qdrant connectivity."""
    providers = {
        "hf_inference": bool(settings.hf_token),
        "openrouter": bool(settings.openrouter_api_key),
        "groq": bool(settings.groq_api_key),
        "cerebras": bool(settings.cerebras_api_key),
        "google": bool(settings.google_api_key),
        "nvidia": bool(settings.nvidia_api_key),
        "tavily": bool(settings.tavily_api_key),
    }

    qdrant_ok = False
    try:
        from retrieval.qdrant_store import QdrantStore

        store = QdrantStore()
        store.collection_info()
        qdrant_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        providers=providers,
        qdrant=qdrant_ok,
    )


@app.get("/", tags=["health"])
async def root():
    """Root endpoint — redirect to docs."""
    return {
        "message": "Financial Research Analyst API",
        "docs": "/docs",
        "health": "/health",
    }
