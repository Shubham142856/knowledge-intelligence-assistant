"""
src/api/app.py — VYOR AI FastAPI application

Startup validates Qdrant and pre-warms the embedding model so the
first upload request is fast (no cold-start delay).
"""

import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("vyor_ai")


# ── Startup / shutdown lifespan ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("VYOR AI starting up ...")

    # Pre-warm the embedder so first upload has no cold-start delay
    try:
        from src.rag.ingestion import get_embedder
        get_embedder()   # loads all-MiniLM-L6-v2 into memory once
        log.info("Embedder pre-warmed")
    except Exception as e:
        log.warning(f"Embedder pre-warm failed: {e}")

    # Validate Qdrant
    try:
        path = os.getenv("QDRANT_PATH")
        if path:
            from qdrant_client import QdrantClient
            QdrantClient(path=path).get_collections()
            log.info(f"Qdrant connected (local: {path})")
        else:
            from qdrant_client import QdrantClient
            QdrantClient(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                timeout=5,
            ).get_collections()
            log.info("Qdrant connected (remote)")
    except Exception as e:
        log.warning(f"Qdrant not ready at startup: {e}")

    yield   # ← app runs here

    log.info("VYOR AI shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="VYOR AI — Knowledge Intelligence Assistant",
    version="1.0.0",
    description=(
        "Internal Q&A powered by Titans Architecture. "
        "Upload documents, ask questions, receive cited answers."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    uncertainty: Optional[bool] = False
    session_id: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infrastructure"])
async def health():
    """Return service health map for all downstream dependencies."""
    from src.vector_db.qdrant_manager import QdrantManager
    try:
        q_ok = QdrantManager().is_healthy()
    except Exception:
        q_ok = False

    is_local = os.getenv("QDRANT_PATH") is not None
    status   = "healthy" if q_ok else "degraded"
    services = {"qdrant": "ok" if q_ok else "unavailable"}

    if not is_local:
        try:
            import redis as _redis
            _redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                socket_connect_timeout=2,
            ).ping()
            r_ok = True
        except Exception:
            r_ok = False

        services["redis"] = "ok" if r_ok else "unavailable"
        if not r_ok:
            status = "degraded"

    return {"status": status, "version": "1.0.0", "services": services}


@app.post("/upload", tags=["Documents"])
async def upload(file: UploadFile = File(...)):
    """Upload a document (PDF/DOCX/PPTX/CSV/TXT/MD) and ingest it."""
    from src.api.routes.upload import handle_upload
    return await handle_upload(file)


@app.post("/query", response_model=QueryResponse, tags=["Q&A"])
async def query(req: QueryRequest):
    """Send a natural-language query; receive an answer with citations."""
    from src.api.routes.query import handle_query
    return await handle_query(req.query, req.session_id)
