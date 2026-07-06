"""
src/api/app.py — VYOR AI FastAPI application v2.0.0

Routes:
  GET  /health          — Service health status
  POST /upload          — Upload + ingest a document
  POST /query           — Semantic Q&A with citations
  POST /forget          — [NEW] GDPR: delete all chunks for a source
  POST /exact-search    — [NEW] Verbatim SQLite keyword lookup
  GET  /sources         — [NEW] List all indexed sources with timestamps

Design note:
  QdrantManager is initialised ONCE at startup via the lifespan context.
  Local (file-mode) Qdrant does not support concurrent client connections,
  so we use a module-level singleton and never re-init per request.
"""

import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Module-level singleton — set during lifespan startup
_qdrant_manager = None
_exact_store    = None

load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
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
    global _qdrant_manager, _exact_store
    log.info("VYOR AI starting up ...")

    # Pre-warm embedder
    try:
        from src.rag.ingestion import get_embedder
        get_embedder()
        log.info("Embedder pre-warmed")
    except Exception as e:
        log.warning(f"Embedder pre-warm failed: {e}")

    # Initialise QdrantManager singleton ONCE — prevents file-lock conflicts
    try:
        from src.vector_db.qdrant_manager import QdrantManager
        _qdrant_manager = QdrantManager()
        _exact_store    = _qdrant_manager.exact_store
        log.info("QdrantManager singleton initialised.")
    except Exception as e:
        log.warning(f"QdrantManager init failed: {e}")

    yield  # ← app runs here

    log.info("VYOR AI shutting down.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="VYOR AI — Knowledge Intelligence Assistant",
    version="2.0.0",
    description=(
        "Enterprise Q&A powered by Titans Neural LTM + Hybrid Qdrant Search. "
        "Upload documents, ask questions, receive cited answers. "
        "Supports GDPR-compliant document forgetting and verbatim exact search."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:      str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer:      str
    citations:   List[str]
    confidence:  float
    uncertainty: Optional[bool] = False
    session_id:  Optional[str] = None


class ForgetRequest(BaseModel):
    source: str   # Exact filename as stored during upload, e.g. "report.pdf"


class ExactSearchRequest(BaseModel):
    keyword: str
    limit:   int = 5


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infrastructure"])
async def health():
    """Return service health for all downstream dependencies."""
    global _qdrant_manager
    try:
        q_ok = _qdrant_manager.is_healthy() if _qdrant_manager else False
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

    return {"status": status, "version": "2.0.0", "services": services}


@app.post("/upload", tags=["Documents"])
async def upload(file: UploadFile = File(...)):
    """Upload a document (PDF/DOCX/PPTX/CSV/TXT/MD) and ingest it."""
    from src.api.routes.upload import handle_upload
    return await handle_upload(file)


@app.post("/query", response_model=QueryResponse, tags=["Q&A"])
async def query(req: QueryRequest):
    """Send a natural-language query; receive a cited, grounded answer."""
    from src.api.routes.query import handle_query
    return await handle_query(req.query, req.session_id)


# ── [NEW] GDPR Forgetting API ─────────────────────────────────────────────────

@app.post("/forget", tags=["GDPR / Privacy"])
async def forget(req: ForgetRequest):
    """
    Delete all chunks for a specific document source from Qdrant and
    the SQLite exact-fact store.

    Satisfies GDPR Article 17 'right to be forgotten' at the document level.

    Body:
        { "source": "filename.pdf" }

    Returns:
        {
          "status": "forgotten",
          "source": str,
          "qdrant_deleted": bool,
          "sqlite_deleted": int
        }
    """
    global _qdrant_manager
    if not req.source or not req.source.strip():
        raise HTTPException(status_code=400, detail="'source' field must not be empty.")
    if not _qdrant_manager:
        raise HTTPException(status_code=503, detail="Knowledge base not initialised yet.")

    result = _qdrant_manager.delete_by_source(req.source.strip())
    log.info(f"GDPR /forget executed for source='{req.source}'")
    return {
        "status":         "forgotten",
        "source":         result["source"],
        "qdrant_deleted": result["qdrant_deleted"],
        "sqlite_deleted": result["sqlite_deleted"],
    }


# ── [NEW] Exact Verbatim Search ───────────────────────────────────────────────

@app.post("/exact-search", tags=["Q&A"])
async def exact_search(req: ExactSearchRequest):
    """
    Verbatim SQLite keyword search.

    Useful for exact-fact retrieval that semantic search may miss:
    employee IDs, project codes, contract dates, precise numbers.

    Body:
        { "keyword": "VYOR-ALPHA-7", "limit": 5 }

    Returns:
        { "keyword": str, "results": [ {text, source, doc_id, created_at} ] }
    """
    global _exact_store
    if not req.keyword or not req.keyword.strip():
        raise HTTPException(status_code=400, detail="'keyword' must not be empty.")
    if not _exact_store:
        return {"keyword": req.keyword, "results": []}

    results = _exact_store.exact_search(req.keyword.strip(), limit=req.limit)
    return {"keyword": req.keyword, "results": results}


# ── [NEW] List Indexed Sources ────────────────────────────────────────────────

@app.get("/sources", tags=["Documents"])
async def list_sources():
    """
    List all documents currently indexed in the knowledge base.

    Returns:
        { "sources": [ {source, chunks, last_seen} ] }
    """
    global _exact_store
    if not _exact_store:
        return {"sources": [], "total": 0}

    sources = _exact_store.list_sources()
    return {"sources": sources, "total": len(sources)}
