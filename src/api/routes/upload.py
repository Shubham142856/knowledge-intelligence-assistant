"""
src/api/routes/upload.py — POST /upload handler

Flow:
  1. Validate file extension.
  2. Enforce 200MB file size limit to prevent OOM/DoS.
  3. Write to temp file.
  4. Parse -> chunk -> embed via DocumentIngestor.
  5. For each chunk, call Contract 1 (process_incoming_chunk).
  6. Route "save_to_qdrant" chunks -> QdrantManager.save().
  7. Return summary JSON.
"""

import os
import tempfile
import logging
from fastapi import UploadFile, File, HTTPException
from src.rag.ingestion import DocumentIngestor
from src.vector_db.qdrant_manager import QdrantManager
from src.integration_interface import process_incoming_chunk

log = logging.getLogger("vyor_ai.upload")

SUPPORTED = {"pdf", "docx", "pptx", "csv", "txt", "md"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB limit (PRD standard)


async def handle_upload(file: UploadFile = File(...)) -> dict:
    """
    Parse, embed, and route a document.

    Returns JSON:
        {
            "status":          "success",
            "filename":        str,
            "total_chunks":    int,
            "saved_to_qdrant": int,
            "memory_updated":  int
        }
    """
    # ── Validate extension ────────────────────────────────────────────────────
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {sorted(SUPPORTED)}",
        )

    # ── Enforce 200MB size limit ──────────────────────────────────────────────
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
        
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / (1024*1024):.1f} MB). Max allowed is 200 MB."
        )

    # ── Write to tempfile ─────────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ── Ingest ───────────────────────────────────────────────────────────
        ingestor         = DocumentIngestor()
        chunks, embeddings = ingestor.ingest(tmp_path)

        qdrant       = QdrantManager()
        saved_count  = 0
        memory_count = 0

        # ── Route each chunk via Contract 1 ───────────────────────────────────
        qdrant_chunks: list[str]  = []
        qdrant_embeddings         = []

        for chunk, emb in zip(chunks, embeddings):
            decision = process_incoming_chunk(chunk, emb.tolist())
            if decision == "save_to_qdrant":
                qdrant_chunks.append(chunk)
                qdrant_embeddings.append(emb)
                saved_count += 1
            elif decision == "memory_updated":
                memory_count += 1
            else:
                log.warning(f"Unknown routing decision '{decision}' — defaulting to Qdrant")
                qdrant_chunks.append(chunk)
                qdrant_embeddings.append(emb)
                saved_count += 1

        # Batch-upsert Qdrant chunks for efficiency
        if qdrant_chunks:
            qdrant.save(qdrant_chunks, qdrant_embeddings, file.filename)

        log.info(
            f"Upload '{file.filename}': {len(chunks)} chunks -> "
            f"{saved_count} Qdrant, {memory_count} LTM"
        )

        return {
            "status":          "success",
            "filename":        file.filename,
            "total_chunks":    len(chunks),
            "saved_to_qdrant": saved_count,
            "memory_updated":  memory_count,
        }

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    finally:
        os.unlink(tmp_path)
