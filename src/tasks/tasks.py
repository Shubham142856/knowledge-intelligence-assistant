"""
src/tasks/tasks.py — Asynchronous Ingestion & Document Processing Tasks
"""

import os
import logging
import numpy as np
from src.tasks.celery_app import celery_app
from src.rag.ingestion import DocumentIngestor, get_embedder
from src.vector_db.qdrant_manager import QdrantManager

log = logging.getLogger("vyor_ai.tasks")


@celery_app.task(name="tasks.ingest_document", bind=True, max_retries=3)
def ingest_document(self, file_path: str, source_name: str, doc_id: str = "") -> dict:
    """
    Celery task to asynchronously parse, chunk, embed, and index files
    into the Qdrant Vector database and ExactFactStore SQLite database.
    """
    log.info(f"Task {self.request.id}: starting ingestion for source='{source_name}' path='{file_path}'")
    
    if not os.path.exists(file_path):
        error_msg = f"Document file not found at path: {file_path}"
        log.error(error_msg)
        return {"success": False, "error": error_msg}

    try:
        # 1. Parse and chunk document
        ingestor = DocumentIngestor()
        chunks = ingestor.parse(file_path)
        
        if not chunks:
            log.warning(f"No text extracted from '{source_name}'. Completing successfully with 0 chunks.")
            return {
                "success": True,
                "chunks_processed": 0,
                "chunks_saved": 0,
                "source": source_name
            }

        # 2. Compute embeddings
        embedder = get_embedder()
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        
        # Ensure embeddings are a numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        # 3. Save to vector db and exact fact store
        db = QdrantManager()
        saved_count = db.save(chunks, embeddings, source=source_name, doc_id=doc_id)

        # 4. Cleanup temp upload file if stored in temporary folder
        if "uploads" in file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                log.info(f"Cleaned up temporary upload file: {file_path}")
            except Exception as cleanup_err:
                log.warning(f"Failed to delete temp file {file_path}: {cleanup_err}")

        return {
            "success": True,
            "chunks_processed": len(chunks),
            "chunks_saved": saved_count,
            "source": source_name,
            "doc_id": doc_id
        }

    except Exception as exc:
        log.error(f"Task {self.request.id} failed: {exc}")
        # Exponential backoff retry on transient issues (e.g. database locks or network timeouts)
        try:
            self.retry(exc=exc, countdown=2 ** self.request.retries * 5)
        except Exception:
            # Reached max retries
            return {"success": False, "error": str(exc)}
