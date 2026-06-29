"""
src/vector_db/qdrant_manager.py — QdrantManager

Manages the Qdrant vector collection: collection creation, idempotent
upserts (content-hash deduplication), and hybrid Dense + BM25 + RRF search.
"""

import os
import hashlib
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from rank_bm25 import BM25Okapi
from src.rag.ingestion import get_embedder

log = logging.getLogger("vyor_ai.qdrant")


class QdrantManager:
    """
    Wraps Qdrant with:
    - Auto-collection creation on first use.
    - SHA-256 content-hash IDs -> no duplicates on re-upload.
    - In-memory BM25 index kept in sync with Qdrant corpus.
    - Reciprocal Rank Fusion (RRF) to blend dense + BM25 results.
    """
    COLLECTION = "vyor_knowledge_base"
    VECTOR_DIM  = 384       # must match all-MiniLM-L6-v2
    RRF_K       = 60        # standard RRF constant

    def __init__(self) -> None:
        path = os.getenv("QDRANT_PATH")
        if path:
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)
            log.info(f"Initialized local in-process Qdrant database at: {path}")
        else:
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            self.client   = QdrantClient(url=url, timeout=20)
            log.info(f"Initialized Qdrant client connecting to: {url}")
        self.embedder = get_embedder()
        self._corpus: list[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist yet."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION not in existing:
            self.client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(
                    size=self.VECTOR_DIM,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            )
            log.info(f"Created Qdrant collection '{self.COLLECTION}'")
        else:
            log.debug(f"Qdrant collection '{self.COLLECTION}' already exists")

    # ── Deduplication ─────────────────────────────────────────────────────────

    @staticmethod
    def _content_hash(text: str) -> str:
        """Deterministic point ID from chunk content — prevents re-upload duplicates."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    # ── Save chunks ───────────────────────────────────────────────────────────

    def save(
        self,
        chunks: list[str],
        embeddings,            # np.ndarray (N, 384)
        source: str,
        doc_id: str = "",
    ) -> int:
        """
        Upsert chunks into Qdrant and rebuild the BM25 index.

        Args:
            chunks:     Text segments to store.
            embeddings: Matching numpy array of shape (N, 384).
            source:     Original filename — stored in payload for citation.
            doc_id:     The EnterpriseRAG-Bench document UUID.

        Returns:
            Number of points upserted.
        """
        if len(chunks) == 0:
            return 0

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=self._content_hash(chunk),
                    vector=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                    payload={
                        "text":         chunk,
                        "source":       source,
                        "doc_id":       doc_id,
                        "chunk_index":  i,
                        "total_chunks": len(chunks),
                    },
                )
            )

        self.client.upsert(collection_name=self.COLLECTION, points=points)
        log.info(f"Upserted {len(points)} points from '{source}' into Qdrant")

        # Rebuild BM25 index with new chunks appended
        self._corpus.extend(chunks)
        tokenised   = [c.lower().split() for c in self._corpus]
        self._bm25  = BM25Okapi(tokenised)

        return len(points)

    # ── Hybrid search ─────────────────────────────────────────────────────────

    def hybrid_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Reciprocal Rank Fusion of dense cosine search + BM25.

        Returns:
            List of dicts: {"text": str, "score": float, "source": str, "doc_id": str}
            Sorted by descending RRF score, length = top_k.
        """
        q_emb = self.embedder.encode(query)

        # — Dense retrieval —
        dense_hits = self.client.search(
            collection_name=self.COLLECTION,
            query_vector=q_emb.tolist(),
            limit=top_k * 2,
        )

        # — BM25 sparse retrieval —
        bm25_ranked: list[tuple[float, str]] = []
        if self._bm25 and self._corpus:
            tokens     = query.lower().split()
            scores     = self._bm25.get_scores(tokens)
            top_idx    = np.argsort(scores)[::-1][: top_k * 2]
            bm25_ranked = [(float(scores[i]), self._corpus[i]) for i in top_idx]

        # — RRF fusion —
        rrf: dict[str, float] = {}
        for rank, hit in enumerate(dense_hits):
            key       = hit.payload["text"]
            rrf[key]  = rrf.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)

        for rank, (_, text) in enumerate(bm25_ranked):
            rrf[text] = rrf.get(text, 0.0) + 1.0 / (self.RRF_K + rank + 1)

        sorted_results = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Enrich with source metadata from dense hits
        meta = {h.payload["text"]: h.payload for h in dense_hits}
        return [
            {
                "text":   text,
                "score":  round(score, 6),
                "source": meta.get(text, {}).get("source", "unknown"),
                "doc_id": meta.get(text, {}).get("doc_id", ""),
            }
            for text, score in sorted_results
        ]

    # ── Health check ──────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """Return True if Qdrant is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
