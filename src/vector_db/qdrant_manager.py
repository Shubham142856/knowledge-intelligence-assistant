"""
src/vector_db/qdrant_manager.py — QdrantManager

Manages the Qdrant vector collection with:
  - Auto-collection creation on first use
  - SHA-256 content-hash deduplication
  - In-memory BM25 index kept in sync
  - Reciprocal Rank Fusion (RRF) hybrid search
  - [NEW] Timestamp payloads on every upsert (ISO-8601 created_at)
  - [NEW] Time-decay scoring: fresher chunks score higher in hybrid results
  - [NEW] delete_by_source() for GDPR forgetting API
  - [NEW] SQLite exact-fact side-store for verbatim retrieval
"""

import os
import hashlib
import logging
import sqlite3
import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv(override=True)

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from rank_bm25 import BM25Okapi
from src.rag.ingestion import get_embedder, EMBEDDING_DIM, EMBEDDER_BACKEND

log = logging.getLogger("vyor_ai.qdrant")


# ── Exact Fact SQLite side-store ───────────────────────────────────────────────

class ExactFactStore:
    """
    SQLite side-store for verbatim exact-match retrieval.

    Complements Qdrant (semantic search) by preserving original chunk text
    for exact lookups: employee IDs, project codes, contract dates, etc.

    Also supports GDPR-compliant selective deletion by source.
    """

    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
        log.info(f"ExactFactStore connected: {db_path}")

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_hash  TEXT PRIMARY KEY,
                text        TEXT NOT NULL,
                source      TEXT NOT NULL,
                doc_id      TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON chunks(created_at)"
        )
        self.conn.commit()

    def insert(
        self,
        chunk_hash: str,
        text: str,
        source: str,
        doc_id: str,
        created_at: str,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_hash, text, source, doc_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (chunk_hash, text, source, doc_id, created_at),
        )
        self.conn.commit()

    def exact_search(self, keyword: str, limit: int = 5) -> list[dict]:
        """
        Case-insensitive LIKE search across all stored chunks.
        Returns verbatim text — not re-ranked, not semantically altered.
        """
        rows = self.conn.execute(
            """SELECT text, source, doc_id, created_at
               FROM chunks
               WHERE text LIKE ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (f"%{keyword}%", limit),
        ).fetchall()
        return [
            {"text": r[0], "source": r[1], "doc_id": r[2], "created_at": r[3]}
            for r in rows
        ]

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a given source (GDPR forget)."""
        cur = self.conn.execute(
            "DELETE FROM chunks WHERE source = ?", (source,)
        )
        self.conn.commit()
        count = cur.rowcount
        log.info(f"ExactFactStore: deleted {count} rows for source='{source}'")
        return count

    def list_sources(self) -> list[dict]:
        """Return all indexed sources with chunk count and last upload time."""
        rows = self.conn.execute(
            """SELECT source, COUNT(*) as chunks, MAX(created_at) as last_seen
               FROM chunks GROUP BY source ORDER BY last_seen DESC"""
        ).fetchall()
        return [
            {"source": r[0], "chunks": r[1], "last_seen": r[2]} for r in rows
        ]


# ── QdrantManager ──────────────────────────────────────────────────────────────

class QdrantManager:
    """
    Wraps Qdrant with:
    - Auto-collection creation on first use.
    - SHA-256 content-hash IDs → no duplicates on re-upload.
    - In-memory BM25 index kept in sync with Qdrant corpus.
    - RRF (dense + BM25) hybrid search.
    - Timestamp on every chunk (created_at) for time-decay scoring.
    - delete_by_source() for GDPR /forget endpoint.
    - ExactFactStore SQLite side-store for verbatim lookups.

    Collection name is backend-specific to avoid dimension conflicts:
      local       → vyor_knowledge_base          (384-dim)
      openrouter  → vyor_knowledge_base_2048     (2048-dim)
    """

    # Dynamic based on active embedder backend
    COLLECTION  = (
        "vyor_knowledge_base_2048"
        if EMBEDDER_BACKEND == "openrouter"
        else "vyor_knowledge_base"
    )
    VECTOR_DIM  = EMBEDDING_DIM   # 384 (local) | 2048 (openrouter)
    RRF_K       = 60
    TIME_DECAY  = 0.05

    _client_instance = None

    def __init__(self) -> None:
        if QdrantManager._client_instance is None:
            path = os.getenv("QDRANT_PATH")
            if path:
                os.makedirs(path, exist_ok=True)
                QdrantManager._client_instance = QdrantClient(path=path)
                log.info(f"Qdrant singleton initialised (local): {path}")
            else:
                url = os.getenv("QDRANT_URL", "http://localhost:6333")
                QdrantManager._client_instance = QdrantClient(url=url, timeout=20)
                log.info(f"Qdrant singleton connected (remote): {url}")

        self.client   = QdrantManager._client_instance
        self.embedder = get_embedder()
        self._corpus:  list[str] = []
        self._meta:    list[dict] = []       # parallel metadata for BM25 corpus
        self._bm25:    Optional[BM25Okapi] = None
        self._ensure_collection()

        # Exact fact SQLite side-store
        db_dir  = os.getenv("QDRANT_PATH", "data/qdrant_db")
        db_path = os.path.join(db_dir, "exact_facts.db")
        self.exact_store = ExactFactStore(db_path)

        # Optional MS-MARCO Cross-Encoder reranker
        self.reranker = None
        if os.getenv("USE_RERANKER", "true").lower() == "true":
            try:
                from sentence_transformers import CrossEncoder
                log.info("Loading Cross-Encoder reranker cross-encoder/ms-marco-MiniLM-L-6-v2...")
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                log.info("Cross-Encoder reranker ready.")
            except Exception as e:
                log.warning(f"Could not initialize Cross-Encoder: {e}. Falling back to standard RRF.")

    # ── Collection setup ──────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
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

    # ── Deduplication ─────────────────────────────────────────────────────────

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    # ── Save chunks ───────────────────────────────────────────────────────────

    def save(
        self,
        chunks:     list[str],
        embeddings,                 # np.ndarray (N, 384)
        source:     str,
        doc_id:     str = "",
    ) -> int:
        """
        Upsert chunks into Qdrant + SQLite exact-fact store.
        Each chunk carries a created_at ISO-8601 timestamp for time-decay.
        """
        if len(chunks) == 0:
            return 0

        now = datetime.datetime.utcnow().isoformat()
        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_hash = self._content_hash(chunk)
            points.append(
                PointStruct(
                    id=chunk_hash,
                    vector=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                    payload={
                        "text":         chunk,
                        "source":       source,
                        "doc_id":       doc_id,
                        "chunk_index":  i,
                        "total_chunks": len(chunks),
                        "created_at":   now,          # [NEW] timestamp
                    },
                )
            )
            # Write to exact-fact SQLite side-store
            self.exact_store.insert(chunk_hash, chunk, source, doc_id, now)

        self.client.upsert(collection_name=self.COLLECTION, points=points)
        log.info(f"Upserted {len(points)} points from '{source}' (ts={now})")

        # Rebuild BM25 index
        self._corpus.extend(chunks)
        self._meta.extend(
            {"source": source, "doc_id": doc_id, "created_at": now}
            for _ in chunks
        )
        tokenised  = [c.lower().split() for c in self._corpus]
        self._bm25 = BM25Okapi(tokenised)

        return len(points)

    # ── GDPR Forgetting ───────────────────────────────────────────────────────

    def delete_by_source(self, source: str) -> dict:
        """
        Delete all chunks from Qdrant and ExactFactStore for a given source.

        Satisfies GDPR Article 17 'right to be forgotten' at the document level.

        Returns:
            {"qdrant_deleted": int, "sqlite_deleted": int, "source": str}
        """
        log.info(f"Forgetting source: '{source}'")

        # Delete from Qdrant by filtering on source payload field
        self.client.delete(
            collection_name=self.COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )

        # Rebuild BM25 corpus excluding deleted source
        remaining      = [
            (c, m) for c, m in zip(self._corpus, self._meta)
            if m.get("source") != source
        ]
        if remaining:
            self._corpus, self._meta = zip(*remaining)
            self._corpus = list(self._corpus)
            self._meta   = list(self._meta)
            self._bm25   = BM25Okapi([c.lower().split() for c in self._corpus])
        else:
            self._corpus = []
            self._meta   = []
            self._bm25   = None

        # Delete from SQLite exact-fact store
        sqlite_count = self.exact_store.delete_by_source(source)

        log.info(f"Forget complete for '{source}': Qdrant ✓, SQLite {sqlite_count} rows")
        return {"source": source, "qdrant_deleted": True, "sqlite_deleted": sqlite_count}

    # ── Hybrid search ─────────────────────────────────────────────────────────

    def hybrid_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        RRF fusion of dense cosine + BM25, with optional time-decay bonus.

        Time-decay: chunks uploaded recently receive a small additive bonus
        to their RRF score, favouring freshness over stale content.

        Returns:
            List of dicts sorted by descending composite score (length=top_k):
            {text, score, source, doc_id, created_at}
        """
        q_emb = self.embedder.encode(query)

        # — Dense retrieval —
        dense_hits = self.client.search(
            collection_name=self.COLLECTION,
            query_vector=q_emb.tolist(),
            limit=top_k * 2,
        )

        # — BM25 sparse retrieval —
        bm25_ranked: list[tuple[float, str, dict]] = []
        if self._bm25 and self._corpus:
            tokens   = query.lower().split()
            scores   = self._bm25.get_scores(tokens)
            top_idx  = np.argsort(scores)[::-1][: top_k * 2]
            bm25_ranked = [
                (float(scores[i]), self._corpus[i], self._meta[i])
                for i in top_idx
            ]

        # — RRF fusion —
        rrf:  dict[str, float] = {}
        meta: dict[str, dict]  = {}

        for rank, hit in enumerate(dense_hits):
            payload = hit.payload or {}
            key     = payload.get("text", "")
            if key:
                rrf[key]  = rrf.get(key, 0.0) + 1.0 / (self.RRF_K + rank + 1)
                meta[key] = payload

        for rank, (_, text, m) in enumerate(bm25_ranked):
            if text:
                rrf[text]  = rrf.get(text, 0.0) + 1.0 / (self.RRF_K + rank + 1)
                if text not in meta:
                    meta[text] = m

        # — Time-decay bonus —
        if self.TIME_DECAY > 0:
            now = datetime.datetime.utcnow()
            for text, payload in meta.items():
                ts_str = payload.get("created_at", "")
                if ts_str:
                    try:
                        ts  = datetime.datetime.fromisoformat(ts_str)
                        age_days = max((now - ts).total_seconds() / 86400.0, 0)
                        # Exponential decay: bonus decreases with age
                        bonus = self.TIME_DECAY * (0.5 ** (age_days / 30.0))
                        rrf[text] = rrf.get(text, 0.0) + bonus
                    except ValueError:
                        pass

        # Rerank the top candidates using Cross-Encoder if available
        sorted_candidates = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]
        if self.reranker and sorted_candidates:
            try:
                pairs = [(query, text) for text, _ in sorted_candidates]
                rerank_scores = self.reranker.predict(pairs)
                reranked = [
                    (text, float(score))
                    for (text, _), score in zip(sorted_candidates, rerank_scores)
                ]
                sorted_results = sorted(reranked, key=lambda x: x[1], reverse=True)[:top_k]
            except Exception as e:
                log.warning(f"Reranking failed: {e}. Falling back to RRF rankings.")
                sorted_results = sorted_candidates[:top_k]
        else:
            sorted_results = sorted_candidates[:top_k]

        return [
            {
                "text":       text,
                "score":      round(score, 6),
                "source":     meta.get(text, {}).get("source", "unknown"),
                "doc_id":     meta.get(text, {}).get("doc_id", ""),
                "created_at": meta.get(text, {}).get("created_at", ""),
            }
            for text, score in sorted_results
        ]

    def exact_search(self, keyword: str, limit: int = 5) -> list[dict]:
        """
        Verbatim SQLite search for exact keywords (IDs, codes, dates, names).
        Returns full original chunks that contain the keyword.
        """
        return self.exact_store.exact_search(keyword, limit=limit)

    def list_sources(self) -> list[dict]:
        """Return all ingested sources with chunk counts and timestamps."""
        return self.exact_store.list_sources()

    # ── Health check ──────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
