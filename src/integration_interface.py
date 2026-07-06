"""
src/integration_interface.py — Real Team 1/2 Integration Bridge (v2)

Connects Team 2's document parser + hybrid search backend with
Team 1's Titans Neural LTM, Surprise Gate, and Multi-Agent Orchestrator.

Changes in v2:
  - Uses real VYORSurpriseGate (Huber Loss + adaptive 80th-percentile threshold)
  - Fallback: if VYORIntegrationInterface (Team 1) is unavailable, uses local
    VYORSurpriseGate + VYOROrchestrator directly
  - Correctly handles dim=384 alignment with VYORNeuralBrain
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path

log = logging.getLogger("vyor_ai.integration")

# ── Try importing Team 1's interface (may or may not be available) ────────────
_TEAM1_AVAILABLE = False
try:
    # Add project root to path so integration_interface.py at root is found
    root = str(Path(__file__).parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from integration_interface import VYORIntegrationInterface
    _TEAM1_AVAILABLE = True
    log.info("Team 1 VYORIntegrationInterface loaded successfully.")
except ImportError:
    log.warning(
        "VYORIntegrationInterface (Team 1) not found. "
        "Falling back to local VYORSurpriseGate + VYOROrchestrator."
    )

# ── Always import our own components ─────────────────────────────────────────
from src.vector_db.qdrant_manager import QdrantManager
from src.rag.ingestion import EMBEDDING_DIM
from surprise_gate import VYORSurpriseGate
from titans_memory import VYORNeuralBrain
from orchestrator import VYOROrchestrator

# ── Singletons ────────────────────────────────────────────────────────────────
_qdrant      = QdrantManager()
_gate        = VYORSurpriseGate(alpha=0.1, delta=1.0)
_brain       = VYORNeuralBrain(dim=EMBEDDING_DIM, chunk_size=64)
_orchestrator: VYOROrchestrator | None = None


def _get_orchestrator() -> VYOROrchestrator:
    """Lazy-load orchestrator (singleton)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = VYOROrchestrator(retrieval_fn=_qdrant_search_fn)
        log.info("VYOROrchestrator initialised with Qdrant retrieval function.")
    return _orchestrator


def _qdrant_search_fn(query: str) -> list[str]:
    """Search Qdrant hybrid store and return formatted context strings."""
    results = _qdrant.hybrid_search(query, top_k=5)
    return [f"Source: {r['source']} | {r['text']}" for r in results]


# ── Team 1 bridge (if available) ──────────────────────────────────────────────
_bridge = None
if _TEAM1_AVAILABLE:
    try:
        _bridge = VYORIntegrationInterface(
            partner_qdrant_search_fn=_qdrant_search_fn,
            partner_qdrant_insert_fn=lambda text: True,  # no-op (we batch-insert in upload.py)
            dim=384,
        )
        log.info("Team 1 bridge initialised.")
    except Exception as exc:
        log.warning(f"Team 1 bridge init failed ({exc}). Using local fallback.")
        _bridge = None


# ── Contract 1: process_incoming_chunk ───────────────────────────────────────

def process_incoming_chunk(chunk_text: str, embedding: list[float]) -> str:
    """
    Routes an incoming chunk to Titans LTM or Qdrant based on surprise score.

    Args:
        chunk_text: Raw text of the chunk.
        embedding:  {EMBEDDING_DIM}-dimensional float list from active embedder.

    Returns:
        "memory_updated"  — chunk was routed to Titans LTM
        "save_to_qdrant"  — chunk should be saved to Qdrant
    """
    # Convert to tensor (1, 1, EMBEDDING_DIM)
    chunk_tensor = torch.tensor(embedding, dtype=torch.float32).view(1, 1, EMBEDDING_DIM)

    # Try Team 1 bridge first
    if _bridge is not None:
        try:
            dest = _bridge.process_incoming_chunk(chunk_tensor, raw_text_meta=chunk_text)
            if dest == "TITANS_NEURAL_MEMORY":
                return "memory_updated"
            return "save_to_qdrant"
        except Exception as exc:
            log.warning(f"Team 1 bridge.process_incoming_chunk failed ({exc}). Using local gate.")

    # ── Local fallback: VYORSurpriseGate + VYORNeuralBrain ───────────────────
    # Use current LTM weights as memory reference vector
    # For comparison, retrieve a zero-state memory vector
    with torch.no_grad():
        try:
            memory_ref = _brain.recall_info(chunk_tensor)
            # Flatten to same shape for Huber Loss
            ref_flat   = memory_ref.view(-1)
            inp_flat   = chunk_tensor.view(-1)

            # Pad/truncate to same length
            min_len = min(len(ref_flat), len(inp_flat))
            ref_flat = ref_flat[:min_len]
            inp_flat = inp_flat[:min_len]

        except Exception:
            ref_flat = torch.zeros_like(chunk_tensor.view(-1))
            inp_flat = chunk_tensor.view(-1)

    huber_loss = _gate.compute_huber_loss(inp_flat, ref_flat)
    decision, threshold = _gate.update_and_route(huber_loss)

    log.debug(
        f"Surprise gate: loss={huber_loss:.4f}, "
        f"threshold={threshold:.4f}, decision={decision}"
    )

    if decision == "memory_updated":
        # Commit to LTM
        _brain.learn_new_info(chunk_tensor)
        return "memory_updated"

    return "save_to_qdrant"


# ── Contract 2: run_orchestrator ──────────────────────────────────────────────

def run_orchestrator(user_query: str) -> dict:
    """
    Runs the multi-agent debate orchestrator to answer a user query.

    Args:
        user_query: Natural language question.

    Returns:
        {
            "answer":     str,
            "citations":  list[str],
            "confidence": float      # 0.95 if Critic-approved, 0.40 otherwise
        }
    """
    # Try Team 1 bridge first
    if _bridge is not None:
        try:
            res = _bridge.run_orchestrator(user_query)
            return {
                "answer":     res.get("answer", ""),
                "citations":  res.get("citations", []),
                "confidence": res.get("confidence", 0.0),
            }
        except Exception as exc:
            log.warning(f"Team 1 bridge.run_orchestrator failed ({exc}). Using local orchestrator.")

    # ── Local fallback: VYOROrchestrator ─────────────────────────────────────
    orch   = _get_orchestrator()
    result = orch.execute_query(user_query)
    return {
        "answer":     result.get("answer", ""),
        "citations":  result.get("citations", []),
        "confidence": result.get("confidence", 0.40),
    }
