"""
src/integration_interface.py — Real Team 1/2 integration bridge.

This connects Team 2's document parser + hybrid search backend with
Team 1's Titans Neural LTM, Surprise Gate, and Multi-Agent Orchestrator.
"""

import torch
import logging
from integration_interface import VYORIntegrationInterface
from src.vector_db.qdrant_manager import QdrantManager

log = logging.getLogger("vyor_ai.integration")

# Initialize QdrantManager to bind search methods
_qdrant = QdrantManager()

def partner_qdrant_search(query: str) -> list[str]:
    """
    Search function passed to Team 1's orchestrator.
    Retrieves top 5 chunks from the hybrid RRF vector store.
    """
    results = _qdrant.hybrid_search(query, top_k=5)
    return [f"Source: {r['source']} | {r['text']}" for r in results]

def partner_qdrant_insert_noop(text: str) -> bool:
    """
    No-op insert function. Since Team 2's upload.py routes chunks and
    batch-upserts them to Qdrant, we do not double-insert inside the hook.
    """
    return True

# Initialize the Team 1 interface with our search and insert functions (matching dim=384)
bridge = VYORIntegrationInterface(
    partner_qdrant_search_fn=partner_qdrant_search,
    partner_qdrant_insert_fn=partner_qdrant_insert_noop,
    dim=384
)

def process_incoming_chunk(chunk_text: str, embedding: list[float]) -> str:
    """
    Routes incoming chunk to the Surprise Gate for LTM or Qdrant decision.
    """
    # Convert list of floats (384-dim) to torch.Tensor of shape (1, 1, 384)
    chunk_tensor = torch.tensor(embedding, dtype=torch.float32).view(1, 1, 384)
    
    # Process through the surprise gate
    dest = bridge.process_incoming_chunk(chunk_tensor, raw_text_meta=chunk_text)
    
    # Map their returns ("TITANS_NEURAL_MEMORY" | "PARTNER_QDRANT_DB") to our contract's returns:
    if dest == "TITANS_NEURAL_MEMORY":
        return "memory_updated"
    else:
        return "save_to_qdrant"

def run_orchestrator(user_query: str) -> dict:
    """
    Passes user query to Team 1's debate-based orchestrator.
    """
    res = bridge.run_orchestrator(user_query)
    return {
        "answer": res.get("answer", ""),
        "citations": res.get("citations", []),
        "confidence": res.get("confidence", 0.0)
    }
