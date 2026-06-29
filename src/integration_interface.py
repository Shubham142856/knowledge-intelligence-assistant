"""
integration_interface.py — Team 1 / Team 2 boundary.

IMPORTANT: This file is PROVIDED BY TEAM 1. Do NOT edit internals.
Team 2 only calls these two functions.

During pre-integration (Days 1-4), stubs are active so infrastructure
can be built and tested independently. Team 1 replaces the bodies on Day 5.
"""

import logging
import random

log = logging.getLogger("vyor_ai.integration")

# ---------------------------------------------------------------------------
# Contract 1 — called by upload.py for each parsed chunk
# ---------------------------------------------------------------------------

def process_incoming_chunk(chunk_text: str, embedding: list[float]) -> str:
    """
    Called by Part B (Team 2) after chunking and embedding each document segment.

    Args:
        chunk_text:  The raw text of this chunk (up to 500 tokens).
        embedding:   384-dimensional float vector from all-MiniLM-L6-v2.

    Returns:
        "memory_updated"  → Titans LTM was updated (high surprise detected).
        "save_to_qdrant"  → Store in Qdrant vector DB (low surprise).

    NOTE: This stub randomly routes 20 % to memory, 80 % to Qdrant.
          Team 1 replaces this logic with the real Surprise gate on Day 5.
    """
    # --- STUB: replace with real Titans surprise gate on Day 5 ---
    decision = "memory_updated" if random.random() < 0.2 else "save_to_qdrant"
    log.debug(f"process_incoming_chunk (stub) → {decision} | len={len(chunk_text)}")
    return decision


# ---------------------------------------------------------------------------
# Contract 2 — called by query.py for every user query
# ---------------------------------------------------------------------------

def run_orchestrator(user_query: str) -> dict:
    """
    Called by Part B (Team 2) when a user sends a natural-language query.

    Args:
        user_query:  Raw question string from the end user.

    Returns:
        {
            "answer":     str   — Plain-language answer.
            "citations":  list  — Source references (filename, chunk index).
            "confidence": float — Score 0.0–1.0 indicating answer reliability.
        }

    NOTE: This stub returns a placeholder response.
          Team 1 replaces this with real multi-agent orchestration on Day 5.
    """
    # --- STUB: replace with real orchestrator call on Day 5 ---
    log.debug(f"run_orchestrator (stub) called | query='{user_query[:60]}'")
    return {
        "answer": (
            f"[STUB — Team 1 not yet integrated] "
            f"Your query was received: '{user_query}'. "
            "Once Team 1's orchestrator is connected on Day 5, "
            "a real cited answer will appear here."
        ),
        "citations": ["stub_source.pdf#chunk_0"],
        "confidence": 0.50,
        "uncertainty": True,
    }
