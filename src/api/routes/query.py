"""
src/api/routes/query.py — POST /query handler

Calls Contract 2 (run_orchestrator) and wraps the response into the
FastAPI QueryResponse schema, adding the uncertainty flag when confidence < 0.65.
"""

import logging
from src.integration_interface import run_orchestrator

log = logging.getLogger("vyor_ai.query")

UNCERTAINTY_THRESHOLD = 0.65


async def handle_query(query: str, session_id: str = None) -> dict:
    """
    Receive a user query, call the orchestrator, and return a structured answer.

    Returns:
        {
            "answer":      str,
            "citations":   list[str],
            "confidence":  float,
            "uncertainty": bool,
            "session_id":  str | None
        }
    """
    if not query or not query.strip():
        return {
            "answer":      "Please enter a question.",
            "citations":   [],
            "confidence":  0.0,
            "uncertainty": False,
            "session_id":  session_id,
        }

    log.info(f"Query | session={session_id} | '{query[:80]}' ...")

    result = run_orchestrator(query)

    confidence  = float(result.get("confidence", 0.75))
    uncertainty = result.get("uncertainty", confidence < UNCERTAINTY_THRESHOLD)

    return {
        "answer":      result["answer"],
        "citations":   result.get("citations", []),
        "confidence":  confidence,
        "uncertainty": uncertainty,
        "session_id":  session_id,
    }
