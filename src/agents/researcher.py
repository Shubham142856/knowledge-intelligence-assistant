"""
src/agents/researcher.py — Researcher Agent
"""

import logging
from src.agents.base_agent import BaseAgent

log = logging.getLogger("vyor_ai.agents.researcher")


class Researcher(BaseAgent):
    """
    Executes search/retrieval for sub-questions and aggregates unique context chunks.
    """

    def __init__(self):
        super().__init__("Researcher", "You are the Researcher agent. Aggregate facts for each sub-question.")

    def run(self, input_data: dict) -> dict:
        """
        Args:
            input_data: {
                'sub_questions': list[str],
                'retrieve_fn': callable (retrieval hook injecting Qdrant search)
            }
        """
        sub_questions = input_data.get("sub_questions", [])
        retrieve_fn = input_data.get("retrieve_fn")

        if not retrieve_fn:
            log.warning("No retrieve_fn passed to Researcher. Returning empty context.")
            return {"context_chunks": [], "sources": []}

        context_chunks = []
        sources = set()

        # Run retrieval for each decomposed sub-question
        for q in sub_questions:
            try:
                # retrieve_fn returns list of dicts: {text, score, source, created_at}
                results = retrieve_fn(q)
                for r in results:
                    if isinstance(r, dict):
                        text = r.get("text", "")
                        src = r.get("source", "unknown")
                    elif isinstance(r, str):
                        text = r
                        src = "unknown"
                    else:
                        continue
                    
                    if text and text not in context_chunks:
                        context_chunks.append(text)
                    if src:
                        sources.add(src)
            except Exception as e:
                log.error(f"Error during Researcher retrieval for '{q}': {e}")

        return {
            "context_chunks": context_chunks,
            "sources": list(sources),
        }
