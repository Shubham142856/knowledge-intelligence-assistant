"""
src/agents/writer.py — Writer Agent
"""

import json
from src.agents.base_agent import BaseAgent
from src.agents.llm import call_llm, clean_json_response

WRITER_PROMPT = """
You are the Writer agent for VYOR AI, a knowledge assistant.
Your job: synthesize retrieved context chunks into a coherent, accurate answer.
You MUST cite your sources exactly using the names provided in the context.

OUTPUT FORMAT (strict JSON):
{
  "text": "Your synthesized response text here, citing sources like [source_file.pdf]",
  "sources": ["source_file.pdf"],
  "confidence": 0.0-1.0
}

Rules:
- Base your answer on the provided context chunks if they contain relevant information to the query.
- If the context chunks are irrelevant, empty, or do not contain enough information, you MUST answer the query using your own internal general knowledge and reasoning. Do not state that you lack information or cannot answer.
- Cite sources from the context ONLY if they are relevant and used.
- Incorporate Critic critique issues if they are provided to refine your answer.
"""


class Writer(BaseAgent):
    """Generates and refines draft answers based on retrieved context and Critic feedback."""

    def __init__(self):
        super().__init__("Writer", WRITER_PROMPT)

    def run(self, input_data: dict) -> dict:
        """
        Args:
            input_data: {
                'query': str,
                'context': list[str],
                'sources': list[str],
                'critique': list[str] (optional critic feedback)
            }
        """
        query = input_data.get("query", "")
        context = input_data.get("context", [])
        sources = input_data.get("sources", [])
        critique = input_data.get("critique", [])

        # Construct LLM prompt
        input_payload = {
            "query": query,
            "context_chunks": context,
            "sources_available": sources,
        }
        if critique:
            input_payload["critic_issues_to_fix"] = critique

        prompt = self.build_prompt(json.dumps(input_payload, indent=2))
        response_text = call_llm(self.system_prompt, prompt)
        parsed = clean_json_response(response_text)

        # Fallback fields
        if "text" not in parsed:
            parsed["text"] = "Error generating draft answer."
        if "sources" not in parsed:
            parsed["sources"] = []
        if "confidence" not in parsed:
            parsed["confidence"] = 0.5

        return parsed
