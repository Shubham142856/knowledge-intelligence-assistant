"""
src/agents/memory_agent.py — Memory Agent
"""

import json
from src.agents.base_agent import BaseAgent
from src.agents.llm import call_llm, clean_json_response

MEMORY_AGENT_PROMPT = """
You are the Memory Agent for VYOR AI.
Your job is to read recalled vectors/raw text retrieved from the Titans Neural LTM and determine if it contains relevant information to directly answer the user's query.

OUTPUT FORMAT (strict JSON):
{
  "recalled_text": "relevant facts recalled, or empty string if not useful",
  "confidence": 0.0-1.0
}
"""


class MemoryAgent(BaseAgent):
    """Directly audits and formats recalled content from Titans LTM."""

    def __init__(self):
        super().__init__("MemoryAgent", MEMORY_AGENT_PROMPT)

    def run(self, input_data: dict) -> dict:
        query = input_data.get("query", "")
        recalled_raw = input_data.get("recalled_raw", "")

        input_payload = {
            "user_query": query,
            "recalled_neural_data": recalled_raw
        }

        prompt = self.build_prompt(json.dumps(input_payload, indent=2))
        response_text = call_llm(self.system_prompt, prompt)
        parsed = clean_json_response(response_text)

        # Fallbacks
        if "recalled_text" not in parsed:
            parsed["recalled_text"] = ""
        if "confidence" not in parsed:
            parsed["confidence"] = 0.0

        return parsed
