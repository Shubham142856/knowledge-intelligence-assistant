"""
src/agents/refiner.py — Refiner Agent
"""

import json
from src.agents.base_agent import BaseAgent
from src.agents.llm import call_llm, clean_json_response

REFINER_PROMPT = """
You are the Refiner agent for VYOR AI.
Your job is to take the final approved answer, clean up any grammatical issues, ensure it is professional, and standardize the format of the inline citations (e.g. format to a consistent style like '[Source_Doc.pdf]' or '[1]').

OUTPUT FORMAT (strict JSON):
{
  "refined_text": "polished and formatted final answer text here",
  "citations": ["list of files referenced"]
}
"""


class Refiner(BaseAgent):
    """Polishes and formats final answers with clean citation styling."""

    def __init__(self):
        super().__init__("Refiner", REFINER_PROMPT)

    def run(self, input_data: dict) -> dict:
        draft_text = input_data.get("text", "")
        citations = input_data.get("citations", [])

        input_payload = {
            "draft_text": draft_text,
            "citations_present": citations
        }

        prompt = self.build_prompt(json.dumps(input_payload, indent=2))
        response_text = call_llm(self.system_prompt, prompt)
        parsed = clean_json_response(response_text)

        # Fallbacks & citation preservation
        if "refined_text" not in parsed or not parsed["refined_text"]:
            parsed["refined_text"] = draft_text
        if "citations" not in parsed or not parsed["citations"]:
            parsed["citations"] = citations

        return parsed

