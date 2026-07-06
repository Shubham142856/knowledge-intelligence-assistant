"""
src/agents/planner.py — Planner Agent
"""

from src.agents.base_agent import BaseAgent
from src.agents.llm import call_llm, clean_json_response

PLANNER_PROMPT = """
You are the Planner agent for VYOR AI, a knowledge assistant.
Your ONLY job: decompose the user query into atomic sub-questions.
OUTPUT FORMAT (strict JSON):
{
  "sub_questions": ["sub-q 1", "sub-q 2", ...],
  "query_type": "factual|reasoning|comparative|procedural",
  "complexity": 1-5
}
Rules:
- Maximum 5 sub-questions
- Each sub-question must be independently answerable
- Do not answer the question yourself
"""


class Planner(BaseAgent):
    """Decomposes complex user queries into atomic search steps."""

    def __init__(self):
        super().__init__("Planner", PLANNER_PROMPT)

    def run(self, input_data: dict) -> dict:
        query = input_data.get("query", "")
        prompt = self.build_prompt(query)
        response_text = call_llm(self.system_prompt, prompt)
        parsed = clean_json_response(response_text)

        # Default fallbacks if parsing fails
        if "sub_questions" not in parsed:
            parsed["sub_questions"] = [query]
        if "query_type" not in parsed:
            parsed["query_type"] = "factual"
        if "complexity" not in parsed:
            parsed["complexity"] = 2

        return parsed
