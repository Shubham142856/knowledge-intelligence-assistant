"""
src/agents/router.py — Router Agent
"""

from src.agents.base_agent import BaseAgent
from src.agents.llm import call_llm, clean_json_response

ROUTER_PROMPT = """
You are the Router agent for VYOR AI.
Your job is to analyze the user's query and decide the best execution route:
1. "ltm" : For simple greetings, conversational questions, or recalling personal preferences/facts mentioned earlier.
2. "rag" : For factual questions that require searching external documents or indexed knowledge.
3. "complex" : For multi-step questions, reasoning, comparative analysis, or mathematical evaluations.

OUTPUT FORMAT (strict JSON):
{
  "route": "ltm|rag|complex",
  "reason": "brief explanation of your decision"
}
"""


class Router(BaseAgent):
    """Analyzes intent and routes the query to the correct execution pipeline."""

    def __init__(self):
        super().__init__("Router", ROUTER_PROMPT)

    def run(self, input_data: dict) -> dict:
        query = input_data.get("query", "")
        prompt = self.build_prompt(query)
        response_text = call_llm(self.system_prompt, prompt)
        parsed = clean_json_response(response_text)

        # Fallbacks
        if "route" not in parsed:
            parsed["route"] = "rag"
        if "reason" not in parsed:
            parsed["reason"] = "Default fallback route to RAG."

        return parsed
