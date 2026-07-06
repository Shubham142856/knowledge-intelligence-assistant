"""
src/agents/critic.py — Critic Agent
"""

import json
from src.agents.base_agent import BaseAgent
from src.agents.llm import call_llm, clean_json_response

CRITIC_PROMPT = """
You are the Critic agent for VYOR AI.
Your job: audit the Writer's draft response against the raw context chunks.
Verify that:
1. Every claim in the draft is directly supported by the context.
2. The draft contains exact numerical metrics (like dates, numbers, counts) if they exist in the context.
3. The citations are correct and grounded in the source documents.

OUTPUT FORMAT (strict JSON):
{
  "score": 0.0-1.0,
  "issues": ["list of factual errors, missing citations, or ungrounded claims"],
  "approved": true|false
}

Rules:
- If there are any ungrounded claims or hallucinated details, set approved to false.
- Set a high score (>= 0.80) only if the draft is fully grounded, correct, and well-cited.
"""


class Critic(BaseAgent):
    """Audits draft answers against raw context to ensure factual accuracy and prevent hallucinations."""

    def __init__(self):
        super().__init__("Critic", CRITIC_PROMPT)

    def run(self, input_data: dict) -> dict:
        """
        Args:
            input_data: {
                'draft': dict (writer's output),
                'context': list[str]
            }
        """
        draft = input_data.get("draft", {})
        context = input_data.get("context", [])

        input_payload = {
            "draft_to_evaluate": draft,
            "raw_ground_truth_context": context,
        }

        prompt = self.build_prompt(json.dumps(input_payload, indent=2))
        response_text = call_llm(self.system_prompt, prompt)
        parsed = clean_json_response(response_text)

        # Fallback fields
        if "score" not in parsed:
            parsed["score"] = 0.5
        if "issues" not in parsed:
            parsed["issues"] = ["Failed to verify groundedness."]
        if "approved" not in parsed:
            parsed["approved"] = False

        return parsed
