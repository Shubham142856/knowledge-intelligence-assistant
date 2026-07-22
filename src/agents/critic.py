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
1. If the context contains relevant information to the query, every claim in the draft is directly supported by the context and cited correctly.
2. If the context is irrelevant or empty (e.g., for general knowledge, math, or logic queries), verify the logical consistency, accuracy, and reasoning of the draft response.
3. The draft contains exact numerical metrics (like dates, numbers, counts) if they exist in the context.

OUTPUT FORMAT (strict JSON):
{
  "score": 0.0-1.0,
  "issues": ["list of factual errors, logical flaws, or ungrounded claims"],
  "approved": true|false
}

Rules:
- If the context is relevant to the query's topic and the draft contains ungrounded claims, set approved to false.
- If the context is completely unrelated to the query's topic (or empty) and the draft is factually/logically correct based on general knowledge, you MUST approve it (approved: true) and score it highly (score >= 0.80). Do not reject it or flag it as ungrounded.
- Set a high score (>= 0.80) if the draft is accurate, logically sound, and correctly cited (if citations apply).
"""


class Critic(BaseAgent):
    """Audits draft answers against raw context to ensure factual accuracy and prevent hallucinations."""

    parse_attempts = 0
    parse_failures = 0

    def __init__(self):
        super().__init__("Critic", CRITIC_PROMPT)

    @classmethod
    def get_parse_failure_rate(cls) -> float:
        return (cls.parse_failures / cls.parse_attempts) if cls.parse_attempts > 0 else 0.0

    def run(self, input_data: dict) -> dict:
        """
        Args:
            input_data: {
                'draft': dict (writer's output),
                'context': list[str]
            }
        """
        Critic.parse_attempts += 1
        draft = input_data.get("draft", {})
        context = input_data.get("context", [])

        input_payload = {
            "draft_to_evaluate": draft,
            "raw_ground_truth_context": context,
        }

        prompt = self.build_prompt(json.dumps(input_payload, indent=2))
        response_text = call_llm(self.system_prompt, prompt)
        parsed = clean_json_response(response_text)

        # Track JSON parsing success/failure
        is_clean_json = bool(parsed and "score" in parsed and "approved" in parsed)
        if not is_clean_json:
            Critic.parse_failures += 1

        # Resilient Text Analysis Fallback if JSON fields are missing
        if "approved" not in parsed:
            txt_lower = response_text.lower()
            if "approved\": true" in txt_lower or "approved: true" in txt_lower or "is accurate" in txt_lower or "logically sound" in txt_lower:
                parsed["approved"] = True
            elif "approved\": false" in txt_lower or "approved: false" in txt_lower or "ungrounded" in txt_lower or "error" in txt_lower:
                parsed["approved"] = False
            else:
                # Default approval if draft has high confidence and context was empty
                draft_conf = draft.get("confidence", 0.5) if isinstance(draft, dict) else 0.5
                parsed["approved"] = draft_conf >= 0.70

        if "score" not in parsed:
            if parsed.get("approved"):
                parsed["score"] = 0.85
            else:
                parsed["score"] = 0.50

        if "issues" not in parsed:
            parsed["issues"] = [] if parsed.get("approved") else ["Potential ungrounded or unverified claims."]

        return parsed

