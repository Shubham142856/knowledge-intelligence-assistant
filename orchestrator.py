"""
orchestrator.py — VYOROrchestrator & AgentOrchestrator

Iterative Multi-Agent Debate Loop coordinating:
  1. Planner Agent — Decomposes query into sub-questions.
  2. Researcher Agent — Retrieves context from Qdrant via hybrid search.
  3. Writer Agent — Synthesizes facts into draft answer with citations.
  4. Critic Agent — Audits draft for hallucination and quality.
"""

import json
import logging
from src.agents.planner import Planner
from src.agents.researcher import Researcher
from src.agents.writer import Writer
from src.agents.critic import Critic

log = logging.getLogger("vyor_ai.orchestrator")


class VYOROrchestrator:
    """
    Multi-Agent coordinator for Q&A reasoning.
    Runs a debate loop between Writer and Critic up to max_debate_iterations.
    """

    def __init__(self, memory_core=None, retrieval_fn=None):
        # Handle arguments positional/keyword swap safely
        # E.g., if initialized as VYOROrchestrator(partner_qdrant_search_fn)
        if callable(memory_core) and retrieval_fn is None:
            retrieval_fn = memory_core
            memory_core = None

        self.memory = memory_core
        self.retrieval_fn = retrieval_fn if retrieval_fn else self._mock_qdrant_search
        self.max_debate_iterations = 1
        self.quality_gate = 0.80

        # Instantiate agents
        self.planner = Planner()
        self.researcher = Researcher()
        self.critic = Critic()
        self.writer = Writer()

    def _mock_qdrant_search(self, query: str) -> list[dict]:
        """Fallback mock search if no real Qdrant function is connected."""
        return [
            {
                "text": "Medical leaves are capped at 15 days per calendar year. Lead approval is mandatory.",
                "source": "HR_Policy_2026.pdf",
                "score": 0.9,
            }
        ]

    def execute_query(self, user_query: str) -> dict:
        """
        Main entrypoint.
        Decomposes query, retrieves context, and runs iterative Critic-Writer debate.
        """
        log.info(f"Orchestrator: Ingesting query -> '{user_query}'")

        # Step 1: Decompose query into sub-questions
        plan = self.planner.run({"query": user_query})
        sub_questions = plan.get("sub_questions", [user_query])
        log.info(f"Orchestrator: sub-questions -> {sub_questions}")

        # Step 2: Retrieve context for each sub-question
        research = self.researcher.run({
            "sub_questions": sub_questions,
            "retrieve_fn": self.retrieval_fn
        })
        context_chunks = research.get("context_chunks", [])
        sources = research.get("sources", [])
        log.info(f"Orchestrator: context chunks -> {len(context_chunks)}, sources -> {sources}")

        # Step 3: Generate initial draft
        draft = self.writer.run({
            "query": user_query,
            "context": context_chunks,
            "sources": sources,
        })
        log.info(f"Orchestrator: Initial draft generated (confidence={draft.get('confidence')})")

        # Step 4: Iterative debate loop with dynamic depth based on complexity
        complexity = int(plan.get("complexity", 2))
        max_iters = 3 if complexity >= 3 else 1
        log.info(f"Orchestrator: query complexity is {complexity}. Setting dynamic debate depth to {max_iters} iterations.")

        feedback = {"score": 0.5, "issues": [], "approved": False}
        for iteration in range(1, max_iters + 1):
            log.info(f"Orchestrator: Debate iteration {iteration}/{max_iters}")
            
            # Critic audits the draft
            feedback = self.critic.run({
                "draft": draft,
                "context": context_chunks,
            })
            log.info(f"Orchestrator: Critic evaluation score={feedback.get('score')} | approved={feedback.get('approved')}")

            # Stop if approved or passes quality gate
            if feedback.get("approved", False) or feedback.get("score", 0.0) >= self.quality_gate:
                break

            # If this is the last iteration, cap reached — set uncertainty
            if iteration == max_iters:
                draft["uncertainty_flag"] = True
                break

            # Writer refines using critic feedback
            draft = self.writer.run({
                "query": user_query,
                "context": context_chunks,
                "sources": sources,
                "critique": feedback.get("issues", []),
            })

        # Step 5: Format final return packet
        confidence = float(feedback.get("score", 0.75))
        return {
            "answer": draft.get("text", "Error generating response."),
            "citations": draft.get("sources", sources),
            "confidence": confidence,
            "uncertainty": draft.get("uncertainty_flag", confidence < 0.65),
        }

    def run(self, user_query: str) -> dict:
        """Alias for AgentOrchestrator compatibility."""
        return self.execute_query(user_query)


# Alias AgentOrchestrator to VYOROrchestrator for absolute compatibility
AgentOrchestrator = VYOROrchestrator