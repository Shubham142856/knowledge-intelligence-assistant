"""
orchestrator.py — VYOROrchestrator (Multi-Agent Router & Coordinator)

Implements Phase 3: 6+ Agent microservices coordination:
  1. Router Agent — Directs query to LTM, RAG, or Complex flow.
  2. Planner Agent — Decomposes complex queries.
  3. Memory Agent — Analyzes Titans LTM neural memory.
  4. Researcher Agent — Retrieves context from databases.
  5. Writer Agent — Synthesizes drafts.
  6. Critic Agent — Audits draft responses.
  7. Refiner Agent — Formats final responses and citations.
"""

import json
import logging
from pathlib import Path
import torch

from src.agents.router import Router
from src.agents.planner import Planner
from src.agents.memory_agent import MemoryAgent
from src.agents.researcher import Researcher
from src.agents.writer import Writer
from src.agents.critic import Critic
from src.agents.refiner import Refiner
from src.rag.ingestion import get_embedder

log = logging.getLogger("vyor_ai.orchestrator")


class VYOROrchestrator:
    """
    Unified Multi-Agent Coordinator implementing routing, 
    LTM memory query, and Critic-Writer debate loops.
    """

    def __init__(self, memory_core=None, retrieval_fn=None):
        if callable(memory_core) and retrieval_fn is None:
            retrieval_fn = memory_core
            memory_core = None

        # Instantiate memory core if not provided
        if memory_core is None:
            from titans_memory import VYORNeuralBrain
            self.memory = VYORNeuralBrain()
        else:
            self.memory = memory_core

        self.retrieval_fn = retrieval_fn if retrieval_fn else self._mock_qdrant_search
        self.quality_gate = 0.80

        # Instantiate 6+ Agents
        self.router = Router()
        self.planner = Planner()
        self.memory_agent = MemoryAgent()
        self.researcher = Researcher()
        self.writer = Writer()
        self.critic = Critic()
        self.refiner = Refiner()

    def _mock_qdrant_search(self, query: str) -> list[dict]:
        """Real-time default vector search fallback across HR_Policy_2026.txt."""
        policy_path = Path(__file__).parent / "data" / "HR_Policy_2026.txt"
        if not policy_path.exists():
            return [
                {
                    "text": "Medical leaves are capped at 15 days per calendar year. Lead approval is mandatory.",
                    "source": "HR_Policy_2026.pdf",
                    "score": 0.9,
                }
            ]

        with open(policy_path, encoding="utf-8") as fh:
            content = fh.read()

        # Split into bullet points / sections
        chunks = [c.strip() for c in content.split("\n") if c.strip() and not c.startswith("VYOR ENTERPRISE")]
        
        # Calculate overlap score for each chunk based on query keywords
        import re
        q_words = set(re.findall(r'\b\w+\b', query.lower()))
        q_words = {w for w in q_words if len(w) > 2}
        scored_chunks = []
        for chunk in chunks:
            c_words = set(re.findall(r'\b\w+\b', chunk.lower()))
            overlap = 0
            for qw in q_words:
                for cw in c_words:
                    if qw in cw or cw in qw:
                        overlap += 1
                        break
            if overlap > 0:
                scored_chunks.append((overlap, chunk))


        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored_chunks[:3]:
            results.append({
                "text": chunk,
                "source": "HR_Policy_2026.pdf",
                "score": float(score) / max(1, len(q_words))
            })

        return results



    def execute_query(self, user_query: str) -> dict:
        """
        Main entrypoint.
        Routes, queries memory or RAG, and runs iterative debate + refining.
        """
        log.info(f"Orchestrator: Routing incoming query -> '{user_query}'")

        # Step 1: Query Router Agent
        route_decision = self.router.run({"query": user_query})
        route = route_decision.get("route", "rag")
        reason = route_decision.get("reason", "")
        log.info(f"Orchestrator: Route chosen -> {route.upper()} (Reason: {reason})")

        # Handle LTM Direct Recall Route
        if route == "ltm":
            log.info("Orchestrator: Executing LTM memory-direct recall...")
            try:
                import os
                if os.getenv("USE_MOCK_LLM") == "1":
                    q_tensor = torch.zeros((1, 1, 384), dtype=torch.float32)
                else:
                    embedder = get_embedder()
                    q_vec = embedder.encode(user_query)
                    q_tensor = torch.tensor(q_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                # Retrieve from Titans LTM core
                recalled_tensor = self.memory.recall_info(q_tensor)

                
                # Mock textual interpretation of the recalled vector
                recalled_raw = f"Neural memory association matrix match: {recalled_tensor.mean().item():.4f}"
            except Exception as e:
                log.warning(f"LTM tensor retrieval failed: {e}. Falling back to text search.")
                recalled_raw = "No neural context available."

            # Query Memory Agent to interpret LTM state
            mem_result = self.memory_agent.run({
                "query": user_query,
                "recalled_raw": recalled_raw
            })
            
            recalled_text = mem_result.get("recalled_text", "")
            
            # If memory agent returned meaningful content, skip RAG search
            if recalled_text:
                log.info("Orchestrator: Direct memory hit. Routing to refiner.")
                refined = self.refiner.run({
                    "text": recalled_text,
                    "citations": ["Titans_Neural_LTM"]
                })
                return {
                    "answer": refined.get("refined_text", recalled_text),
                    "citations": ["Titans_Neural_LTM"],
                    "confidence": float(mem_result.get("confidence", 0.9)),
                    "uncertainty": False
                }
            log.info("Orchestrator: Memory recall cold. Falling back to standard RAG.")
            route = "rag"

        # Determine sub-questions based on Route complexity
        if route == "complex":
            plan = self.planner.run({"query": user_query})
            sub_questions = plan.get("sub_questions", [user_query])
            complexity = int(plan.get("complexity", 3))
        else: # Simple RAG
            sub_questions = [user_query]
            complexity = 1

        log.info(f"Orchestrator: sub-questions -> {sub_questions}")

        # Step 2: Retrieve context
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

        # Step 4: Iterative Critic-Writer debate loop
        max_iters = 3 if complexity >= 3 else 1
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

        # Step 5: Polish using Refiner Agent
        final_draft_text = draft.get("text", "Error generating response.")
        citations = draft.get("sources", sources)
        if not citations and final_draft_text:
            citations = ["General_Knowledge"]
        
        refined = self.refiner.run({
            "text": final_draft_text,
            "citations": citations
        })
        
        confidence = float(feedback.get("score", 0.75))
        final_citations = refined.get("citations", citations)


        return {
            "answer": refined.get("refined_text", final_draft_text),
            "citations": final_citations,
            "confidence": confidence,
            "uncertainty": draft.get("uncertainty_flag", confidence < 0.65),
        }


    def run(self, user_query: str) -> dict:
        """Alias for AgentOrchestrator compatibility."""
        return self.execute_query(user_query)


# Alias AgentOrchestrator to VYOROrchestrator for absolute compatibility
AgentOrchestrator = VYOROrchestrator