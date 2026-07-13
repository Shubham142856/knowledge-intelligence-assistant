"""
benchmarks/baselines/no_debate_rag.py — Full Pipeline with Single-Pass (No Iterative Debate)

Tests the contribution of multi-round refinement. The Critic runs once to score the
initial draft, but the Writer never revises it. Max iterations is capped at 1.
"""

import time
from orchestrator import VYOROrchestrator

class NoDebateOrchestrator(VYOROrchestrator):
    def execute_query(self, user_query: str) -> dict:
        # Patch Planner to return complexity 1 (which limits Critic-Writer debate to 1 loop)
        orig_planner_run = self.planner.run
        
        def mock_planner_run(input_data):
            res = orig_planner_run(input_data)
            res["complexity"] = 1
            return res
            
        self.planner.run = mock_planner_run
        
        t0 = time.perf_counter()
        res = super().execute_query(user_query)
        latency = time.perf_counter() - t0
        res["latency_s"] = latency
        
        # Restore Planner
        self.planner.run = orig_planner_run
        return res

def run_no_debate_rag(query: str) -> dict:
    orchestrator = NoDebateOrchestrator()
    return orchestrator.execute_query(query)
