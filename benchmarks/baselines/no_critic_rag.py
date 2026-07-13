"""
benchmarks/baselines/no_critic_rag.py — Full Pipeline WITHOUT Critic Agent

Tests the contribution of adversarial auditing by bypassing the critic review.
The writer's initial draft is accepted immediately.
"""

import time
from orchestrator import VYOROrchestrator

class NoCriticOrchestrator(VYOROrchestrator):
    def execute_query(self, user_query: str) -> dict:
        # Patch Critic to always approve the first draft instantly
        orig_critic_run = self.critic.run
        
        def mock_critic_run(input_data):
            return {
                "score": 1.0,
                "issues": [],
                "approved": True
            }
            
        self.critic.run = mock_critic_run
        
        t0 = time.perf_counter()
        res = super().execute_query(user_query)
        latency = time.perf_counter() - t0
        res["latency_s"] = latency
        
        # Restore Critic
        self.critic.run = orig_critic_run
        return res

def run_no_critic_rag(query: str) -> dict:
    orchestrator = NoCriticOrchestrator()
    return orchestrator.execute_query(query)
