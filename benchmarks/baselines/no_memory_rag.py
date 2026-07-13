"""
benchmarks/baselines/no_memory_rag.py — Full Agent Pipeline WITHOUT Titans Neural Memory

Tests contribution of neural LTM by forcing all conversational queries to RAG
and bypassing Titans LTM direct lookup.
"""

import time
from orchestrator import VYOROrchestrator

class NoMemoryOrchestrator(VYOROrchestrator):
    def execute_query(self, user_query: str) -> dict:
        # Save original router run method
        orig_run = self.router.run
        
        # Override to force "rag" instead of "ltm" route
        def forced_run(input_data):
            res = orig_run(input_data)
            if res.get("route") == "ltm":
                res["route"] = "rag"
                res["reason"] = "Forced RAG route for No-Memory baseline."
            return res
        
        self.router.run = forced_run
        
        # Run the standard query flow
        t0 = time.perf_counter()
        res = super().execute_query(user_query)
        latency = time.perf_counter() - t0
        res["latency_s"] = latency
        
        # Restore router in case of reuse
        self.router.run = orig_run
        return res

def run_no_memory_rag(query: str) -> dict:
    orchestrator = NoMemoryOrchestrator()
    return orchestrator.execute_query(query)
