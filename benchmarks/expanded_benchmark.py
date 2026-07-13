"""
benchmarks/expanded_benchmark.py — Unified Benchmark & Baseline Comparison Runner

Runs evaluation queries against the following configurations:
1. Full VYOR System (Orchestrator)
2. Vanilla RAG (Dense cosine retrieval + single LLM call)
3. Hybrid RAG (Dense + BM25 + RRF + Cross-Encoder + single LLM call)
4. Agent Pipeline WITHOUT Memory (LTM bypassed)
5. Agent Pipeline WITHOUT Critic (Auditing bypassed)
6. Agent Pipeline WITHOUT Debate (Capped at 1 iteration)

Saves results to benchmarks/results/expanded_results.json.
"""

import sys
import os
import time
import json
import argparse
import statistics
from pathlib import Path

# Add project root directory to python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from orchestrator import VYOROrchestrator
from benchmarks.baselines.vanilla_rag import run_vanilla_rag
from benchmarks.baselines.hybrid_rag import run_hybrid_rag
from benchmarks.baselines.no_memory_rag import run_no_memory_rag
from benchmarks.baselines.no_critic_rag import run_no_critic_rag
from benchmarks.baselines.no_debate_rag import run_no_debate_rag

# Default evaluation queries (robust sample covering different question intents)
DEFAULT_QUERIES = [
    "What is the maximum duration of a remote work contract?",
    "How many medical leaves are employees entitled to per year?",
    "Who approves overtime requests for senior engineers?",
    "What are the data retention policies for customer records?",
    "What are the security requirements for remote access to production systems?",
    "What is the policy on intellectual property developed by contractors?",
    "Who handles emergency hardware replacements in the datacenter?",
    "What is the standard onboarding duration for new engineering hires?",
    "How are customer database backups encrypted?",
    "What are the consequences of violating the code of conduct policy?"
]

# Simple heuristic to flag hallucinations: no citations but asserts factual claims
_CLAIM_KEYWORDS = ["is", "was", "has", "will", "are", "were", "have", "can", "does"]

def check_hallucination(res: dict, answer: str) -> bool:
    if res.get("citations") or res.get("sources"):
        return False
    answer_lower = answer.lower()
    has_claim = any(f" {kw} " in answer_lower for kw in _CLAIM_KEYWORDS)
    return len(answer) > 50 and has_claim

def run_evaluation(system_name: str, run_fn, queries: list) -> dict:
    print(f"\n[Expanded Benchmark]: Running evaluation for '{system_name}'...")
    latencies = []
    confidences = []
    citations_count = []
    uncertainty_count = 0
    hallucination_count = 0
    success_count = 0

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        try:
            res = run_fn(q)
            latency = time.perf_counter() - t0
            latencies.append(latency)
            
            # Unify keys for response format
            answer = res.get("answer", "")
            cites = res.get("citations", res.get("sources", []))
            conf = res.get("confidence", 0.5)
            uncert = res.get("uncertainty", False)
            
            confidences.append(conf)
            citations_count.append(len(cites))
            if uncert:
                uncertainty_count += 1
            if check_hallucination(res, answer):
                hallucination_count += 1
                
            success_count += 1
            print(f"  [{i+1}/{len(queries)}] Latency: {latency:.2f}s | Confidence: {conf:.2f} | Citations: {len(cites)}")
        except Exception as e:
            print(f"  [ERROR] Query '{q}' failed: {e}")

    total = len(queries)
    return {
        "system": system_name,
        "avg_latency_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "p95_latency_s": round(statistics.quantiles(latencies, n=20)[18], 4) if len(latencies) > 1 else 0.0,
        "avg_confidence": round(statistics.mean(confidences), 4) if confidences else 0.0,
        "avg_citations": round(statistics.mean(citations_count), 4) if citations_count else 0.0,
        "uncertainty_triggers": uncertainty_count,
        "hallucination_rate": round(hallucination_count / total, 4) if total else 0.0,
        "success_rate": round(success_count / total, 4) if total else 0.0
    }

def main():
    parser = argparse.ArgumentParser(description="Expanded Benchmark Suite")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of queries to run")
    parser.add_argument("--use-bench-dataset", action="store_true", help="Load queries from questions.jsonl")
    args = parser.parse_args()

    # Determine query set
    queries = DEFAULT_QUERIES[:args.num_queries]
    
    if args.use_bench_dataset:
        q_path = Path("data/enterprise_rag_bench/questions.jsonl")
        if q_path.exists():
            print(f"Loading queries from {q_path}...")
            loaded_queries = []
            with open(q_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        loaded_queries.append(json.loads(line).get("question"))
            if loaded_queries:
                queries = [q for q in loaded_queries if q][:args.num_queries]
        else:
            print(f"Dataset path {q_path} not found. Defaulting to sample queries.")

    print(f"Running benchmark with {len(queries)} queries.")

    # Initialize standard orchestrator
    orchestrator = VYOROrchestrator()

    # Systems registry
    systems = [
        ("Full VYOR System", lambda q: orchestrator.execute_query(q)),
        ("Vanilla RAG Baseline", lambda q: run_vanilla_rag(q)),
        ("Hybrid RAG Baseline", lambda q: run_hybrid_rag(q)),
        ("No Memory Baseline", lambda q: run_no_memory_rag(q)),
        ("No Critic Baseline", lambda q: run_no_critic_rag(q)),
        ("No Debate Baseline", lambda q: run_no_debate_rag(q)),
    ]

    results = []
    for name, run_fn in systems:
        res = run_evaluation(name, run_fn, queries)
        results.append(res)

    # ── Render Markdown Report ───────────────────────────────────────────────
    report_md = "# Expanded Benchmark & Baseline Comparison Report\n\n"
    report_md += "| System / Baseline | Avg Latency (s) | p95 Latency (s) | Avg Confidence | Avg Citations | Uncertainty Triggers | Hallucination Rate |\n"
    report_md += "| --- | --- | --- | --- | --- | --- | --- |\n"
    for r in results:
        report_md += f"| {r['system']} | {r['avg_latency_s']:.3f}s | {r['p95_latency_s']:.3f}s | {r['avg_confidence']:.2%} | {r['avg_citations']:.1f} | {r['uncertainty_triggers']} | {r['hallucination_rate']:.2%} |\n"

    print("\n" + report_md)

    # Save results
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "expanded_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        
    with open(out_dir / "expanded_report.md", "w", encoding="utf-8") as fh:
        fh.write(report_md)

    print(f"\n[Expanded Benchmark]: All results saved to {out_dir}/")

if __name__ == "__main__":
    main()
