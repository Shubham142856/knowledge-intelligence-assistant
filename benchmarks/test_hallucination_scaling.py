"""
benchmarks/test_hallucination_scaling.py - Scaling Evaluation of Hallucination Rate

Evaluates hallucination rates across increasing query counts (e.g., 200, 1000, 5000, 20000).
Calls run_orchestrator for evaluation queries and computes hallucination rate metrics.

Target: < 5% hallucination rate across scaling tiers.

Usage:
    python benchmarks/test_hallucination_scaling.py --queries 200 1000 5000 20000
"""

import sys
import os
import time
import json
import argparse
import statistics
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from integration_interface import run_orchestrator
except (ImportError, AttributeError):
    try:
        from integration_interface import VYORIntegrationInterface
        _iface = VYORIntegrationInterface()
        def run_orchestrator(query: str):
            return _iface.run_orchestrator(query)
    except (ImportError, AttributeError):
        from orchestrator import VYOROrchestrator
        _orch = VYOROrchestrator()
        def run_orchestrator(query: str):
            return _orch.execute_query(query)


# Keywords that suggest a factual claim - simple heuristic
_CLAIM_KEYWORDS = [
    "is", "was", "has", "will", "the", "in", "are", "were", "have",
    "can", "does", "did", "would", "could", "should",
]


def _is_hallucinated(result: dict, answer: str) -> bool:
    """
    Heuristic: flag an answer as a potential hallucination if:
    - No citations are provided, AND
    - The answer is > 50 chars and contains a factual-claim keyword.
    """
    if result.get("citations") or result.get("sources"):
        return False  # cited -> not flagged
    answer_lower = answer.lower()
    has_claim = any(f" {kw} " in answer_lower for kw in _CLAIM_KEYWORDS)
    return len(answer) > 50 and has_claim


def generate_extended_queries(base_queries: list, target_count: int) -> list:
    """
    Extend base query set to target_count using variations if base query set is smaller.
    """
    if len(base_queries) >= target_count:
        return base_queries[:target_count]

    result = list(base_queries)
    base_len = len(base_queries)
    idx = 0
    multiplier = 1
    while len(result) < target_count:
        orig = base_queries[idx % base_len]
        new_q = dict(orig)
        new_q["id"] = f"{orig.get('id', 'q')}_scale_{multiplier}_{idx}"
        # Slight variation in query string to maintain variety
        new_q["query"] = f"{orig['query']} (variant {multiplier})"
        result.append(new_q)
        idx += 1
        if idx % base_len == 0:
            multiplier += 1

    return result[:target_count]


def run_scaling_benchmark(query_counts: list, queries_path: str = "data/halumem/evaluation_queries.json"):
    print(f"\n{'='*75}")
    print(f"  VYOR AI - Hallucination Scaling Benchmark")
    print(f"  Target Query Counts: {query_counts}")
    print(f"{'='*75}\n")

    q_path = Path(queries_path)
    base_queries = []

    if q_path.exists():
        with open(q_path, encoding="utf-8") as fh:
            base_queries = json.load(fh)
        print(f"  [Dataset] Loaded {len(base_queries)} base queries from {q_path}")
    else:
        print(f"  [Warning] Queries file not found at {q_path}. Generating synthetic base queries.")
        base_queries = [
            {"id": "synth_1", "query": "What are the remote work policy guidelines?", "type": "grounded"},
            {"id": "synth_2", "query": "What secret features exist in system?", "type": "open_ended"},
            {"id": "synth_3", "query": "How is customer data encrypted at rest?", "type": "grounded"},
            {"id": "synth_4", "query": "What is the overtime rate for holidays?", "type": "grounded"},
            {"id": "synth_5", "query": "Who is the CEO of the universe?", "type": "open_ended"},
        ]

    benchmark_results = []

    for count in query_counts:
        print(f"\n--- Evaluating Query Tier: {count:,} Queries ---")
        queries = generate_extended_queries(base_queries, count)
        
        hallucinated_count = 0
        latencies = []
        t_start = time.perf_counter()

        # Log progress interval depending on count
        log_interval = max(1, count // 10)

        import io
        from contextlib import redirect_stdout
        from src.agents.critic import Critic

        for i, item in enumerate(queries):
            q_text = item["query"]
            t0 = time.perf_counter()
            f_null = io.StringIO()
            with redirect_stdout(f_null):
                res = run_orchestrator(q_text)
            dt = time.perf_counter() - t0
            latencies.append(dt)

            answer = res.get("answer", "")
            if _is_hallucinated(res, answer):
                hallucinated_count += 1

            if (i + 1) % log_interval == 0 or (i + 1) == count:
                cur_rate = hallucinated_count / (i + 1)
                elapsed = time.perf_counter() - t_start
                qps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Progress: {i+1:>6}/{count} ({((i+1)/count)*100:>5.1f}%) | "
                      f"Flagged: {hallucinated_count:>4} | Current Rate: {cur_rate:>6.2%} | "
                      f"Speed: {qps:>6.1f} q/s")


        total_time = time.perf_counter() - t_start
        final_rate = hallucinated_count / count if count > 0 else 0.0
        avg_lat_ms = (statistics.mean(latencies) * 1000.0) if latencies else 0.0
        p95_lat_ms = (statistics.quantiles(latencies, n=20)[18] * 1000.0) if len(latencies) > 1 else avg_lat_ms
        passed = final_rate <= 0.05

        status_str = "PASS" if passed else "FAIL (Target <= 5%)"
        print(f"\n  Tier {count:,} Results:")
        print(f"    - Total Queries    : {count:,}")
        print(f"    - Hallucinations   : {hallucinated_count:,}")
        print(f"    - Hallucination %  : {final_rate:.2%}")
        print(f"    - Avg Latency      : {avg_lat_ms:.2f} ms")
        print(f"    - p95 Latency      : {p95_lat_ms:.2f} ms")
        print(f"    - Total Duration   : {total_time:.2f} s")
        print(f"    - Status           : {status_str}")

        benchmark_results.append({
            "target_queries": count,
            "actual_queries": count,
            "hallucinations": hallucinated_count,
            "hallucination_rate_pct": round(final_rate * 100, 2),
            "avg_latency_ms": round(avg_lat_ms, 2),
            "p95_latency_ms": round(p95_lat_ms, 2),
            "total_duration_s": round(total_time, 2),
            "throughput_qps": round(count / total_time, 2) if total_time > 0 else 0,
            "pass": passed
        })

    # Render Summary Table
    report_md = "# Hallucination Rate Scaling Benchmark Report\n\n"
    report_md += "| Query Count | Hallucinations | Hallucination Rate | Avg Latency (ms) | p95 Latency (ms) | Throughput (q/s) | Status |\n"
    report_md += "| --- | --- | --- | --- | --- | --- | --- |\n"
    for r in benchmark_results:
        status = "PASS" if r["pass"] else "FAIL"
        report_md += (f"| {r['target_queries']:,} | {r['hallucinations']:,} | "
                      f"{r['hallucination_rate_pct']:.2f}% | {r['avg_latency_ms']:.2f}ms | "
                      f"{r['p95_latency_ms']:.2f}ms | {r['throughput_qps']:.1f} | {status} |\n")

    print("\n" + "="*75)
    print("  SUMMARY SCALING RESULTS")
    print("="*75 + "\n")
    print(report_md)

    # Save outputs
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "hallucination_scaling.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(benchmark_results, fh, indent=2)

    md_path = out_dir / "hallucination_scaling_report.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(report_md)

    print(f"  [Saved] JSON results -> {json_path}")
    print(f"  [Saved] Markdown report -> {md_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Hallucination Rate Scaling Benchmark")
    parser.add_argument("--queries", nargs="+", type=int, default=[200, 1000, 5000, 20000],
                        help="List of query counts to benchmark (e.g. --queries 200 1000 5000 20000)")
    parser.add_argument("--queries-path", type=str, default="data/halumem/evaluation_queries.json",
                        help="Path to evaluation queries JSON")
    parser.add_argument("--fast", "--mock", action="store_true",
                        help="Use fast mock LLM fallbacks to avoid cloud API rate limits and network latency")

    args = parser.parse_args()
    
    if args.fast:
        os.environ["USE_MOCK_LLM"] = "1"
        import src.agents.llm as llm_module
        llm_module.call_llm = lambda sys_prompt, user_content, **kwargs: llm_module.get_mock_fallback(sys_prompt, user_content)

    run_scaling_benchmark(query_counts=args.queries, queries_path=args.queries_path)



if __name__ == "__main__":
    main()
