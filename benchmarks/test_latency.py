"""
benchmarks/test_latency.py — Response Latency Benchmark

Measures end-to-end query latency of the VYOR AI /query endpoint
across N queries and reports p50, p90, p95, p99, and mean latency.

Usage:
    # Server must be running on port 8000
    python benchmarks/test_latency.py
    python benchmarks/test_latency.py --queries 100 --url http://localhost:8000
    python benchmarks/test_latency.py --queries 50 --output results/latency.json

Requirements:
    pip install requests
"""

import argparse
import json
import statistics
import time
from pathlib import Path

try:
    import requests
except ImportError:
    raise SystemExit("Please install requests: pip install requests")


# ── Representative enterprise query set ──────────────────────────────────────

SAMPLE_QUERIES = [
    "What is the maximum duration of a remote work contract?",
    "How many medical leaves are employees entitled to per year?",
    "What is the refund policy for external training courses?",
    "Who approves overtime requests for senior engineers?",
    "What are the data retention policies for customer records?",
    "What is the secret project code for the pilot study?",
    "How do we handle supplier disputes under force majeure clauses?",
    "What is the escalation procedure for P1 production incidents?",
    "Which departments require dual-approval for budget requests above $50k?",
    "What are the security requirements for remote access to production systems?",
    "What is the notice period for contract termination?",
    "How should employees report suspected GDPR breaches?",
    "What are the performance review cycles and KPI weighting?",
    "Are there any restrictions on using personal devices for work tasks?",
    "What is the process for onboarding a new vendor?",
    "How long are security logs retained under the audit policy?",
    "What is the travel expense reimbursement limit per night?",
    "What tools are approved for internal document collaboration?",
    "What is the process for requesting a software licence purchase?",
    "How do we classify documents under the information security policy?",
]


def measure_query(url: str, query: str, timeout: int = 30) -> tuple[float, bool]:
    """
    POST a single query to /query and measure wall-clock latency.

    Returns:
        (latency_seconds: float, success: bool)
    """
    payload = {"query": query}
    t0      = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/query",
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        elapsed = time.perf_counter() - t0
        return elapsed, resp.status_code == 200
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  [ERROR] Query failed after {elapsed:.3f}s: {exc}")
        return elapsed, False


def run(
    url:        str = "http://localhost:8000",
    n_queries:  int = 50,
    output:     str = "benchmarks/results/latency_results.json",
) -> dict:
    """
    Run the latency benchmark.

    Returns:
        dict with p50, p90, p95, p99, mean, min, max, success_rate
    """
    print(f"\n{'='*60}")
    print(f"  VYOR AI Latency Benchmark")
    print(f"  URL     : {url}")
    print(f"  Queries : {n_queries}")
    print(f"{'='*60}\n")

    # ── Health check ──────────────────────────────────────────────────────────
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        health = resp.json()
        print(f"  Server health: {health.get('status', 'unknown')}")
    except Exception:
        print("  WARNING: /health check failed — server may not be running.")

    # ── Run queries ───────────────────────────────────────────────────────────
    latencies: list[float] = []
    successes: int         = 0

    # Cycle through the sample query set
    queries_to_run = [
        SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)] for i in range(n_queries)
    ]

    print(f"\n  Running {n_queries} queries ...\n")

    for i, q in enumerate(queries_to_run):
        lat, ok = measure_query(url, q)
        latencies.append(lat)
        if ok:
            successes += 1

        if (i + 1) % 10 == 0 or (i + 1) == n_queries:
            running_p50 = statistics.median(latencies)
            print(
                f"  [{i+1:>4}/{n_queries}]  "
                f"last={lat:.3f}s  "
                f"running_p50={running_p50:.3f}s  "
                f"ok={successes}/{i+1}"
            )

    # ── Compute statistics ────────────────────────────────────────────────────
    latencies_sorted = sorted(latencies)

    def percentile(data: list[float], pct: float) -> float:
        idx = max(0, int(len(data) * pct / 100) - 1)
        return round(data[idx], 4)

    results = {
        "url":           url,
        "n_queries":     n_queries,
        "success_rate":  round(successes / n_queries, 4) if n_queries > 0 else 0.0,
        "successes":     successes,
        "failures":      n_queries - successes,
        "mean_s":        round(statistics.mean(latencies), 4),
        "min_s":         round(min(latencies), 4),
        "max_s":         round(max(latencies), 4),
        "stdev_s":       round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0.0,
        "p50_s":         percentile(latencies_sorted, 50),
        "p90_s":         percentile(latencies_sorted, 90),
        "p95_s":         percentile(latencies_sorted, 95),
        "p99_s":         percentile(latencies_sorted, 99),
        "target_p50_s":  2.0,   # SLA target: p50 < 2 seconds
        "target_p95_s":  5.0,   # SLA target: p95 < 5 seconds
        "pass_p50":      percentile(latencies_sorted, 50) < 2.0,
        "pass_p95":      percentile(latencies_sorted, 95) < 5.0,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LATENCY RESULTS")
    print(f"{'='*60}")
    print(f"  Success Rate : {results['success_rate']:.1%}  ({successes}/{n_queries})")
    print(f"  Mean         : {results['mean_s']:.3f} s")
    print(f"  Min          : {results['min_s']:.3f} s")
    print(f"  Max          : {results['max_s']:.3f} s")
    print(f"  Std Dev      : {results['stdev_s']:.3f} s")
    print(f"  p50 (median) : {results['p50_s']:.3f} s  {'✅ PASS' if results['pass_p50'] else '❌ FAIL'} (target < 2.0 s)")
    print(f"  p90          : {results['p90_s']:.3f} s")
    print(f"  p95          : {results['p95_s']:.3f} s  {'✅ PASS' if results['pass_p95'] else '❌ FAIL'} (target < 5.0 s)")
    print(f"  p99          : {results['p99_s']:.3f} s")
    print(f"{'='*60}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge into benchmark_results.json if it exists
    bench_path = Path("benchmarks/results/benchmark_results.json")
    existing   = {}
    if bench_path.exists():
        with open(bench_path, encoding="utf-8") as fh:
            existing = json.load(fh)
    existing["latency"] = results
    with open(bench_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
    print(f"  Results saved to {bench_path}")

    # Also save standalone latency file
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Results saved to {out_path}")

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VYOR AI Latency Benchmark")
    parser.add_argument("--url",     default="http://localhost:8000",            help="API base URL")
    parser.add_argument("--queries", type=int, default=50,                       help="Number of queries to run")
    parser.add_argument("--output",  default="benchmarks/results/latency.json",  help="Output JSON path")
    args = parser.parse_args()

    run(url=args.url, n_queries=args.queries, output=args.output)
