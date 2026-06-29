"""
benchmarks/test_hallucination.py - Hallucination Rate Evaluation

Calls Contract 2 (run_orchestrator) for each HaluMem evaluation query
and counts answers that contain no citations but make factual claims.

Target: < 5% hallucination rate on 200-query evaluation set.

Usage:
    python benchmarks/test_hallucination.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integration_interface import run_orchestrator

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
    if result.get("citations"):
        return False                    # cited -> not flagged
    answer_lower = answer.lower()
    has_claim    = any(f" {kw} " in answer_lower for kw in _CLAIM_KEYWORDS)
    return len(answer) > 50 and has_claim


def run(
    queries_path: str = "data/halumem/evaluation_queries.json",
    max_queries: int  = 200,
) -> float:
    """
    Evaluate hallucination rate.

    Returns:
        rate (float) - fraction of flagged responses.
    """
    q_path = Path(queries_path)
    if not q_path.exists():
        print(f"ERROR: HaluMem queries not found at {q_path}")
        print("Download from the research repository or generate synthetic queries.")
        return 0.0

    with open(q_path, encoding="utf-8") as fh:
        queries = json.load(fh)

    hallucinated = 0
    total        = min(len(queries), max_queries)

    print(f"\nHallucination Rate Evaluation (HaluMem)")
    print(f"  Queries: {total}")
    print()

    for i, item in enumerate(queries[:total]):
        result = run_orchestrator(item["query"])
        answer = result.get("answer", "")
        if _is_hallucinated(result, answer):
            hallucinated += 1
            print(f"  [{i+1}] FLAGGED: {answer[:80]} ...")

    rate = hallucinated / total if total > 0 else 0.0
    print(f"\n  Hallucinated : {hallucinated}/{total}")
    print(f"  Rate         : {rate:.1%}")
    print(f"  Target       : < 5%")
    print(f"  Result       : {'PASS' if rate <= 0.05 else 'FAIL - target is < 5%'}")

    results_path = Path("benchmarks/results/benchmark_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as fh:
            existing = json.load(fh)
    existing["hallucination"] = {
        "rate": rate,
        "hallucinated": hallucinated,
        "total": total,
        "pass": rate <= 0.05,
    }
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)

    return rate


if __name__ == "__main__":
    run()
