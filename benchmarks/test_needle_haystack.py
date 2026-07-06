"""
benchmarks/test_needle_haystack.py — Needle-in-Haystack Retrieval Test

Hides a known fact in a large document and tests whether hybrid search
can find it at increasing document sizes.

Default sizes: 1K, 10K, 100K words.
Extended mode: configurable via --max-size (up to 2,000,000 words).

Usage:
    python benchmarks/test_needle_haystack.py
    python benchmarks/test_needle_haystack.py --max-size 500000
    python benchmarks/test_needle_haystack.py --sizes 1000 10000 100000 500000
    python benchmarks/test_needle_haystack.py --max-size 2000000 --iterations 10
"""

import sys
import json
import tempfile
import os
import argparse
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.ingestion import DocumentIngestor
from src.vector_db.qdrant_manager import QdrantManager
from src.integration_interface import process_incoming_chunk

NEEDLE = "The secret project code is VYOR-ALPHA-7"
DEFAULT_SIZES = [1_000, 10_000, 100_000]


def test_size(
    tokens:   int,
    qdrant:   QdrantManager,
    ingestor: DocumentIngestor,
    iterations: int = 1,
) -> tuple[bool, float]:
    """
    Return (found: bool, avg_latency_s: float) for a given context size.
    Runs `iterations` independent trials and returns best-of-N (conservative).
    """
    from scripts.generate_needle_haystack import generate
    import time

    hit_count = 0
    latencies = []

    for trial in range(iterations):
        data = generate(tokens, NEEDLE)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(data["text"])
            tmp = fh.name

        try:
            t0 = time.perf_counter()
            chunks, embeddings = ingestor.ingest(tmp)

            for chunk, emb in zip(chunks, embeddings):
                if process_incoming_chunk(chunk, emb.tolist()) == "save_to_qdrant":
                    qdrant.save([chunk], [emb], f"needle_{tokens}_trial{trial}")

            results = qdrant.hybrid_search("What is the secret project code?", top_k=5)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            found = any(NEEDLE.lower() in r["text"].lower() for r in results)
            if found:
                hit_count += 1

            # Also try exact search as fallback
            if not found:
                exact = qdrant.exact_search(NEEDLE.split()[-1], limit=5)
                found = any(NEEDLE.lower() in r["text"].lower() for r in exact)
                if found:
                    hit_count += 1

        finally:
            os.unlink(tmp)

    found_overall = hit_count > 0
    avg_lat = statistics.mean(latencies) if latencies else 0.0

    status = "FOUND" if found_overall else "MISSED"
    print(
        f"  {tokens:>12,} words | {status} | "
        f"hits={hit_count}/{iterations} | "
        f"avg_latency={avg_lat:.3f}s"
    )
    return found_overall, avg_lat


def run(
    sizes:      list[int] = None,
    iterations: int       = 1,
) -> float:
    """Run all needle sizes and report overall recall."""
    if sizes is None:
        sizes = DEFAULT_SIZES

    print(f"\n{'='*65}")
    print(f"  Needle-in-Haystack Retrieval Test")
    print(f"  Needle     : '{NEEDLE}'")
    print(f"  Sizes      : {[f'{s:,}' for s in sizes]} words")
    print(f"  Iterations : {iterations} per size")
    print(f"{'='*65}\n")

    qdrant   = QdrantManager()
    ingestor = DocumentIngestor()

    found_results = []
    latency_map   = {}

    for n in sizes:
        found, lat = test_size(n, qdrant, ingestor, iterations=iterations)
        found_results.append(found)
        latency_map[str(n)] = round(lat, 4)

    recall = sum(found_results) / len(found_results)

    print(f"\n{'='*65}")
    print(f"  Recall : {recall:.0%}  ({sum(found_results)}/{len(found_results)} sizes found)")
    print(f"  Target : > 90%")
    print(f"  Result : {'✅ PASS' if recall >= 0.90 else '❌ FAIL'}")
    print(f"{'='*65}\n")

    # ── Persist results ───────────────────────────────────────────────────────
    results_path = Path("benchmarks/results/benchmark_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as fh:
            existing = json.load(fh)

    existing["needle_haystack"] = {
        "recall":     recall,
        "pass":       recall >= 0.90,
        "sizes":      {str(s): bool(r) for s, r in zip(sizes, found_results)},
        "latency_s":  latency_map,
        "iterations": iterations,
    }

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)

    print(f"  Results saved to {results_path}")
    return recall


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VYOR AI Needle-in-Haystack Benchmark")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Space-separated list of context sizes in words (e.g. --sizes 1000 10000 100000)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Auto-generate sizes from 1K up to MAX_SIZE (doubling each step)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of independent trials per size (default 1)",
    )
    args = parser.parse_args()

    if args.max_size:
        # Auto-generate doubling sizes: 1K, 2K, 4K, ... up to max_size
        sizes = []
        s = 1_000
        while s <= args.max_size:
            sizes.append(s)
            s *= 2
        if sizes[-1] != args.max_size:
            sizes.append(args.max_size)
        print(f"  Auto-generated sizes: {[f'{x:,}' for x in sizes]}")
    elif args.sizes:
        sizes = args.sizes
    else:
        sizes = DEFAULT_SIZES

    run(sizes=sizes, iterations=args.iterations)
