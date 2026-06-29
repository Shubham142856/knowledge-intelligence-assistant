"""
benchmarks/test_needle_haystack.py — Needle-in-Haystack Retrieval Test

Hides a known fact in a large document and tests whether hybrid search can
find it at increasing document sizes.

Sizes: 1 K, 10 K, 100 K words.
Target: > 90% recall across sizes.

Usage:
    python benchmarks/test_needle_haystack.py
"""

import sys
import json
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.ingestion import DocumentIngestor
from src.vector_db.qdrant_manager import QdrantManager
from src.integration_interface import process_incoming_chunk

NEEDLE = "The secret project code is VYOR-ALPHA-7"
SIZES  = [1_000, 10_000, 100_000]


def test_size(tokens: int, qdrant: QdrantManager, ingestor: DocumentIngestor) -> bool:
    """Return True if the needle is found in top-5 results."""
    from scripts.generate_needle_haystack import generate

    data = generate(tokens, NEEDLE)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(data["text"])
        tmp = fh.name

    try:
        chunks, embeddings = ingestor.ingest(tmp)

        for chunk, emb in zip(chunks, embeddings):
            if process_incoming_chunk(chunk, emb.tolist()) == "save_to_qdrant":
                qdrant.save([chunk], [emb], f"needle_{tokens}")

        results = qdrant.hybrid_search("What is the secret project code?", top_k=5)
        found   = any(NEEDLE.lower() in r["text"].lower() for r in results)
        status  = "FOUND" if found else "MISSED"
        print(f"  {tokens:>8,} words: {status}")
        return found

    finally:
        os.unlink(tmp)


def run() -> float:
    """Run all needle sizes and report overall recall."""
    print("\nNeedle-in-Haystack Retrieval Test")
    print(f"  Needle: '{NEEDLE}'")
    print()

    qdrant   = QdrantManager()
    ingestor = DocumentIngestor()

    results = [test_size(n, qdrant, ingestor) for n in SIZES]
    recall  = sum(results) / len(results)

    print(f"\n  Recall : {recall:.0%}  ({sum(results)}/{len(results)} found)")
    print(f"  Target : > 90%")
    print(f"  Result : {'PASS' if recall >= 0.90 else 'FAIL'}")

    # Persist
    results_path = Path("benchmarks/results/benchmark_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as fh:
            existing = json.load(fh)
    existing["needle_haystack"] = {
        "recall": recall,
        "pass": recall >= 0.90,
        "sizes": {str(s): bool(r) for s, r in zip(SIZES, results)},
    }
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)

    return recall


if __name__ == "__main__":
    run()
