"""
benchmarks/test_retrieval.py — EnterpriseRAG-Bench Retrieval Accuracy

Tests hybrid search recall against the EnterpriseRAG-Bench ground truth.
Target: > 80% recall on 500-question test set.

Real dataset format (questions.jsonl):
  {
    "question_id": "qst_0001",
    "question_type": "basic",
    "source_types": ["github"],
    "question": "What are the default size limits...",
    "expected_doc_ids": ["dsid_ae068ee4aa9640159427cd941bef0238"],
    "gold_answer": "The default limits are...",
    "answer_facts": ["fact 1", "fact 2"]
  }

Run AFTER ingesting documents with:
    python scripts/load_enterprise_bench.py --smart-limit 175
Then:
    python benchmarks/test_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_db.qdrant_manager import QdrantManager


def run(
    questions_path: str = "data/enterprise_rag_bench/questions.jsonl",
    top_k: int = 10,
    max_questions: int = 500,
) -> float:
    """
    Run retrieval recall benchmark against the real EnterpriseRAG-Bench dataset.

    Recall is measured as: fraction of questions where the correct source document ID
    (expected_doc_ids) was retrieved in the top-k results.

    Returns:
        recall (float): fraction of questions where at least one expected doc ID
                        was found in the top-k retrieved chunks.
    """
    q_path = Path(questions_path)

    if not q_path.exists():
        print(f"ERROR: Questions file not found: {q_path}")
        print("Clone the dataset first:")
        print("  git clone https://github.com/onyx-dot-app/EnterpriseRAG-Bench data/enterprise_rag_bench")
        return 0.0

    # Load JSONL (one JSON object per line)
    questions = []
    with open(q_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    # Use only "basic" question types for simpler recall evaluation
    basic_questions = [q for q in questions if q.get("question_type") == "basic"]
    test_set = basic_questions[:max_questions]
    total = len(test_set)

    if total == 0:
        print("No questions found to test.")
        return 0.0

    print(f"\nEnterpriseRAG-Bench Retrieval Recall")
    print(f"  Total available: {len(questions)} questions")
    print(f"  Testing on    : {total} basic questions")
    print(f"  Top-K         : {top_k}")
    print()

    qdrant = QdrantManager()
    hits   = 0

    for i, item in enumerate(test_set):
        question         = item["question"]
        expected_doc_ids = set(item.get("expected_doc_ids", []))

        results = qdrant.hybrid_search(question, top_k=top_k)
        
        # Collect all retrieved document IDs
        retrieved_doc_ids = {r["doc_id"] for r in results if r.get("doc_id")}

        # Hit = at least one expected doc ID is among the retrieved doc IDs
        hit = len(expected_doc_ids.intersection(retrieved_doc_ids)) > 0

        if hit:
            hits += 1

        if (i + 1) % 50 == 0 or (i + 1) == total:
            current_recall = hits / (i + 1)
            print(f"  [{i+1}/{total}] Running recall: {current_recall:.1%}")

    recall = hits / total if total > 0 else 0.0

    print(f"\n  Questions tested : {total}")
    print(f"  Hits             : {hits}")
    print(f"  Recall           : {recall:.1%}")
    print(f"  Target           : > 80%")
    print(f"  Result           : {'PASS' if recall >= 0.80 else 'FAIL - target is > 80%'}")

    # Save results
    results_path = Path("benchmarks/results/benchmark_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as fh:
            existing = json.load(fh)
    existing["retrieval_recall"] = {
        "recall": recall,
        "hits": hits,
        "total": total,
        "pass": recall >= 0.80,
    }
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)

    print(f"\n  Results saved to {results_path}")
    return recall


if __name__ == "__main__":
    run()
