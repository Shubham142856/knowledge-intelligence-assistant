"""
benchmarks/debug_retrieval.py

Diagnostic script to print actual points, search results, and match status
for the first question in the dataset, using a single Qdrant client instance.
"""

import sys, os, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from src.rag.ingestion import get_embedder

def run():
    path = os.getenv("QDRANT_PATH", "data/qdrant_db")
    print(f"QDRANT_PATH: {path}")

    # Check via Client (instantiated once)
    client = QdrantClient(path=path)
    collections = [c.name for c in client.get_collections().collections]
    print(f"Collections: {collections}")

    if "vyor_knowledge_base" not in collections:
        print("Collection 'vyor_knowledge_base' not found!")
        return

    count_res = client.count("vyor_knowledge_base")
    print(f"Point count in vyor_knowledge_base: {count_res.count}")

    # Load first question
    q_path = Path("data/enterprise_rag_bench/questions.jsonl")
    if not q_path.exists():
        print("Questions file not found!")
        return

    with open(q_path, encoding="utf-8") as f:
        first_q = json.loads(f.readline().strip())

    question = first_q["question"]
    gold_answer = first_q.get("gold_answer", "")
    answer_facts = first_q.get("answer_facts", [])

    print("\n--- Diagnostic Question ---")
    print(f"Question: {question}")
    print(f"Gold Answer: {gold_answer}")
    print(f"Answer Facts: {answer_facts}")

    # Run search directly
    embedder = get_embedder()
    q_emb = embedder.encode(question)
    dense_hits = client.search(
        collection_name="vyor_knowledge_base",
        query_vector=q_emb.tolist(),
        limit=5
    )

    print(f"\n--- Search Results ({len(dense_hits)} returned) ---")
    for hit in dense_hits:
        print(f"Source: {hit.payload.get('source')}")
        print(f"Score : {hit.score}")
        print(f"Text  : {hit.payload.get('text', '')[:150]}...")
        print("-" * 40)

if __name__ == "__main__":
    run()
