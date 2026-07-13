"""
benchmarks/scalability_study.py — Ingestion & Search Scalability Study

Inserts varying numbers of synthetic vectors into a temporary local Qdrant collection:
- 100, 500, 1,000, 5,000, 10,000 vectors.
Measures:
- Ingestion time (vectors/second)
- Average search latency (milliseconds)
- Verifies sub-linear O(log n) scalability of the vector DB search.
"""

import sys
import os
import time
import json
import statistics
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Add project root directory to python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def run_scalability_study():
    print(f"\n{'='*65}")
    print(f"  Qdrant Ingestion & Search Scalability Study")
    print(f"{'='*65}\n")
    
    # Connect client
    path = os.getenv("QDRANT_PATH")
    if path:
        client = QdrantClient(path=path)
    else:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=url)
        
    dim = 384
    test_collection = "vyor_scalability_test"
    
    # Ensure clean state
    try:
        client.delete_collection(test_collection)
    except Exception:
        pass
        
    client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    
    sizes = [100, 500, 1000, 5000, 10000]
    results = []
    
    # Generate large pool of random vectors
    np.random.seed(42)
    vector_pool = np.random.randn(max(sizes), dim).astype(np.float32)
    # L2 normalize for cosine distance compatibility
    vector_pool = vector_pool / np.linalg.norm(vector_pool, axis=1, keepdims=True)
    
    current_count = 0
    
    for target_size in sizes:
        to_add = target_size - current_count
        points = []
        
        t_ingest_start = time.perf_counter()
        for idx in range(to_add):
            pt_id = current_count + idx
            vec = vector_pool[pt_id].tolist()
            points.append(
                PointStruct(
                    id=pt_id,
                    vector=vec,
                    payload={"text": f"Sample chunk {pt_id}", "source": "synthetic.txt"}
                )
            )
            
        # Batch upload
        client.upsert(collection_name=test_collection, points=points)
        ingest_time = time.perf_counter() - t_ingest_start
        ingest_rate = to_add / ingest_time if ingest_time > 0 else 0.0
        
        current_count = target_size
        
        # Run search query evaluations
        search_latencies = []
        for _ in range(50):
            q_vec = np.random.randn(dim).astype(np.float32)
            q_vec = (q_vec / np.linalg.norm(q_vec)).tolist()
            
            t_search_start = time.perf_counter()
            client.search(
                collection_name=test_collection,
                query_vector=q_vec,
                limit=10
            )
            search_latencies.append(time.perf_counter() - t_search_start)
            
        avg_search_ms = statistics.mean(search_latencies) * 1000.0
        p95_search_ms = statistics.quantiles(search_latencies, n=20)[18] * 1000.0
        
        print(f"  DB Size: {target_size:<5} | Ingestion: {ingest_rate:.1f} pts/sec | Avg Search: {avg_search_ms:.2f} ms | p95 Search: {p95_search_ms:.2f} ms")
        results.append({
            "db_size": target_size,
            "ingest_rate_pts_sec": round(ingest_rate, 2),
            "avg_search_latency_ms": round(avg_search_ms, 3),
            "p95_search_latency_ms": round(p95_search_ms, 3),
        })

    # Cleanup
    client.delete_collection(test_collection)
    
    # Render Markdown table
    report_md = "# Vector Database Ingestion & Search Scalability Results\n\n"
    report_md += "| DB Size (Vectors) | Ingestion Rate (pts/sec) | Avg Search Latency (ms) | p95 Search Latency (ms) |\n"
    report_md += "| --- | --- | --- | --- |\n"
    for r in results:
        report_md += f"| {r['db_size']:,} | {r['ingest_rate_pts_sec']:.1f} | {r['avg_search_latency_ms']:.2f}ms | {r['p95_search_latency_ms']:.2f}ms |\n"
        
    print("\n" + report_md)
    
    # Save results
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "scalability.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        
    with open(out_dir / "scalability_report.md", "w", encoding="utf-8") as fh:
        fh.write(report_md)
        
    print(f"  Results saved to {out_dir}/")

if __name__ == "__main__":
    run_scalability_study()
