"""
benchmarks/ablation_suite.py — Titans Memory Variant Ablation Runner

Compares standard RAG vs. Titans MAC vs. Titans MAG vs. Titans MAL variants.
Measures and reports:
  1. Average Latency (s)
  2. Average Factual Confidence Score
  3. Context Recall / Citation Coverage
  4. Uncertainty Flags Triggered
"""

import sys
import os
import time
import json
import statistics
from pathlib import Path

# Add project root directory to python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from orchestrator import VYOROrchestrator

# Sample queries testing direct facts, multi-hop reasoning, and safety boundaries
ABLATION_QUERIES = [
    "What is the maximum duration of a remote work contract?",
    "How many medical leaves are employees entitled to per year?",
    "Who approves overtime requests for senior engineers?",
    "What are the data retention policies for customer records?",
    "What are the security requirements for remote access to production systems?",
]


def run_ablation_for_variant(variant_name: str) -> dict:
    """
    Sets the active memory variant and runs evaluation queries to collect metrics.
    """
    print(f"\n[Ablation Suite]: Starting evaluation for variant -> {variant_name}")
    
    # Configure environment variant
    os.environ["TITANS_VARIANT"] = variant_name
    
    # Reinitialize orchestrator to bind new variant configuration
    orchestrator = VYOROrchestrator()
    
    latencies = []
    confidences = []
    citations_count = []
    uncertainties = 0
    
    for q in ABLATION_QUERIES:
        t0 = time.perf_counter()
        try:
            res = orchestrator.execute_query(q)
            latency = time.perf_counter() - t0
            latencies.append(latency)
            
            conf = res.get("confidence", 0.5)
            confidences.append(conf)
            
            cites = len(res.get("citations", []))
            citations_count.append(cites)
            
            if res.get("uncertainty", False):
                uncertainties += 1
                
        except Exception as e:
            print(f"  [ERROR] Query '{q}' failed: {e}")
            
    # Compute aggregates
    return {
        "variant": variant_name,
        "avg_latency_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "p95_latency_s": round(statistics.quantiles(latencies, n=20)[18], 4) if len(latencies) > 1 else 0.0,
        "avg_confidence": round(statistics.mean(confidences), 4) if confidences else 0.0,
        "avg_citations": round(statistics.mean(citations_count), 4) if citations_count else 0.0,
        "uncertainty_triggers": uncertainties,
        "success_rate": round((len(latencies) / len(ABLATION_QUERIES)), 4)
    }


def run_suite():
    # Run variants
    variants = ["MAC", "MAG", "MAL"]
    results = []
    
    for v in variants:
        res = run_ablation_for_variant(v)
        results.append(res)
        
    # Standard RAG simulation (Disable LTM / Routing)
    print("\n[Ablation Suite]: Starting evaluation for Standard RAG (baseline)...")
    os.environ["TITANS_VARIANT"] = "MAC" # use default memory shell
    # Standard RAG doesn't route to LTM (force bypass LTM in custom setup if desired, 
    # but here we can mock it by setting memory tokens to 0 to simulate baseline)
    os.environ["TITANS_PERSISTENT_TOKENS"] = "0"
    
    rag_orchestrator = VYOROrchestrator()
    rag_latencies = []
    rag_confidences = []
    rag_citations = []
    rag_uncertainties = 0
    
    for q in ABLATION_QUERIES:
        t0 = time.perf_counter()
        try:
            res = rag_orchestrator.execute_query(q)
            latency = time.perf_counter() - t0
            rag_latencies.append(latency)
            rag_confidences.append(res.get("confidence", 0.5))
            rag_citations.append(len(res.get("citations", [])))
            if res.get("uncertainty", False):
                rag_uncertainties += 1
        except Exception as e:
            print(f"  [ERROR] Baseline query failed: {e}")
            
    results.append({
        "variant": "Standard RAG (Baseline)",
        "avg_latency_s": round(statistics.mean(rag_latencies), 4) if rag_latencies else 0.0,
        "p95_latency_s": round(statistics.quantiles(rag_latencies, n=20)[18], 4) if len(rag_latencies) > 1 else 0.0,
        "avg_confidence": round(statistics.mean(rag_confidences), 4) if rag_confidences else 0.0,
        "avg_citations": round(statistics.mean(rag_citations), 4) if rag_citations else 0.0,
        "uncertainty_triggers": rag_uncertainties,
        "success_rate": round((len(rag_latencies) / len(ABLATION_QUERIES)), 4)
    })
    
    # ── Render Markdown Report ───────────────────────────────────────────────
    report_md = "# Ablation Study Results: Titans Memory Variants vs. Baseline RAG\n\n"
    report_md += "| Configuration | Avg Latency (s) | p95 Latency (s) | Avg Confidence | Avg Citations | Uncertainty Triggers |\n"
    report_md += "| --- | --- | --- | --- | --- | --- |\n"
    for r in results:
        report_md += f"| {r['variant']} | {r['avg_latency_s']:.3f}s | {r['p95_latency_s']:.3f}s | {r['avg_confidence']:.2%} | {r['avg_citations']:.1f} | {r['uncertainty_triggers']} |\n"
        
    print("\n" + report_md)
    
    # Save results
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON data
    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        
    # Save Markdown Report
    with open(out_dir / "ablation_report.md", "w", encoding="utf-8") as fh:
        fh.write(report_md)
        
    print(f"[Ablation Suite]: Ablation files saved successfully to benchmarks/results/")


if __name__ == "__main__":
    run_suite()
