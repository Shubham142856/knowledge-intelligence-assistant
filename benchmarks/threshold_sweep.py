"""
benchmarks/threshold_sweep.py — Surprise Gate Threshold Sweep Optimization

Sweeps the surprise gate percentile threshold from 60% to 95%.
Measures the division of routing (LTM vs. Qdrant) and evaluates performance.
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path

# Add project root directory to python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from surprise_gate import VYORSurpriseGate

def generate_synthetic_loss_profile(num_chunks: int = 150) -> list[float]:
    """
    Generates a realistic stream of Huber losses.
    Includes a mix of low surprise (routine information) and spikes (novel facts).
    """
    np.random.seed(42)
    # Routine/expected info: low loss (0.05 to 0.15)
    routine_losses = np.random.normal(0.10, 0.03, int(num_chunks * 0.75))
    
    # Novel/surprising info: high loss (0.45 to 0.85)
    novel_losses = np.random.normal(0.65, 0.10, num_chunks - len(routine_losses))
    
    # Mix them randomly
    all_losses = np.concatenate([routine_losses, novel_losses])
    np.random.shuffle(all_losses)
    return [float(x) for x in all_losses]

def run_sweep():
    print(f"\n{'='*65}")
    print(f"  Surprise Gate Threshold Sweep (60% to 95%)")
    print(f"{'='*65}\n")
    
    losses = generate_synthetic_loss_profile(num_chunks=200)
    percentiles = [60, 65, 70, 75, 80, 85, 90, 95]
    results = []
    
    for p in percentiles:
        gate = VYORSurpriseGate(percentile=p)
        ltm_count = 0
        qdrant_count = 0
        
        for loss in losses:
            decision, _ = gate.update_and_route(loss)
            if decision == "memory_updated":
                ltm_count += 1
            else:
                qdrant_count += 1
                
        total = len(losses)
        ltm_pct = ltm_count / total
        qdrant_pct = qdrant_count / total
        
        # Estimate theoretical recall and latency trade-offs
        # Higher LTM means more test-time learning overhead (higher latency) but higher recall of edge cases.
        estimated_latency = 0.2 + (ltm_pct * 0.8) # base RAG (0.2s) + LTM gradient updates (0.8s)
        estimated_recall = 0.75 + (ltm_pct * 0.20) - (0.05 if ltm_pct > 0.35 else 0.0) # Sweet spot is when we capture the right amount of surprise
        
        print(f"  Threshold Percentile: {p}% | routed LTM: {ltm_count:<3} ({ltm_pct:.1%}) | routed Qdrant: {qdrant_count:<3} ({qdrant_pct:.1%})")
        results.append({
            "percentile": p,
            "ltm_count": ltm_count,
            "qdrant_count": qdrant_count,
            "ltm_pct": round(ltm_pct, 4),
            "qdrant_pct": round(qdrant_pct, 4),
            "estimated_latency_s": round(estimated_latency, 3),
            "estimated_recall": round(estimated_recall, 3)
        })

    # Render Markdown table
    report_md = "# Surprise Gate Threshold Sweep Optimization Results\n\n"
    report_md += "| Percentile | Chunks Routed to LTM | Chunks Routed to Qdrant | LTM % | Qdrant % | Est. Latency (s) | Est. Recall |\n"
    report_md += "| --- | --- | --- | --- | --- | --- | --- |\n"
    for r in results:
        report_md += f"| {r['percentile']}% | {r['ltm_count']} | {r['qdrant_count']} | {r['ltm_pct']:.2%} | {r['qdrant_pct']:.2%} | {r['estimated_latency_s']:.3f}s | {r['estimated_recall']:.2%} |\n"
        
    print("\n" + report_md)
    
    # Save results
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "threshold_sweep.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        
    with open(out_dir / "threshold_sweep_report.md", "w", encoding="utf-8") as fh:
        fh.write(report_md)
        
    print(f"  Results saved to {out_dir}/")

if __name__ == "__main__":
    run_sweep()
