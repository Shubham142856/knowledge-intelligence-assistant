"""
benchmarks/memory_capacity.py — Titans Memory Capacity & Saturation Study

Tests the limits of the Neural LTM reconstruction capability under consecutive updates.
Compares:
1. Standard test-time learning without forgetting (alpha_t = 0.0)
2. Adaptive test-time learning with forgetting decay (alpha_t = 1.0)

Measures and logs:
- Reconstruction loss over 1000 sequential updates.
- LTM parameter L2 norms (to measure weight saturation).
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path

# Add project root directory to python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.titans_memory_core import NeuralLTM

def calculate_parameter_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def run_study():
    print(f"\n{'='*65}")
    print(f"  Neural LTM Capacity & Weight Saturation Study")
    print(f"{'='*65}\n")
    
    dim = 384
    num_updates = 500
    
    # Generate constant input data for test-time updates
    torch.manual_seed(42)
    inputs = [torch.randn(1, 1, dim) for _ in range(num_updates)]
    
    # ── Scenario A: No forgetting/decay (alpha_t = 0.0) ──────────────────────
    print("Running Scenario A (Alpha = 0.0: standard learning without forgetting)...")
    ltm_no_decay = NeuralLTM(dim=dim)
    losses_no_decay = []
    norms_no_decay = []
    
    for i, x in enumerate(inputs):
        loss = ltm_no_decay.update_memory(x, alpha_t=0.0)
        losses_no_decay.append(loss)
        if (i + 1) % 50 == 0 or i == 0:
            norm = calculate_parameter_norm(ltm_no_decay)
            norms_no_decay.append((i + 1, norm))
            print(f"  Update {i+1:<4} | Loss: {loss:.4f} | Param Norm: {norm:.4f}")
            
    # ── Scenario B: With forgetting/decay (alpha_t = 0.5) ─────────────────────
    print("\nRunning Scenario B (Alpha = 0.5: adaptive forgetting weight decay)...")
    ltm_decay = NeuralLTM(dim=dim)
    losses_decay = []
    norms_decay = []
    
    for i, x in enumerate(inputs):
        # We simulate varying alpha_t values; here we set it to 0.5 for uniform comparison
        loss = ltm_decay.update_memory(x, alpha_t=0.5)
        losses_decay.append(loss)
        if (i + 1) % 50 == 0 or i == 0:
            norm = calculate_parameter_norm(ltm_decay)
            norms_decay.append((i + 1, norm))
            print(f"  Update {i+1:<4} | Loss: {loss:.4f} | Param Norm: {norm:.4f}")

    # Summarize results
    results = {
        "updates": [x[0] for x in norms_decay],
        "no_decay": {
            "final_loss": losses_no_decay[-1],
            "norms": [x[1] for x in norms_no_decay]
        },
        "with_decay": {
            "final_loss": losses_decay[-1],
            "norms": [x[1] for x in norms_decay]
        }
    }
    
    # Render Markdown table
    report_md = "# Neural LTM Capacity & Weight Saturation Study Results\n\n"
    report_md += "| Step | No Decay (Alpha=0.0) Norm | With Decay (Alpha=0.5) Norm | No Decay Loss | With Decay Loss |\n"
    report_md += "| --- | --- | --- | --- | --- |\n"
    
    steps = [0] + [x[0] for x in norms_decay]
    for idx, step in enumerate(steps[1:]):
        # Find closest loss
        loss_nd = losses_no_decay[step-1]
        loss_d = losses_decay[step-1]
        norm_nd = norms_no_decay[idx][1]
        norm_d = norms_decay[idx][1]
        report_md += f"| {step} | {norm_nd:.4f} | {norm_d:.4f} | {loss_nd:.4f} | {loss_d:.4f} |\n"
        
    print("\n" + report_md)
    
    # Save results
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "memory_capacity.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        
    with open(out_dir / "memory_capacity_report.md", "w", encoding="utf-8") as fh:
        fh.write(report_md)
        
    print(f"  Results saved to {out_dir}/")

if __name__ == "__main__":
    run_study()
