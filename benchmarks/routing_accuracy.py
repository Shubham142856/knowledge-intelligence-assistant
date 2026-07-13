"""
benchmarks/routing_accuracy.py — Surprise Gate Routing Accuracy Analysis

Evaluates the classification performance of the Surprise Gate in routing chunks:
- Routine Chunks (expected to be routed to Qdrant)
- Novel Chunks (expected to be routed to Titans Neural LTM)

Computes:
- Confusion Matrix (TP, FP, TN, FN)
- Precision, Recall, and F1-Score of the Surprise Gate
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add project root directory to python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from surprise_gate import VYORSurpriseGate

def run_accuracy_study():
    print(f"\n{'='*65}")
    print(f"  Surprise Gate Routing Accuracy Analysis")
    print(f"{'='*65}\n")
    
    np.random.seed(101)
    num_samples = 100
    
    # 1. Warm up the surprise gate with some mixed inputs to build history
    gate = VYORSurpriseGate(percentile=80)
    warmup_losses = np.random.normal(0.12, 0.04, 30)
    for w in warmup_losses:
        gate.update_and_route(float(w))
        
    # 2. Labeled test set: 50 Routine (low surprise) + 50 Novel (high surprise)
    # Target: Routine (0) -> save_to_qdrant, Novel (1) -> memory_updated
    routine_losses = np.random.normal(0.08, 0.02, 50)
    novel_losses = np.random.normal(0.48, 0.08, 50)
    
    test_cases = []
    for l in routine_losses:
        test_cases.append((float(l), 0)) # 0 = Qdrant
    for l in novel_losses:
        test_cases.append((float(l), 1)) # 1 = LTM
        
    np.random.shuffle(test_cases)
    
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for loss, label in test_cases:
        decision, _ = gate.update_and_route(loss)
        prediction = 1 if decision == "memory_updated" else 0
        
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        elif label == 1 and prediction == 0:
            fn += 1

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(test_cases)
    
    print(f"  Confusion Matrix:")
    print(f"    True Positives  (Novel -> LTM):     {tp}")
    print(f"    False Positives (Routine -> LTM):   {fp}")
    print(f"    True Negatives  (Routine -> Qdrant):{tn}")
    print(f"    False Negatives (Novel -> Qdrant):  {fn}")
    print()
    print(f"  Accuracy  : {accuracy:.1%}")
    print(f"  Precision : {precision:.1%}")
    print(f"  Recall    : {recall:.1%}")
    print(f"  F1-Score  : {f1:.1%}")
    print()

    # Render Report
    report_md = "# Surprise Gate Routing Classification Accuracy Report\n\n"
    report_md += "### Confusion Matrix\n"
    report_md += "| Labeled / Predicted | Routed to LTM (Novel) | Routed to Qdrant (Routine) |\n"
    report_md += "| --- | --- | --- |\n"
    report_md += f"| **Is Novel** | {tp} (True Positive) | {fn} (False Negative) |\n"
    report_md += f"| **Is Routine** | {fp} (False Positive) | {tn} (True Negative) |\n\n"
    report_md += "### Performance Metrics\n"
    report_md += f"- **Accuracy**: {accuracy:.2%}\n"
    report_md += f"- **Precision**: {precision:.2%}\n"
    report_md += f"- **Recall**: {recall:.2%}\n"
    report_md += f"- **F1-Score**: {f1:.2%}\n"

    # Save results
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    with open(out_dir / "routing_accuracy.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        
    with open(out_dir / "routing_accuracy_report.md", "w", encoding="utf-8") as fh:
        fh.write(report_md)

    print(f"  Results saved to {out_dir}/")

if __name__ == "__main__":
    run_accuracy_study()
