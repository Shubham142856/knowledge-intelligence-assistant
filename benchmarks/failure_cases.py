"""
benchmarks/failure_cases.py — Adversarial Query & Failure Case Analysis

Evaluates system resilience by executing queries designed to break typical RAG pipelines:
1. Out-of-domain query (No relevant knowledge ingested).
2. Contradictory information query (Conflicting source documents).
3. Extremely long query (> 500 tokens).
4. Malformed/Ambiguous query.

Measures how the Critic and Refiner handle failure, checking:
- Answer factual accuracy (does it hallucinate or state lack of info?).
- Confidence scores.
- Assertion of the uncertainty flag.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add project root directory to python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from orchestrator import VYOROrchestrator

ADVERSARIAL_CASES = [
    {
        "category": "Out-of-Domain",
        "query": "What is the capital city of ancient Atlantis according to the legendary dialogues of Plato?"
    },
    {
        "category": "Contradictory Information",
        "query": "How many total vacation days do employees receive per year?" 
        # (This query intentionally triggers confusion if multiple conflicting files exist)
    },
    {
        "category": "Extremely Long Query",
        "query": "Tell me about the work hours " + ("and remote work policies " * 50) + "for engineering contractors?"
    },
    {
        "category": "Ambiguous/Malformed",
        "query": "When does the... you know, the policy about the stuff... apply to people?"
    }
]

def run_adversarial_suite():
    print(f"\n{'='*65}")
    print(f"  VYOR AI Adversarial Resilience & Failure Case Study")
    print(f"{'='*65}\n")
    
    orchestrator = VYOROrchestrator()
    results = []
    
    for case in ADVERSARIAL_CASES:
        cat = case["category"]
        q = case["query"]
        print(f"  Evaluating Case: [{cat}] -> Query: '{q[:50]}...'")
        
        try:
            res = orchestrator.execute_query(q)
            ans = res.get("answer", "")
            cites = res.get("citations", [])
            conf = res.get("confidence", 0.5)
            uncert = res.get("uncertainty", False)
            
            # Heuristic to check if system correctly admitted lack of info
            admitted_no_info = any(kw in ans.lower() for kw in ["cannot find", "do not know", "not provided", "not mention", "no information", "not contain", "does not contain", "unable to"])
            
            print(f"    Confidence: {conf:.2f} | Uncertainty Flag: {uncert} | Admitted Lack of Info: {admitted_no_info}")
            print(f"    Answer: '{ans[:120]}...'\n")
            
            results.append({
                "category": cat,
                "query": q,
                "answer": ans,
                "citations": cites,
                "confidence": conf,
                "uncertainty_flag": uncert,
                "admitted_lack_of_info": admitted_no_info
            })
        except Exception as e:
            print(f"    [CRITICAL FAILURE] Query failed entirely: {e}\n")
            results.append({
                "category": cat,
                "query": q,
                "error": str(e),
                "failed": True
            })

    # Render Report
    report_md = "# VYOR AI Adversarial Robustness & Failure Modes Analysis\n\n"
    report_md += "| Adversarial Category | Query Sample | Uncertainty Flagged? | Confidence Score | Citation Count | Admitted Lack of Info? |\n"
    report_md += "| --- | --- | --- | --- | --- | --- |\n"
    for r in results:
        if r.get("failed"):
            report_md += f"| {r['category']} | '{r['query'][:40]}...' | **CRASHED** | N/A | N/A | N/A |\n"
        else:
            report_md += f"| {r['category']} | '{r['query'][:40]}...' | {'YES ⚠️' if r['uncertainty_flag'] else 'NO'} | {r['confidence']:.2%} | {len(r['citations'])} | {'YES' if r['admitted_lack_of_info'] else 'NO'} |\n"

    print("\n" + report_md)
    
    # Save results
    out_dir = Path("benchmarks/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "failure_cases.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        
    with open(out_dir / "failure_cases_report.md", "w", encoding="utf-8") as fh:
        fh.write(report_md)
        
    print(f"  Results saved to {out_dir}/")

if __name__ == "__main__":
    run_adversarial_suite()
