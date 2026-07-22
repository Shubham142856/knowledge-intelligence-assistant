"""
scripts/test_realtime_hr.py — Test Real-Time HR Policy Grounding Solution
"""

import sys
import os
import json
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["USE_MOCK_LLM"] = "1"

from integration_interface import run_orchestrator

test_queries = [
    "How many medical leaves are employees entitled to per year?",
    "What is the maximum duration of a remote work contract?",
    "Who approves overtime requests for senior engineers?",
    "How are customer database backups encrypted?",
    "Who handles emergency hardware replacements in the datacenter?",
    "What is the policy on intellectual property developed by contractors?"
]

def main():
    print(f"\n{'='*75}")
    print(f"  REAL-TIME HR POLICY GROUNDING & HALLUCINATION 0 VERIFICATION")
    print(f"{'='*75}\n")

    for i, q in enumerate(test_queries, 1):
        res = run_orchestrator(q)
        answer = res.get("answer", "")
        citations = res.get("citations", [])
        confidence = res.get("confidence", 0.0)

        print(f"[{i}] Query: '{q}'")
        print(f"    Answer    : {answer}")
        print(f"    Citations : {citations}")
        print(f"    Confidence: {confidence:.2%}")
        print(f"    Cites HR  : {'YES' if 'HR_Policy_2026.pdf' in citations else 'NO'}\n")

if __name__ == "__main__":
    main()
