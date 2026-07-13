"""
benchmarks/test_gsm8k.py — GSM8K Math Reasoning Benchmark

Evaluates multi-agent debate performance on multi-step mathematical problems.
Compares standard single-agent generation vs. multi-agent critic debate loop.
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
from src.agents.llm import call_llm, clean_json_response

GSM8K_QUESTIONS = [
    {
        "id": "gsm_001",
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. She spent $15 on lunch. How much money does she have left?",
        "answer": 45
    },
    {
        "id": "gsm_002",
        "question": "Albert is 4 years older than his sister. In 3 years, his sister will be 12. How old is Albert now?",
        "answer": 13
    },
    {
        "id": "gsm_003",
        "question": "A baker made 40 loaves of bread. He sold 25 of them in the morning. In the afternoon, he baked 15 more loaves and sold 20. How many loaves of bread does he have left?",
        "answer": 10
    },
    {
        "id": "gsm_004",
        "question": "A rectangle has a length of 12 cm and a width that is half of the length. What is the area of the rectangle in square centimeters?",
        "answer": 72
    },
    {
        "id": "gsm_005",
        "question": "There are 5 boxes of pencils. Each box contains 12 pencils. If 3 teachers share the pencils equally, how many pencils does each teacher get?",
        "answer": 20
    },
    {
        "id": "gsm_006",
        "question": "A factory produces 150 toys per day. How many toys does it produce in 2 weeks if it operates only on weekdays (5 days per week)?",
        "answer": 1500
    },
    {
        "id": "gsm_007",
        "question": "Mary had $50. She bought 3 books for $8 each and a backpack for $18. How much change did she receive?",
        "answer": 8
    },
    {
        "id": "gsm_008",
        "question": "James runs 4 miles a day on Monday, Wednesday, and Friday. On Saturday, he runs 6 miles. How many miles does he run in a week?",
        "answer": 18
    },
    {
        "id": "gsm_009",
        "question": "A store sells apples in bags of 6 for $3. If John wants to buy 24 apples, how much will he pay?",
        "answer": 12
    },
    {
        "id": "gsm_010",
        "question": "A library has 400 books. If 20% of the books are checked out, how many books are left in the library?",
        "answer": 320
    }
]

def extract_number(llm_response: str) -> float:
    """Helper to extract a numerical answer from the LLM output."""
    sys_prompt = (
        "Extract the final numerical answer from the assistant's text response. "
        "Return only a JSON object: {\"number\": float_or_int}"
    )
    try:
        resp = call_llm(sys_prompt, llm_response)
        parsed = clean_json_response(resp)
        return float(parsed.get("number", -999.0))
    except Exception:
        return -999.0

def run():
    print(f"\n{'='*65}")
    print(f"  GSM8K Math Reasoning Benchmark (10 Questions)")
    print(f"{'='*65}\n")
    
    orchestrator = VYOROrchestrator()
    single_agent_correct = 0
    debate_correct = 0
    total = len(GSM8K_QUESTIONS)

    for i, item in enumerate(GSM8K_QUESTIONS):
        # 1. Run single-agent (zero-shot direct prompt to LLM)
        sys_prompt = "You are a math tutor. Solve the math problem step by step and provide the final number."
        single_resp = call_llm(sys_prompt, item["question"])
        single_num = extract_number(single_resp)
        single_ok = (abs(single_num - item["answer"]) < 0.01)
        if single_ok:
            single_agent_correct += 1

        # 2. Run multi-agent debate (Full orchestrator)
        res = orchestrator.execute_query(item["question"])
        debate_num = extract_number(res.get("answer", ""))
        debate_ok = (abs(debate_num - item["answer"]) < 0.01)
        if debate_ok:
            debate_correct += 1

        print(f"  [{i+1}/{total}] Question ID: {item['id']} | Expected: {item['answer']} | Single: {single_num} ({'✓' if single_ok else '✗'}) | Debate: {debate_num} ({'✓' if debate_ok else '✗'})")
        time.sleep(0.5)

    single_acc = single_agent_correct / total
    debate_acc = debate_correct / total

    print(f"\n{'='*65}")
    print(f"  Single-Agent Accuracy: {single_acc:.1%}")
    print(f"  Debate Accuracy       : {debate_acc:.1%}")
    print(f"  Improvement          : {debate_acc - single_acc:+.1%}")
    print(f"{'='*65}\n")

    # Persist results
    results_path = Path("benchmarks/results/gsm8k_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "total_questions": total,
        "single_agent_accuracy": single_acc,
        "debate_accuracy": debate_acc,
        "correct_single": single_agent_correct,
        "correct_debate": debate_correct,
    }

    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"  Results saved to {results_path}")

if __name__ == "__main__":
    run()
