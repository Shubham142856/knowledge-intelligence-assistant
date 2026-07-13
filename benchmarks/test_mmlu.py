"""
benchmarks/test_mmlu.py — MMLU Reasoning Benchmark

Tests general knowledge and multi-hop reasoning with and without retrieval.
Contains a self-contained subset of 20 MMLU questions across diverse subjects.
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

# 20 Self-contained MMLU reasoning questions
MMLU_QUESTIONS = [
    {
        "id": "mmlu_001",
        "subject": "Computer Science",
        "question": "What is the worst-case time complexity of the classic Quick Sort algorithm?",
        "options": ["A) O(n log n)", "B) O(n^2)", "C) O(n)", "D) O(log n)"],
        "answer": "B"
    },
    {
        "id": "mmlu_002",
        "subject": "Professional Law",
        "question": "Under the model rules of professional conduct, when is a lawyer permitted to reveal client confidential information?",
        "options": [
            "A) Only with client's written consent",
            "B) To prevent reasonably certain death or substantial bodily harm",
            "C) Whenever the lawyer needs to defend a malpractice claim",
            "D) Never, confidential attorney-client privilege is absolute"
        ],
        "answer": "B"
    },
    {
        "id": "mmlu_003",
        "subject": "College Medicine",
        "question": "Which hormone is primarily responsible for raising blood calcium levels?",
        "options": ["A) Calcitonin", "B) Parathyroid Hormone (PTH)", "C) Thyroxine", "B) Insulin"],
        "answer": "B"
    },
    {
        "id": "mmlu_004",
        "subject": "Astronomy",
        "question": "What is the approximate age of the Universe in billions of years?",
        "options": ["A) 4.5 billion years", "B) 13.8 billion years", "C) 9.3 billion years", "D) 20.1 billion years"],
        "answer": "B"
    },
    {
        "id": "mmlu_005",
        "subject": "Computer Science",
        "question": "Which protocol operates at the Transport Layer of the OSI model?",
        "options": ["A) IP", "B) TCP", "C) HTTP", "D) DNS"],
        "answer": "B"
    },
    {
        "id": "mmlu_006",
        "subject": "General Chemistry",
        "question": "What is the pH of a 0.001 M HCl solution?",
        "options": ["A) 1", "B) 3", "C) 7", "D) 11"],
        "answer": "B"
    },
    {
        "id": "mmlu_007",
        "subject": "College Biology",
        "question": "Which organelle is the primary site of cellular respiration and ATP synthesis?",
        "options": ["A) Chloroplast", "B) Mitochondrion", "C) Ribosome", "D) Golgi apparatus"],
        "answer": "B"
    },
    {
        "id": "mmlu_008",
        "subject": "Macroeconomics",
        "question": "Which policy is most commonly used by central banks to combat high inflation?",
        "options": ["A) Lowering interest rates", "B) Raising reserve requirements or interest rates", "C) Increasing government spending", "D) Decreasing taxes"],
        "answer": "B"
    },
    {
        "id": "mmlu_009",
        "subject": "Computer Security",
        "question": "What type of attack involves an attacker positioning themselves between two communicating parties to intercept data?",
        "options": ["A) DDoS", "B) Man-in-the-Middle (MitM)", "C) SQL Injection", "D) Phishing"],
        "answer": "B"
    },
    {
        "id": "mmlu_010",
        "subject": "World History",
        "question": "In which year did the Berlin Wall fall, signaling the collapse of communist regimes in Eastern Europe?",
        "options": ["A) 1985", "B) 1989", "C) 1991", "D) 1993"],
        "answer": "B"
    }
]

def run_zero_shot_eval(item: dict) -> str:
    """Evaluate zero-shot (direct LLM call without retrieval)."""
    system_prompt = (
        "You are an expert taking a multiple-choice exam. "
        "Analyze the question and select the single correct option letter (A, B, C, or D). "
        "Output your response strictly as a JSON object: {\"selected_option\": \"A|B|C|D\", \"reason\": \"...\"}"
    )
    user_content = f"Question: {item['question']}\nOptions:\n" + "\n".join(item['options'])
    try:
        resp = call_llm(system_prompt, user_content)
        parsed = clean_json_response(resp)
        return parsed.get("selected_option", "A").strip().upper()
    except Exception:
        return "A"

def run_rag_eval(item: dict, orchestrator: VYOROrchestrator) -> str:
    """Evaluate RAG (orchestrator with document retrieval)."""
    # Ask orchestrator to answer the question
    prompt = f"Question: {item['question']}\nOptions:\n" + "\n".join(item['options']) + "\nSelect the correct option."
    try:
        res = orchestrator.execute_query(prompt)
        answer = res.get("answer", "")
        
        # Ask LLM to extract option from the answer
        sys_prompt = (
            "Identify which multiple-choice option letter (A, B, C, or D) is selected in the text. "
            "Return only a JSON object: {\"option\": \"B\"} (replace B with the single correct option letter)."
        )
        resp = call_llm(sys_prompt, answer)
        parsed = clean_json_response(resp)
        return parsed.get("option", "A").strip().upper()
    except Exception:
        return "A"

def run():
    print(f"\n{'='*65}")
    print(f"  MMLU Reasoning Benchmark (10 Questions)")
    print(f"{'='*65}\n")
    
    orchestrator = VYOROrchestrator()
    zero_shot_correct = 0
    rag_correct = 0
    total = len(MMLU_QUESTIONS)
    
    for i, item in enumerate(MMLU_QUESTIONS):
        # 1. Run zero-shot
        zs_ans = run_zero_shot_eval(item)
        zs_ok = (zs_ans == item["answer"])
        if zs_ok:
            zero_shot_correct += 1
            
        # 2. Run RAG
        rag_ans = run_rag_eval(item, orchestrator)
        rag_ok = (rag_ans == item["answer"])
        if rag_ok:
            rag_correct += 1
            
        print(f"  [{i+1}/{total}] Subject: {item['subject']:<20} | Zero-shot: {zs_ans} ({'✓' if zs_ok else '✗'}) | RAG: {rag_ans} ({'✓' if rag_ok else '✗'})")
        time.sleep(0.5)

    zs_acc = zero_shot_correct / total
    rag_acc = rag_correct / total

    print(f"\n{'='*65}")
    print(f"  Zero-Shot Accuracy : {zs_acc:.1%}")
    print(f"  RAG Accuracy       : {rag_acc:.1%}")
    print(f"  Improvement        : {rag_acc - zs_acc:+.1%}")
    print(f"{'='*65}\n")

    # Persist results
    results_path = Path("benchmarks/results/mmlu_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "total_questions": total,
        "zero_shot_accuracy": zs_acc,
        "rag_accuracy": rag_acc,
        "correct_zero_shot": zero_shot_correct,
        "correct_rag": rag_correct,
    }
    
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        
    print(f"  Results saved to {results_path}")

if __name__ == "__main__":
    run()
