"""
scripts/generate_needle_haystack.py

Generate a synthetic needle-in-a-haystack test document.
A known "needle" sentence is inserted at a random position inside a
large block of filler words.

Usage:
    python scripts/generate_needle_haystack.py --tokens 100000 --output data/needle_test.json
"""

import random
import json
import argparse

# Simple word list for filler — avoids lorem_text dependency
_WORDS = (
    "the quick brown fox jumps over lazy dog a an is was are were be been being "
    "have has had do does did will would could should may might must shall can "
    "and or but if so yet for nor because although however therefore thus hence "
    "system data document knowledge information query answer result process "
    "enterprise retrieval memory context neural vector embedding model response "
    "latency accuracy hallucination confidence citation source pipeline chunk "
    "ingestion parsing extraction transformation validation benchmark evaluation "
    "infrastructure container service api endpoint deployment monitoring health "
).split()


def _filler(n_words: int) -> str:
    return " ".join(random.choice(_WORDS) for _ in range(n_words))


def generate(
    total_tokens: int,
    needle: str = "The secret project code is VYOR-ALPHA-7",
) -> dict:
    """
    Build a haystack document with the needle inserted at a random position.

    Args:
        total_tokens: Approximate target word count (tokens ≈ words here).
        needle:       The specific sentence to hide in the haystack.

    Returns:
        {
            "text":         full combined text,
            "needle":       needle string,
            "position":     word index of needle insertion,
            "total_tokens": actual total word count
        }
    """
    needle_words  = needle.split()
    filler_target = max(total_tokens - len(needle_words), 10)
    words         = _filler(filler_target).split()

    pos = random.randint(0, len(words))
    words[pos:pos] = needle_words

    text = " ".join(words)
    return {
        "text":         text,
        "needle":       needle,
        "position":     pos,
        "total_tokens": len(words),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate needle-in-a-haystack test.")
    parser.add_argument("--tokens", type=int, default=100_000, help="Approx word count")
    parser.add_argument("--output", type=str, default="data/needle_tests/needle_test.json")
    parser.add_argument(
        "--needle",
        type=str,
        default="The secret project code is VYOR-ALPHA-7",
    )
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    result = generate(args.tokens, args.needle)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"Generated {result['total_tokens']:,}-word haystack → {args.output}")
    print(f"Needle at word position {result['position']:,}: '{args.needle}'")
