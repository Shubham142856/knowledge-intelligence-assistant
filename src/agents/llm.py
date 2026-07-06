"""
src/agents/llm.py — OpenRouter API Helper for Multi-Agent System

Sends prompts to the configured OpenRouter model with retry logic for rate limits.
"""

import os
import json
import time
import logging
import requests
from dotenv import load_dotenv

# Load environment variables from .env at import time
load_dotenv()

log = logging.getLogger("vyor_ai.agents.llm")

OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openrouter/free").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Resilient list of free models to try in sequence if the primary model fails or blocks
CHAT_MODEL_FALLBACKS = [
    OPENROUTER_CHAT_MODEL,
    "cohere/north-mini-code:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]


def call_llm(system_prompt: str, user_content: str, retries: int = 3) -> str:
    """
    Call the OpenRouter API chat completions endpoint.
    If the primary model fails, times out, or returns a safety block,
    automatically falls back to secondary free models.
    """
    if not OPENROUTER_API_KEY:
        log.warning("OPENROUTER_API_KEY not set. Using offline fallback mock response.")
        return get_mock_fallback(system_prompt, user_content)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
        "X-Title": os.getenv("OPENROUTER_SITE_NAME", "VYOR-AI"),
    }

    # Attempt to query with fallback models if the first choice fails or blocks
    for model_candidate in CHAT_MODEL_FALLBACKS:
        log.info(f"LLM Call: using model '{model_candidate}'")
        payload = {
            "model": model_candidate,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

        for attempt in range(retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=25)
                
                # Check for rate-limiting
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    log.warning(f"OpenRouter 429 for model {model_candidate}. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code == 200:
                    body = resp.json()
                    content = body["choices"][0]["message"]["content"]
                    
                    # Check if response text looks like a safety block message or OpenRouter glitch
                    if "User Safety:" in content or "safety policy" in content.lower():
                        log.warning(f"Model '{model_candidate}' returned a safety block/glitch message. Trying next fallback.")
                        break # break the retry loop to try the next model candidate
                        
                    return content.strip()
                
                # If model is not found or other non-429 error, try next candidate
                log.warning(f"Model '{model_candidate}' returned error {resp.status_code}. Trying next fallback.")
                break
                
            except Exception as e:
                log.warning(f"Request failed for model '{model_candidate}' on attempt {attempt+1}/{retries}: {e}")
                if attempt == retries - 1:
                    log.warning(f"All retries failed for model '{model_candidate}'. Trying next fallback.")

    log.error("All fallback models failed on OpenRouter. Returning offline fallback.")
    return get_mock_fallback(system_prompt, user_content)


def clean_json_response(text: str) -> dict:
    """
    Clean LLM response text (removes markdown ```json ... ``` blocks)
    and parse as dictionary. Includes robust heuristics to fix common JSON syntax errors.
    """
    cleaned = text.strip()
    
    # 1. Remove markdown wrappers
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Quick return on empty
    if not cleaned:
        return {}

    # 2. Try raw parsing
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3. Apply heuristics to fix common JSON issues
    processed = cleaned
    try:
        # Replace python style Booleans/None if LLM generated them
        processed = processed.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
        processed = processed.replace(":True", ":true").replace(":False", ":false").replace(":None", ":null")
        
        # Replace trailing commas before closing braces/brackets, e.g. [1, 2,] -> [1, 2]
        import re
        processed = re.sub(r',\s*\}', '}', processed)
        processed = re.sub(r',\s*\]', ']', processed)
        
        # Handle unescaped backslashes (but don't mess up valid escapes like \n or \")
        # Just escape general backslashes
        processed = processed.replace('\\', '\\\\')
        # Restore valid JSON escape sequences
        processed = processed.replace('\\\\n', '\\n').replace('\\\\"', '\\"').replace('\\\\t', '\\t')
        processed = processed.replace('\\\\r', '\\r').replace('\\\\b', '\\b').replace('\\\\f', '\\f')
        processed = processed.replace('\\\\/', '\\/')

        return json.loads(processed)
    except Exception as e:
        log.error(f"Heuristics parse failed: {e}")

    # 4. Regex fallback search for a brace-enclosed block
    import re
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            # Try single-quote to double-quote replacement
            try:
                # Replace single quotes with double quotes for keys and values
                # E.g. 'approved': false -> "approved": false
                candidate_fixed = re.sub(r"'(\w+)'\s*:", r'"\1":', candidate)
                candidate_fixed = re.sub(r":\s*'([^']*)'", r': "\1"', candidate_fixed)
                candidate_fixed = candidate_fixed.replace("'", '"')
                return json.loads(candidate_fixed)
            except Exception:
                pass

    log.error(f"Failed to parse LLM JSON after all cleaning attempts.\nRaw Text: {text}")
    return {}


def get_mock_fallback(system_prompt: str, user_content: str) -> str:
    """
    Offline/Fallback mock responses when API is down or key is missing.
    Matches the schema requirements of each agent.
    """
    if "Planner" in system_prompt or "PLANNER_PROMPT" in system_prompt:
        return json.dumps({
            "sub_questions": ["Find medical leave limits", "Identify approval workflow"],
            "query_type": "factual",
            "complexity": 2
        })
    elif "Critic" in system_prompt:
        # Grounded check fallback
        if "15 days" in user_content or "medical" in user_content.lower():
            return json.dumps({
                "score": 0.95,
                "issues": [],
                "approved": True
            })
        return json.dumps({
            "score": 0.50,
            "issues": ["Draft lacks precise metrics (e.g. 15 days) or source citations."],
            "approved": False
        })
    elif "Writer" in system_prompt:
        return json.dumps({
            "text": "Based on the HR policy documents, employees are entitled to a maximum of 15 medical leaves per calendar year, requiring Lead approval.",
            "sources": ["HR_Policy_2026.pdf"],
            "confidence": 0.85
        })
    # Default fallback
    return "{}"
