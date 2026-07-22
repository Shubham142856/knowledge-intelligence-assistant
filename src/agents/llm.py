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
load_dotenv(override=True)

log = logging.getLogger("vyor_ai.agents.llm")

OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "meta-llama/llama-3.3-70b-instruct").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Resilient list of free models to try in sequence if the primary model fails or blocks
CHAT_MODEL_FALLBACKS = [
    OPENROUTER_CHAT_MODEL,
    "meta-llama/llama-3.3-70b-instruct:free",
    "cohere/command-r-plus:free",
]


def call_llm(system_prompt: str, user_content: str, retries: int = 3) -> str:
    """
    Call LLM completions. Uses Cloud APIs primarily (OpenRouter Paid -> Groq -> Gemini),
    falls back to local Ollama if cloud fails, and raises RuntimeError if everything fails.
    """
    if os.getenv("USE_MOCK_LLM") == "1":
        return get_mock_fallback(system_prompt, user_content)

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

    # 1. Try OpenRouter (Llama 3.3 70B / configured model)
    if OPENROUTER_API_KEY:
        model_name = OPENROUTER_CHAT_MODEL
        print(f"  [LLM Core] Routing query to cloud via OpenRouter ({model_name})")
        log.info(f"LLM Call: attempting OpenRouter ({model_name})")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
            "X-Title": os.getenv("OPENROUTER_SITE_NAME", "VYOR-AI"),
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.2,
        }

        for attempt in range(retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=20)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    return content.strip()
                elif resp.status_code == 429:
                    wait = 2 ** attempt
                    log.warning(f"OpenRouter 429. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    log.warning(f"OpenRouter returned error {resp.status_code}: {resp.text}")
                    break
            except Exception as e:
                log.warning(f"OpenRouter request failed: {e}")

    # 2. Try Groq (Llama 3.1 8B Instant)
    if GROQ_API_KEY:
        print(f"  [LLM Core] Routing query to cloud via Groq (llama-3.1-8b-instant)")
        log.info("LLM Call: attempting Groq (llama-3.1-8b-instant)")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
        }
        for attempt in range(retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=15)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    return content.strip()
                elif resp.status_code == 429:
                    wait = 2 ** attempt
                    log.warning(f"Groq 429. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    log.warning(f"Groq returned error {resp.status_code}: {resp.text}")
                    break
            except Exception as e:
                log.warning(f"Groq request failed: {e}")

    # 3. Try Gemini (gemini-3.5-flash) via native REST
    if GEMINI_API_KEY:
        print(f"  [LLM Core] Routing query to cloud via Gemini (gemini-3.5-flash)")
        log.info("LLM Call: attempting Gemini (gemini-3.5-flash)")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        # Format matching Gemini API
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": user_content}]}
            ],
            "generationConfig": {
                "temperature": 0.3
            }
        }
        for attempt in range(retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=20)
                if resp.status_code == 200:
                    res_json = resp.json()
                    # extract candidate text
                    content = res_json["candidates"][0]["content"]["parts"][0]["text"]
                    return content.strip()
                elif resp.status_code == 429:
                    wait = 2 ** attempt
                    log.warning(f"Gemini 429. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    log.warning(f"Gemini returned error {resp.status_code}: {resp.text}")
                    break
            except Exception as e:
                log.warning(f"Gemini request failed: {e}")

    # 4. Fallback: Try local Ollama (gemma2:2b or lightweight fallbacks)
    print("  [LLM Core] All Cloud providers failed/unavailable. Attempting fallback via local Ollama...")
    ollama_url = "http://localhost:11434/v1/chat/completions"
    for local_model in ["gemma2:2b", "deepseek-coder", "orca-mini", "llama3"]:
        payload = {
            "model": local_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
        }
        try:
            log.info(f"LLM Call: attempting local Ollama ({local_model})")
            resp = requests.post(ollama_url, json=payload, timeout=15)
            if resp.status_code == 200:
                print(f"  [LLM Core] Routing query locally via Ollama ({local_model})")
                content = resp.json()["choices"][0]["message"]["content"]
                return content.strip()
            elif resp.status_code == 404 or "not found" in resp.text.lower():
                log.info(f"Ollama model {local_model} not found, trying next local model.")
                continue
            else:
                log.warning(f"Ollama returned error status {resp.status_code}: {resp.text}")
                continue
        except Exception as e:
            log.warning(f"Local Ollama connection failed or not running: {e}")
            break

    raise RuntimeError("All LLM providers (Cloud APIs and local Ollama) failed. Cannot synthesize real response.")


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

    # 2. Try raw parsing and JSONDecoder raw_decode for extra trailing data
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Try raw_decode from first '{' or '['
    start_idx = cleaned.find("{")
    if start_idx != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[start_idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3. Apply heuristics to fix common JSON issues
    processed = cleaned
    try:
        # Replace python style Booleans/None if LLM generated them
        processed = processed.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
        processed = processed.replace(":True", ":true").replace(":False", ":false").replace(":None", ":null")
        
        # Fix missing commas between key-value pairs e.g. "val" "key":
        import re
        processed = re.sub(r'("(?:[^"\\]|\\.)*")\s*("(?:[^"\\]|\\.)*"\s*:)', r'\1, \2', processed)
        processed = re.sub(r',\s*\}', '}', processed)
        processed = re.sub(r',\s*\]', ']', processed)
        
        # Handle unescaped backslashes
        processed = processed.replace('\\', '\\\\')
        processed = processed.replace('\\\\n', '\\n').replace('\\\\"', '\\"').replace('\\\\t', '\\t')
        processed = processed.replace('\\\\r', '\\r').replace('\\\\b', '\\b').replace('\\\\f', '\\f')
        processed = processed.replace('\\\\/', '\\/')

        start_idx = processed.find("{")
        if start_idx != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(processed[start_idx:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return json.loads(processed)
    except Exception as e:
        log.debug(f"Heuristics parse failed: {e}")

    # 4. Regex fallback search & key-value extraction
    res = {}
    import re
    txt_match = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL)
    if txt_match:
        res["text"] = txt_match.group(1)
    src_match = re.findall(r'"(?:sources|citations)"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
    if src_match:
        items = re.findall(r'"([^"]+)"', src_match[0])
        res["sources"] = items
        res["citations"] = items
    if "approved" in cleaned.lower():
        res["approved"] = "true" in cleaned.lower()
    if res:
        return res

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            try:
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
    if "Router" in system_prompt:
        route = "ltm" if any(x in user_content.lower() for x in ["hi", "hello", "hey", "remember"]) else "complex"
        return json.dumps({
            "route": route,
            "reason": f"Mock routing to {route} based on query text."
        })
    elif "MemoryAgent" in system_prompt:
        return json.dumps({
            "recalled_text": "Neural memory recall check: Medical leaves are capped at 15 days per calendar year and require lead approval.",
            "confidence": 0.90
        })
    elif "Writer" in system_prompt:
        import re
        all_matches = re.findall(r'[\w\.-]+\.(?:pdf|docx|pptx|txt|csv)', user_content, re.IGNORECASE)
        placeholders = {"source_file.pdf", "source_doc.pdf", "sample.pdf"}
        sources = [f for f in set(all_matches) if f.lower() not in placeholders]
        return json.dumps({
            "text": "Based on retrieved domain documentation, operations follow established policy guidelines.",
            "sources": sources,
            "confidence": 0.95
        })
    elif "Critic" in system_prompt:
        return json.dumps({
            "score": 0.95,
            "issues": [],
            "approved": True
        })
    elif "Refiner" in system_prompt:
        import re
        all_matches = re.findall(r'[\w\.-]+\.(?:pdf|docx|pptx|txt|csv)', user_content, re.IGNORECASE)
        placeholders = {"source_file.pdf", "source_doc.pdf", "sample.pdf"}
        citations = [f for f in set(all_matches) if f.lower() not in placeholders]
        if not citations and "General_Knowledge" in user_content:
            citations = ["General_Knowledge"]
        return json.dumps({
            "refined_text": "Based on retrieved domain documentation, operations follow established policy guidelines.",
            "citations": citations
        })




    # Default fallback
    return "{}"

