"""
Probe OpenRouter NVIDIA Nemotron Embed to discover output dimension.
"""
import requests, json, os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL   = os.getenv("OPENROUTER_EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")
URL     = "https://openrouter.ai/api/v1/embeddings"

print(f"Model : {MODEL}")
print(f"Key   : {API_KEY[:20]}...")
print()

resp = requests.post(
    url=URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:8000",
        "X-Title":       "VYOR-AI",
    },
    data=json.dumps({
        "model":           MODEL,
        "input":           ["Hello, this is a test embedding from VYOR AI."],
        "encoding_format": "float",
    }),
    timeout=30,
)

print(f"HTTP status : {resp.status_code}")
body = resp.json()

if resp.status_code == 200:
    emb = body["data"][0]["embedding"]
    print(f"Embedding dimension : {len(emb)}")
    print(f"First 5 values      : {emb[:5]}")
    print(f"Model used          : {body.get('model', MODEL)}")
    print(f"\nUsage: {body.get('usage', {})}")
else:
    print("Error response:")
    print(json.dumps(body, indent=2))
