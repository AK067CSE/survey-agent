from typing import Optional, Dict, Any
import os
import json
import requests

from .config import OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, HF_API_KEY, HF_URL, MODEL_CONFIG, HF_MODEL

# Lightweight provider adapters. We keep to chat/completions-like interface returning text.

class ProviderError(Exception):
    pass


def call_all_providers(prompt: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    # Hugging Face
    try:
        results["huggingface"] = {"ok": True, "text": call_huggingface_inference(prompt)}
    except Exception as e:
        results["huggingface"] = {"ok": False, "error": str(e)}
    # OpenAI
    try:
        results["openai"] = {"ok": True, "text": call_openai(prompt, model="gpt-4o-mini")}
    except Exception as e:
        results["openai"] = {"ok": False, "error": str(e)}
    # Groq
    try:
        results["groq"] = {"ok": True, "text": call_groq(prompt, model="llama3-8b-8192")}
    except Exception as e:
        results["groq"] = {"ok": False, "error": str(e)}
    # Gemini
    try:
        results["gemini"] = {"ok": True, "text": call_gemini(prompt, model="gemini-1.5-pro")}
    except Exception as e:
        results["gemini"] = {"ok": False, "error": str(e)}
    return results


def _headers_json(api_key: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def call_openai(prompt: str, model: str = "gpt-4o-mini", json_object: bool = False) -> str:
    if not OPENAI_API_KEY:
        raise ProviderError("OPENAI_API_KEY missing")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": MODEL_CONFIG["temperature"],
        "max_tokens": MODEL_CONFIG["max_new_tokens"],
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(url, headers=_headers_json(OPENAI_API_KEY), json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def call_groq(prompt: str, model: str = "llama3-8b-8192", json_object: bool = False) -> str:
    if not GROQ_API_KEY:
        raise ProviderError("GROQ_API_KEY missing")
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": MODEL_CONFIG["temperature"],
        "max_tokens": MODEL_CONFIG["max_new_tokens"],
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    r = requests.post(url, headers=_headers_json(GROQ_API_KEY), json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def call_gemini(prompt: str, model: str = "gemini-1.5-pro") -> str:
    if not GEMINI_API_KEY:
        raise ProviderError("GEMINI_API_KEY missing")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Extract text safely
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return json.dumps(data)


def call_huggingface_inference(prompt: str, model: Optional[str] = None) -> str:
    # Uses Inference API if HF key set; fallback to echo
    selected_model = model or HF_MODEL
    if not HF_API_KEY:
        # Fallback stub
        return f"[HF local stub] {prompt[:200]}"
    url = f"https://api-inference.huggingface.co/models/{selected_model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": MODEL_CONFIG["max_new_tokens"], "temperature": MODEL_CONFIG["temperature"]}}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # HF responses vary; normalize to text
    if isinstance(data, list) and len(data) and "generated_text" in data[0]:
        return data[0]["generated_text"][len(prompt):].strip()
    return json.dumps(data)
