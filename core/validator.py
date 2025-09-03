import json
from typing import Dict, Any, Optional
from .llm_providers import call_openai, call_groq, call_gemini

# Try multiple providers to validate/normalize JSON

def _prompt(raw_text: str, expected_schema: Optional[Dict[str, Any]] = None) -> str:
    return (
        "Extract valid JSON from text, fix errors. Return only JSON.\n"
        + (f"Expected schema: {json.dumps(expected_schema)}\n" if expected_schema else "")
        + f"Text: {raw_text}"
    )

def validate_json(raw_text: str, expected_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prompt = _prompt(raw_text, expected_schema)
    # 1. OpenAI JSON mode
    try:
        text = call_openai(prompt, model="gpt-4o-mini", json_object=True)
        return json.loads(text)
    except Exception:
        pass
    # 2. Groq (Llama3)
    try:
        text = call_groq(prompt, model="llama3-8b-8192")
        return json.loads(text)
    except Exception:
        pass
    # 3. Gemini (plain text, attempt JSON parse)
    try:
        text = call_gemini(prompt, model="gemini-1.5-pro")
        return json.loads(text)
    except Exception:
        pass
    # Fallback minimal structure
    return {"response": raw_text.strip()[:400], "confidence": 0.3}