import os
from dotenv import load_dotenv

load_dotenv()

# Read from environment first, then fallback to Streamlit secrets if available
try:
    import streamlit as st  # type: ignore
    _SECRETS = getattr(st, "secrets", {})
except Exception:
    _SECRETS = {}


def _get(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is not None and val != "":
        return val
    if isinstance(_SECRETS, dict):
        sec_val = _SECRETS.get(name)
        if sec_val is not None and str(sec_val) != "":
            return str(sec_val)
    return default


def _get_any(names: list[str], default: str | None = None) -> str | None:
    for n in names:
        v = _get(n)
        if v:
            return v
    return default

OPENAI_API_KEY = _get("OPENAI_API_KEY")
GROQ_API_KEY = _get("GROQ_API_KEY")
GEMINI_API_KEY = _get("GEMINI_API_KEY")
# Prefer HF_API_KEY, keep backward-compatible fallbacks
HF_API_KEY = _get_any(["HF_API_KEY", "HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"]) 
MONGO_URI = _get_any(["MONGO_URI", "MONGODB_URI"]) 

# Hugging Face model selection and URL (default to Mistral as requested)
HF_MODEL = _get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2") or "mistralai/Mistral-7B-Instruct-v0.2"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
# Backward compatibility
DEFAULT_HF_MODEL = HF_MODEL
EMBED_MODEL = _get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2") or "sentence-transformers/all-MiniLM-L6-v2"

SURVEY_DOMAINS = ["agriculture", "education", "healthcare"]
INDIAN_REGIONS = {
    "north": ["Delhi", "Haryana", "Punjab", "Uttar Pradesh", "Uttarakhand", "Himachal Pradesh", "Jammu & Kashmir"],
    "south": ["Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana"],
    "east": ["Bihar", "Jharkhand", "Odisha", "West Bengal"],
    "west": ["Gujarat", "Maharashtra", "Rajasthan", "Madhya Pradesh"],
    "northeast": ["Assam", "Arunachal Pradesh", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Tripura", "Sikkim"],
    "central": ["Chhattisgarh", "Madhya Pradesh"],
}

MODEL_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}
