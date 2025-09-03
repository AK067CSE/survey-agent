import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Prefer HF_API_KEY, keep backward-compatible fallbacks
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")

# Hugging Face model selection and URL (default to Mistral as requested)
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
# Backward compatibility
DEFAULT_HF_MODEL = HF_MODEL
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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