from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
from ..config import MODEL_CONFIG, HF_MODEL
from ..llm_providers import call_huggingface_inference

class BaseAgent(ABC):
    name: str = "BaseAgent"
    domain: str = "generic"

    @abstractmethod
    def build_prompt(self, question: str, region: str, context: Optional[Dict[str, Any]]) -> str:
        ...

    def run(self, prompt: str) -> str:
        # Per-domain override via AGRICULTURE_HF_MODEL / EDUCATION_HF_MODEL / HEALTHCARE_HF_MODEL
        dom_key = f"{self.domain.upper()}_HF_MODEL"
        model = os.getenv(dom_key) or HF_MODEL
        return call_huggingface_inference(prompt, model=model)

    def process(self, question: str, region: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prompt = self.build_prompt(question, region, context)
        raw = self.run(prompt)
        return {
            "agent": self.name,
            "domain": self.domain,
            "region": region,
            "question": question,
            "raw_response": raw,
            "status": "success",
        }