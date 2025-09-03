import json
from typing import Dict, Any, Optional
from .base import BaseAgent

class EducationAgent(BaseAgent):
    name = "EducationAgent"
    domain = "education"

    def build_prompt(self, question: str, region: str, context: Optional[Dict[str, Any]]) -> str:
        ctx = f"\nContext: {json.dumps(context, ensure_ascii=False)}" if context else ""
        return f"""You are an education survey agent for India, region {region}.
Question: {question}{ctx}

Respond with JSON:
{{
 "student_response": "...",
 "confidence": 0.85,
 "key_insights": ["..."],
 "recommendations": ["..."],
 "region_specific_factors": ["..."],
 "follow_up_questions": ["..."],
 "education_level": "primary/secondary/higher",
 "infrastructure_needs": ["..."]
}}
Only output JSON:"""