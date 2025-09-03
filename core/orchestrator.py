from typing import Dict, Any, Optional
import time
import json
from .agents.agriculture import AgricultureAgent
from .agents.education import EducationAgent
from .agents.healthcare import HealthAgent
from .validator import validate_json
from .db import db
from .llm_providers import call_openai

AGENTS = {
    "agriculture": AgricultureAgent(),
    "education": EducationAgent(),
    "healthcare": HealthAgent(),
}

_session_ctx: Dict[str, Dict[str, Any]] = {}


def process_question(domain: str, question: str, region: str, session_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    t0 = time.time()
    sid = session_id or str(int(time.time() * 1000))
    ctx = _session_ctx.get(sid, {})
    if context:
        ctx.update(context)
    agent = AGENTS.get(domain)
    if not agent:
        return {"status": "error", "error": f"Unknown domain {domain}"}

    raw = agent.process(question, region, ctx)
    cleaned = validate_json(raw.get("raw_response", ""))

    ctx.setdefault("history", []).append({"q": question, "a": cleaned})
    ctx["domain"] = domain
    ctx["region"] = region
    _session_ctx[sid] = ctx

    result = {
        "session_id": sid,
        "domain": domain,
        "region": region,
        "question": question,
        "agent_response": cleaned,
        "processing_time": time.time() - t0,
        "status": "success",
    }
    db.save_response(sid, result)
    return result


def get_history(session_id: str):
    return _session_ctx.get(session_id, {}).get("history", [])


def finalize_session(session_id: str) -> Dict[str, Any]:
    ctx = _session_ctx.get(session_id)
    if not ctx:
        return {"status": "error", "error": "No session"}
    domain = ctx.get("domain", "unknown")
    region = ctx.get("region", "unknown")
    history = ctx.get("history", [])
    # Build a concise final JSON by summarizing history
    joined = "\n".join([f"Q: {h['q']}\nA: {json.dumps(h['a'], ensure_ascii=False)}" for h in history])
    try:
        prompt = f"Summarize the survey conversation into a final JSON object capturing key_insights, recommendations, follow_up_questions.\nDomain: {domain}\nRegion: {region}\nConversation:\n{joined}\nReturn JSON only with keys: final_summary, key_insights, recommendations, follow_up_questions."
        text = call_openai(prompt, model="gpt-4o-mini", json_object=True)
        return json.loads(text)
    except Exception:
        # Fallback minimal
        return {
            "final_summary": f"{len(history)} turns in {domain}/{region}",
            "key_insights": [],
            "recommendations": [],
            "follow_up_questions": [],
        }