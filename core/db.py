from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from .config import MONGO_URI

logger = logging.getLogger(__name__)

class DB:
    def __init__(self):
        self.client = None
        self.db = None
        self.mem = {"responses": [], "logs": []}
        if MONGO_URI:
            try:
                self.client = MongoClient(MONGO_URI)
                self.client.admin.command("ping")
                self.db = self.client["survey_ai"]
                logger.info("✅ Mongo connected")
            except Exception as e:
                logger.warning(f"⚠️ Mongo connection failed: {e}. Using memory store.")
                self.client = None
                self.db = None

    def save_response(self, session_id: str, payload: Dict[str, Any]) -> str:
        doc = {
            **payload,
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
        }
        if self.db:
            return str(self.db["responses"].insert_one(doc).inserted_id)
        self.mem["responses"].append(doc)
        return "mem_response"

    def list_responses(self, limit: int = 100) -> List[Dict[str, Any]]:
        if self.db:
            cur = self.db["responses"].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
            return list(cur)
        return self.mem["responses"][-limit:]

    def log(self, agent: str, input_data: str, output_data: str, duration: float, session_id: Optional[str] = None):
        doc = {"agent": agent, "input": input_data, "output": output_data, "duration": duration, "session_id": session_id, "timestamp": datetime.utcnow()}
        if self.db:
            self.db["logs"].insert_one(doc)
        else:
            self.mem["logs"].append(doc)


db = DB()