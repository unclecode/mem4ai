from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from litellm import completion
import os, json
from .base import KnowledgeExtractionStrategy
from ...utils.config_manager import config_manager

class EchoKnowledgeStrategy(KnowledgeExtractionStrategy):
    def __init__(self):
        pass

    def extract_knowledge(
        self, user_message: str, assistant_response: str
    ) -> Dict[str, Any]:
        return {
            # "user_message": user_message,
            # "assistant_response": assistant_response,
            "timestamp": datetime.now().isoformat(),
        }