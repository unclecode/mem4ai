from abc import ABC, abstractmethod
from typing import Dict, Any

class KnowledgeExtractionStrategy(ABC):
    @abstractmethod
    def extract_knowledge(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        """Extract knowledge from a conversation exchange."""
        pass