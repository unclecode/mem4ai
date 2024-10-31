from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from litellm import completion
import os, json
from .base import KnowledgeExtractionStrategy
from ...utils.config_manager import config_manager


class SimpleSummaryContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str = Field(..., description="Timestamp of the interaction")
    summary: str = Field(..., description="Concise summary of the interaction")
    keywords: List[str] = Field(
        ..., description="Key terms and concepts from the conversation"
    )
    interaction_type: Literal["task", "discussion", "query"] = Field(
        ..., description="Basic type of the interaction"
    )


class SimpleLLMExtractionStrategy(KnowledgeExtractionStrategy):
    def __init__(self):
        self.model = config_manager.get("extraction.model", "gpt-4o-mini")
        self.api_key = config_manager.get(
            "extraction.api_key", os.getenv("OPENAI_API_KEY")
        )

        if not self.api_key:
            raise ValueError("No API key found for LLM extraction")

        os.environ["OPENAI_API_KEY"] = self.api_key

    def _create_system_prompt(self) -> str:
        return """You are a knowledge extraction system focusing on creating concise summaries and extracting keywords from conversations. Your task is to analyze conversations and produce a simple, focused summary of what was discussed or accomplished.

Given a conversation between a user and an assistant, you must:
1. Create a brief, informative summary of the interaction
2. Extract relevant keywords that capture the main concepts
3. Determine the basic type of interaction (task, discussion, or query)

Example 1:
User: "Can you recommend some classic noir films from the 1940s?"
Assistant: "Here are some essential film noir classics from the 1940s: 'Double Indemnity' (1944), 'The Maltese Falcon' (1941), and 'The Big Sleep' (1946). These films defined the genre with their dark themes, cynical characters, and stylish cinematography."

{
    "timestamp": "2024-10-31T14:30:00Z",
    "summary": "User requested recommendations for 1940s film noir classics, receiving suggestions of three defining movies from the genre",
    "keywords": ["film noir", "1940s", "movie recommendations", "Double Indemnity", "Maltese Falcon", "Big Sleep", "classics"],
    "interaction_type": "query"
}

Example 2:
User: "Add 'The Godfather' and 'Goodfellas' to my 'Crime Classics' watchlist"
Assistant: "I've added both 'The Godfather' and 'Goodfellas' to your 'Crime Classics' watchlist. Would you like me to suggest similar crime films?"

{
    "timestamp": "2024-10-31T14:35:00Z",
    "summary": "User requested addition of two classic crime films to their 'Crime Classics' watchlist",
    "keywords": ["The Godfather", "Goodfellas", "watchlist", "crime movies", "list management"],
    "interaction_type": "task"
}

Required Response Format:
{
    "timestamp": string (ISO format),
    "summary": string (concise description of the interaction),
    "keywords": [string] (list of relevant terms and concepts),
    "interaction_type": "task" | "discussion" | "query"
}"""

    def extract_knowledge(
        self, user_message: str, assistant_response: str
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
            {
                "role": "user",
                "content": f"User message: {user_message}\nAssistant response: {assistant_response}",
            },
        ]

        try:
            response = completion(
                model=self.model,
                messages=messages,
                response_format=SimpleSummaryContext,
            )

            return json.loads(response.model_dump()["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"Error in simple knowledge extraction: {str(e)}")
            # Return a basic context on error
            return SimpleSummaryContext(
                timestamp=datetime.now().isoformat(),
                summary="Error occurred during extraction",
                keywords=["error"],
                interaction_type="discussion",
            ).model_dump()
