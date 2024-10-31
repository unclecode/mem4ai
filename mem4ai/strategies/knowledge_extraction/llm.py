from typing import Dict, Any, List, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from litellm import completion
import os
from .base import KnowledgeExtractionStrategy
from ...utils.config_manager import config_manager
import json

# Pydantic models for structured knowledge extraction
class ActionDetails(BaseModel):    
    type: Literal['list_modification', 'search', 'creation', 'deletion'] = Field(
        ...,  # Make it required
        description="The type of action performed"
    )
    target: Literal['favorites_list', 'watch_later', 'custom_list'] = Field(
        ...,
        description="The target of the action"
    )
    status: Literal['completed', 'failed', 'partial'] = Field(
        ...,
        description="The status of the action"
    )
    
    class Config:
        extra = 'forbid'

class ModifiedElements(BaseModel):
    lists: List[str] = Field(
        ...,  # Changed from default_factory=list to required
        description="Names of modified lists"
    )
    movies: List[str] = Field(
        ...,  # Changed from default_factory=list to required
        description="Titles of modified movies"
    )
    preferences: List[str] = Field(
        ...,  # Changed from default_factory=list to required
        description="Modified preferences"
    )
    class Config:
        extra = 'forbid'

class ActionInfo(BaseModel):
    primary_action: ActionDetails = Field(
        ...,
        description="Primary action details"
    )
    modified_elements: ModifiedElements = Field(
        ...,
        description="Elements modified by the action"
    )
    class Config:
        extra = 'forbid'
        
class ReferencedValues(BaseModel):
    movies: List[str] = Field(
        ...,  # Changed from default_factory=list to required
        description="Movie titles referenced"
    )
    ratings: List[str] = Field(
        ...,  # Changed from default_factory=list to required
        description="Ratings referenced"
    )
    dates: List[str] = Field(
        ...,  # Changed from default_factory=list to required
        description="Dates referenced"
    )
    other_values: List[str] = Field(
        ...,  # Changed from default_factory=list to required
        description="Other referenced values"
    )
    class Config:
        extra = 'forbid'
        
class KeyInformation(BaseModel):
    explicit_mentions: List[str] = Field(
        ...,
        description="Explicitly mentioned terms"
    )
    implicit_context: List[str] = Field(
        ...,
        description="Implicitly derived context"
    )
    referenced_values: ReferencedValues = Field(
        ...,
        description="Values referenced in the conversation"
    )
    class Config:
        extra = 'forbid'

class ConversationDetails(BaseModel):
    intent: Literal['movie_organization', 'information_query', 'general_discussion'] = Field(
        ...,
        description="The intent of the conversation"
    )
    topic: str = Field(
        ...,
        description="The main topic of conversation"
    )
    key_information: KeyInformation = Field(
        ...,
        description="Key information extracted from the conversation"
    )
    user_message: str = Field(
        ...,
        description="Original user message"
    )
    assistant_response: str = Field(
        ...,
        description="Original assistant response"
    )
    class Config:
        extra = 'forbid'

class Summary(BaseModel):
    request_essence: str = Field(
        ...,
        description="Essence of the user's request"
    )
    response_essence: str = Field(
        ...,
        description="Essence of the assistant's response"
    )
    key_points: List[str] = Field(
        ...,
        description="Key points from the conversation"
    )
    class Config:
        extra = 'forbid'

class MemoryContext(BaseModel):
    timestamp: str = Field(  # Changed from datetime to str for easier serialization
        ...,
        description="Timestamp of the interaction"
    )
    interaction_type: Literal['action_based', 'conversational', 'information_seeking'] = Field(
        ...,
        description="Type of interaction"
    )
    action_details: Optional[ActionInfo] = Field(
        None,
        description="Details of any actions performed"
    )
    conversation_details: ConversationDetails = Field(
        ...,
        description="Details of the conversation"
    )
    summary: Summary = Field(
        ...,
        description="Summary of the interaction"
    )
    
    class Config:
        extra = 'forbid'

class LLMExtractionStrategy(KnowledgeExtractionStrategy):
    def __init__(self):
        self.model = config_manager.get('extraction.model', 'gpt-4o-mini')
        self.api_key = config_manager.get('extraction.api_key', os.getenv('OPENAI_API_KEY'))
        
        if not self.api_key:
            raise ValueError("No API key found for LLM extraction")
            
        os.environ['OPENAI_API_KEY'] = self.api_key

    def _create_system_prompt(self) -> str:
        return """You are a knowledge extraction system for a movie application assistant. Your task is to analyze conversations between users and the assistant, extracting structured information about both action-based interactions (like creating lists or modifying preferences) and general conversations about movies.

    Given a conversation, you must:
    1. Identify if this is an action-based interaction (e.g., creating lists, adding movies) or a conversational interaction (e.g., discussing movies, asking opinions)
    2. Extract all relevant information including explicit mentions and implicit context
    3. Preserve important referenced values (movies, ratings, dates) for future use
    4. Create a concise summary capturing the essence of the interaction

    For action-based interactions, focus on:
    - Specific actions performed (creation, modification, deletion)
    - Changes made to lists, movies, or preferences
    - Status of the actions performed

    For conversational interactions, focus on:
    - Main topics and themes discussed
    - Movie-related information shared
    - User preferences and opinions expressed

    Example 1 (Action-based):
    User message: "Create a new list called 'Horror Nights' and add some recent psychological horror movies with good ratings."
    Assistant response: {"action":"create_list", "list_id":"hn_123", "movies":[{"title":"Talk to Me","year":2023,"rating":7.1},{"title":"Hereditary","year":2018,"rating":7.3}]} I've created your 'Horror Nights' list and added some highly-rated psychological horror films. Would you like to see more options?

    {
        "timestamp": "2024-10-31T14:30:00Z",
        "interaction_type": "action_based",
        "action_details": {
            "primary_action": {
                "type": "creation",
                "target": "custom_list",
                "status": "completed"
            },
            "modified_elements": {
                "lists": ["Horror Nights"],
                "movies": ["Talk to Me", "Hereditary"],
                "preferences": ["psychological horror"]
            }
        },
        "conversation_details": {
            "intent": "movie_organization",
            "topic": "horror movies",
            "key_information": {
                "explicit_mentions": ["horror", "psychological", "recent"],
                "implicit_context": ["user prefers quality over quantity"],
                "referenced_values": {
                    "movies": ["Talk to Me", "Hereditary"],
                    "ratings": ["7.1", "7.3"],
                    "dates": ["2023", "2018"],
                    "other_values": []
                }
            },
            "user_message": "Create a new list called 'Horror Nights' and add some recent psychological horror movies with good ratings.",
            "assistant_response": "I've created your 'Horror Nights' list and added some highly-rated psychological horror films. Would you like to see more options?"
        },
        "summary": {
            "request_essence": "Create new horror movie list with recent, well-rated films",
            "response_essence": "Created list with two psychological horror movies",
            "key_points": ["List creation", "Focus on psychological horror", "Recent releases", "High ratings"]
        }
    }

    Example 2 (Conversational):
    User message: "What did you think about the ending of Hereditary? Was it too disturbing?"
    Assistant response: "The ending of Hereditary is particularly impactful because it completes the family's tragic arc. The final treehouse scene reveals the cult's true purpose. While disturbing, it serves the story's themes of family trauma and fate perfectly."

    {
        "timestamp": "2024-10-31T14:35:00Z",
        "interaction_type": "conversational",
        "action_details": null,
        "conversation_details": {
            "intent": "information_query",
            "topic": "movie analysis",
            "key_information": {
                "explicit_mentions": ["Hereditary", "ending", "disturbing"],
                "implicit_context": ["user seeking opinion", "concerned about content intensity"],
                "referenced_values": {
                    "movies": ["Hereditary"],
                    "ratings": [],
                    "dates": [],
                    "other_values": ["treehouse scene", "family trauma", "cult"]
                }
            },
            "user_message": "What did you think about the ending of Hereditary? Was it too disturbing?",
            "assistant_response": "The ending of Hereditary is particularly impactful because it completes the family's tragic arc. The final treehouse scene reveals the cult's true purpose. While disturbing, it serves the story's themes of family trauma and fate perfectly."
        },
        "summary": {
            "request_essence": "Opinion request about Hereditary's ending",
            "response_essence": "Analysis of ending's significance and themes",
            "key_points": ["Movie analysis", "Thematic discussion", "Content intensity"]
        }
    }

    Required Response Format:
    Your response must strictly follow this schema:
    {
        "timestamp": string (ISO format),
        "interaction_type": "action_based" | "conversational" | "information_seeking",
        "action_details": {  // Optional, null for non-action interactions
            "primary_action": {
                "type": "list_modification" | "search" | "creation" | "deletion",
                "target": "favorites_list" | "watch_later" | "custom_list",
                "status": "completed" | "failed" | "partial"
            },
            "modified_elements": {
                "lists": [string],  // Required, can be empty
                "movies": [string],  // Required, can be empty
                "preferences": [string]  // Required, can be empty
            }
        },
        "conversation_details": {
            "intent": "movie_organization" | "information_query" | "general_discussion",
            "topic": string,
            "key_information": {
                "explicit_mentions": [string],  // Required
                "implicit_context": [string],  // Required
                "referenced_values": {
                    "movies": [string],  // Required
                    "ratings": [string],  // Required
                    "dates": [string],  // Required
                    "other_values": [string]  // Required
                }
            },
            "user_message": string,  // Original message
            "assistant_response": string  // Original response
        },
        "summary": {
            "request_essence": string,
            "response_essence": string,
            "key_points": [string]  // Required
        }
    }"""


    def extract_knowledge(self, user_message: str, assistant_response: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
            {"role": "user", "content": f"User message: {user_message}\nAssistant response: {assistant_response}"}
        ]
      
        try:
            response = completion(
                model=self.model,
                messages=messages,
                response_format=MemoryContext
            )
            
            return json.loads(response.model_dump()['choices'][0]['message']['content'])
        except Exception as e:
            print(f"Error in knowledge extraction: {str(e)}")
            # Return a basic context on error
            return MemoryContext(
                timestamp=datetime.now(),
                interaction_type="conversational",
                conversation_details=ConversationDetails(
                    intent="general_discussion",
                    topic="general",
                    original_exchange={
                        "user_message": user_message,
                        "assistant_response": assistant_response
                    }
                ),
                summary=Summary(
                    request_essence="Error in extraction",
                    response_essence="Error in extraction",
                    key_points=["Error in knowledge extraction"]
                )
            ).model_dump()