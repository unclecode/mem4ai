import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

class Memory:
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None, 
                 embedding: Optional[Any] = None, context: Optional[Dict[str, Any]] = None,
                 user_id: Optional[str] = None, session_id: Optional[str] = None, 
                 agent_id: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.embedding = embedding
        self.context = context or {}  # New field for extracted knowledge
        self.metadata = metadata or {}
        self.update_history = []
        self.timestamp = datetime.now()
        
        if user_id:
            self.metadata['user_id'] = user_id
        if session_id:
            self.metadata['session_id'] = session_id
        if agent_id:
            self.metadata['agent_id'] = agent_id

    def update(self, new_content: str, new_metadata: Dict[str, Any] = None, 
               new_context: Dict[str, Any] = None):
        self.update_history.append({
            "content": self.content,
            "metadata": self.metadata.copy(),
            "context": self.context.copy()  # Also store context history
        })
        self.content = new_content
        if new_metadata:
            self.metadata.update(new_metadata)
        if new_context:
            self.context.update(new_context)

    def __repr__(self):
        return f"Memory(id={self.id}, content={self.content[:50]}...)"