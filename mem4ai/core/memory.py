import uuid
from typing import Any, Dict, List

class Memory:
    def __init__(self, content, metadata=None, embedding=None, user_id=None, session_id=None, agent_id=None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.embedding = embedding
        self.metadata = metadata or {}
        self.update_history = []
        
        if user_id:
            self.metadata['user_id'] = user_id
        if session_id:
            self.metadata['session_id'] = session_id
        if agent_id:
            self.metadata['agent_id'] = agent_id

    def update(self, new_content: str, new_metadata: Dict[str, Any] = None):
        self.update_history.append({
            "content": self.content,
            "metadata": self.metadata.copy()
        })
        self.content = new_content
        if new_metadata:
            self.metadata.update(new_metadata)

    def __repr__(self):
        return f"Memory(id={self.id}, content={self.content[:50]}...)"