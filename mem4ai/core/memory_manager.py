from typing import List, Dict, Any, Optional
from .memory import Memory
from ..strategies.embedding_strategy import *
from ..strategies.storage_strategy import *
from ..strategies.search_strategy import *
from ..utils.config_manager import config_manager

class MemoryManager:
    def __init__(self, embedding_strategy: EmbeddingStrategy, 
                 storage_strategy: StorageStrategy,
                 search_strategy: SearchStrategy):
        self.embedding_strategy = embedding_strategy or self._get_default_embedding_strategy()
        self.storage_strategy = storage_strategy or self._get_default_storage_strategy()
        self.search_strategy = search_strategy or self._get_default_search_strategy()
        self.max_history = config_manager.get('memory.max_history', 5)

    def _get_default_embedding_strategy(self):
        return get_embedding_strategy()

    def _get_default_storage_strategy(self):
        return get_storage_strategy()

    def _get_default_search_strategy(self):
        return get_search_strategy()
        
    def add_memory(self, content: str, user_id: Optional[str] = None, 
                   session_id: Optional[str] = None, agent_id: Optional[str] = None, 
                   metadata: Dict[str, Any] = None) -> str:
        metadata = metadata or {}
        if user_id:
            metadata['user_id'] = user_id
        if session_id:
            metadata['session_id'] = session_id
        if agent_id:
            metadata['agent_id'] = agent_id

        memory = Memory(content, metadata)
        memory.embedding = self.embedding_strategy.embed(content)
        self.storage_strategy.save(memory)
        return memory.id

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        return self.storage_strategy.load(memory_id)

    def update_memory(self, memory_id: str, new_content: str, metadata: Dict[str, Any] = None) -> bool:
        memory = self.get_memory(memory_id)
        if memory:
            memory.update(new_content, metadata)
            memory.embedding = self.embedding_strategy.embed(new_content)
            self.storage_strategy.update(memory_id, memory)
            return True
        return False

    def delete_memory(self, memory_id: str) -> bool:
        return self.storage_strategy.delete(memory_id)

    def list_memories(self, user_id: Optional[str] = None, session_id: Optional[str] = None, 
                      agent_id: Optional[str] = None, metadata_filters: List[tuple] = None) -> List[Memory]:
        all_memories = self.storage_strategy.list_all()
        filtered_memories = self.storage_strategy.apply_filters(all_memories, metadata_filters or [])

        return [mem for mem in filtered_memories 
                if (not user_id or mem.metadata.get('user_id') == user_id) and
                   (not session_id or mem.metadata.get('session_id') == session_id) and
                   (not agent_id or mem.metadata.get('agent_id') == agent_id)]

    def search_memories(self, query: str, top_k: int = 10, user_id: Optional[str] = None, 
                    session_id: Optional[str] = None, agent_id: Optional[str] = None, 
                    keywords: List[str] = None, metadata_filters: List[Tuple[str, str, Any]] = None) -> List[Memory]:
        top_k = top_k or config_manager.get('search.top_k', 10)
        all_memories = self.list_memories(user_id, session_id, agent_id, metadata_filters)
        return self.search_strategy.search(query, all_memories, top_k, keywords or [], metadata_filters or [])