from typing import List, Optional, Dict, Any, Tuple
from .core.embedding_manager import EmbeddingManager
from .strategies.embedding_strategy import get_embedding_strategy, EmbeddingStrategy
from .strategies.storage_strategy import get_storage_strategy, StorageStrategy
from .strategies.search_strategy import get_search_strategy, SearchStrategy
from .core.memory import Memory
from .utils.config_manager import config_manager

class Memtor:
    def __init__(self, embedding_strategy=None, storage_strategy=None, search_strategy=None):
        """
        Initialize the Memtor instance with the specified strategies.
        If strategies are not provided, default ones will be used based on the configuration.
        """
        self.embedding_manager = EmbeddingManager(embedding_strategy)
        self.storage_strategy = storage_strategy or get_storage_strategy()
        self.search_strategy = search_strategy or get_search_strategy(self.embedding_manager)


    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None,
                   user_id: Optional[str] = None, session_id: Optional[str] = None,
                   agent_id: Optional[str] = None) -> str:
        """
        Add a new memory to the storage.

        :param content: The content of the memory.
        :param metadata: Additional metadata for the memory.
        :param user_id: Optional user ID.
        :param session_id: Optional session ID.
        :param agent_id: Optional agent ID.
        :return: The ID of the newly created memory.
        """
        if not isinstance(content, str):
            raise TypeError("Content must be a string")

        metadata = metadata or {}
        if user_id:
            metadata['user_id'] = user_id
        if session_id:
            metadata['session_id'] = session_id
        if agent_id:
            metadata['agent_id'] = agent_id

        embedding = self.embedding_manager.embed(content)
        memory = Memory(content, metadata, embedding, user_id, session_id, agent_id)
        self.storage_strategy.save(memory)
        return memory.id

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by its ID.

        :param memory_id: The ID of the memory to retrieve.
        :return: The Memory object if found, None otherwise.
        """
        if not isinstance(memory_id, str):
            raise TypeError("Memory ID must be a string")

        return self.storage_strategy.load(memory_id)

    def update_memory(self, memory_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing memory.

        :param memory_id: The ID of the memory to update.
        :param content: The new content for the memory.
        :param metadata: New metadata to merge with existing metadata.
        :return: True if the update was successful, False otherwise.
        """
        if not isinstance(memory_id, str) or not isinstance(content, str):
            raise TypeError("Memory ID and content must be strings")

        memory = self.get_memory(memory_id)
        if memory:
            memory.update(content, metadata)
            memory.embedding = self.embedding_manager.embed(content)
            return self.storage_strategy.update(memory_id, memory)
        return False

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by its ID.

        :param memory_id: The ID of the memory to delete.
        :return: True if the deletion was successful, False otherwise.
        """
        if not isinstance(memory_id, str):
            raise TypeError("Memory ID must be a string")

        return self.storage_strategy.delete(memory_id)

    def delete_memories_by_user(self, user_id: str) -> int:
        """
        Delete all memories associated with a specific user.

        :param user_id: The ID of the user whose memories should be deleted.
        :return: The number of memories deleted.
        """
        memories = self.list_memories(user_id=user_id)
        deleted_count = 0
        for memory in memories:
            if self.delete_memory(memory.id):
                deleted_count += 1
        return deleted_count

    def delete_memories_by_session(self, session_id: str) -> int:
        """
        Delete all memories associated with a specific session.

        :param session_id: The ID of the session whose memories should be deleted.
        :return: The number of memories deleted.
        """
        memories = self.list_memories(session_id=session_id)
        deleted_count = 0
        for memory in memories:
            if self.delete_memory(memory.id):
                deleted_count += 1
        return deleted_count
    
    def clear_all_storage(self) -> bool:
        """
        Clear all memories from storage.

        :return: True if the operation was successful, False otherwise.
        """
        try:
            self.storage_strategy.clear_all()
            return True
        except Exception as e:
            print(f"Error clearing storage: {str(e)}")
            return False

    def list_memories(self, user_id: Optional[str] = None, session_id: Optional[str] = None,
                      agent_id: Optional[str] = None, metadata_filters: Optional[List[Tuple[str, str, Any]]] = None) -> List[Memory]:
        """
        List memories based on optional filters.

        :param user_id: Optional user ID to filter memories.
        :param session_id: Optional session ID to filter memories.
        :param agent_id: Optional agent ID to filter memories.
        :param metadata_filters: Optional list of metadata filters.
        :return: List of Memory objects that match the filters.
        """
        all_memories = self.storage_strategy.list_all()
        filtered_memories = [
            mem for mem in all_memories
            if (not user_id or mem.metadata.get('user_id') == user_id) and
               (not session_id or mem.metadata.get('session_id') == session_id) and
               (not agent_id or mem.metadata.get('agent_id') == agent_id)
        ]

        if metadata_filters:
            filtered_memories = self.storage_strategy.apply_filters(filtered_memories, metadata_filters)

        return filtered_memories

    def search_memories(self, query: str, top_k: int = 10, user_id: Optional[str] = None,
                        session_id: Optional[str] = None, agent_id: Optional[str] = None,
                        keywords: Optional[List[str]] = None, metadata_filters: Optional[List[Tuple[str, str, Any]]] = None) -> List[Memory]:
        """
        Search memories based on a query and optional filters.

        :param query: The search query.
        :param top_k: The number of top results to return.
        :param user_id: Optional user ID to filter memories.
        :param session_id: Optional session ID to filter memories.
        :param agent_id: Optional agent ID to filter memories.
        :param keywords: Optional list of keywords to boost in the search.
        :param metadata_filters: Optional list of metadata filters.
        :return: List of Memory objects that match the search criteria.
        """
        if not isinstance(query, str):
            raise TypeError("Query must be a string")

        all_memories = self.list_memories(user_id, session_id, agent_id, metadata_filters)
        return self.search_strategy.search(query, all_memories, top_k, keywords or [], metadata_filters or [])

    # def configure(self, **kwargs):
    #     """
    #     Update the configuration of the Memtor instance.

    #     :param kwargs: Configuration key-value pairs to update.
    #     """
    #     for key, value in kwargs.items():
    #         config_manager.config[key].update(value)
    #     config_manager.save()
    #     self.embedding_manager.dimension = config_manager.get('embedding.dimension', 512)
        # Add other updates here as needed

    def __repr__(self):
        return f"Memtor(embedding_strategy={self.embedding_manager.embedding_strategy.__class__.__name__}, " \
               f"storage_strategy={self.storage_strategy.__class__.__name__}, " \
               f"search_strategy={self.search_strategy.__class__.__name__})"

if __name__ == "__main__":
    # Usage example
    memtor = Memtor()

    # Add some memories
    memory1_id = memtor.add_memory("The quick brown fox jumps over the lazy dog", {"tag": "animals"})
    memory2_id = memtor.add_memory("To be or not to be, that is the question", {"tag": "literature"})

    # Search memories
    results = memtor.search_memories("animal fox", top_k=1)
    if results:
        print(f"Top search result: {results[0].content}")

    # Update a memory
    memtor.update_memory(memory1_id, "The quick brown fox leaps over the lazy dog", {"tag": "animals", "updated": True})

    # List memories with a filter
    filtered_memories = memtor.list_memories(metadata_filters=[("tag", "==", "animals")])
    print(f"Number of animal-related memories: {len(filtered_memories)}")

    # Delete a memory
    memtor.delete_memory(memory2_id)

    print("Memtor usage example completed successfully!")