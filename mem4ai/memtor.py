from typing import List, Optional, Dict, Any, Tuple
from .core.embedding_manager import EmbeddingManager
from .strategies.embedding_strategy import get_embedding_strategy, EmbeddingStrategy
from .strategies.storage_strategy import get_storage_strategy, StorageStrategy
from .strategies.search_strategy import get_search_strategy, SearchStrategy
from .strategies.knowledge_extraction import get_extraction_strategy
from .core.memory import Memory
from .utils.config_manager import config_manager
from datetime import datetime

class Memtor:
    def __init__(self, embedding_strategy=None, storage_strategy=None, 
             search_strategy=None, extraction_strategy=None):
        """
        Initialize the Memtor instance with the specified strategies.
        
        :param embedding_strategy: Strategy for creating embeddings
        :param storage_strategy: Strategy for storing memories
        :param search_strategy: Strategy for searching memories
        :param extraction_strategy: Strategy for knowledge extraction. 
                                Can be None to disable extraction
        """
        self.embedding_manager = EmbeddingManager(embedding_strategy)
        self.storage_strategy = storage_strategy or get_storage_strategy()
        self.search_strategy = search_strategy or get_search_strategy(self.embedding_manager)
        
        # Handle extraction strategy like other strategies
        if extraction_strategy is None:
            # Check config to see if extraction should be enabled
            if config_manager.get('extraction.enabled', True):
                self.extraction_strategy = get_extraction_strategy()
            else:
                self.extraction_strategy = None
        else:
            # If strategy is provided directly, use it
            self.extraction_strategy = extraction_strategy

    def add_memory(self, user_message: str, assistant_response: str, 
                   metadata: Optional[Dict[str, Any]] = None,
                   user_id: Optional[str] = None, session_id: Optional[str] = None,
                   agent_id: Optional[str] = None) -> str:
        """
        Add a new memory with both user message and assistant response.
        """
        if not isinstance(user_message, str) or not isinstance(assistant_response, str):
            raise TypeError("Messages must be strings")

        # Combine messages for embedding
        combined_content = f"User: {user_message}\nAssistant: {assistant_response}"
        
        # Extract knowledge context if strategy exists
        context = None
        if self.extraction_strategy is not None:
            try:
                context = self.extraction_strategy.extract_knowledge(user_message, assistant_response)
            except Exception as e:
                print(f"Warning: Knowledge extraction failed: {str(e)}")
                context = None
        
        # Create embedding
        embedding = self.embedding_manager.embed(combined_content)
        
        # Create memory
        metadata = metadata or {}
        memory = Memory(
            content=combined_content,
            metadata=metadata,
            embedding=embedding,
            context=context,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id
        )
        
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

    def search_memories(self, query: Optional[str] = None, top_k: int = 10, 
                    start_time: Optional[datetime] = None, end_time: Optional[datetime] = None,
                    user_id: Optional[str] = None, session_id: Optional[str] = None,
                    agent_id: Optional[str] = None, keywords: Optional[List[str]] = None,
                    metadata_filters: Optional[List[Tuple[str, str, Any]]] = None,
                    sort_by: str = 'relevance') -> List[Memory]:
        """
        Enhanced search function that supports multiple search modes and combinations.
        
        :param query: Optional search query for semantic search
        :param top_k: Maximum number of results to return
        :param start_time: Optional start time for time-range filtering
        :param end_time: Optional end time for time-range filtering
        :param user_id: Optional user ID filter
        :param session_id: Optional session ID filter
        :param agent_id: Optional agent ID filter
        :param keywords: Optional list of keywords to boost in search
        :param metadata_filters: Optional list of metadata filters as (key, op, value) tuples
        :param sort_by: How to sort results ('relevance', 'time_desc', 'time_asc')
        :return: List of Memory objects matching the search criteria
        """
        # Input validation
        if sort_by not in ('relevance', 'time_desc', 'time_asc'):
            raise ValueError("sort_by must be one of: 'relevance', 'time_desc', 'time_asc'")

        # Build metadata filters dictionary
        meta_dict = {}
        if user_id:
            meta_dict['user_id'] = user_id
        if session_id:
            meta_dict['session_id'] = session_id
        if agent_id:
            meta_dict['agent_id'] = agent_id

        # Determine search mode and get initial results
        if query:
            # Semantic search mode
            if start_time or end_time:
                # Get time-filtered memories first
                memories = self.storage_strategy.find_by_time(
                    start_time or datetime.min,
                    end_time or datetime.max,
                    **meta_dict
                )
                # Then apply semantic search
                results = self.search_strategy.search(
                    query=query,
                    memories=memories,
                    top_k=top_k,
                    keywords=keywords or [],
                    metadata_filters=metadata_filters or []
                )
            else:
                # Get filtered memories and apply semantic search
                memories = self.storage_strategy.find_by_meta(meta_dict) if meta_dict else self.storage_strategy.list_all()
                if metadata_filters:
                    memories = self.storage_strategy.apply_filters(memories, metadata_filters)
                results = self.search_strategy.search(
                    query=query,
                    memories=memories,
                    top_k=top_k,
                    keywords=keywords or [],
                    metadata_filters=[]  # Already applied
                )
        else:
            # Non-semantic search mode
            if start_time or end_time:
                # Time-based search
                results = self.storage_strategy.find_by_time(
                    start_time or datetime.min,
                    end_time or datetime.max,
                    **meta_dict
                )
            elif meta_dict or metadata_filters:
                # Metadata-based search
                results = self.storage_strategy.find_by_meta(meta_dict)
                if metadata_filters:
                    results = self.storage_strategy.apply_filters(results, metadata_filters)
            else:
                # Recent memories
                results = self.storage_strategy.find_recent(top_k, **meta_dict)

        # Sort results if needed
        if sort_by != 'relevance' or not query:
            if sort_by == 'time_desc' or not query:
                results = sorted(results, key=lambda x: x.timestamp, reverse=True)
            elif sort_by == 'time_asc':
                results = sorted(results, key=lambda x: x.timestamp)

        return results[:top_k]

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