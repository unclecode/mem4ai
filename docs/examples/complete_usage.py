"""
Memtor: Complete Usage Guide

This script demonstrates all the features and functionalities of the Mem4AI library.
It covers initialization, memory operations, searching, filtering, and advanced usage.
"""

from mem4ai import Memtor, Memory
from mem4ai.strategies import CustomEmbeddingStrategy, CustomStorageStrategy, CustomSearchStrategy
import numpy as np

def main():
    # Initialize Mem4AI Memtor
    print("Initializing Memtor...")
    memtor = Memtor()
    print("Memtor initialized successfully.\n")

    # Basic Memory Operations
    print("Performing basic memory operations...")
    
    # Adding memories
    memory1_id = memtor.add_memory(
        "The quick brown fox jumps over the lazy dog",
        metadata={"tag": "animals", "complexity": "simple"},
        user_id="user1",
        session_id="session1"
    )
    print(f"Added memory 1 with ID: {memory1_id}")

    memory2_id = memtor.add_memory(
        "To be or not to be, that is the question",
        metadata={"tag": "literature", "author": "Shakespeare"},
        user_id="user1",
        session_id="session1"
    )
    print(f"Added memory 2 with ID: {memory2_id}")

    memory3_id = memtor.add_memory(
        "E = mc^2",
        metadata={"tag": "science", "field": "physics"},
        user_id="user2",
        session_id="session2"
    )
    print(f"Added memory 3 with ID: {memory3_id}")

    # Retrieving memories
    retrieved_memory = memtor.get_memory(memory1_id)
    print(f"Retrieved memory: {retrieved_memory.content}")

    # Updating memories
    updated = memtor.update_memory(
        memory1_id,
        "The quick brown fox leaps over the lazy dog",
        metadata={"tag": "animals", "complexity": "simple", "action": "leap"}
    )
    print(f"Memory updated: {updated}")

    # Deleting memories
    deleted = memtor.delete_memory(memory3_id)
    print(f"Memory deleted: {deleted}")

    print("Basic memory operations completed.\n")

    # Searching and Filtering
    print("Demonstrating search and filter capabilities...")

    # Simple search
    results = memtor.search_memories("fox", top_k=1)
    print(f"Search result for 'fox': {results[0].content}")

    # Search with user filter
    user_results = memtor.search_memories("question", top_k=1, user_id="user1")
    print(f"Search result for 'question' (user1): {user_results[0].content}")

    # Search with metadata filter
    metadata_results = memtor.search_memories(
        "animal",
        top_k=1,
        metadata_filters=[("tag", "==", "animals")]
    )
    print(f"Search result with metadata filter: {metadata_results[0].content}")

    print("Search and filter demonstrations completed.\n")

    # Advanced Usage
    print("Demonstrating advanced features...")

    # List all memories
    all_memories = memtor.list_memories()
    print(f"Total memories: {len(all_memories)}")

    # List memories by user
    user_memories = memtor.list_memories(user_id="user1")
    print(f"Memories for user1: {len(user_memories)}")

    # List memories by session
    session_memories = memtor.list_memories(session_id="session1")
    print(f"Memories for session1: {len(session_memories)}")

    # Delete memories by user
    deleted_count = memtor.delete_memories_by_user("user2")
    print(f"Deleted {deleted_count} memories for user2")

    # Delete memories by session
    deleted_count = memtor.delete_memories_by_session("session2")
    print(f"Deleted {deleted_count} memories for session2")

    # Clear all storage (use with caution!)
    # memtor.clear_all_storage()
    # print("All storage cleared")

    print("Advanced feature demonstrations completed.\n")

    # Custom Strategies
    print("Demonstrating custom strategies...")

    # Custom Embedding Strategy
    class SimpleEmbeddingStrategy(CustomEmbeddingStrategy):
        def embed(self, text):
            # Simple embedding: use the length of words as features
            return [len(word) for word in text.split()]

    # Custom Storage Strategy
    class InMemoryStorageStrategy(CustomStorageStrategy):
        def __init__(self):
            self.storage = {}

        def save(self, memory):
            self.storage[memory.id] = memory

        def load(self, memory_id):
            return self.storage.get(memory_id)

        def update(self, memory_id, memory):
            if memory_id in self.storage:
                self.storage[memory_id] = memory
                return True
            return False

        def delete(self, memory_id):
            if memory_id in self.storage:
                del self.storage[memory_id]
                return True
            return False

        def list_all(self):
            return list(self.storage.values())

        def clear_all(self):
            self.storage.clear()

    # Custom Search Strategy
    class SimpleSearchStrategy(CustomSearchStrategy):
        def search(self, query, memories, top_k, keywords, metadata_filters):
            # Simple search: rank by number of common words
            query_words = set(query.lower().split())
            scored_memories = [
                (memory, len(set(memory.content.lower().split()) & query_words))
                for memory in memories
            ]
            sorted_memories = sorted(scored_memories, key=lambda x: x[1], reverse=True)
            return [memory for memory, score in sorted_memories[:top_k]]

    # Initialize Memtor with custom strategies
    custom_memtor = Memtor(
        embedding_strategy=SimpleEmbeddingStrategy(),
        storage_strategy=InMemoryStorageStrategy(),
        search_strategy=SimpleSearchStrategy()
    )

    # Use custom Memtor
    custom_memory_id = custom_memtor.add_memory("Custom strategies are powerful and flexible")
    custom_results = custom_memtor.search_memories("powerful strategies", top_k=1)
    print(f"Custom search result: {custom_results[0].content}")

    print("Custom strategy demonstrations completed.\n")

    # Configuration
    print("Demonstrating configuration...")

    # Update configuration
    memtor.configure(embedding={"dimension": 128}, search={"algorithm": "cosine"})
    print("Configuration updated")

    print("Configuration demonstration completed.\n")

    print("Mem4AI complete usage guide finished.")

if __name__ == "__main__":
    main()