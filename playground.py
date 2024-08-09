from mem4ai.strategies.embedding_strategy import LiteLLMEmbeddingStrategy
from mem4ai.strategies.storage_strategy import LMDBStorageStrategy
from mem4ai.strategies.search_strategy import DefaultSearchStrategy
from mem4ai.core.embedding_manager import EmbeddingManager
from mem4ai.core.memory import Memory
from mem4ai.memtor import Memtor
import pytest
from mem4ai import Memtor, Memory

def test_embedding():
    # Test the LiteLLMEmbeddingStrategy
    embedding_strategy = LiteLLMEmbeddingStrategy()
    
    # Test single string embedding
    test_string = "This is a test sentence for embedding."
    embedding = embedding_strategy.embed(test_string)
    print(f"Single string embedding shape: {len(embedding)}")
    assert len(embedding[0]) == embedding_strategy.dimension, "Embedding dimension mismatch"

    # Test list of strings embedding
    test_strings = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "I think, therefore I am."
    ]
    embeddings = embedding_strategy.embed(test_strings)
    print(f"List of strings embedding shape: {len(embeddings)}x{len(embeddings[0])}")
    assert all(len(emb) == embedding_strategy.dimension for emb in embeddings), "Embedding dimension mismatch"

    # Test embedding similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"Cosine similarity between first two sentences: {similarity}")

    print("All tests passed successfully!")

def test_storage():
    # Test the LMDBStorageStrategy
    storage = LMDBStorageStrategy()
    
    # Test save and load
    test_memory = Memory("Test content", {"tag": "test"})
    storage.save(test_memory)
    loaded_memory = storage.load(test_memory.id)
    assert loaded_memory is not None, "Failed to load saved memory"
    assert loaded_memory.content == test_memory.content, "Loaded memory content does not match"
    
    # Test update
    test_memory.content = "Updated content"
    assert storage.update(test_memory.id, test_memory), "Failed to update memory"
    updated_memory = storage.load(test_memory.id)
    assert updated_memory is not None, "Failed to load updated memory"
    assert updated_memory.content == "Updated content", "Memory content was not updated"
    
    # Test delete
    assert storage.delete(test_memory.id), "Failed to delete memory"
    assert storage.load(test_memory.id) is None, "Memory was not deleted"
    
    # Test list_all and apply_filters
    memory1 = Memory("Content 1", {"value": 10})
    memory2 = Memory("Content 2", {"value": 20})
    storage.save(memory1)
    storage.save(memory2)
    all_memories = storage.list_all()
    assert len(all_memories) == 2, "list_all did not return all memories"
    filtered_memories = storage.apply_filters(all_memories, [("value", ">", 15)])
    assert len(filtered_memories) == 1, "apply_filters did not filter correctly"
    assert filtered_memories[0].metadata["value"] == 20, "apply_filters returned wrong memory"
    
    print("All tests passed successfully!")    

def test_search_strategy():
    # Create EmbeddingManager
    embedding_manager = EmbeddingManager()

    # Create SearchStrategy with EmbeddingManager
    search_strategy = DefaultSearchStrategy(embedding_manager)

    # Create test memories
    test_memories = [
        Memory("The quick brown fox jumps over the lazy dog", {"tag": "animals", "year": 2020}),
        Memory("A journey of a thousand miles begins with a single step", {"tag": "motivation", "year": 2019}),
        Memory("To be or not to be, that is the question", {"tag": "literature", "year": 2021}),
        Memory("I think, therefore I am", {"tag": "philosophy", "year": 2018})
    ]

    # Assign embeddings to test memories using EmbeddingManager
    for memory in test_memories:
        memory.embedding = embedding_manager.embed(memory.content)

    # Test search with string query and metadata filter
    query = "What animal jumps?"
    metadata_filters = [("year", ">=", 2020)]
    results = search_strategy.search(query, test_memories, top_k=2, keywords=["animal"], metadata_filters=metadata_filters)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert "fox" in results[0].content.lower(), f"Expected 'fox' in top result, got: {results[0].content}"
    assert all(m.metadata["year"] >= 2020 for m in results), "Metadata filter not applied correctly"

    # Test search with embedding query and metadata filter
    query_embedding = embedding_manager.embed("A philosophical question")
    metadata_filters = [("tag", "==", "philosophy")]
    results = search_strategy.search(query_embedding, test_memories, top_k=1, keywords=[], metadata_filters=metadata_filters)
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0].metadata["tag"] == "philosophy", "Metadata filter not applied correctly"

    # Test error handling
    with pytest.raises(TypeError, match="Query must be a string or a list of floats"):
        search_strategy.search(123, test_memories, top_k=2, keywords=[], metadata_filters=[])

    print("All tests passed successfully!")
    print("All tests passed successfully!")

def test_memtor():
    # Initialize Memtor
    memtor = Memtor()
    print("Memtor initialized successfully.")
    
    # Remove previous memroy for users and sessions
    memtor.storage_strategy.clear_all()

    # Add memories
    memory1_id = memtor.add_memory("The quick brown fox jumps over the lazy dog", 
                                   metadata={"tag": "animals"}, 
                                   user_id="user1", 
                                   session_id="session1")
    memory2_id = memtor.add_memory("To be or not to be, that is the question", 
                                   metadata={"tag": "literature"}, 
                                   user_id="user1", 
                                   session_id="session1")
    memory3_id = memtor.add_memory("E = mc^2", 
                                   metadata={"tag": "science"}, 
                                   user_id="user2", 
                                   session_id="session2")
    print("Three memories added successfully.")

    # List memories
    all_memories = memtor.list_memories()
    assert len(all_memories) == 3, f"Expected 3 memories, got {len(all_memories)}"
    print("List memories successful.")

    # Search memories
    search_results = memtor.search_memories("fox", top_k=1, user_id="user1")
    assert len(search_results) == 1, f"Expected 1 search result, got {len(search_results)}"
    assert "fox" in search_results[0].content, "Search result doesn't contain 'fox'"
    print("Search memories successful.")

    # Update memory
    updated = memtor.update_memory(memory1_id, "The quick brown fox leaps over the lazy dog", 
                                   metadata={"tag": "animals", "updated": True})
    assert updated, "Memory update failed"
    updated_memory = memtor.get_memory(memory1_id)
    assert "leaps" in updated_memory.content, "Memory content not updated correctly"
    assert updated_memory.metadata.get("updated") == True, "Memory metadata not updated correctly"
    print("Update memory successful.")

    # Delete memory
    deleted = memtor.delete_memory(memory3_id)
    assert deleted, "Memory deletion failed"
    remaining_memories = memtor.list_memories()
    assert len(remaining_memories) == 2, f"Expected 2 memories after deletion, got {len(remaining_memories)}"
    print("Delete memory successful.")

    # Test metadata filtering
    filtered_memories = memtor.list_memories(user_id="user1", metadata_filters=[("tag", "==", "literature")])
    assert len(filtered_memories) == 1, f"Expected 1 filtered memory, got {len(filtered_memories)}"
    assert filtered_memories[0].metadata["tag"] == "literature", "Metadata filtering failed"
    print("Metadata filtering successful.")

    print("All tests passed successfully!")

if __name__ == "__main__":
    # test_embedding()
    # test_storage()
    # test_search_strategy()  
    test_memtor()