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

def test_knowledge_extraction():
    """Test the knowledge extraction functionality of Memtor"""
    # Initialize Memtor
    memtor = Memtor()
    
    # Clear previous memories
    memtor.storage_strategy.clear_all()
    
    # Sample user message asking to create a horror movie list
    user_message = """Can you create a new favorites list called 'Horror Nights' and add some top-rated recent horror movies? I prefer psychological horror over gore."""

    # Sample assistant response with both JSON data and natural language
    assistant_response = """{
        "action": "create_and_populate_list",
        "list": {
            "name": "Horror Nights",
            "id": "hn_123",
            "movies": [
                {"id": "m1", "title": "Talk to Me", "year": 2023, "rating": 7.1},
                {"id": "m2", "title": "Hereditary", "year": 2018, "rating": 7.3},
                {"id": "m3", "title": "The Black Phone", "year": 2021, "rating": 7.0}
            ]
        }
    }
    I've created a new list called 'Horror Nights' and added some highly-rated psychological horror movies. I included 'Talk to Me' (2023) which deals with supernatural communication, 'Hereditary' (2018) which is a masterpiece of psychological horror, and 'The Black Phone' (2021) which blends supernatural elements with psychological tension. Would you like me to add more movies or adjust the selection based on your preferences?"""

    # Add memory with the conversation
    memory_id = memtor.add_memory(user_message, assistant_response)

    # Retrieve the memory to check the extracted context
    memory = memtor.get_memory(memory_id)
    assert memory is not None, "Failed to retrieve memory"
    
    # Verify context structure
    assert 'context' in memory.__dict__, "Memory doesn't have context field"
    context = memory.context
    
    # Basic structure checks
    assert 'timestamp' in context, "Context missing timestamp"
    assert 'interaction_type' in context, "Context missing interaction_type"
    assert context['interaction_type'] == 'action_based', "Incorrect interaction type"
    
    # Check action details
    assert 'action_details' in context, "Context missing action_details"
    action_details = context['action_details']
    assert action_details['primary_action']['type'] == 'creation', "Incorrect action type"
    assert action_details['primary_action']['target'] == 'favorite_list', "Incorrect action target"
    assert 'Horror Nights' in action_details['modified_elements']['lists'], "Created list not in modified elements"
    
    # Check conversation details
    assert 'conversation_details' in context, "Context missing conversation_details"
    conv_details = context['conversation_details']
    assert conv_details['intent'] == 'movie_organization', "Incorrect conversation intent"
    assert 'horror' in conv_details['key_information']['explicit_mentions'], "Missing explicit mention of horror"
    
    # Check referenced values
    assert 'Talk to Me' in conv_details['referenced_values']['movies'], "Missing referenced movie"
    assert '7.1' in conv_details['referenced_values']['ratings'], "Missing referenced rating"
    assert '2023' in conv_details['referenced_values']['dates'], "Missing referenced date"
    
    # Check summary
    assert 'summary' in context, "Context missing summary"
    assert len(context['summary']['key_points']) > 0, "Summary missing key points"
    
    # Test searching with extracted knowledge
    search_results = memtor.search_memories("psychological horror movies")
    assert len(search_results) > 0, "Failed to find memory using extracted knowledge"
    assert memory_id == search_results[0].id, "Search didn't return the correct memory"
    
    # Test another type of interaction (conversational)
    user_message_2 = "What did you think about the ending of Hereditary?"
    assistant_response_2 = "The ending of Hereditary is particularly haunting because it shows the culmination of the family's tragic fate. The final scene in the treehouse reveals that the entire series of events was orchestrated by the cult of Paimon. The possession of Peter and his ritualistic crowning as Paimon's vessel creates a deeply disturbing conclusion that fits perfectly with the film's themes of family trauma and predestination."
    
    memory_id_2 = memtor.add_memory(user_message_2, assistant_response_2)
    memory_2 = memtor.get_memory(memory_id_2)
    
    # Verify conversational context
    context_2 = memory_2.context
    assert context_2['interaction_type'] == 'conversational', "Incorrect interaction type for conversation"
    assert 'Hereditary' in context_2['conversation_details']['referenced_values']['movies'], "Missing movie reference"
    assert context_2['conversation_details']['intent'] == 'information_query', "Incorrect conversation intent"
    
    print("All knowledge extraction tests passed successfully!")

def test_summary_knowledge_extraction():
    """Test the simple summary knowledge extraction functionality of Memtor"""
    # Initialize Memtor with simple extraction strategy
    from mem4ai.strategies.knowledge_extraction import SimpleLLMExtractionStrategy
    memtor = Memtor(extraction_strategy=SimpleLLMExtractionStrategy())
    
    # Clear previous memories
    memtor.storage_strategy.clear_all()
    
    # Sample user message asking to create a horror movie list
    user_message = """Can you create a new favorites list called 'Horror Nights' and add some top-rated recent horror movies? I prefer psychological horror over gore."""

    # Sample assistant response with both JSON data and natural language
    assistant_response = """{
        "action": "create_and_populate_list",
        "list": {
            "name": "Horror Nights",
            "id": "hn_123",
            "movies": [
                {"id": "m1", "title": "Talk to Me", "year": 2023, "rating": 7.1},
                {"id": "m2", "title": "Hereditary", "year": 2018, "rating": 7.3},
                {"id": "m3", "title": "The Black Phone", "year": 2021, "rating": 7.0}
            ]
        }
    }
    I've created a new list called 'Horror Nights' and added some highly-rated psychological horror movies. I included 'Talk to Me' (2023) which deals with supernatural communication, 'Hereditary' (2018) which is a masterpiece of psychological horror, and 'The Black Phone' (2021) which blends supernatural elements with psychological tension. Would you like me to add more movies or adjust the selection based on your preferences?"""

    # Add memory with the conversation
    memory_id = memtor.add_memory(user_message, assistant_response)

    # Retrieve the memory to check the extracted context
    memory = memtor.get_memory(memory_id)
    assert memory is not None, "Failed to retrieve memory"
    
    # Verify context structure
    assert 'context' in memory.__dict__, "Memory doesn't have context field"
    context = memory.context
    
    # Basic structure checks
    assert 'timestamp' in context, "Context missing timestamp"
    assert 'interaction_type' in context, "Context missing interaction_type"
    assert context['interaction_type'] == 'task', "Incorrect interaction type"
    
    # Check summary
    assert 'summary' in context, "Context missing summary"
    assert isinstance(context['summary'], str), "Summary should be a string"
    assert 'Horror Nights' in context['summary'], "Summary should mention the created list"
    
    # Check keywords
    assert 'keywords' in context, "Context missing keywords"
    assert isinstance(context['keywords'], list), "Keywords should be a list"
    assert any('horror' in keyword.lower() for keyword in context['keywords']), "Keywords should include 'horror'"
    assert any('psychological' in keyword.lower() for keyword in context['keywords']), "Keywords should include 'psychological'"
    
    # Test conversational interaction
    user_message_2 = "What did you think about the ending of Hereditary?"
    assistant_response_2 = "The ending of Hereditary is particularly haunting because it shows the culmination of the family's tragic fate. The final scene in the treehouse reveals that the entire series of events was orchestrated by the cult of Paimon. The possession of Peter and his ritualistic crowning as Paimon's vessel creates a deeply disturbing conclusion that fits perfectly with the film's themes of family trauma and predestination."
    
    memory_id_2 = memtor.add_memory(user_message_2, assistant_response_2)
    memory_2 = memtor.get_memory(memory_id_2)
    
    # Verify conversational context
    context_2 = memory_2.context
    assert context_2['interaction_type'] == 'discussion', "Incorrect interaction type for conversation"
    assert isinstance(context_2['summary'], str), "Summary should be a string"
    assert 'Hereditary' in context_2['summary'], "Summary should mention the movie"
    assert any('ending' in keyword.lower() for keyword in context_2['keywords']), "Keywords should include 'ending'"
    assert any('hereditary' in keyword.lower() for keyword in context_2['keywords']), "Keywords should include 'hereditary'"
    
    # Test searching with extracted knowledge
    search_results = memtor.search_memories("psychological horror")
    assert len(search_results) > 0, "Failed to find memory using extracted knowledge"
    assert memory_id == search_results[0].id, "Search didn't return the correct memory"
    
    print("All summary knowledge extraction tests passed successfully!")
    
if __name__ == "__main__":
    # test_embedding()
    # test_storage()
    # test_search_strategy()  
    # test_memtor()
    # test_knowledge_extraction()
    test_summary_knowledge_extraction()