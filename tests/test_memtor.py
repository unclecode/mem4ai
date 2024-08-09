import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from mem4ai import Memtor, Memory
from typing import List, Dict, Any

@pytest.fixture
def memtor():
    return Memtor()

@pytest.fixture
def clean_memtor():
    m = Memtor()
    # Clear memories for test users and sessions
    m.delete_memories_by_user("test_user")
    m.delete_memories_by_session("test_session")
    return m

@pytest.fixture
def sample_memories():
    return [
        {"content": "The quick brown fox jumps over the lazy dog", "metadata": {"tag": "animals", "year": 2020}},
        {"content": "To be or not to be, that is the question", "metadata": {"tag": "literature", "year": 2021}},
        {"content": "I think, therefore I am", "metadata": {"tag": "philosophy", "year": 2019}},
        {"content": "The only way to do great work is to love what you do", "metadata": {"tag": "motivation", "year": 2022}}
    ]

def test_add_memory(clean_memtor: Memtor):
    memory_id = clean_memtor.add_memory("Test content", {"tag": "test"}, user_id="test_user")
    assert isinstance(memory_id, str)
    assert len(memory_id) > 0

def test_get_memory(clean_memtor: Memtor):
    content = "Test content for get"
    memory_id = clean_memtor.add_memory(content, {"tag": "test"}, user_id="test_user")
    retrieved_memory = clean_memtor.get_memory(memory_id)
    assert isinstance(retrieved_memory, Memory)
    assert retrieved_memory.content == content
    assert retrieved_memory.metadata["tag"] == "test"

def test_update_memory(clean_memtor: Memtor):
    memory_id = clean_memtor.add_memory("Original content", {"tag": "original"}, user_id="test_user")
    updated = clean_memtor.update_memory(memory_id, "Updated content", {"tag": "updated"})
    assert updated is True
    updated_memory = clean_memtor.get_memory(memory_id)
    assert updated_memory.content == "Updated content"
    assert updated_memory.metadata["tag"] == "updated"

def test_delete_memory(clean_memtor: Memtor):
    memory_id = clean_memtor.add_memory("Content to delete", {"tag": "delete"}, user_id="test_user")
    deleted = clean_memtor.delete_memory(memory_id)
    assert deleted is True
    assert clean_memtor.get_memory(memory_id) is None

def test_list_memories(clean_memtor: Memtor, sample_memories: List[Dict[str, Any]]):
    for mem in sample_memories:
        clean_memtor.add_memory(mem["content"], mem["metadata"], user_id="test_user", session_id="test_session")
    
    all_memories = clean_memtor.list_memories(user_id="test_user")
    assert len(all_memories) == len(sample_memories)

    animal_memories = clean_memtor.list_memories(user_id="test_user", metadata_filters=[("tag", "==", "animals")])
    assert len(animal_memories) == 1
    assert animal_memories[0].metadata["tag"] == "animals"

    recent_memories = clean_memtor.list_memories(user_id="test_user", metadata_filters=[("year", ">=", 2021)])
    assert len(recent_memories) == 2
    assert all(mem.metadata["year"] >= 2021 for mem in recent_memories)

def test_search_memories(clean_memtor: Memtor, sample_memories: List[Dict[str, Any]]):
    for mem in sample_memories:
        clean_memtor.add_memory(mem["content"], mem["metadata"], user_id="test_user", session_id="test_session")

    results = clean_memtor.search_memories("fox", top_k=1, user_id="test_user")
    assert len(results) == 1
    assert "fox" in results[0].content.lower()

    results = clean_memtor.search_memories("philosophy", top_k=2, user_id="test_user")
    assert len(results) == 2
    assert any("think" in mem.content.lower() for mem in results)

    results = clean_memtor.search_memories("work", top_k=1, user_id="test_user", metadata_filters=[("year", ">", 2020)])
    assert len(results) == 1
    assert "work" in results[0].content.lower()
    assert results[0].metadata["year"] > 2020

def test_error_handling(clean_memtor: Memtor):
    with pytest.raises(TypeError):
        clean_memtor.add_memory(123)  # type: ignore

    with pytest.raises(TypeError):
        clean_memtor.get_memory(123)  # type: ignore

    with pytest.raises(TypeError):
        clean_memtor.update_memory(123, "content")  # type: ignore

    with pytest.raises(TypeError):
        clean_memtor.delete_memory(123)  # type: ignore

    with pytest.raises(TypeError):
        clean_memtor.search_memories(123)  # type: ignore

def test_memory_persistence(clean_memtor: Memtor):
    memory_id = clean_memtor.add_memory("Persistent content", {"tag": "persist"}, user_id="test_user")
    
    # Create a new Memtor instance to check if the memory persists
    new_memtor = Memtor()
    retrieved_memory = new_memtor.get_memory(memory_id)
    assert retrieved_memory is not None
    assert retrieved_memory.content == "Persistent content"
    assert retrieved_memory.metadata["tag"] == "persist"

def test_delete_memories_by_user(clean_memtor: Memtor):
    for i in range(5):
        clean_memtor.add_memory(f"Test content {i}", {"tag": "test"}, user_id="test_user")
    
    deleted_count = clean_memtor.delete_memories_by_user("test_user")
    assert deleted_count == 5
    
    remaining_memories = clean_memtor.list_memories(user_id="test_user")
    assert len(remaining_memories) == 0

def test_delete_memories_by_session(clean_memtor: Memtor):
    for i in range(3):
        clean_memtor.add_memory(f"Test content {i}", {"tag": "test"}, session_id="test_session")
    
    deleted_count = clean_memtor.delete_memories_by_session("test_session")
    assert deleted_count == 3
    
    remaining_memories = clean_memtor.list_memories(session_id="test_session")
    assert len(remaining_memories) == 0

if __name__ == "__main__":
    pytest.main([__file__])