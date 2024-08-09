from mem4ai import Memtor

def main():
    # Initialize Memtor
    memtor = Memtor()
    print("Memtor initialized successfully.")

    # Add memories
    print("\nAdding memories...")
    memory1_id = memtor.add_memory(
        "The quick brown fox jumps over the lazy dog",
        metadata={"tag": "animals"},
        user_id="user1",
        session_id="session1"
    )
    print(f"Memory 1 added with ID: {memory1_id}")

    memory2_id = memtor.add_memory(
        "To be or not to be, that is the question",
        metadata={"tag": "literature"},
        user_id="user1",
        session_id="session1"
    )
    print(f"Memory 2 added with ID: {memory2_id}")

    # Search memories
    print("\nSearching memories...")
    search_results = memtor.search_memories("fox", top_k=1, user_id="user1")
    print(f"Search result: {search_results[0].content}")

    # Update memory
    print("\nUpdating memory...")
    updated = memtor.update_memory(
        memory1_id,
        "The quick brown fox leaps over the lazy dog",
        metadata={"tag": "animals", "updated": True}
    )
    if updated:
        print("Memory updated successfully")
    
    # Retrieve updated memory
    updated_memory = memtor.get_memory(memory1_id)
    print(f"Updated memory content: {updated_memory.content}")
    print(f"Updated memory metadata: {updated_memory.metadata}")

    # List memories
    print("\nListing memories for user1...")
    user_memories = memtor.list_memories(user_id="user1")
    for memory in user_memories:
        print(f"- {memory.content}")

    # Delete memory
    print("\nDeleting memory...")
    deleted = memtor.delete_memory(memory2_id)
    if deleted:
        print("Memory deleted successfully")

    # Verify deletion
    remaining_memories = memtor.list_memories(user_id="user1")
    print(f"Remaining memories: {len(remaining_memories)}")

    # Clean up (optional)
    print("\nCleaning up...")
    memtor.delete_memories_by_user("user1")
    print("All test memories deleted")

if __name__ == "__main__":
    main()