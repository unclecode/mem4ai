Basic Usage Example
===================

This example demonstrates the basic usage of Memtor, including how to add, retrieve, update, and delete memories.

.. code-block:: python

   from mem4ai import Memtor

   # Initialize Memtor
   memtor = Memtor()

   # Add a memory
   memory_id = memtor.add_memory(
       "The quick brown fox jumps over the lazy dog",
       metadata={"tag": "animals"},
       user_id="user1",
       session_id="session1"
   )
   print(f"Added memory with ID: {memory_id}")

   # Retrieve the memory
   memory = memtor.get_memory(memory_id)
   print(f"Retrieved memory: {memory.content}")

   # Update the memory
   memtor.update_memory(
       memory_id,
       "The quick brown fox leaps over the lazy dog",
       metadata={"tag": "animals", "updated": True}
   )

   # Search for memories
   results = memtor.search_memories("fox", top_k=1, user_id="user1")
   print(f"Search result: {results[0].content}")

   # Delete the memory
   memtor.delete_memory(memory_id)
   print("Memory deleted")

This script covers the basic operations you can perform with Memtor. It demonstrates how to:

1. Initialize a Memtor instance
2. Add a new memory with content, metadata, user ID, and session ID
3. Retrieve a memory by its ID
4. Update an existing memory
5. Search for memories
6. Delete a memory

You can run this example by executing the script in the `examples` directory.