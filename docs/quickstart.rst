Quick Start Guide
=================

This guide will help you get started with Memtor quickly.

Creating a Memtor Instance
--------------------------

First, import Memtor and create an instance:

.. code-block:: python

   from mem4ai import Memtor

   memtor = Memtor()

Adding Memories
---------------

You can add memories like this:

.. code-block:: python

   memory_id = memtor.add_memory(
       "The quick brown fox jumps over the lazy dog",
       metadata={"tag": "animals"},
       user_id="user1",
       session_id="session1"
   )

Searching Memories
------------------

To search for memories:

.. code-block:: python

   results = memtor.search_memories("fox", top_k=1, user_id="user1")
   print(results[0].content)

Updating and Deleting Memories
------------------------------

Update a memory:

.. code-block:: python

   memtor.update_memory(
       memory_id,
       "The quick brown fox leaps over the lazy dog",
       metadata={"tag": "animals", "updated": True}
   )

Delete a memory:

.. code-block:: python

   memtor.delete_memory(memory_id)

For more detailed information, check out the full documentation and API reference.