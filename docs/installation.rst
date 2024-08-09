Installation
============

This section covers the installation process for Mem4AI.

Requirements
------------

- Python 3.7+
- pip

Installing Mem4AI
-----------------

You can install Mem4AI using pip:

.. code-block:: bash

   pip install mem4ai

Or, if you're installing from source:

.. code-block:: bash

   git clone https://github.com/unclecode/mem4ai.git
   cd mem4ai
   pip install -e .

Verifying Installation
----------------------

After installation, you can verify that Mem4AI is installed correctly by running:

.. code-block:: python

   import mem4ai
   print(mem4ai.__version__)

This should print the version number of Mem4AI.