"""
Mem4AI: A memory management library for LLMs and AI systems.

This library provides functionality for storing, retrieving, and searching memories,
with support for embedding-based similarity search and metadata filtering.
"""

from .memtor import Memtor
from .core.memory import Memory
from .strategies.embedding_strategy import EmbeddingStrategy
from .strategies.storage_strategy import StorageStrategy
from .strategies.search_strategy import SearchStrategy
from .strategies.knowledge_extraction import LLMExtractionStrategy, SummaryExtractionStrategy, EchoKnowledgeStrategy, SummaryExtractionStrategy
from .strategies.knowledge_extraction import KnowledgeExtractionStrategy
from .utils.config_manager import config_manager

# Version of the memtor package
__version__ = "0.1.1"

# Define what should be importable from the package
__all__ = ['Memtor', 'Memory', 'EmbeddingStrategy', 'StorageStrategy', 'SearchStrategy', 'config_manager', 'LLMExtractionStrategy', 'SummaryExtractionStrategy', 'EchoKnowledgeStrategy', 'KnowledgeExtractionStrategy']

# Package level initialization code (if any)
def initialize():
    """
    Perform any necessary package-level initialization.
    This function is called when the package is imported.
    """
    # For now, we don't need any initialization, but we can add code here if needed in the future
    pass

# Call the initialize function
initialize()