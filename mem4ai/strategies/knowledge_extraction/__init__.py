from .base import KnowledgeExtractionStrategy
from .llm import LLMExtractionStrategy
from .summary import SummaryExtractionStrategy
from .echo import EchoKnowledgeStrategy
from ...utils.config_manager import config_manager

def get_extraction_strategy() -> KnowledgeExtractionStrategy:
    """
    Factory function to get the appropriate knowledge extraction strategy based on configuration.
    """
    strategy_name = config_manager.get('extraction.strategy', 'llm')
    
    if strategy_name == 'llm':
        return LLMExtractionStrategy()
    elif strategy_name == 'simple':
        return SummaryExtractionStrategy()
    elif strategy_name == 'none':
        return None
    else:
        raise ValueError(f"Unknown extraction strategy: {strategy_name}")

# Make these accessible when importing from knowledge_extraction
__all__ = [
    'KnowledgeExtractionStrategy',
    'LLMExtractionStrategy',
    'SummaryExtractionStrategy',
    'get_extraction_strategy',
    'EchoKnowledgeStrategy',
]