from abc import ABC, abstractmethod
from typing import List, Union
import os
from litellm import embedding
from ..utils.config_manager import config_manager
import numpy as np

class EmbeddingStrategy(ABC):
    @abstractmethod
    def embed(self, input: Union[str, List[str]]) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

class LiteLLMEmbeddingStrategy(EmbeddingStrategy):
    def __init__(self):
        self.model = config_manager.get('embedding.model', 'text-embedding-3-small')
        self._dimension = config_manager.get('embedding.dimension', 1536)  # Default for text-embedding-3-small
        self.api_key = config_manager.get('embedding.api_key', os.getenv('OPENAI_API_KEY'))
        self.input_type = config_manager.get('embedding.input_type', None)

        if 'openai' in self.model.lower():
            os.environ['OPENAI_API_KEY'] = self.api_key
        elif 'huggingface' in self.model.lower():
            os.environ['HUGGINGFACE_API_KEY'] = self.api_key

    def embed(self, input: Union[str, List[str]]) -> np.ndarray:
        if isinstance(input, str):
            input = [input]

        kwargs = {
            "model": self.model,
            "input": input,
        }

        if 'text-embedding-3' in self.model:
            kwargs["dimensions"] = self._dimension

        if self.input_type and self.model.startswith('huggingface'):
            kwargs["input_type"] = self.input_type

        response = embedding(**kwargs)
        data = [item['embedding'] for item in response['data']]
        return np.array(data)

    @property
    def dimension(self) -> int:
        return self._dimension

def get_embedding_strategy() -> EmbeddingStrategy:
    strategy_name = config_manager.get('embedding.strategy', 'litellm')
    if strategy_name == 'litellm':
        return LiteLLMEmbeddingStrategy()
    else:
        raise ValueError(f"Unknown embedding strategy: {strategy_name}")