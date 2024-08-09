from ..strategies.embedding_strategy import get_embedding_strategy, EmbeddingStrategy
from ..utils.config_manager import config_manager
import numpy as np

class EmbeddingManager:
    def __init__(self, embedding_strategy: EmbeddingStrategy = None):
        self.embedding_strategy = embedding_strategy or get_embedding_strategy()

    def embed(self, text) -> np.ndarray:
        return self.embedding_strategy.embed(text)

    @property
    def dimension(self) -> int:
        return self.embedding_strategy.dimension