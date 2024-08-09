from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..core.memory import Memory
from ..utils.config_manager import config_manager
from ..core.embedding_manager import EmbeddingManager

class SearchStrategy(ABC):
    @abstractmethod
    def search(self, query: Union[str, np.ndarray], memories: List[Memory], top_k: int, 
               keywords: List[str], metadata_filters: List[Tuple[str, str, Any]]) -> List[Memory]:
        pass

class DefaultSearchStrategy:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.k1: float = config_manager.get('search.bm25_k1', 1.5)
        self.b: float = config_manager.get('search.bm25_b', 0.75)

    def search(self, query: Union[str, np.ndarray], memories: List[Memory], top_k: int, 
               keywords: List[str], metadata_filters: List[Tuple[str, str, Any]]) -> List[Memory]:
        if not isinstance(memories, list) or not all(isinstance(m, Memory) for m in memories):
            raise TypeError("memories must be a list of Memory objects")
        if not memories:
            return []
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        
        if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
            raise TypeError("keywords must be a list of strings")
        
        if not isinstance(metadata_filters, list):
            raise TypeError("metadata_filters must be a list")
        
        filtered_memories = self._apply_metadata_filters(memories, metadata_filters)
        if not filtered_memories:
            return []

        # Stage 1: Cosine Similarity
        try:
            if isinstance(query, str):
                query_embedding = self.embedding_manager.embed(query)
            elif isinstance(query, np.ndarray) and query.dtype == float:
                query_embedding = query
            else:
                raise TypeError("query must be a string or a numpy array of floats")
            
            cosine_scores = self._calculate_cosine_similarity(query_embedding, filtered_memories)
            top_k = min(top_k, len(filtered_memories))
            top_cosine_indices = np.argsort(cosine_scores)[-top_k:][::-1]
            top_memories = [filtered_memories[i] for i in top_cosine_indices]
        except Exception as e:
            print(f"Error during cosine similarity calculation: {str(e)}")
            return []

        # Stage 2: BM25 Re-ranking (only for string queries)
        if isinstance(query, str):
            try:
                bm25_scores = self._calculate_bm25_scores(query, top_memories, keywords)
                final_indices = np.argsort(bm25_scores)[::-1]
                if final_indices == [0]:
                    return top_memories
                return [top_memories[i] for i in final_indices]
            except Exception as e:
                print(f"Error during BM25 re-ranking: {str(e)}")
                return top_memories  # Fall back to cosine similarity results if BM25 fails
        else:
            return top_memories

    def _apply_metadata_filters(self, memories, filters):
        if not filters:
            return memories

        def passes_filter(memory, filter_tuple):
            key, op, value = filter_tuple
            if key not in memory.metadata:
                return False
            mem_value = memory.metadata[key]
            if op == '==':
                return mem_value == value
            elif op == '!=':
                return mem_value != value
            elif op == '>':
                return mem_value > value
            elif op == '>=':
                return mem_value >= value
            elif op == '<':
                return mem_value < value
            elif op == '<=':
                return mem_value <= value
            else:
                raise ValueError(f"Unknown operator: {op}")

        return [mem for mem in memories if all(passes_filter(mem, f) for f in filters)]

    def _calculate_cosine_similarity(self, query_embedding : np.ndarray, memories : List[Memory]) -> np.ndarray:
        memory_embeddings = np.array([memory.embedding[0] for memory in memories])
        
        if memory_embeddings.shape[0] == 0:
            return np.array([])
        
        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure memory_embeddings is 2D
        if memory_embeddings.ndim == 1:
            memory_embeddings = memory_embeddings.reshape(1, -1)
        
        # return cosine_similarity(query_embedding, memory_embeddings)[0]
        return np.dot(memory_embeddings, query_embedding.T).flatten()

    def _calculate_bm25_scores(self, query : str, memories : List[Memory], keywords : List[str]) -> np.ndarray:
        try:
            corpus = [memory.content for memory in memories]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            doc_lens = tfidf_matrix.sum(axis=1).flatten()  # Sum across rows and flatten
            avg_doc_len = doc_lens.mean()

            query_vec = self.tfidf_vectorizer.transform([query])
            query_terms = query_vec.indices

            keyword_boost = np.zeros(len(memories))
            for keyword in keywords:
                if keyword in self.tfidf_vectorizer.vocabulary_:
                    keyword_idx = self.tfidf_vectorizer.vocabulary_[keyword]
                    keyword_boost += tfidf_matrix[:, keyword_idx].toarray().flatten()

            scores = np.zeros(len(memories))
            for idx in query_terms:
                qi = np.array(query_vec[0, idx])
                fi = tfidf_matrix[:, idx].toarray().flatten()
                numerator = fi * (self.k1 + 1)
                denominator = fi + self.k1 * (1 - self.b + self.b * doc_lens / avg_doc_len)
                x = qi * (numerator / denominator)
                scores = scores + x

            final_scores = scores + keyword_boost
            return final_scores.tolist()[0]
        except Exception as e:
                print(f"Error in BM25 calculation: {str(e)}")
                return [0.0] * len(memories)  # Return neutral scores if calculation fails

def get_search_strategy(embedding_manager=None):
    strategy_name = config_manager.get('search.strategy', 'default')
    if strategy_name == 'default':
        return DefaultSearchStrategy(embedding_manager or EmbeddingManager())
    else:
        raise ValueError(f"Unknown search strategy: {strategy_name}")