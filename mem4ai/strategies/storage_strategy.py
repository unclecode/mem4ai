from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import os
import lmdb
import pickle
from ..core.memory import Memory
from ..utils.config_manager import config_manager

class StorageStrategy(ABC):
    @abstractmethod
    def save(self, memory: Memory) -> None:
        pass

    @abstractmethod
    def load(self, memory_id: str) -> Optional[Memory]:
        pass

    @abstractmethod
    def update(self, memory_id: str, memory: Memory) -> bool:
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def list_all(self) -> List[Memory]:
        pass

    @abstractmethod
    def apply_filters(self, memories: List[Memory], filters: List[tuple]) -> List[Memory]:
        pass
    
    @abstractmethod
    def clear_all(self):
        """Clear all data from the storage."""
        pass

class LMDBStorageStrategy(StorageStrategy):
    def __init__(self):
        self.path: str = config_manager.get('storage.path', './mem4ai_storage')
        self.map_size: int = config_manager.get('storage.map_size', 10 * 1024 * 1024 * 1024)  # 10GB default
        self._ensure_directory()
        self.env = lmdb.open(self.path, map_size=self.map_size)

    def _ensure_directory(self) -> None:
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, memory: Memory) -> None:
        if not isinstance(memory, Memory):
            raise TypeError("Expected Memory object, got {type(memory)}")
        
        with self.env.begin(write=True) as txn:
            txn.put(memory.id.encode(), pickle.dumps(memory))

    def load(self, memory_id: str) -> Optional[Memory]:
        if not isinstance(memory_id, str):
            raise TypeError(f"Expected string for memory_id, got {type(memory_id)}")
        
        with self.env.begin() as txn:
            data = txn.get(memory_id.encode())
            if data is None:
                return None
            return pickle.loads(data)

    def update(self, memory_id: str, memory: Memory) -> bool:
        if not isinstance(memory_id, str) or not isinstance(memory, Memory):
            raise TypeError("Invalid types for update operation")
        
        with self.env.begin(write=True) as txn:
            if txn.get(memory_id.encode()) is None:
                return False
            txn.put(memory_id.encode(), pickle.dumps(memory))
            return True

    def delete(self, memory_id: str) -> bool:
        if not isinstance(memory_id, str):
            raise TypeError(f"Expected string for memory_id, got {type(memory_id)}")
        
        with self.env.begin(write=True) as txn:
            return txn.delete(memory_id.encode())

    def list_all(self) -> List[Memory]:
        memories: List[Memory] = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                memories.append(pickle.loads(value))
        return memories

    def clear_all(self):
        with self.env.begin(write=True) as txn:
            txn.drop(self.env.open_db())
            
    def apply_filters(self, memories: List[Memory], filters: List[tuple]) -> List[Memory]:
        if not isinstance(memories, list) or not isinstance(filters, list):
            raise TypeError("Invalid types for apply_filters operation")
        
        def passes_filter(memory: Memory, filter_tuple: tuple) -> bool:
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

def get_storage_strategy() -> StorageStrategy:
    strategy_name: str = config_manager.get('storage.strategy', 'lmdb')
    if strategy_name == 'lmdb':
        return LMDBStorageStrategy()
    else:
        raise ValueError(f"Unknown storage strategy: {strategy_name}")

