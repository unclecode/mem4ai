from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import os
import lmdb
import pickle
from datetime import datetime, timedelta
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
    def clear_all(self) -> None:
        pass

    @abstractmethod
    def find_recent(self, limit: int, **kwargs) -> List[Memory]:
        """Get the most recent memories with optional filters"""
        pass

    @abstractmethod
    def find_by_time(self, start_time: datetime, end_time: datetime, **kwargs) -> List[Memory]:
        """Get memories within a time range with optional filters"""
        pass

    @abstractmethod
    def find_by_meta(self, metadata_filters: Dict[str, Any]) -> List[Memory]:
        """Get memories by metadata filters"""
        pass

class LMDBStorageStrategy(StorageStrategy):
    METADATA_KEYS = ['user_id', 'session_id', 'agent_id']

    def __init__(self):
        self.path: str = config_manager.get('storage.path', './mem4ai_storage')
        self.map_size: int = config_manager.get('storage.map_size', 10 * 1024 * 1024 * 1024)  # 10GB default
        self._ensure_directory()
        
        # Main environment for storing memories
        self.env = lmdb.open(self.path, map_size=self.map_size)
        
        # Separate environments for indices
        self.timestamp_env = lmdb.open(f"{self.path}_timestamp_index", map_size=1024 * 1024 * 1024)
        self.metadata_env = lmdb.open(f"{self.path}_metadata_index", map_size=1024 * 1024 * 1024)
        
        self._init_indices()

    def _ensure_directory(self) -> None:
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _init_indices(self) -> None:
        """Initialize index structures if they don't exist"""
        with self.timestamp_env.begin(write=True) as txn:
            if not txn.get(b'timestamp_index'):
                txn.put(b'timestamp_index', pickle.dumps({}))

        with self.metadata_env.begin(write=True) as txn:
            for key in self.METADATA_KEYS:
                if not txn.get(key.encode()):
                    txn.put(key.encode(), pickle.dumps({}))

    def save(self, memory: Memory) -> None:
        if not isinstance(memory, Memory):
            raise TypeError(f"Expected Memory object, got {type(memory)}")
        
        with self.env.begin(write=True) as txn:
            txn.put(memory.id.encode(), pickle.dumps(memory))
        
        # Update indices
        with self.timestamp_env.begin(write=True) as txn:
            self._update_timestamp_index(memory, txn)
        
        with self.metadata_env.begin(write=True) as txn:
            self._update_metadata_index(memory, txn)

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
            
            # Update indices
            with self.timestamp_env.begin(write=True) as ts_txn:
                self._update_timestamp_index(memory, ts_txn)
            
            with self.metadata_env.begin(write=True) as meta_txn:
                self._update_metadata_index(memory, meta_txn)
            
            return True

    def delete(self, memory_id: str) -> bool:
        if not isinstance(memory_id, str):
            raise TypeError(f"Expected string for memory_id, got {type(memory_id)}")
        
        memory = self.load(memory_id)
        if not memory:
            return False
        
        # Remove from main storage and indices
        with self.env.begin(write=True) as txn:
            txn.delete(memory_id.encode())
        
        self._remove_from_indices(memory)
        return True

    def list_all(self) -> List[Memory]:
        memories = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                memories.append(pickle.loads(value))
        return memories

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

    def clear_all(self) -> None:
        with self.env.begin(write=True) as txn:
            txn.drop(self.env.open_db())
        
        with self.timestamp_env.begin(write=True) as txn:
            txn.drop(self.timestamp_env.open_db())
        
        with self.metadata_env.begin(write=True) as txn:
            txn.drop(self.metadata_env.open_db())
        
        self._init_indices()

    def _update_timestamp_index(self, memory: Memory, txn) -> None:
        timestamp_key = memory.timestamp.isoformat().encode()
        txn.put(f"ts:{memory.id}".encode(), timestamp_key)
        
        index = pickle.loads(txn.get(b'timestamp_index') or pickle.dumps({}))
        if timestamp_key not in index:
            index[timestamp_key] = set()
        index[timestamp_key].add(memory.id)
        txn.put(b'timestamp_index', pickle.dumps(index))

    def _update_metadata_index(self, memory: Memory, txn) -> None:
        for key in self.METADATA_KEYS:
            if key in memory.metadata:
                value = memory.metadata[key]
                index_key = f"{key}:{value}".encode()
                index = pickle.loads(txn.get(index_key) or pickle.dumps(set()))
                index.add(memory.id)
                txn.put(index_key, pickle.dumps(index))

    def _remove_from_indices(self, memory: Memory) -> None:
        # Remove from timestamp index
        with self.timestamp_env.begin(write=True) as txn:
            index = pickle.loads(txn.get(b'timestamp_index'))
            timestamp_key = memory.timestamp.isoformat().encode()
            if timestamp_key in index:
                index[timestamp_key].discard(memory.id)
                if not index[timestamp_key]:
                    del index[timestamp_key]
            txn.put(b'timestamp_index', pickle.dumps(index))
        
        # Remove from metadata indices
        with self.metadata_env.begin(write=True) as txn:
            for key in self.METADATA_KEYS:
                if key in memory.metadata:
                    index_key = f"{key}:{memory.metadata[key]}".encode()
                    index = pickle.loads(txn.get(index_key) or pickle.dumps(set()))
                    index.discard(memory.id)
                    if index:
                        txn.put(index_key, pickle.dumps(index))
                    else:
                        txn.delete(index_key)

    def find_recent(self, limit: int, **kwargs) -> List[Memory]:
        memories = []
        metadata_filters = {k: v for k, v in kwargs.items() 
                          if k in self.METADATA_KEYS and v is not None}
        
        with self.timestamp_env.begin() as ts_txn:
            index = pickle.loads(ts_txn.get(b'timestamp_index'))
            sorted_timestamps = sorted(index.keys(), reverse=True)
            
            with self.env.begin() as mem_txn:
                for ts_key in sorted_timestamps:
                    memory_ids = index[ts_key]
                    if metadata_filters:
                        memory_ids = self._filter_by_metadata(memory_ids, metadata_filters)
                    
                    for memory_id in memory_ids:
                        memory = pickle.loads(mem_txn.get(memory_id.encode()))
                        if memory:
                            memories.append(memory)
                            if len(memories) >= limit:
                                return memories[:limit]
        return memories

    def find_by_time(self, start_time: datetime, end_time: datetime, **kwargs) -> List[Memory]:
        memories = []
        start_key = start_time.isoformat().encode()
        end_key = end_time.isoformat().encode()
        metadata_filters = {k: v for k, v in kwargs.items() 
                          if k in self.METADATA_KEYS and v is not None}
        
        with self.timestamp_env.begin() as ts_txn:
            index = pickle.loads(ts_txn.get(b'timestamp_index'))
            
            with self.env.begin() as mem_txn:
                for ts_key in index:
                    if start_key <= ts_key <= end_key:
                        memory_ids = index[ts_key]
                        if metadata_filters:
                            memory_ids = self._filter_by_metadata(memory_ids, metadata_filters)
                        
                        for memory_id in memory_ids:
                            memory = pickle.loads(mem_txn.get(memory_id.encode()))
                            if memory:
                                memories.append(memory)
        
        return sorted(memories, key=lambda x: x.timestamp)

    def _filter_by_metadata(self, memory_ids: set, metadata_filters: Dict[str, Any]) -> set:
        result_ids = memory_ids
        for key, value in metadata_filters.items():
            with self.metadata_env.begin() as txn:
                index_key = f"{key}:{value}".encode()
                current_ids = pickle.loads(txn.get(index_key) or pickle.dumps(set()))
                result_ids = result_ids.intersection(current_ids)
                if not result_ids:  # Short circuit if no matches
                    break
        return result_ids

    def find_by_meta(self, metadata_filters: Dict[str, Any]) -> List[Memory]:
        valid_filters = {k: v for k, v in metadata_filters.items() 
                        if k in self.METADATA_KEYS and v is not None}
        
        if not valid_filters:
            return []

        key, value = next(iter(valid_filters.items()))
        with self.metadata_env.begin() as meta_txn:
            memory_ids = pickle.loads(meta_txn.get(f"{key}:{value}".encode()) or pickle.dumps(set()))

        if len(valid_filters) > 1:
            memory_ids = self._filter_by_metadata(memory_ids, valid_filters)

        memories = []
        with self.env.begin() as mem_txn:
            for memory_id in memory_ids:
                memory_data = mem_txn.get(memory_id.encode())
                if memory_data:
                    memories.append(pickle.loads(memory_data))

        return sorted(memories, key=lambda x: x.timestamp, reverse=True)
    
    
def get_storage_strategy() -> StorageStrategy:
    strategy_name: str = config_manager.get('storage.strategy', 'lmdb')
    if strategy_name == 'lmdb':
        return LMDBStorageStrategy()
    else:
        raise ValueError(f"Unknown storage strategy: {strategy_name}")

