import os
import yaml
from typing import Any, Dict

class ConfigManager:
    DEFAULT_CONFIG = {
        'embedding': {
            'model': 'text-embedding-3-small',
            'dimension': 768,
        },
        'storage': {
            'type': 'lmdb',
            'path': './memtor_storage',
        },
        'search': {
            'algorithm': 'cosine_bm25',
            'top_k': 10,
        },
        'memory': {
            'max_history': 5,
        }
    }

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.expanduser('~/.mem4ai/config.yaml')
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            return self.merge_configs(self.DEFAULT_CONFIG, user_config)
        return self.DEFAULT_CONFIG.copy()

    def merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in user.items():
            if isinstance(value, dict) and key in default:
                default[key] = self.merge_configs(default[key], value)
            else:
                default[key] = value
        return default

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        return value

    def save(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

config_manager = ConfigManager()