from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class BaseConfig:
    """Base configuration class with common utilities."""
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = getattr(self, '__dict__', {}).copy()
        
        # Helper to recursively convert Path objects or other non-serializable objects
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj
            
        config_dict = convert_paths(config_dict)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)