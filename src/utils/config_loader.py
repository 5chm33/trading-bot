import yaml
from typing import Dict, Any
import os
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class ConfigLoader:
    """Enhanced YAML configuration loader with environment variable support"""
    
    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """
        Load and validate configuration file with environment variable substitution
        Example ${VAR_NAME} in YAML gets replaced with os.environ['VAR_NAME']
        """
        try:
            with open(path) as f:
                raw_config = f.read()
            
            # Handle environment variables
            config = yaml.safe_load(
                os.path.expandvars(raw_config))
            
            # Validate required sections
            ConfigLoader._validate_structure(config)
            
            logger.info(f"Loaded config from {path}")
            return config
            
        except Exception as e:
            logger.critical(f"Config loading failed: {str(e)}")
            raise

    @staticmethod
    def _validate_structure(config: Dict) -> None:
        """Ensure required sections exist"""
        required = {
            'meta': ['version'],
            'tickers': ['primary'],
            'rl': ['algorithm', 'training']
        }
        
        for section, keys in required.items():
            if section not in config:
                raise ValueError(f"Missing config section: {section}")
            for key in keys:
                if key not in config[section]:
                    raise ValueError(f"Missing {section}.{key} in config")

    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict:
        """Deep merge two config dictionaries"""
        merged = base.copy()
        for key, value in override.items():
            if (key in merged and isinstance(merged[key], dict) 
                    and isinstance(value, dict)):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged