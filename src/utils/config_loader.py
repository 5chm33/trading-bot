<<<<<<< HEAD
# src/utils/config_loader.py
import yaml
import os
import copy
from typing import Dict, Any, Optional, Union
from pathlib import Path
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class ConfigLoader:
    """
    Production-grade configuration loader with:
    - Environment variable substitution
    - Schema validation
    - Deep merging
    - Multiple file formats support
    """

    @staticmethod
    def load_config(
        path: Union[str, Path],
        validate: bool = True,
        env_substitution: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration file with advanced features
        
        Args:
            path: Path to config file
            validate: Whether to validate config structure
            env_substitution: Whether to substitute environment variables
            
        Returns:
            Parsed configuration dictionary
        """
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")

            with open(path) as f:
                raw_config = f.read()

            # Environment variable substitution
            if env_substitution:
                raw_config = os.path.expandvars(raw_config)

            # Parse YAML
            config = yaml.safe_load(raw_config) or {}

            # Validate structure if requested
            if validate:
                ConfigLoader._validate_structure(config)

            logger.info(f"Successfully loaded config from {path}")
            return config

        except yaml.YAMLError as e:
            logger.critical(f"YAML parsing error in {path}: {str(e)}")
            raise
        except Exception as e:
            logger.critical(f"Config loading failed: {str(e)}")
            raise

    @staticmethod
    def _validate_structure(config: Dict) -> None:
        """Validate configuration structure with required sections"""
        required = {
            'meta': ['version', 'description'],
            'tickers': ['primary'],
            'trading': ['initial_balance'],
            'rl': ['algorithm', 'training'],
            'brokers': ['alpaca']
        }

        for section, keys in required.items():
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
            for key in keys:
                if key not in config[section]:
                    raise ValueError(f"Missing required key: {section}.{key}")

    @staticmethod
    def deep_merge(base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries (override takes precedence)
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = copy.deepcopy(base)
        for key, value in override.items():
            if (key in result and isinstance(result[key], dict) 
                    and isinstance(value, dict)):
                result[key] = ConfigLoader.deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    @staticmethod
    def merge_configs(*configs: Dict) -> Dict:
        """
        Merge multiple configs sequentially (last one has highest priority)
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        if not configs:
            return {}
            
        merged = copy.deepcopy(configs[0])
        for config in configs[1:]:
            merged = ConfigLoader.deep_merge(merged, config)
        return merged

    @staticmethod
    def save_config(config: Dict, path: Union[str, Path]) -> None:
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary
            path: Destination file path
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                yaml.safe_dump(config, f, sort_keys=False)
                
            logger.info(f"Successfully saved config to {path}")
        except Exception as e:
            logger.error(f"Failed to save config: {str(e)}")
            raise

    @staticmethod
    def get_nested(config: Dict, key_path: str, default: Any = None) -> Any:
        """
        Safely get nested config value using dot notation
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated path (e.g. 'rl.training.batch_size')
            default: Default value if key not found
            
        Returns:
            Config value or default if not found
        """
        keys = key_path.split('.')
        current = config
        for key in keys:
            if key not in current:
                return default
            current = current[key]
        return current
=======
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
>>>>>>> 60870aec3b9ed2c2cb804ceb4f1eeb5c6af9d852
