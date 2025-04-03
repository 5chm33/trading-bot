import os
import numpy as np
import torch
import psutil
from typing import Dict, Any
from pathlib import Path
from prometheus_client import Gauge

# Stable-Baselines3 callback import with version compatibility
try:
    from stable_baselines3.common.callbacks import BaseCallback as Callback
except ImportError:
    from stable_baselines3.common.callbacks import Callback

class SACParamValidator:
    """Robust SAC configuration validator with safe defaults"""

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and sanitizes the complete configuration with:
        - Auto-populated default values
        - Hardware compatibility checks
        - Parameter boundary enforcement
        """
        validated = config.copy()

        # Ensure all required sections exist
        validated.setdefault('rl', {}).setdefault('hyperparams', {}).setdefault('sac', {})
        validated['rl'].setdefault('algorithm', {}).setdefault('policy_kwargs', {})

        cls._validate_learning_params(validated)
        cls._validate_network(validated)
        cls._check_hardware_compatibility(validated)
        return validated

    @staticmethod
    def _validate_learning_params(config: Dict) -> None:
        # Ensure SAC params section exists
        sac_params = config['rl'].setdefault('hyperparams', {}).setdefault('sac', {})

        # Handle backward compatibility
        training_params = config['rl'].get('training', {})

        # Parameter migration with defaults
        param_mapping = {
            'buffer_size': (training_params.get('buffer_size'), 1000000),
            'batch_size': (training_params.get('batch_size'), 256),
            'learning_rate': (training_params.get('learning_rate'), 3e-4),
            'tau': (None, 0.005),
            'gamma': (None, 0.99),
            'ent_coef': (None, "auto"),
            'target_entropy': (None, "auto"),
            'use_sde': (None, True),
            'sde_sample_freq': (None, 64)
        }

        for param, (value, default) in param_mapping.items():
            sac_params[param] = sac_params.get(param, value if value is not None else default)

        # Enforce bounds
        sac_params['learning_rate'] = np.clip(
            float(sac_params['learning_rate']),
            1e-5,  # min
            1e-2   # max
        )

    @staticmethod
    def _validate_network(config: Dict[str, Any]) -> None:
        """Network architecture validation with defaults"""
        net_arch = config['rl']['algorithm']['policy_kwargs'].setdefault(
            'net_arch',
            {'pi': [256, 256], 'qf': [256, 256]}
        )

        # Validate each network type
        for net_type in ['pi', 'qf']:
            net_arch[net_type] = [
                min(max(int(u), 32), 1024)  # Clamp to 32-1024 units
                for u in net_arch.get(net_type, [256, 256])
            ]

    @staticmethod
    def _check_hardware_compatibility(config: Dict[str, Any]) -> None:
        """Verify hardware settings match available resources"""
        device = config['rl']['algorithm'].setdefault('device', 'auto')

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but not available. "
                "Set device: 'auto' or 'cpu' in config"
            )

class MemoryMonitor(Callback):
    """Real-time resource monitor with Prometheus integration"""

    def __init__(self):
        super().__init__()
        self.metrics = {
            'ram': Gauge('training_ram_usage', 'RAM usage (MB)'),
            'gpu': Gauge('training_gpu_usage', 'GPU memory (MB)'),
            'cpu': Gauge('training_cpu_usage', 'CPU utilization %')
        }

    def _on_step(self) -> bool:
        """Record metrics at each training step"""
        process = psutil.Process()

        # RAM tracking
        self.metrics['ram'].set(process.memory_info().rss / 1024**2)

        # GPU tracking
        if torch.cuda.is_available():
            self.metrics['gpu'].set(torch.cuda.memory_allocated() / 1024**2)

        # CPU utilization
        self.metrics['cpu'].set(process.cpu_percent())

        return True

def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    """Safe config loader with validation"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path.absolute()}\n"
            f"Expected location: {Path('config').absolute()}/config.yaml"
        )

    with open(path) as f:
        config = SACParamValidator.validate(yaml.safe_load(f))

    return config
