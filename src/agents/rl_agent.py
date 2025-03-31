# src/agents/rl_agent.py
import os
import gymnasium as gym
import torch
import numpy as np
import logging
from typing import Dict, Any
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class RLAgent:
    """Optimized RL Trading Agent with Safety Features"""
    
    def __init__(self, env, config: Dict[str, Any]):
        self._setup_hardware()
        self._validate_config(config)
        self.config = config
        self.env = env
        
        # Initialize components
        self._process_config()
        self._initialize_model()
        
        logger.info(f"RLAgent initialized on {self.device}")

    def _setup_hardware(self):
        """Configure hardware settings"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')

    def _validate_config(self, config: Dict):
        """Validate critical config sections"""
        required = {
            'rl': ['hyperparams', 'training'],
            'risk': ['max_drawdown'],
            'trading': ['position_limits']
        }
        for section, keys in required.items():
            if section not in config:
                raise ValueError(f"Missing config section: {section}")
            for key in keys:
                if key not in config[section]:
                    logger.warning(f"Missing recommended key: {section}.{key}")

    def _process_config(self):
        """Process and set default config values"""
        self.train_config = {
            'learning_rate': float(self.config['rl']['training'].get('learning_rate', 3e-4)),
            'batch_size': int(self.config['rl']['training'].get('batch_size', 256)),
            'total_steps': int(self.config['rl']['training'].get('total_steps', 100000))
        }

    def _initialize_model(self):
        """Initialize SAC model with proper configuration"""
        policy_kwargs = {
            "net_arch": self._get_network_architecture(),
            "features_extractor_class": self._build_feature_extractor(),
            "features_extractor_kwargs": {"features_dim": 128},
            "optimizer_kwargs": {"eps": 1e-8, "weight_decay": 1e-6}
        }

        self.model = SAC(
            policy="MultiInputPolicy",
            env=self.env,
            learning_rate=self.train_config['learning_rate'],
            batch_size=self.train_config['batch_size'],
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config['rl']['training'].get('tensorboard_dir', "logs/"),
            device=self.device,
            verbose=1
        )

    def _get_network_architecture(self) -> Dict:
        """Get validated network architecture"""
        net_arch = self.config['rl']['hyperparams'].get('net_arch', {'pi': [256,256], 'qf': [256,256]})
        return {
            'pi': self._validate_arch(net_arch.get('pi', [256, 256])),
            'qf': self._validate_arch(net_arch.get('qf', [256, 256]))
        }

    def _validate_arch(self, arch) -> list:
        """Ensure valid architecture dimensions"""
        return [int(x) for x in arch] if isinstance(arch, (list, tuple)) else [256, 256]

    def _build_feature_extractor(self):
        """Dynamic feature extractor construction"""
        class TradingFeatureExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space, features_dim=128):
                super().__init__(observation_space, features_dim)
                self.market_net = nn.Sequential(
                    nn.Linear(observation_space['market_data'].shape[0], 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.LayerNorm(128)
                )
                self.portfolio_net = nn.Linear(observation_space['portfolio'].shape[0], 32)
                self.output_net = nn.Linear(160, features_dim)
            
            def forward(self, obs):
                market = self.market_net(obs['market_data'])
                portfolio = self.portfolio_net(obs['portfolio'])
                return self.output_net(torch.cat([market, portfolio], dim=1))
        
        return TradingFeatureExtractor

    def train(self, total_timesteps: int, callback=None):
        """Optimized training with phase management"""
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=10
            )
        finally:
            self._clear_gpu_cache()

    def predict(self, observation, deterministic=False):
        """Safe action prediction with clipping"""
        action, _ = self.model.predict(observation, deterministic)
        return np.clip(action, -1, 1)

    def save(self, path: str):
        """Save model with directory creation"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def _clear_gpu_cache(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()