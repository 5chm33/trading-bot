import os
import gymnasium as gym
import torch
import pandas as pd
import numpy as np
import logging
import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import DictReplayBuffer
from torch.nn.functional import huber_loss
from src.utils.logging import setup_logger
from src.models.rl.env import TradingEnv
from src.models.transformer.trainer import TransformerTrainer, TransformerFeatureProcessor
from src.monitoring.trading_metrics import TradeMetrics
from src.utils.config_loader import ConfigLoader

logger = setup_logger(__name__)

class RLAgent:
    """
    Production-Grade RL Trading Agent with:
    - Transformer feature processing
    - Adaptive risk management
    - Hardware optimization
    - Comprehensive monitoring
    """

    def __init__(self, env: gym.Env, config: Dict[str, Any], metrics: TradeMetrics = None):
        """
        Initialize agent with environment and config
        
        Args:
            env: Trading environment (live or backtest)
            config: Agent configuration dictionary
            metrics: Optional metrics tracker (default: new instance)
        """
        # 1. System setup
        self._setup_hardware()
        self._validate_config(config)
        
        # 2. Core components
        self.env = env
        self.config = config
        self.metrics = metrics if metrics else TradeMetrics()  # Use provided or create new
        self.safety_monitor = SafetyMonitor(config)
        
        # 3. Feature processing pipeline
        self.feature_processor = self._init_feature_processor(env)
        
        # 4. Model initialization
        self._process_config()
        self.model = self._initialize_model()
        
        # Record initialization metrics
        self._record_initialization()
        
        self.greek_weights = {
            'delta': 0.6,  # Price sensitivity
            'theta': 0.3,  # Time decay
            'vega': 0.1   # Volatility
        }
        
        logger.info(f"RLAgent initialized on {self.device.type.upper()}")

    def _process_config(self) -> None:
        """Process and normalize configuration"""
        defaults = {
            'rl': {
                'policy': "MultiInputPolicy",
                'learning_rate': 3e-4,
                'buffer_size': 100000,
                'batch_size': 256,
                'gamma': 0.99,
                'tau': 0.005
            },
            'risk': {
                'max_drawdown': 0.25,
                'position_limits': {'max': 0.5}
            }
        }
        
        from src.utils.config_loader import ConfigLoader
        self.config = ConfigLoader.deep_merge(defaults, self.config)
        
        # Validate RL config
        self._validate_rl_config()
        
        # Type conversion
        self.config['rl']['learning_rate'] = float(self.config['rl']['learning_rate'])
        self.config['rl']['buffer_size'] = int(self.config['rl']['buffer_size'])
        
    
    def _init_feature_processor(self, env: gym.Env) -> TransformerFeatureProcessor:
        """Initialize feature processor with metrics instrumentation"""
        try:
            processor = TransformerFeatureProcessor(self.config)
            
            if hasattr(env, 'broker'):  # Live trading
                logger.info("Live trading processor initialized")
                self.metrics.PROCESSOR_INIT.labels(mode='live').inc()
            else:  # Backtesting
                logger.info("Fitting processor with historical data")
                sample_data = env.data.head(1000).copy()
                
                if 'rsi' not in sample_data.columns:
                    trainer = TransformerTrainer(self.config)
                    sample_data = trainer.add_technical_indicators(
                        sample_data,
                        target_col=f"{self.config['tickers']['primary'][0].lower()}_close"
                    )
                
                processor.fit(sample_data)
                self.metrics.PROCESSOR_INIT.labels(mode='backtest').inc()
            
            return processor
            
        except Exception as e:
            self.metrics.PROCESSOR_INIT_FAILURES.labels(stage='initialization').inc()
            logger.error(f"Feature processor failed: {str(e)}")
            raise RuntimeError("Feature processor initialization failed") from e

    def _setup_hardware(self) -> None:
        """Optimize hardware configuration"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure"""
        required = {
            'rl': ['policy', 'learning_rate', 'buffer_size'],
            'risk': ['max_drawdown', 'position_limits'],
            'trading': ['tickers', 'initial_balance']
        }
        
        for section, keys in required.items():
            if section not in config:
                raise ValueError(f"Missing config section: {section}")
                
            missing = [k for k in keys if k not in config[section]]
            if missing:
                logger.warning(f"Missing recommended keys in {section}: {missing}")

    def _initialize_model(self) -> SAC:
        """Initialize SAC model with proper configuration for dictionary observations"""
        policy_kwargs = {
            "net_arch": self._get_network_architecture(),
            "features_extractor_class": FinancialFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "optimizer_class": torch.optim.AdamW,
            "optimizer_kwargs": {"weight_decay": 1e-6}
        }
        
        return SAC(
            policy="MultiInputPolicy",  # Changed from MlpPolicy
            env=self.env,
            learning_rate=self.config['rl']['learning_rate'],
            buffer_size=self.config['rl']['buffer_size'],
            batch_size=self.config['rl']['batch_size'],
            gamma=self.config['rl']['gamma'],
            tau=self.config['rl']['tau'],
            policy_kwargs=policy_kwargs,
            device=self.device,
            tensorboard_log=self.config['rl'].get('tensorboard_dir'),
            verbose=1
        )

    def _initialize_model(self) -> SAC:
        """Initialize SAC model with proper configuration"""
        policy_kwargs = {
            "net_arch": self._get_network_architecture(),
            "features_extractor_class": FinancialFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "optimizer_class": torch.optim.AdamW,
            "optimizer_kwargs": {"weight_decay": 1e-6}
        }
        
        return SAC(
            policy=self.config['rl']['policy'],
            env=self.env,
            learning_rate=self.config['rl']['learning_rate'],
            buffer_size=self.config['rl']['buffer_size'],
            batch_size=self.config['rl']['batch_size'],
            gamma=self.config['rl']['gamma'],
            tau=self.config['rl']['tau'],
            policy_kwargs=policy_kwargs,
            device=self.device,
            tensorboard_log=self.config['rl'].get('tensorboard_dir'),
            verbose=1
        )

    def _record_initialization(self):
        """Record initialization metrics with proper error handling"""
        try:
            # Record agent initialization using the new metrics interface
            self.metrics.record_component_init(
                component=f"agent_{self.config['rl']['policy']}",
                success=True
            )
            
            # Record device-specific metrics
            device_info = {
                'type': self.device.type,
                'policy': self.config['rl']['policy'],
                'version': self.config.get('meta', {}).get('version', 'unknown')
            }
            
            if self.device.type == 'cuda' and torch.cuda.is_available():
                device_info.update({
                    'gpu_name': torch.cuda.get_device_name(0),
                    'gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB"
                })
                
                # Record GPU-specific metrics
                self.metrics.record_trade(
                    ticker='system',
                    direction='hardware',
                    status=f"gpu_{torch.cuda.get_device_name(0)}"
                )

            # Record the full device configuration
            self.metrics.record_position(
                ticker='system_config',
                size=1.0  # Using size field to indicate presence
            )
            
            logger.info(f"Agent initialized with config: {device_info}")

        except Exception as e:
            logger.error(f"Initialization metrics recording failed: {str(e)}")
            # Still record the failure
            self.metrics.record_component_init(
                component='agent_initialization',
                success=False
            )
            # Don't raise to prevent agent startup failure
            
    def _get_network_architecture(self) -> Dict[str, list]:
        """Get validated network architecture"""
        default = {'pi': [256, 256], 'qf': [256, 256]}
        
        if 'net_arch' not in self.config['rl']:
            return default
            
        arch = self.config['rl']['net_arch']
        if isinstance(arch, dict):
            return {**default, **arch}
        return {'pi': arch, 'qf': arch} if isinstance(arch, list) else default

    def decide(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Make trading decision with full metrics instrumentation
        
        Args:
            state: Current market state dictionary
            
        Returns:
            Normalized action array clipped to action space bounds
        """
        with self.metrics.HISTO_DECISION_TIME.time():
            try:
                # Get primary ticker from config
                primary_ticker = self.config['tickers']['primary'][0]
                
                # Process and predict
                processed_state = self._process_observation(state)
                action, uncertainty = self._predict_action(processed_state)
                
                # Risk management
                if self.safety_monitor.check(self.env) or uncertainty > 0.2:
                    action = action * 0.5  # Reduce position
                    self.metrics.record_risk_reduction(f"uncertainty_{uncertainty:.2f}")
                
                # Record metrics using configured ticker
                self.metrics.POSITION_GAUGE.labels(ticker=primary_ticker).set(float(action[0]))
                
                # Clip to action space bounds
                return np.clip(
                    action,
                    self.config['rl']['action_space']['low'],
                    self.config['rl']['action_space']['high']
                )
                
            except Exception as e:
                error_type = type(e).__name__
                self.metrics.DECISION_FAILURE.labels(error_type=error_type).inc()
                logger.error(f"Decision failed ({error_type}): {str(e)}", exc_info=True)
                
                # Return neutral action
                return np.zeros(self.env.action_space.shape[0])

    def _predict_action(self, state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """Predict action with uncertainty estimation"""
        # Base prediction
        action, _ = self.model.predict(state, deterministic=False)
        
        # Uncertainty estimation
        with torch.no_grad():
            actions = np.array([self.model.predict(state)[0] for _ in range(3)])
            uncertainty = np.std(actions, axis=0).mean()
            
        return action, uncertainty
    
    def _adjust_for_greeks(self, action: np.ndarray, greeks: Dict) -> np.ndarray:
        """Modify actions based on options Greeks"""
        return action * np.array([
            self.greek_weights['delta'] * (1 - abs(greeks['delta'])),
            self.greek_weights['theta'] * greeks['theta'],
            self.greek_weights['vega'] * greeks['vega']
        ])
        
    def train(self, total_timesteps: int, callback=None) -> None:
        """Phase-aware training with monitoring"""
        self._log_training_start(total_timesteps)
        
        try:
            for phase in [
                {"name": "warmup", "steps": min(10000, total_timesteps)},
                {"name": "main", "steps": max(0, total_timesteps - 10000)}
            ]:
                if phase["steps"] <= 0:
                    continue
                    
                self._clear_gpu_cache()
                self.model.learn(
                    total_timesteps=phase['steps'],
                    callback=callback,
                    reset_num_timesteps=False
                )
                
                if self.safety_monitor.check(self.env):
                    logger.warning("Training paused by circuit breaker")
                    break
                    
        finally:
            self._clear_gpu_cache()

    def save_model(self, path: str) -> None:
        """Save model with directory creation"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model with validation"""
        try:
            self.model = SAC.load(path, device=self.device)
            logger.info(f"Model loaded from {path}")
            self._verify_model_compatibility()
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def _verify_model_compatibility(self) -> None:
        """Ensure model matches environment"""
        if hasattr(self.env, 'observation_space'):
            model_obs = self.model.observation_space.shape
            env_obs = self.env.observation_space.shape
            if model_obs != env_obs:
                raise ValueError(
                    f"Model expects {model_obs} inputs but environment provides {env_obs}"
                )

    def _clear_gpu_cache(self) -> None:
        """Clear GPU memory and log status"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(
                f"GPU cache cleared | "
                f"Allocated: {torch.cuda.memory_allocated(self.device)/1e6:.1f}MB | "
                f"Cached: {torch.cuda.memory_reserved(self.device)/1e6:.1f}MB"
            )
    def _validate_rl_config(self) -> None:
        """Validate RL-specific config parameters"""
        required_rl_keys = [
            'policy', 'learning_rate', 'buffer_size',
            'batch_size', 'gamma', 'tau'
        ]
        
        # Check required keys exist
        for key in required_rl_keys:
            if key not in self.config['rl']:
                raise ValueError(f"Missing required RL config key: {key}")
        
        # Force MultiInputPolicy for dict observations
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            if self.config['rl']['policy'] != "MultiInputPolicy":
                logger.warning("Forcing policy to MultiInputPolicy for dictionary observations")
                self.config['rl']['policy'] = "MultiInputPolicy"
                
class FinancialFeaturesExtractor(BaseFeaturesExtractor):
    """Advanced feature extractor for financial time series with dict observations"""
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        # We need to calculate the total features dimension
        market_data_space = observation_space.spaces['market_data']
        portfolio_space = observation_space.spaces['portfolio']
        
        # Calculate input dimensions
        market_data_dim = market_data_space.shape[0] * market_data_space.shape[1]
        portfolio_dim = portfolio_space.shape[0]
        total_dim = market_data_dim + portfolio_dim
        
        super().__init__(observation_space, features_dim)
        
        # Temporal processing
        self.temporal_net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1))
        
        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.Tanh()
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Flatten and concatenate all observations
        market_data = observations['market_data'].flatten(start_dim=1)
        portfolio = observations['portfolio'].flatten(start_dim=1)
        combined = torch.cat([market_data, portfolio], dim=1)
        
        # Process through networks
        temporal = self.temporal_net(combined)
        return self.output_net(temporal)

class SafetyMonitor:
    """Real-time risk management system"""
    def __init__(self, config: Dict[str, Any]):
        self.max_drawdown = config['risk']['max_drawdown']
        self.position_limits = config['risk']['position_limits']
        self.active = False

    def check(self, env: gym.Env) -> bool:
        """Check risk thresholds"""
        if self.active:
            return True
            
        current_drawdown = getattr(env, '_current_drawdown', lambda: 0)()
        if current_drawdown > self.max_drawdown:
            logger.critical(f"Drawdown breach: {current_drawdown:.2%}")
            self.active = True
            return True
            
        positions = getattr(env, 'positions', np.array([0]))
        if np.max(np.abs(positions)) > self.position_limits['max']:
            logger.critical("Position limit breached")
            self.active = True
            return True
            
        return False
