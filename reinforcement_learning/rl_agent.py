import os
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, Any
from utils.custom_logging import setup_logger
import psutil

logger = setup_logger(__name__)

class RLAgent:
    def __init__(self, env, config: Dict[str, Any]):
        """Initialize with optimized settings and enhanced debugging"""
        self.config = config
        self.config['rl']['hyperparams']['learning_rate'] = float(self.config['rl']['hyperparams']['learning_rate'])
        self.env = env  # Store env reference for debugging
        rl_config = config['rl']['hyperparams']
        training_config = config['rl']['training']

        # Network architecture with debugging info
        policy_kwargs = {
            "net_arch": {
                "pi": rl_config['net_arch'],
                "qf": rl_config['net_arch']
            },
            "optimizer_class": torch.optim.AdamW,
            "optimizer_kwargs": {
                "eps": 1e-8,
                "weight_decay": 1e-6
            }
        }

        # Enhanced hardware logging
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"\n=== Agent Initialization ===")
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
        logger.info(f"RAM Available: {psutil.virtual_memory().available/1e9:.2f}GB")

        # Force GPU tensor allocation
        torch.set_default_tensor_type('torch.cuda.FloatTensor'
                                    if self.device.type == 'cuda'
                                    else 'torch.FloatTensor')

        # Model initialization with debug info
        logger.info(f"\n=== Model Configuration ===")
        logger.info(f"Policy: {rl_config['policy']}")
        logger.info(f"Network Architecture: {policy_kwargs['net_arch']}")
        logger.info(f"Learning Rate: {rl_config['learning_rate']}")
        logger.info(f"Batch Size: {rl_config['batch_size']}")

        self.model = SAC(
            policy=rl_config['policy'],
            env=env,
            learning_rate=rl_config['learning_rate'],
            buffer_size=rl_config['buffer_size'],
            batch_size=rl_config['batch_size'],
            tau=rl_config['tau'],
            gamma=rl_config['gamma'],
            ent_coef=rl_config['ent_coef'],
            target_entropy=rl_config.get('target_entropy', 'auto'),
            train_freq=training_config['train_freq'],
            gradient_steps=training_config['gradient_steps'],
            verbose=0,  # We use our own logging
            policy_kwargs=policy_kwargs,
            tensorboard_log=training_config['tensorboard_dir'],
            device=self.device
        )

        self._setup_gradient_clipping(rl_config['learning_rate'])
        logger.info("Agent initialized successfully\n")

    def _setup_gradient_clipping(self, lr: float):
        """Configure gradient clipping with debug logging"""
        clip_value = self.config['model']['transformer']['regularization']['gradient_clip']
        logger.debug(f"Setting gradient clipping to {clip_value}")

        for param in self.model.policy.parameters():
            param.register_hook(
                lambda grad: torch.nan_to_num(
                    grad.clamp_(-clip_value, clip_value),
                    nan=0.0, posinf=clip_value, neginf=-clip_value
                )
            )

        # Update optimizers with debug info
        self.model.actor.optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=lr,
            eps=1e-8,
            weight_decay=1e-6
        )
        self.model.critic.optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=lr,
            eps=1e-8,
            weight_decay=1e-6
        )

    def choose_action(self, state, deterministic: bool = True) -> np.ndarray:
        """Enhanced action selection with debugging"""
        action, _ = self.model.predict(state, deterministic=deterministic)
        clipped_action = np.clip(
            action,
            self.config['rl']['action_space']['low'],
            self.config['rl']['action_space']['high']
        )
        logger.debug(f"Action chosen: {clipped_action} (raw: {action})")
        return clipped_action

    def train(self, total_timesteps: int, callback=None, progress_bar: bool = True):
        """Complete training loop with parameter verification and enhanced logging"""
        try:
            # 1. Parameter verification
            logger.info("\n=== Training Parameter Verification ===")
            config_lr = float(self.config['rl']['training']['learning_rate'])
            config_batch = int(self.config['rl']['training']['batch_size'])

            logger.info(f"Config Training LR: {config_lr}")
            logger.info(f"Config Training Batch Size: {config_batch}")
            logger.info(f"Model Current LR: {self.model.learning_rate}")
            logger.info(f"Model Batch Size: {self.model.batch_size}")

            # Verify critical parameters match config
            if abs(self.model.learning_rate - config_lr) > 1e-8:
                logger.warning(f"Learning rate mismatch! Model: {self.model.learning_rate} vs Config: {config_lr}")

            if self.model.batch_size != config_batch:
                logger.warning(f"Batch size mismatch! Model: {self.model.batch_size} vs Config: {config_batch}")

            # 2. Training phases
            logger.info("\n=== Starting Warmup Phase (10,000 steps) ===")
            self.model.learn(
                total_timesteps=10000,
                callback=callback,
                log_interval=10,
                progress_bar=progress_bar
            )

            # 3. Main training loop
            logger.info("\n=== Starting Main Training ===")
            for chunk in range(total_timesteps // 10000):
                chunk_start = chunk * 10000
                chunk_end = (chunk + 1) * 10000

                # Log reward components before each chunk
                if callback and hasattr(callback, 'last_reward_components'):
                    components = callback.last_reward_components
                    self.model.logger.record("reward/tech_adjusted", components.get('tech_adjusted', 0))
                    self.model.logger.record("reward/rsi_adjustment", components.get('rsi_adj', 0))
                    self.model.logger.record("reward/macd_signal", components.get('macd_signal', 0))
                    self.model.logger.record("reward/sharpe", components.get('sharpe', 0))
                    self.model.logger.record("reward/raw", components.get('raw', 0))

                logger.info(f"\n--- Training Chunk {chunk + 1} ({chunk_start}-{chunk_end}) ---")
                logger.info(f"Current Portfolio Value: ${self.env.portfolio_value:.2f}")

                # Train on this chunk
                self.model.learn(
                    total_timesteps=10000,
                    callback=callback,
                    reset_num_timesteps=False,
                    log_interval=10,
                    progress_bar=progress_bar
                )

                # Force log metrics at chunk boundaries
                self.model.logger.dump()

                # Periodic parameter verification
                if chunk % 5 == 0:  # Every 50,000 steps
                    logger.info("\n=== Parameter Check ===")
                    logger.info(f"Current LR: {self.model.learning_rate}")
                    logger.info(f"Current batch size: {self.model.batch_size}")
                    logger.info(f"Buffer size: {len(self.model.replay_buffer)}")

            # 4. Final reporting
            logger.info("\n=== Training Complete ===")
            logger.info(f"Final Portfolio Value: ${self.env.portfolio_value:.2f}")
            logger.info(f"Total Steps Completed: {self.model.num_timesteps}")

            # Final reward component logging
            if callback and hasattr(callback, 'last_reward_components'):
                components = callback.last_reward_components
                logger.info("\n=== Final Reward Components ===")
                logger.info(f"Tech-Adjusted Return: {components.get('tech_adjusted', 0):.4f}")
                logger.info(f"RSI Adjustment: {components.get('rsi_adj', 0):.4f}")
                logger.info(f"MACD Signal: {components.get('macd_signal', 0):.4f}")

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    def save(self, path: str):
        """Enhanced model saving with validation"""
        try:
            self.model.save(path)
            logger.info(f"Model successfully saved to {path}")
            # Verify save
            if not os.path.exists(path + ".zip"):
                raise FileNotFoundError(f"Model file not found at {path}.zip")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}", exc_info=True)
            raise

    def load(self, path: str, env=None):
        """Enhanced model loading with validation"""
        try:
            if not os.path.exists(path + ".zip"):
                raise FileNotFoundError(f"Model file not found at {path}.zip")

            self.model = SAC.load(
                path,
                env=env if env else self.model.env,
                device=self.device
            )
            self._setup_gradient_clipping(self.model.learning_rate)
            logger.info(f"Model successfully loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
