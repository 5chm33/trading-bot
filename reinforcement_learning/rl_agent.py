import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, Any
from utils.custom_logging import setup_logger

logger = setup_logger(__name__)

class RLAgent:
    def __init__(self, env, config: Dict[str, Any]):
        """Initialize RL agent with dynamic config."""
        rl_config = config['rl']['hyperparams']
        training_config = config['rl']['training']
        
        self.model = SAC(
            policy=rl_config['policy'],
            env=env,
            learning_rate=rl_config['learning_rate'],
            buffer_size=rl_config['buffer_size'],
            batch_size=rl_config['batch_size'],
            tau=rl_config['tau'],
            gamma=rl_config['gamma'],
            ent_coef=rl_config['ent_coef'],
            verbose=config['execution']['logging']['level'] == 'DEBUG',
            policy_kwargs={
                "net_arch": dict(pi=rl_config['net_arch'], qf=rl_config['net_arch']),
                "optimizer_class": torch.optim.AdamW,
                "optimizer_kwargs": {
                    "eps": 1e-8,
                    "weight_decay": 1e-6
                }
            },
            target_entropy="auto",
            use_sde=False,
            train_freq=(training_config['train_freq'], "step"),
            gradient_steps=training_config['gradient_steps'],
            tensorboard_log=config['execution']['logging']['tensorboard_dir']
        )
        
        self._setup_gradient_clipping(rl_config['learning_rate'])
    
    def _setup_gradient_clipping(self, lr: float):
        """Configure gradient clipping from config."""
        clip_value = self.config['model']['transformer']['regularization']['gradient_clip']
        
        for param in self.model.policy.parameters():
            param.register_hook(
                lambda grad: torch.nan_to_num(
                    grad.clamp_(-clip_value, clip_value),
                    nan=0.0, posinf=clip_value, neginf=-clip_value
                )
            )
        
        optimizer_kwargs = {
            "lr": lr,
            "eps": 1e-8,
            "weight_decay": 1e-6
        }
        
        self.model.actor.optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            **optimizer_kwargs
        )
        self.model.critic.optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            **optimizer_kwargs
        )

    def choose_action(self, state, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(state, deterministic=deterministic)
        return np.clip(
            action,
            self.config['rl']['action_space']['low'],
            self.config['rl']['action_space']['high']
        )

    def train(self, total_timesteps: int = None, callback=None, progress_bar: bool = True):
        """Train with config-based parameters."""
        if total_timesteps is None:
            total_timesteps = self.config['rl']['training']['total_steps']
            
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=progress_bar,
                reset_num_timesteps=False,
                tb_log_name="sac_trading"
            )
        except Exception as e:
            logger.error(f"Training interrupted: {str(e)}", exc_info=True)
            self.save("autosave_interrupted_model")
            raise

    def save(self, path: str):
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}", exc_info=True)
            raise

    def load(self, path: str, env=None):
        self.model = SAC.load(
            path,
            env=env if env else self.model.env,
            device=self.config['execution']['device']
        )
        self._setup_gradient_clipping(self.model.learning_rate)