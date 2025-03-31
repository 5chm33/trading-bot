import os
import numpy as np
from typing import Dict, Tuple
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    CheckpointCallback
)
from stable_baselines3.common.type_aliases import GymEnv
from src.models.rl.env import TradingEnv
from src.utils.config_validation import SACParamValidator, MemoryMonitor
from src.utils.logging import setup_logger

# Disable unnecessary TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logger = setup_logger(__name__)

class EnhancedTrainingCallback(BaseCallback):
    """Tracks Sharpe ratio and saves best model"""
    def __init__(self, config: Dict, eval_env: GymEnv):
        super().__init__()
        self.config = config
        self.eval_env = eval_env
        self.best_sharpe = -np.inf

    def _on_step(self) -> bool:
        if len(self.locals['infos']) > 0:  # Safe access
            current_sharpe = self.locals['infos'][0].get('sharpe', 0)
            if current_sharpe > self.best_sharpe:
                self.best_sharpe = current_sharpe
                self.model.save(f"models/best_sharpe_{current_sharpe:.2f}")
        return True

def initialize_envs(config: Dict) -> Tuple[GymEnv, GymEnv]:
    """Creates training and evaluation environments"""
    from src.data.processors.data_utils import DataFetcher
    from src.utils.data_schema import ColumnSchema
    
    fetcher = DataFetcher(config)
    train_data, eval_data = fetcher.prepare_dataset()
    
    # Validate data schema
    ColumnSchema.validate(train_data, config['tickers']['primary'])
    ColumnSchema.validate(eval_data, config['tickers']['primary'])
    
    return TradingEnv(train_data, config), TradingEnv(eval_data, config)

def train_agent(config: Dict) -> SAC:
    """Complete training pipeline with version-safe implementations"""
    # Validate and prepare
    config = SACParamValidator.validate(config)
    train_env, eval_env = initialize_envs(config)
    
    # Callback setup
    callbacks = CallbackList([
        EnhancedTrainingCallback(config, eval_env),
        EvalCallback(
            eval_env,
            best_model_save_path="models/",
            eval_freq=config['rl']['evaluation']['freq'],
            deterministic=True,
            verbose=1
        ),
        CheckpointCallback(
            save_freq=config['rl']['checkpointing']['save_freq'],
            save_path="checkpoints/",
            name_prefix="rl_model"
        ),
        MemoryMonitor()
    ])
    
    # Model initialization (SB3 2.5.0 + PyTorch 2.6.0 optimized)
    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=config['rl']['hyperparams']['sac']['learning_rate'],
        buffer_size=config['rl']['hyperparams']['sac']['buffer_size'],
        batch_size=config['rl']['hyperparams']['sac']['batch_size'],
        tau=config['rl']['hyperparams']['sac']['tau'],
        gamma=config['rl']['hyperparams']['sac']['gamma'],
        ent_coef=config['rl']['hyperparams']['sac']['ent_coef'],
        target_entropy=config['rl']['hyperparams']['sac']['target_entropy'],
        use_sde=config['rl']['hyperparams']['sac']['use_sde'],
        sde_sample_freq=config['rl']['hyperparams']['sac']['sde_sample_freq'],
        tensorboard_log=config['rl']['training']['tensorboard_dir'],
        policy_kwargs=config['rl']['algorithm']['policy_kwargs'],
        device="cuda" if torch.cuda.is_available() else "auto",
        verbose=config['rl']['training']['verbose'],
        seed=config['rl']['training']['seed']
    )
    
    # Training loop
    try:
        model.learn(
            total_timesteps=config['rl']['training']['total_steps'],
            callback=callbacks,
            progress_bar=True,
            tb_log_name=f"SAC_{config['meta']['version']}"
        )
    except KeyboardInterrupt:
        model.save("models/interrupted_model")
        logger.warning("Training interrupted - model saved")
    finally:
        train_env.close()
        eval_env.close()
    
    return model

if __name__ == "__main__":
    from src.utils.config_loader import ConfigLoader
    
    try:
        config = ConfigLoader.load_config("config/config.yaml")
        trained_model = train_agent(config)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        raise