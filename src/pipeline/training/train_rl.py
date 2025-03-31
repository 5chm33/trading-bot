# src/training/train_rl.py
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    CallbackList
)
from src.models.rl.env import TradingEnv
from src.data.processors.data_utils import DataFetcher
from src.utils.logging import setup_logger
from src.utils.data_schema import ColumnSchema

logger = setup_logger(__name__)

class EnhancedTrainingCallback:
    """Advanced training monitoring with regime awareness"""
    
    def __init__(self, config: Dict, eval_env: TradingEnv):
        self.config = config
        self.eval_env = eval_env
        self.best_sharpe = -np.inf
        
    def __call__(self, locals_, globals_):
        # Track portfolio metrics
        info = locals_['infos'][0]
        current_sharpe = info.get('sharpe', 0)
        
        # Save best model
        if current_sharpe > self.best_sharpe:
            self.best_sharpe = current_sharpe
            locals_['self'].save(f"models/best_sharpe_{current_sharpe:.2f}")
            
        return True

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load and validate configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    for section in ['rl', 'trading', 'time_settings']:
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    
    return config

def initialize_envs(config: Dict) -> Tuple[TradingEnv, TradingEnv]:
    """Prepare training and evaluation environments"""
    fetcher = DataFetcher(config)
    train_data, eval_data = fetcher.prepare_dataset()
    
    # Validate data schema
    ColumnSchema.validate(train_data, config['tickers']['primary'])
    ColumnSchema.validate(eval_data, config['tickers']['primary'])
    
    return (
        TradingEnv(train_data, config),
        TradingEnv(eval_data, config)
    )

def train_agent(config: Dict):
    """Complete RL training pipeline"""
    # Initialize environments
    train_env, eval_env = initialize_envs(config)
    
    # Setup callbacks
    callbacks = CallbackList([
        EnhancedTrainingCallback(config, eval_env),
        EvalCallback(
            eval_env,
            best_model_save_path="models/",
            eval_freq=config['rl']['training']['eval_freq'],
            deterministic=True
        ),
        CheckpointCallback(
            save_freq=10000,
            save_path="checkpoints/",
            name_prefix="rl_model"
        )
    ])
    
    # Initialize agent
    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log="logs/tensorboard",
        **config['rl']['hyperparams']
    )
    
    # Train
    logger.info(f"Starting training for {config['rl']['training']['total_steps']} steps")
    model.learn(
        total_timesteps=config['rl']['training']['total_steps'],
        callback=callbacks,
        tb_log_name="sac_trading"
    )
    
    # Cleanup
    train_env.close()
    eval_env.close()
    return model

def validate_config(config: dict) -> bool:
    """Ensure required config sections exist"""
    required = {
        'time_settings': ['train', 'test'],
        'tickers': ['primary']
    }
    
    for section, subsections in required.items():
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
        if isinstance(subsections, list):
            for sub in subsections:
                if sub not in config[section]:
                    raise ValueError(f"Missing {section}.{sub} in config")
    return True

def main():
    """Entry point for RL training"""
    try:
        config = load_config()
        validate_config(config)
        train_agent(config)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.critical(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()