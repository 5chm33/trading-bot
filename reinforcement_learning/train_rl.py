import os
import sys
import yaml
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
    CheckpointCallback
)
from reinforcement_learning.trading_env import TradingEnv
from reinforcement_learning.rl_agent import RLAgent
from utils.custom_logging import setup_logger
from models.transformer_trainer import TransformerTrainer
from data_preprocessing.fetch_data import fetch_data
import warnings
import torch
import psutil

# Initialize custom logger
logger = setup_logger(__name__)
warnings.filterwarnings('ignore')

# Enhanced hardware logging
logger.info(f"\n=== System Configuration ===")
logger.info(f"Python Version: {sys.version}")
logger.info(f"PyTorch Version: {torch.__version__}")
logger.info(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
logger.info(f"RAM Available: {psutil.virtual_memory().available/1e9:.2f}GB")

class TrainingCallback(BaseCallback):
    def __init__(self, config, verbose=0):
        super().__init__(verbose)
        self.config = config
        self.episode_rewards = []
        self.reward_history = []  # Track all rewards for metrics
        self.tech_feature_history = []  # Track technical indicators
        self.position_history = []  # Track position sizes
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        try:
            # Access the environment
            env = self.training_env.envs[0].env

            # Store rewards and metrics
            if "reward" in self.locals:
                reward = self.locals["reward"][0]
                self.reward_history.append(reward)
                self.episode_rewards.append(reward)

                # Store technical features if available
                if "infos" in self.locals and len(self.locals["infos"]) > 0:
                    info = self.locals["infos"][-1]
                    if 'reward_components' in info:
                        self.last_reward_components = info['reward_components']
                    self.position_history.append(info.get('positions', []))
                    if 'technical_features' in info:
                        self.tech_feature_history.append(info['technical_features'])

            # Enhanced logging every 100 steps
            if self.num_timesteps % 100 == 0 and len(self.reward_history) > 0:
                recent_rewards = self.reward_history[-100:]
                recent_positions = self.position_history[-100:] if self.position_history else []

                logger.info(
                    f"\n=== Training Metrics ({self.num_timesteps} steps) ==="
                    f"\nAvg Reward (last 100): {np.mean(recent_rewards):.4f}"
                    f"\nReward StdDev: {np.std(recent_rewards):.4f}"
                    f"\nMax Position Size: {np.max(np.abs(np.concatenate(recent_positions))) if recent_positions else 0:.2f}"
                    f"\nAvg Transaction Costs: {env.transaction_cost * np.mean([np.sum(np.abs(p)) for p in recent_positions]) if recent_positions else 0:.4f}"
                    f"\nCurrent Portfolio Value: ${env.portfolio_value:.2f}"
                    f"\nTechnical Features (avg):"
                    f"\n  RSI: {np.mean([t[0] for t in self.tech_feature_history[-100:]]) if self.tech_feature_history else 0:.2f}"
                    f"\n  MACD: {np.mean([t[1] for t in self.tech_feature_history[-100:]]) if self.tech_feature_history else 0:.2f}"
                    f"\n  ATR: {np.mean([t[2] for t in self.tech_feature_history[-100:]]) if self.tech_feature_history else 0:.2f}"
                )

            # Early stopping logic
            if reward > self.best_reward:
                self.best_reward = reward
            elif (self.best_reward - reward) > self.config['rl']['training']['early_stopping']['min_reward']:
                logger.warning(f"Early stopping triggered at step {self.num_timesteps}")
                return False

        except Exception as e:
            logger.error(f"Callback error: {str(e)}", exc_info=True)

        return True

def load_config():
    """Enhanced config loading with validation and type conversion"""
    try:
        config_path = os.path.join("config", "config.yaml")
        logger.info(f"Loading config from {config_path}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate and convert RL training parameters
        rl_training = config['rl']['training']
        rl_training['learning_rate'] = float(rl_training['learning_rate'])
        rl_training['batch_size'] = int(rl_training['batch_size'])
        rl_training['gradient_steps'] = int(rl_training['gradient_steps'])
        rl_training['total_steps'] = int(rl_training['total_steps'])

        # Convert hyperparams
        rl_hyper = config['rl']['hyperparams']
        rl_hyper['learning_rate'] = float(rl_hyper['learning_rate'])
        rl_hyper['batch_size'] = int(rl_hyper['batch_size'])
        rl_hyper['buffer_size'] = int(rl_hyper['buffer_size'])

        logger.info("Config validation and type conversion complete")
        return config

    except Exception as e:
        logger.error(f"Config loading failed: {str(e)}", exc_info=True)
        raise

def prepare_training_data(config):
    """Enhanced data preparation with progress tracking"""
    logger.info("\n=== Preparing Training Data ===")
    all_data = []

    for ticker in config['tickers']:
        try:
            logger.info(f"Processing {ticker}...")
            train_cfg = config['time_settings']['train']
            data = fetch_data(
                ticker,
                train_cfg['start_date'],
                train_cfg['end_date'],
                config['time_settings']['interval']
            )

            if data is not None:
                trainer = TransformerTrainer(data, config, f"{ticker.lower()}_close")
                processed_data = trainer.add_technical_indicators(data, ticker.lower())
                if processed_data is not None:
                    processed_data = trainer.handle_missing_data(processed_data)
                    all_data.append(processed_data)
                    logger.info(f"Processed {ticker} | Shape: {processed_data.shape}")

        except Exception as e:
            logger.error(f"Failed {ticker}: {str(e)}", exc_info=True)

    if not all_data:
        raise ValueError("No valid ticker data processed")

    combined_data = pd.concat(all_data, axis=1).ffill().bfill()
    logger.info(f"\n=== Data Summary ===")
    logger.info(f"Final data shape: {combined_data.shape}")
    logger.info(f"Date range: {combined_data.index[0]} to {combined_data.index[-1]}")
    logger.info(f"Features per ticker: {len([c for c in combined_data.columns if c.startswith(config['tickers'][0].lower()+'_')])}")

    return combined_data

def main():
    try:
        # Load and log config
        config = load_config()
        logger.info(f"\n=== Training Configuration ===")
        logger.info(f"Tickers: {config['tickers']}")
        logger.info(f"Time Range: {config['time_settings']['train']['start_date']} to {config['time_settings']['train']['end_date']}")
        logger.info(f"Total Steps: {config['rl']['training']['total_steps']}")

        # Prepare data
        data = prepare_training_data(config)

        # Initialize environments
        logger.info("\nInitializing environments...")
        env = TradingEnv(data=data, tickers=config['tickers'], config=config)
        eval_env = TradingEnv(data=data, tickers=config['tickers'], config=config)
        logger.info("Environments initialized successfully")

        # Initialize agent
        logger.info("\nInitializing agent...")
        agent = RLAgent(env=env, config=config)

        # Setup callbacks
        logger.info("\nSetting up callbacks...")
        training_callback = TrainingCallback(config)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="checkpoints/",
            log_path="logs/",
            eval_freq=config['rl']['training']['eval_freq'],
            deterministic=True,
            render=False,
            verbose=0
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="checkpoints/",
            name_prefix="rl_model"
        )

        # Start training
        logger.info("\n=== Starting Training ===")
        agent.train(
            total_timesteps=config['rl']['training']['total_steps'],
            callback=CallbackList([training_callback, eval_callback, checkpoint_callback]),
            progress_bar=True
        )

        # Final save and cleanup
        agent.save("final_model")
        logger.info("\n=== Training Complete ===")
        logger.info(f"Final Portfolio Value: {env.portfolio_value:.2f}")

    except Exception as e:
        logger.critical(f"\n!!! Training Failed !!!\nReason: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
