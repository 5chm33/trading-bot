import os
import yaml
import numpy as np
import pandas as pd
from reinforcement_learning.trading_env import TradingEnv
from reinforcement_learning.rl_agent import RLAgent
from utils.custom_logging import setup_logger
from stable_baselines3.common.callbacks import BaseCallback
from models.transformer_trainer import TransformerTrainer
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = setup_logger(__name__)

class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        if "episode" in self.locals:
            reward = self.locals["episode"]["r"]
            self.episode_rewards.append(reward)
            logger.info(f"Episode reward: {reward:.2f}")
        return True

def load_config():
    config_path = os.path.join("config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def prepare_training_data(config):
    """Process all tickers and combine data with validation"""
    all_data = []
    
    for ticker in config['tickers']:
        try:
            # Initialize with empty data - will be filled by TransformerTrainer
            trainer = TransformerTrainer(pd.DataFrame(), config, f"{ticker.lower()}_close")
            data = trainer.process_single_ticker(ticker)
            
            if data is not None:
                all_data.append(data)
                logger.info(f"Processed {ticker} | Shape: {data.shape}")
        except Exception as e:
            logger.error(f"Failed {ticker}: {str(e)}", exc_info=True)
    
    if not all_data:
        raise ValueError("No valid ticker data processed")
    
    combined = pd.concat(all_data, axis=1).ffill().bfill()
    return combined

def main():
    try:
        # Load and validate config
        config = load_config()
        required_keys = ['tickers', 'rl', 'trading']
        if not all(k in config for k in required_keys):
            raise ValueError(f"Missing required config keys: {required_keys}")

        # Prepare data
        data = prepare_training_data(config)
        logger.info(f"Final training data shape: {data.shape}")

        # Initialize environment
        env = TradingEnv(
            data=data,
            tickers=config['tickers'],
            config=config
        )

        # Initialize agent with full config
        agent = RLAgent(env=env, config=config)

        # Training parameters from config
        training_config = config['rl']['training']
        total_steps = training_config['total_steps']
        eval_freq = training_config['eval_freq']

        # Train with checkpointing
        best_reward = -np.inf
        for chunk in range(0, total_steps, eval_freq):
            agent.train(
                total_timesteps=min(eval_freq, total_steps - chunk),
                callback=TrainingCallback()
            )
            
            # Save checkpoint
            agent.save(f"checkpoint_step_{chunk + eval_freq}")

        # Final save
        agent.save("final_model")
        logger.info("Training completed successfully")

    except Exception as e:
        logger.critical(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()