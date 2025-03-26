import numpy as np
import yaml
import os
import pandas as pd
import yfinance as yf
from reinforcement_learning.trading_env import TradingEnv
from reinforcement_learning.rl_agent import RLAgent
from utils.custom_logging import setup_logger
from stable_baselines3.common.callbacks import BaseCallback

logger = setup_logger()

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if "episode" in self.locals:
            episode_reward = self.locals["episode"]["r"]
            self.episode_rewards.append(episode_reward)
            logger.info(f"Episode: {len(self.episode_rewards)}, Total Reward: {episode_reward}")
        return True

def load_config(config_path):
    """Load the configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def fetch_historical_data(symbol, start_date, end_date, interval="1d"):
    """Fetch historical price data using yfinance."""
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        data[f"Volatility_{symbol}"] = data["Close"].pct_change().rolling(window=14).std()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None

if __name__ == "__main__":
    # Load the config file
    config_path = os.path.join("config", "config.yaml")
    config = load_config(config_path)

    # Fetch historical price data
    symbol = "AAPL"  # Ticker symbol
    start_date = "2021-01-01"  # Start date
    end_date = "2023-01-01"  # End date

    data = fetch_historical_data(symbol, start_date, end_date)
    if data is None:
        raise ValueError("Failed to fetch historical data. Check the yfinance connection.")

    # Initialize the environment
    env = TradingEnv(
        data=data,
        close_column="Close",
        volatility_column=f"Volatility_{symbol}",
    )

    # Define the best hyperparameters from Ray Tune
    best_hyperparams = {
        "learning_rate": 0.00012279146367978207,
        "batch_size": 128,
        "ent_coef": 0.01989499304974788,
        "buffer_size": 10000,
        "tau": 0.09839764358193134,
        "gamma": 0.9868272831478102
    }

    # Initialize the RL agent with the best hyperparameters
    agent = RLAgent(
        env,
        learning_rate=best_hyperparams["learning_rate"],
        batch_size=best_hyperparams["batch_size"],
        ent_coef=best_hyperparams["ent_coef"],
        buffer_size=best_hyperparams["buffer_size"],
        tau=best_hyperparams["tau"],
        gamma=best_hyperparams["gamma"]
    )

    # Train the RL agent with a custom callback
    callback = CustomCallback()
    agent.train(total_timesteps=100000, callback=callback)

    # Save the trained model
    agent.save("sac_trading_model")
    logger.info("Training complete. Model saved to 'sac_trading_model'.")