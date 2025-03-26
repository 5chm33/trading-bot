from ray import tune
import logging
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from reinforcement_learning.trading_env import TradingEnv
from reinforcement_learning.rl_agent import RLAgent
from utils.custom_logging import setup_logger
import os
import yfinance as yf
import pandas as pd

logger = setup_logger()

# Copy and paste the fetch_historical_data function here
def fetch_historical_data(symbol, start_date, end_date, interval="1d"):
    """
    Fetch historical price data using yfinance.

    Args:
        symbol (str): Ticker symbol (e.g., "AAPL").
        start_date (str): Start date in "YYYY-MM-DD" format.
        end_date (str): End date in "YYYY-MM-DD" format.
        interval (str): Timeframe for the data (e.g., "1d").

    Returns:
        pd.DataFrame: Historical price data.
    """
    try:
        # Fetch historical data
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

        # Add volatility column
        data[f"Volatility_{symbol}"] = data["Close"].pct_change().rolling(window=14).std()

        # Drop NaN values
        data.dropna(inplace=True)

        return data
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None

def train_with_ray_tune_rl(config, checkpoint_dir=None):
    """Train the RL model with Ray Tune and save the best model."""
    try:
        # Initialize the environment
        env = TradingEnv(
            data=fetch_historical_data("AAPL", "2021-01-01", "2023-01-01"),
            close_column="Close",
            volatility_column="Volatility_AAPL"
        )

        # Initialize the RL agent with the suggested hyperparameters
        agent = RLAgent(
            env,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            ent_coef=config["ent_coef"],
            buffer_size=config["buffer_size"],
            tau=config["tau"],
            gamma=config["gamma"]
        )

        # Train the agent
        agent.train(total_timesteps=50000)

        # Evaluate the agent
        mean_reward, _ = evaluate_policy(agent.model, env, n_eval_episodes=10)

        # Save the best model
        if checkpoint_dir:
            agent.save(os.path.join(checkpoint_dir, "sac_trading_model"))

        # Report metrics to Ray Tune
        tune.report({"mean_reward": mean_reward})

    except Exception as e:
        logger.error(f"Error during training with Ray Tune: {e}", exc_info=True)
        raise
    finally:
        # Ensure Ray Tune receives a reward value even if an error occurs
        tune.report({"mean_reward": mean_reward if 'mean_reward' in locals() else -float('inf')})

def custom_trial_dirname_creator(trial):
    """Create a shorter trial directory name."""
    return f"trial_{trial.trial_id}"

if __name__ == "__main__":
    # Define the search space for RL hyperparameters
    search_space = {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "ent_coef": tune.loguniform(0.001, 0.1),
        "buffer_size": tune.choice([10000, 50000, 100000]),
        "tau": tune.uniform(0.001, 0.1),
        "gamma": tune.uniform(0.9, 0.9999),
    }

    # Set a custom log directory with a shorter path
    custom_log_dir = os.path.abspath("./ray_results")
    os.makedirs(custom_log_dir, exist_ok=True)

    # Run the hyperparameter search
    analysis = tune.run(
        train_with_ray_tune_rl,
        config=search_space,
        num_samples=200,  # Number of trials
        resources_per_trial={"cpu": 8, "gpu": 1},
        name="rl_tuning",
        metric="mean_reward",
        mode="max",
        trial_dirname_creator=custom_trial_dirname_creator,  # Shorten trial directory names
        storage_path=custom_log_dir  # Use storage_path instead of local_dir
    )

    # Print the best hyperparameters
    best_config = analysis.get_best_config(metric="mean_reward", mode="max")
    logger.info(f"Best hyperparameters: {best_config}")