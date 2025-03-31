from ray import tune
import logging
import os
import numpy as np
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error
from utils.custom_logging import setup_logger
from ray.tune.experiment import Trial

logger = setup_logger()

def kalman_tuning(config, y_pred, y_true):
    """
    Custom objective function for tuning the Kalman Filter.
    """
    # Apply the Kalman Filter with the given parameters
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=y_pred[0],
        initial_state_covariance=1.0,
        observation_covariance=config["observation_covariance"],
        transition_covariance=config["transition_covariance"],
    )
    y_pred_smoothed, _ = kf.filter(y_pred)

    # Calculate the evaluation metric (e.g., MSE)
    mse = mean_squared_error(y_true, y_pred_smoothed)
    return mse

def train_with_ray_tune_filter(config, y_pred, y_true):
    """
    Train the Kalman Filter with Ray Tune.
    """
    # Evaluate the Kalman Filter with the given configuration
    mse = kalman_tuning(config, y_pred, y_true)

    # Report the metric to Ray Tune
    tune.report({"mse": mse})

def custom_trial_dirname_creator(trial: Trial) -> str:
    """
    Create a shorter trial directory name to avoid Windows path length issues.
    """
    return f"trial_{trial.trial_id}"

def main(y_pred, y_true):
    # Define the search space for Kalman Filter parameters
    search_space = {
        "observation_covariance": tune.loguniform(0.01, 1.0),  # Range for observation noise
        "transition_covariance": tune.loguniform(0.001, 0.1),  # Range for process noise
    }

    # Set a custom log directory with a shorter path
    custom_log_dir = os.path.abspath("./ray_results_kalman")
    os.makedirs(custom_log_dir, exist_ok=True)

    # Run the hyperparameter search
    analysis = tune.run(
        tune.with_parameters(train_with_ray_tune_filter, y_pred=y_pred, y_true=y_true),
        config=search_space,
        num_samples=50,  # Number of trials
        metric="mse",
        mode="min",  # Minimize the MSE
        trial_dirname_creator=custom_trial_dirname_creator,  # Shorten trial directory names
        storage_path=custom_log_dir  # Use a custom storage path
    )

    # Get the best configuration
    best_config = analysis.get_best_config(metric="mse", mode="min")
    print(f"Best Kalman Filter parameters: {best_config}")
    return best_config

if __name__ == "__main__":
    # Example usage
    y_pred = np.random.rand(100)  # Replace with your model's predictions
    y_true = np.random.rand(100)  # Replace with the actual values

    # Run the Kalman Filter tuning
    best_kalman_config = main(y_pred, y_true)