import os
import sys
import json
import logging
import yaml

# Third-party imports
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.experiment import Trial
from bayes_opt import BayesianOptimization

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Custom module imports
from utils.custom_logging import setup_logger
from data_preprocessing.fetch_data import fetch_data
from data_preprocessing.combine_data import combine_data
from models.transformer_trainer import TransformerTrainer
from config.search_space import get_search_space

# Set up the logger
logger = setup_logger()

def custom_trial_dirname_creator(trial: Trial) -> str:
    """Custom function to create shorter trial directory names."""
    return f"trial_{trial.trial_id}"

def load_config(config_path):
    """
    Load the configuration from a YAML file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Loaded configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def fetch_and_combine_data(tickers, start_date, end_date, interval, config):
    """
    Fetch and combine data for multiple tickers.
    """
    all_data = []
    for ticker in tickers:
        logger.info(f"Fetching data for {ticker}.")
        data = fetch_data(ticker, start_date, end_date, interval)
        if not data.empty:
            logger.info(f"Data fetched for {ticker}. Shape: {data.shape}")
            all_data.append(data)
        else:
            logger.warning(f"No data fetched for {ticker}. Skipping.")

    if not all_data:
        logger.error("No data fetched for any ticker. Check the tickers and dates.")
        return None

    # Combine data for all tickers
    try:
        combined_data = combine_data(all_data, config)
        if combined_data is None or combined_data.empty:
            logger.error("Failed to combine data.")
            return None

        logger.info(f"Combined data shape: {combined_data.shape}")
        return combined_data
    except Exception as e:
        logger.error(f"Error combining data: {e}")
        return None

def build_model(config, num_features):
    """
    Build the Transformer-based model.

    Args:
        config (dict): Hyperparameter configuration.
        num_features (int): Number of features in the input data.

    Returns:
        tf.keras.Model: The compiled model.
    """
    try:
        # Ensure sequence_length is in the config
        sequence_length = config.get("sequence_length", 30)  # Default to 30 if not provided

        # Input layer
        inputs = Input(shape=(sequence_length, num_features))
        
        # Transformer layers
        x = MultiHeadAttention(
            num_heads=config.get("num_heads", 2),  # Default to 2 if not provided
            key_dim=num_features
        )(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(config.get("dropout_rate", 0.1))(x)
        
        # Global average pooling to reduce sequence to a single value
        x = GlobalAveragePooling1D()(x)
        
        # Output layer
        outputs = Dense(1, activation='linear', kernel_regularizer=l2(config.get("l2_reg", 0.01)))(x)
        
        # Create the model
        model = Model(inputs, outputs)
        return model
    except Exception as e:
        logger.error(f"Error building model: {e}", exc_info=True)
        return None

def optimize_hyperparameters(X_train, y_train, X_val, y_val, num_features):
    """
    Optimize hyperparameters using Bayesian Optimization.

    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        X_val (np.array): Validation features.
        y_val (np.array): Validation labels.
        num_features (int): Number of features in the input data.

    Returns:
        dict: Best hyperparameters found.
    """
    def train_model(learning_rate, dropout_rate, l2_reg):
        # Build and train the model with the given hyperparameters
        model = build_model(
            {"learning_rate": learning_rate, "dropout_rate": dropout_rate, "l2_reg": l2_reg},
            num_features
        )
        if model is None:
            return -1e4  # Return a large negative value if model building fails

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,  # Use a small number of epochs for quick evaluation
            batch_size=32,  # Use a fixed batch size for simplicity
            verbose=0
        )

        # Evaluate the model on the validation set
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        return -val_loss  # Minimize loss

    # Define the parameter bounds
    pbounds = {
        'learning_rate': (1e-6, 1e-3),
        'dropout_rate': (0.1, 0.5),
        'l2_reg': (1e-4, 1e-2),
    }

    # Run Bayesian Optimization
    optimizer = BayesianOptimization(f=train_model, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=5, n_iter=25)

    # Get the best hyperparameters
    best_params = optimizer.max['params']
    return best_params

def main():
    """
    Main function to run the hyperparameter tuning pipeline.
    """
    try:
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
        config = load_config(config_path)

        # Get the list of tickers from the config
        tickers = config.get("tickers", [])
        if not tickers:
            raise ValueError("No tickers found in the config file.")

        # Fetch and combine data for all tickers
        combined_data = fetch_and_combine_data(
            tickers, config['start_date'], config['end_date'], config['interval'], config
        )

        # Initialize the trainer with the combined data
        close_column = f"Close_{tickers[0]}"  # Use the first ticker's close column
        trainer = TransformerTrainer(combined_data, config, close_column=close_column)

        # Add technical indicators for each ticker
        for ticker in tickers:
            logger.info(f"Adding technical indicators for {ticker}.")
            combined_data = trainer.add_technical_indicators(combined_data, ticker)

        # Handle missing data
        combined_data = trainer.handle_missing_data(combined_data)

        # Prepare data for training
        logger.info("Preparing data for training.")
        prepared_data = trainer.prepare_data(combined_data, lookback=config['lookback'])
        if prepared_data is None:
            logger.error("Data preparation failed. Cannot proceed with hyperparameter tuning.")
            return

        X_train, X_val, y_train, y_val, _, _, num_features = prepared_data

        # Optimize hyperparameters using Bayesian Optimization
        logger.info("Optimizing hyperparameters using Bayesian Optimization.")
        best_hyperparams = optimize_hyperparameters(X_train, y_train, X_val, y_val, num_features)
        logger.info(f"Best hyperparameters from Bayesian Optimization: {best_hyperparams}")

        # Update the search space with the best hyperparameters
        search_space = get_search_space()
        search_space.update(best_hyperparams)

        # Initialize Ray
        ray.init()

        # Define the ASHA scheduler
        scheduler = ASHAScheduler(
            metric="loss",  # Metric to optimize
            mode="min",     # Minimize the loss
            max_t=100,      # Maximum number of epochs per trial
            grace_period=10,  # Minimum number of epochs to run before early stopping
            reduction_factor=2,  # Factor by which to reduce the number of trials
        )

        # Set a custom log directory with a shorter path
        custom_log_dir = os.path.abspath("./ray_short")
        os.makedirs(custom_log_dir, exist_ok=True)

        # Run the hyperparameter search
        analysis = tune.run(
            tune.with_parameters(trainer.train_with_ray_tune, data=combined_data),
            config=search_space,
            num_samples=4000,  # Number of trials
            scheduler=scheduler,
            resources_per_trial={"cpu": 8, "gpu": 1},
            storage_path=custom_log_dir,
            name="transformer_tuning",
            trial_dirname_creator=custom_trial_dirname_creator,
        )

        # Fetch the best configuration
        best_config = analysis.get_best_config(metric="loss", mode="min")
        logger.info(f"Best hyperparameters found: {best_config}")

        # Save the best hyperparameters
        with open("best_params.json", "w") as f:
            json.dump(best_config, f, indent=4)
        logger.info(f"Best hyperparameters saved to 'best_params.json': {best_config}")

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}", exc_info=True)
    finally:
        # Shutdown Ray
        ray.shutdown()

if __name__ == "__main__":
    main()