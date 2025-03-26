import os
import yaml
import logging
import joblib
import numpy as np
import tensorflow as tf
from data_preprocessing.load_and_preprocess import load_and_preprocess
from models.transformer_trainer import TransformerTrainer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from reinforcement_learning.trading_env import TradingEnv
from utils.custom_logging import setup_logger
import matplotlib.pyplot as plt 

# Set up logging
logger = setup_logger()

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

def calculate_volatility(data, close_column, window=14):
    """
    Calculate volatility (standard deviation of returns) for a given window.

    Args:
        data (pd.DataFrame): The input data.
        close_column (str): The name of the column containing the closing prices.
        window (int): The window size for calculating volatility.

    Returns:
        pd.Series: The volatility values.
    """
    returns = data[close_column].pct_change()  # Calculate daily returns
    volatility = returns.rolling(window=window).std()  # Calculate rolling standard deviation
    return volatility

def train_with_rl(model, data, close_column, sequence_length=30):
    """
    Train the model using reinforcement learning.

    Args:
        model (tf.keras.Model): The trained model.
        data (pd.DataFrame): The input data.
        close_column (str): The name of the column containing the closing prices.
        sequence_length (int): The sequence length required by the model.

    Returns:
        float: Total reward from RL training.
    """
    # Define the volatility column name
    ticker = close_column.split("_")[1]  # Extract ticker from close_column
    volatility_column = f'Volatility_{ticker}'

    # Initialize the environment
    env = TradingEnv(data, close_column, volatility_column)  # Pass the correct close_column and volatility_column
    state = env.reset()
    done = False
    total_reward = 0

    # Initialize a buffer to store the last `sequence_length` states
    state_buffer = []

    while not done:
        # Append the current state to the buffer
        state_buffer.append(state)

        # If the buffer has enough states, predict an action
        if len(state_buffer) >= sequence_length:
            # Convert the buffer to a sequence of shape (1, sequence_length, num_features)
            state_sequence = np.array(state_buffer[-sequence_length:])  # Shape: (sequence_length, num_features)

            # Reshape to match the model's input shape: (1, sequence_length, num_features)
            state_sequence = state_sequence[np.newaxis, :, :]  # Add batch dimension

            # Log the shape of the state_sequence
            logger.debug(f"state_sequence shape: {state_sequence.shape}")

            # Ensure the number of features matches the model's input shape
            if state_sequence.shape[2] != model.input_shape[2]:
                raise ValueError(f"Input shape mismatch: expected {model.input_shape[2]} features, got {state_sequence.shape[2]}")

            # Predict the action using the model
            action = model.predict(state_sequence, verbose=0)  # Predict action using the model

            # Extract a single numeric value from the action (e.g., the first element of the prediction)
            action = action[0][0]  # Assuming the model outputs a single value
        else:
            # If the buffer is not full, take a random action
            action = np.random.uniform(-1, 1)  # Single numeric value

        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        # Log the step details
        logger.debug(f"Step: {env.current_step}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}, Done: {done}")

        # Update the state
        state = next_state

    return total_reward

def main():
    try:
        # Define the path to config.yaml in the config folder
        config_path = os.path.join("config", "config.yaml")
        
        # Check if config.yaml exists
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at: {config_path}. Please ensure 'config.yaml' is in the 'config' folder.")
            return

        # Load config
        logger.info("Loading configuration file.")
        config = load_config(config_path)

        # Define the best hyperparameters (from Ray Tune results)
        best_hyperparams = {
            "learning_rate": 0.0007535014124895449,
            "num_heads": 4,
            "dropout_rate": 0.4219361311205221,
            "num_layers": 1,
            "ff_dim": 512,
            "batch_size": 16,
            "l2_reg": 0.0007728032325182345,
            "lookback": 30,
            "sequence_length": 30
        }

        # Load and preprocess data
        logger.info("Loading and preprocessing data.")
        combined_data = load_and_preprocess(config)
        if combined_data is None or combined_data.empty:
            logger.error("Data loading and preprocessing failed. Cannot proceed with training.")
            return

        # Initialize the trainer
        logger.info("Initializing the TransformerTrainer.")
        close_column = f"Close_{config['tickers'][0]}"  # Define close_column here
        trainer = TransformerTrainer(combined_data, config, close_column=close_column)

        # Add technical indicators for each ticker
        for ticker in config['tickers']:
            logger.info(f"Adding technical indicators for {ticker}.")
            combined_data = trainer.add_technical_indicators(combined_data, ticker)

        # Log the columns after adding technical indicators
        logger.info(f"Columns after adding technical indicators: {combined_data.columns}")

        # Handle missing data
        logger.info("Handling missing data.")
        combined_data = trainer.handle_missing_data(combined_data)

        # Prepare data for training
        logger.info("Preparing data for training.")
        prepared_data = trainer.prepare_data(combined_data, lookback=best_hyperparams["lookback"])
        if prepared_data is None:
            logger.error("Data preparation failed. Cannot proceed with training.")
            return

        X_train, X_val, y_train, y_val, feature_scaler, target_scaler, num_features = prepared_data

        # Log the shapes of the prepared data
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_val shape: {X_val.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"y_val shape: {y_val.shape}")
        logger.info(f"Number of features: {num_features}")

        # Build the model with the best hyperparameters
        logger.info("Building the model with best hyperparameters.")
        final_model = trainer.build_model(best_hyperparams, num_features)
        if final_model is None:
            logger.error("Model building failed. Cannot proceed with training.")
            return

        # Compile the model
        logger.info("Compiling the model.")
        final_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_hyperparams["learning_rate"]),
            loss='mean_squared_error'
        )

        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        ]

        # Debug logs to verify shapes before training
        logger.debug(f"X_train shape before training: {X_train.shape}")
        logger.debug(f"y_train shape before training: {y_train.shape}")

        # Train the final model
        logger.info("Training the final model.")
        history = final_model.fit(
            X_train, y_train,  # Use the scaled target values
            validation_data=(X_val, y_val),  # Use the scaled validation target values
            epochs=2000,  # Adjust as needed
            batch_size=best_hyperparams["batch_size"],
            callbacks=callbacks,
            verbose=1
        )

        # Log training and validation loss
        logger.info(f"Training loss: {history.history['loss']}")
        logger.info(f"Validation loss: {history.history['val_loss']}")

        # Plot training and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Save the final model and scalers
        logger.info("Saving the final model and scalers.")
        final_model.save("best_model.h5")
        joblib.dump(feature_scaler, 'final_scaler.pkl')
        joblib.dump(target_scaler, 'final_target_scaler.pkl')  # Save the fitted target scaler

        # Save the feature names used during training
        feature_names = combined_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_names = [name.lower() for name in feature_names]  # Normalize to lowercase
        joblib.dump(feature_names, 'feature_names.pkl')
        logger.info(f"Feature names saved to 'feature_names.pkl': {feature_names}")
        logger.info("Model and scalers saved successfully.")

    except Exception as e:
        logger.error(f"Error in final_model_trainer: {e}", exc_info=True)

if __name__ == "__main__":
    main()