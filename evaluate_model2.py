import os
import logging
import joblib
import numpy as np
import pandas as pd
import yaml
import json
import tensorflow as tf
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from stable_baselines3 import SAC
from reinforcement_learning.trading_env import TradingEnv
from models.transformer_trainer import TransformerTrainer
from utils.custom_logging import setup_logger

# Set up the logger
logger = setup_logger()

def load_config(config_path):
    """Load the configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}.")
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

        if data.empty:
            logger.error(f"No data fetched for {symbol}.")
            return None

        # Log the original column names
        logger.debug(f"Original columns: {data.columns}")

        # Flatten the MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]

        # Log the updated column names
        logger.debug(f"Updated columns: {data.columns}")

        # Add volatility column
        data[f"Volatility_{symbol}"] = data[f"Close_{symbol}"].pct_change().rolling(window=14).std()

        # Drop NaN values
        data.dropna(inplace=True)

        logger.info(f"Data fetched successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sharpe Ratio.
    
    Args:
        returns (pd.Series or np.array): Daily returns.
        risk_free_rate (float): Risk-free rate (default: 0.02).
    
    Returns:
        float: Sharpe Ratio.
    """
    try:
        returns = np.array(returns)
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) == 0:
            raise ValueError("No valid returns provided.")
        
        excess_returns = returns - risk_free_rate
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)

        if std_excess_return < 1e-10:
            return 0.0
        
        return mean_excess_return / std_excess_return
    except Exception as e:
        logger.error(f"Error calculating Sharpe Ratio: {e}")
        return np.nan

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sortino Ratio.
    
    Args:
        returns (pd.Series or np.array): Daily returns.
        risk_free_rate (float): Risk-free rate (default: 0.02).
    
    Returns:
        float: Sortino Ratio.
    """
    try:
        returns = np.array(returns)
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) == 0:
            raise ValueError("No valid returns provided.")
        
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        mean_excess_return = np.mean(excess_returns)
        std_downside_return = np.std(downside_returns, ddof=1)

        if std_downside_return < 1e-10:
            return np.inf
        
        return mean_excess_return / std_downside_return
    except Exception as e:
        logger.error(f"Error calculating Sortino Ratio: {e}")
        return np.nan

def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns.
    
    Args:
        returns (pd.Series or np.array): Daily returns.
    
    Returns:
        np.array: Cumulative returns.
    """
    try:
        returns = np.array(returns)
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) == 0:
            raise ValueError("No valid returns provided.")
        
        cumulative_returns = np.cumprod(1 + returns) - 1
        return cumulative_returns
    except Exception as e:
        logger.error(f"Error calculating cumulative returns: {e}")
        return np.nan

def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy.
    
    Args:
        y_true (pd.Series or np.array): True values.
        y_pred (pd.Series or np.array): Predicted values.
    
    Returns:
        float: Directional accuracy (percentage).
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isinf(y_true) & ~np.isinf(y_pred)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) < 2 or len(y_pred) < 2:
            raise ValueError("Insufficient data to calculate directional accuracy.")
        
        true_changes = np.sign(y_true[1:] - y_true[:-1])
        pred_changes = np.sign(y_pred[1:] - y_pred[:-1])
        
        accuracy = np.mean(true_changes == pred_changes) * 100
        return accuracy
    except Exception as e:
        logger.error(f"Error calculating directional accuracy: {e}")
        return np.nan

# In evaluate_model.py
from pykalman import KalmanFilter

def prepare_and_evaluate(data, model, feature_scaler, target_scaler, lookback=30):
    try:
        logger.info("Preparing data for evaluation.")

        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty.")

        # Load the saved feature names
        feature_names = joblib.load('feature_names.pkl')
        logger.debug(f"Feature names from feature_names.pkl: {feature_names}")

        # Normalize feature names in the evaluation dataset to match the training feature names
        data.columns = [col.strip().lower() for col in data.columns]
        logger.debug(f"Columns in evaluation dataset after normalization: {data.columns}")

        # Ensure the evaluation data has the same features as the training data
        feature_names = [col.lower() for col in feature_names]
        missing_features = set(feature_names) - set(data.columns)
        if missing_features:
            logger.warning(f"Missing features in evaluation data: {missing_features}")
            feature_names = [col for col in feature_names if col in data.columns]

        # Ensure the evaluation data has the same features as the training data
        features = data[feature_names]

        # Extract the 6 features for the target (e.g., 'close_aapl', 'volatility_aapl', etc.)
        target_cols = ['close_aapl', 'volatility_aapl', 'rsi_aapl', 'macd_aapl', 'atr_aapl', 'obv_aapl']
        if not all(col in data.columns for col in target_cols):
            logger.error(f"Target columns not found in data. Available columns: {data.columns}")
            return None

        target = data[target_cols].values  # Shape: (n_samples, 6)

        # Scale the features using the saved feature scaler
        scaled_features = feature_scaler.transform(features)

        # Create sequences for evaluation
        X, y = [], []
        for i in range(lookback, len(scaled_features)):
            X.append(scaled_features[i - lookback:i])
            y.append(target[i])  # Target: 6-dimensional observation at the current timestep

        X, y = np.array(X), np.array(y)

        # Log the shape of the sequences
        logger.debug(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")

        # Extract the relevant 6 features from each sequence
        X_reshaped = X[:, -1, -6:]  # Shape: (n_samples, 6)
        logger.debug(f"X_reshaped shape: {X_reshaped.shape}")

        # Make predictions
        predictions = model.predict(X_reshaped)  # Returns a tuple (action, state)
        y_pred_scaled = predictions[0]  # Extract the action (first element of the tuple)
        logger.debug(f"y_pred_scaled shape: {y_pred_scaled.shape}")
        logger.debug(f"First 10 scaled predictions: {y_pred_scaled[:10]}")

        # Ensure y_pred_scaled has the same number of samples as y_test_original
        if y_pred_scaled.shape[0] != y.shape[0]:
            logger.warning(f"Reshaping y_pred_scaled to match y_test_original shape: {y.shape}")
            y_pred_scaled = y_pred_scaled[:y.shape[0]]  # Truncate y_pred_scaled to match y_test_original

        # Reshape y_pred_scaled to (num_samples, 1) if necessary
        if len(y_pred_scaled.shape) > 1:
            y_pred_scaled = y_pred_scaled[:, 0]  # Only use the first column (close price)

        # Reshape y_pred_scaled to (n_samples, 1)
        y_pred_scaled_reshaped = y_pred_scaled.reshape(-1, 1)  # Shape: (n_samples, 1)

        # Pad the predictions with zeros for the other 5 features
        y_pred_scaled_padded = np.zeros((y_pred_scaled_reshaped.shape[0], 6))  # Shape: (n_samples, 6)
        y_pred_scaled_padded[:, 0] = y_pred_scaled_reshaped[:, 0]  # Only the first column is used for predictions

        # Inverse transform the padded predictions using the target scaler
        y_pred = target_scaler.inverse_transform(y_pred_scaled_padded)[:, 0]  # Only use the first column (close price)

        # Scale the actual values using the target scaler
        y_scaled = target_scaler.transform(y)  # Scale the actual values
        y_test_original = target_scaler.inverse_transform(y_scaled)[:, 0]  # Inverse transform to get the original values

        # Clip predictions to a reasonable range
        y_pred = np.clip(y_pred, y_test_original.min(), y_test_original.max())

        # Apply Kalman Filter smoothing with the best parameters
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=y_pred[0],
            initial_state_covariance=1.0,
            observation_covariance=0.960132060625243,  # Best observation_covariance
            transition_covariance=0.0013551825565085705,  # Best transition_covariance
        )
        y_pred_smoothed, _ = kf.filter(y_pred)

        # Use the smoothed predictions for evaluation
        y_pred = y_pred_smoothed

        # Flatten y_pred and y_test_original to 1D arrays
        y_pred = y_pred.flatten()
        y_test_original = y_test_original.flatten()

        logger.debug(f"First 10 inverse transformed actual values: {y_test_original[:10]}")
        logger.debug(f"First 10 smoothed predictions: {y_pred[:10]}")

        # Verify shapes
        logger.debug(f"y_test_original shape: {y_test_original.shape}")
        logger.debug(f"y_pred shape: {y_pred.shape}")

        # Calculate returns (using only the 'close' price)
        close_returns = np.diff(y_test_original) / y_test_original[:-1]  # Percentage returns
        logger.debug(f"First 10 returns: {close_returns[:10]}")

        # Calculate cumulative returns
        cumulative_returns = calculate_cumulative_returns(close_returns)
        logger.debug(f"Cumulative returns: {cumulative_returns}")

        # Calculate metrics (using only the 'close' price)
        mse = mean_squared_error(y_test_original, y_pred)  # Use only the 'close' predictions for metrics
        mae = mean_absolute_error(y_test_original, y_pred)  # Use only the 'close' predictions for metrics
        r2 = r2_score(y_test_original, y_pred)  # Use only the 'close' predictions for metrics
        mape = mean_absolute_percentage_error(y_test_original, y_pred)  # Use only the 'close' predictions for metrics
        directional_acc = directional_accuracy(y_test_original, y_pred)  # Ensure 1D arrays
        sharpe_ratio = calculate_sharpe_ratio(close_returns)
        sortino_ratio = calculate_sortino_ratio(close_returns)

        # Log the metrics
        logger.info(f"Mean Squared Error (MSE): {mse}")
        logger.info(f"Mean Absolute Error (MAE): {mae}")
        logger.info(f"R-squared (R²): {r2}")
        logger.info(f"Mean Absolute Percentage Error (MAPE): {mape}")
        logger.info(f"Directional Accuracy: {directional_acc:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio}")
        logger.info(f"Sortino Ratio: {sortino_ratio}")
        logger.info(f"Cumulative Returns: {cumulative_returns[-1]}")

        # Plot actual vs. predicted values (using only the 'close' price)
        dates = data.index[-len(y_test_original):]  # Get the corresponding dates for the test set
        plt.figure(figsize=(12, 6))
        plt.plot(dates, y_test_original, label='Actual', color='blue')
        plt.plot(dates, y_pred, label='Predicted (Smoothed)', color='red', linestyle='--')
        plt.title("Actual vs. Predicted Prices (Smoothed with Kalman Filter)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        # Plot residuals
        residuals = y_test_original - y_pred
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title("Residuals Plot (Smoothed with Kalman Filter)")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.show()

        return {
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Directional_Accuracy': directional_acc,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Cumulative_Returns': cumulative_returns[-1],
        }

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise
    
def main():
    try:
        # Load configuration
        config_path = os.path.join("config", "config.yaml")
        config = load_config(config_path)

        # Load the saved SAC model
        logger.info("Loading the saved SAC model.")
        model = SAC.load("sac_trading_model")

        # Load the saved scalers
        logger.info("Loading the saved scalers.")
        feature_scaler = joblib.load("final_scaler.pkl")
        target_scaler = joblib.load("final_target_scaler.pkl")
        logger.debug(f"Target scaler data_min_: {target_scaler.data_min_}")
        logger.debug(f"Target scaler data_max_: {target_scaler.data_max_}")
        logger.info(f"Feature scaler: {feature_scaler}")
        logger.info(f"Target scaler: {target_scaler}")

        # Fetch and combine test data
        logger.info("Fetching and combining test data.")
        test_data = fetch_historical_data(
            config['tickers'][0],  # Use the first ticker
            config['start_date'],
            config['end_date'],
            config['interval']
        )

        if test_data is None:
            raise ValueError("Failed to fetch test data. Check the ticker and date range.")

        # Initialize the trainer
        close_column = f"Close_{config['tickers'][0]}"  # Use the first ticker's close column
        trainer = TransformerTrainer(test_data, config, close_column=close_column)

        # Add technical indicators
        for ticker in config['tickers']:
            logger.info(f"Adding technical indicators for {ticker}.")
            test_data = trainer.add_technical_indicators(test_data, ticker)

            if test_data is None:
                raise ValueError(f"Failed to add technical indicators for {ticker}.")

        # Handle missing data
        test_data = trainer.handle_missing_data(test_data)

        if test_data is None:
            raise ValueError("Failed to handle missing data.")

        # Prepare and evaluate the model
        metrics = prepare_and_evaluate(test_data, model, feature_scaler, target_scaler, lookback=config['lookback'])

        # Save evaluation results
        with open("evaluation_results.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Evaluation results saved to 'evaluation_results.json'.")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)

if __name__ == "__main__":
    main()