import os
import logging
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import yaml
import talib
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingRegressor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Layer
from tensorflow.keras.models import Model
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from collections import deque
import random
import requests
from textblob import TextBlob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import Env
from gym.spaces import Box
from tqdm import tqdm
import smogn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import shap
from scipy.signal import savgol_filter
from scipy.stats import norm
from ray.tune.experiment import Trial
import ray 
import json 

# Suppress TensorFlow warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging

# Global logger instance
logger = logging.getLogger(__name__)

# Configure the logger only if it hasn't been configured yet
if not logger.handlers:
    logger.setLevel(logging.DEBUG)

    # Create a file handler with UTF-8 encoding
    file_handler = logging.FileHandler('trading_bot.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)  # Use sys.stdout for UTF-8 support
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent duplicate logs by disabling propagation to the root logger
    logger.propagate = False
    
    # Custom function to create shorter trial directory names
def custom_trial_dirname_creator(trial: Trial) -> str:
    """Custom function to create shorter trial directory names."""
    # Use a shorter naming scheme, e.g., trial_<trial_id>
    return f"trial_{trial.trial_id}"

class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, logger=None):
        super(TrainingLogger, self).__init__()
        if logger is None:
            self.logger = logging.getLogger(__name__)  # Use the existing logger
        else:
            self.logger = logger  # Use the provided logger

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.logger.info(f"Epoch {epoch + 1}: loss={logs.get('loss', 'N/A')}, val_loss={logs.get('val_loss', 'N/A')}")
        
class ExpandDimsLayer(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis
        self.logger = logging.getLogger(__name__)  # Use the existing logger

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({'axis': self.axis})
        return config
    
class TradingEnv(Env):
    """Custom Trading Environment for reinforcement learning."""
    def __init__(self, data, transaction_cost=0.001, risk_free_rate=0.02):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.n_steps = len(data) - 1
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)  # Use the existing logger

        # Define action and observation space
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)  # Continuous action space
        self.observation_space = Box(low=0, high=1, shape=(data.shape[1],), dtype=np.float32)  # Observation space

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_step = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        """Take a step in the environment."""
        self.current_step += 1

        if self.current_step >= self.n_steps:
            done = True
        else:
            done = False

        # Calculate reward (e.g., based on profit/loss)
        reward = self._calculate_reward(action)

        # Get the next observation
        observation = self.data.iloc[self.current_step].values
        return observation, reward, done, {}

    def _calculate_reward(self, action):
        """Calculate the reward based on the action taken."""
        # Calculate price change
        price_change = self.data.iloc[self.current_step]['Close'] - self.data.iloc[self.current_step - 1]['Close']
        
        # Calculate raw profit/loss
        raw_profit = price_change * action[0]
        
        # Apply transaction cost
        transaction_cost = abs(action[0]) * self.transaction_cost
        net_profit = raw_profit - transaction_cost
        
        # Calculate risk-adjusted return (Sharpe Ratio)
        # Assuming daily returns, we can calculate the Sharpe Ratio over a window
        window_size = 30  # Adjust as needed
        if self.current_step >= window_size:
            returns = self.data['Close'].pct_change().iloc[self.current_step - window_size:self.current_step]
            sharpe_ratio = (returns.mean() - self.risk_free_rate) / returns.std()
            risk_adjusted_profit = net_profit * sharpe_ratio
        else:
            risk_adjusted_profit = net_profit
        
        return risk_adjusted_profit

def fetch_data(ticker, start_date, end_date, interval):
    """Fetch historical stock data using yfinance and check for missing values."""
    logging.info(f"Fetching data for {ticker} from {start_date} to {end_date}.")
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    if data.empty:
        logging.error(f"No data fetched for {ticker}. Check the ticker symbol, dates, and interval.")
        return pd.DataFrame()
    
    # Rename columns to include the ticker suffix
    # Handle tuple-based column names (e.g., ('Close', 'AAPL'))
    new_columns = []
    for col in data.columns:
        if isinstance(col, tuple):
            # If the column name is a tuple, join the elements with an underscore
            new_col = "_".join(col)
        else:
            # If the column name is a string, append the ticker suffix
            new_col = f"{col}_{ticker}"
        new_columns.append(new_col)
    
    data.columns = new_columns
    
    # Log the renamed columns
    logging.debug(f"Renamed columns: {data.columns}")
    
    # Add ticker column to the data
    data['Ticker'] = ticker
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.any():
        logging.warning(f"Missing values found in data for {ticker}:\n{missing_values}")
        logging.debug(f"Rows with missing values:\n{data[data.isnull().any(axis=1)]}")
    else:
        logging.info("No missing values found in the raw data.")
    
    # Log the fetched data
    logging.debug(f"Fetched data for {ticker}:\n{data.head()}")
    logging.debug(f"Fetched data columns:\n{data.columns}")
    
    return data

def flatten_columns(df):
    """
    Flatten MultiIndex columns into a single level.
    Ensure the 'Ticker' column is not duplicated.
    """
    logging.debug(f"Columns before flattening:\n{df.columns}")
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            # Flatten columns like ('Close', 'AAPL') to 'Close_AAPL'
            new_columns.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
        else:
            # Keep non-MultiIndex columns as-is
            new_columns.append(col)
    df.columns = new_columns
    logging.debug(f"Columns after flattening:\n{df.columns}")
    return df

def remove_duplicated_columns(df):
    """
    Remove duplicated columns from the DataFrame.
    """
    df = df.loc[:, ~df.columns.duplicated()]
    return df  # Return the modified DataFrame

def combine_data(all_data, config):
    """
    Combine data from multiple tickers and handle missing values.
    Also, flatten MultiIndex columns properly and remove duplicated columns.
    """
    # Combine data from all tickers
    combined_data = pd.concat(all_data, axis=1)

    # Log the combined data before flattening
    logger.debug(f"Combined data before flattening:\n{combined_data.head()}")
    logger.debug(f"Combined data columns before flattening:\n{combined_data.columns}")

    # Flatten MultiIndex columns
    combined_data = flatten_columns(combined_data)

    # Log the combined data after flattening
    logger.debug(f"Combined data after flattening:\n{combined_data.head()}")
    logger.debug(f"Combined data columns after flattening:\n{combined_data.columns}")

    # Remove duplicated columns
    combined_data = remove_duplicated_columns(combined_data)

    # Log the combined data after removing duplicates
    logger.debug(f"Combined data after removing duplicates:\n{combined_data.head()}")
    logger.debug(f"Combined data columns after removing duplicates:\n{combined_data.columns}")

    # Initialize the trainer to use its handle_missing_data method
    trainer = TransformerTrainer(combined_data, config, close_column='Close_AAPL')  # Use the first ticker's close column as a placeholder

    # Handle missing data using the trainer's method
    combined_data = trainer.handle_missing_data(combined_data)

    # Log the combined data after handling missing data
    logger.debug(f"Combined data after handling missing data:\n{combined_data.head()}")
    logger.debug(f"Combined data columns after handling missing data:\n{combined_data.columns}")

    # Drop columns with all NaN values
    combined_data = combined_data.dropna(axis=1, how='all')

    # Log the DataFrame shape and columns after dropping NaN columns
    logger.debug(f"DataFrame shape after dropping NaN columns: {combined_data.shape}")
    logger.debug(f"Columns after dropping NaN columns:\n{combined_data.columns}")

    # Check if DataFrame is empty
    if combined_data.empty:
        logger.error("Combined DataFrame is empty. Check the input data.")
        return combined_data

    # Log the final DataFrame
    logger.debug(f"Final DataFrame:\n{combined_data}")
    logger.debug(f"DataFrame columns:\n{combined_data.columns}")

    return combined_data

def fetch_news_sentiment(ticker, api_key):
    """Fetch news sentiment using NewsAPI."""
    logging.info(f"Fetching news sentiment for {ticker}.")
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    sentiments = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        text = f"{title} {description}".strip()
        if text:
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    logging.info(f"Average sentiment for {ticker}: {avg_sentiment}")
    return avg_sentiment

@register_keras_serializable()
class TileLayer(Layer):
    """Custom layer to tile positions for positional encoding."""
    def __init__(self, **kwargs):
        super(TileLayer, self).__init__(**kwargs)

    def call(self, inputs):
        positions, batch_size = inputs
        # Ensure batch_size is of type int32 for tf.tile
        batch_size = tf.cast(batch_size, dtype=tf.int32)
        return tf.tile(positions, [batch_size, 1, 1])

    def compute_output_shape(self, input_shape):
        # input_shape is a list of shapes for each input tensor
        positions_shape, batch_size_shape = input_shape
        # If batch_size_shape is a scalar (shape ()), we assume it represents the batch size
        if len(batch_size_shape) == 0:
            return (None, positions_shape[1], positions_shape[2])
        else:
            return (batch_size_shape[0], positions_shape[1], positions_shape[2])

    def get_config(self):
        config = super(TileLayer, self).get_config()
        return config

@register_keras_serializable()
class BatchSizeLayer(Layer):
    """Custom layer to get the batch size dynamically."""
    def __init__(self, **kwargs):
        super(BatchSizeLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.shape(inputs)[0]  
    
class TransformerTrainer:
    def __init__(self, data, config, close_column):
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty.")
        self.data = data
        self.config = config
        self.close_column = close_column
        self.scaler = MinMaxScaler(feature_range=(0, 1))  
        self.input_shape = None
        self.feature_names = []
        self.online_buffer = deque(maxlen=1000)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.logger = logging.getLogger(__name__)
        self._prepared_data = None

    def train_with_ray_tune(self, config, checkpoint_dir=None, data=None):
        """Train the model with Ray Tune."""
        try:
            # Ensure data is passed
            if data is None or data.empty:
                raise ValueError("Data cannot be None or empty.")

            # Log the data being passed
            self.logger.debug(f"Data columns in train_with_ray_tune: {data.columns}")
            self.logger.debug(f"Data head in train_with_ray_tune:\n{data.head()}")

            # Initialize the trainer with the provided data
            trainer = TransformerTrainer(data, config, close_column='Close_AAPL')

            # Prepare data
            self.logger.info("Preparing data...")
            prepared_data = trainer.prepare_data(ticker='AAPL')  # Replace with your ticker
            if prepared_data is None:
                raise ValueError("Data preparation failed.")
            X_train, X_val, X_test, y_train, y_val, y_test = prepared_data
            self.logger.info("Data preparation successful.")

            # Build the model
            logger.info("Building the model...")
            model = trainer.build_model(config)
            if model is None:
                raise ValueError("Model building failed.")
            logger.info("Model built successfully.")

            # Compile the model
            optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"], clipvalue=1.0)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            # Train the model
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

            history = model.fit(
                X_train, y_train,
                epochs=100,  # Adjust as needed
                batch_size=config["batch_size"],
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=0,
            )

            # Evaluate the model on the test set
            test_loss = model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Test Loss: {test_loss}")

            # Report metrics to Ray Tune
            val_loss = history.history['val_loss'][-1]
            logger.info(f"Validation loss: {val_loss}")
            tune.report({"loss": val_loss, "test_loss": test_loss})  # Include test loss in the report

        except Exception as e:
            logger.error(f"Error during training with Ray Tune: {e}", exc_info=True)
            raise

    def inverse_transform_predictions(self, predictions, feature_index=0):
        """
        Inverse transform the scaled predictions back to the original price range.
        """
        self.logger.info("Inverse transforming predictions.")
        try:
            # Ensure predictions are in the correct shape
            predictions = predictions.flatten()
            self.logger.debug(f"Predictions shape before flattening: {predictions.shape}")
            self.logger.debug(f"First 5 predictions: {predictions[:5]}")

            # Check if the scaler is fitted
            if not hasattr(self.scaler, 'center_') or not hasattr(self.scaler, 'scale_'):
                self.logger.error("Scaler is not fitted. Call `scaler.fit()` before inverse transformation.")
                return None

            # Log the scaler's attributes
            self.logger.debug(f"Scaler center: {self.scaler.center_}")
            self.logger.debug(f"Scaler scale: {self.scaler.scale_}")

            # Determine the number of features in the scaler
            n_features = len(self.scaler.center_)  # Use len(scaler.center_) as the number of features
            self.logger.debug(f"Number of features in scaler: {n_features}")

            # Check if the feature index is valid
            if feature_index >= n_features:
                self.logger.error(f"Feature index {feature_index} is out of bounds for scaler with {n_features} features.")
                return None

            # Create a dummy array with the same number of features as the scaler was trained on
            dummy_array = np.zeros((len(predictions), n_features))
            self.logger.debug(f"Dummy array shape: {dummy_array.shape}")

            # Fill the dummy array with predictions
            dummy_array[:, feature_index] = predictions
            self.logger.debug(f"Dummy array after filling predictions:\n{dummy_array[:5]}")

            # Inverse transform the predictions
            inverse_transformed = self.scaler.inverse_transform(dummy_array)
            self.logger.debug(f"Inverse transformed predictions (first 5 rows):\n{inverse_transformed[:5, feature_index]}")
            self.logger.debug(f"Inverse transformed predictions shape: {inverse_transformed.shape}")

            return inverse_transformed[:, feature_index]
        except Exception as e:
            self.logger.error(f"Error during inverse transformation: {e}", exc_info=True)
            return None
    def _add_fibonacci_and_chaikin(self, data, ticker):
        """
        Add Fibonacci retracement levels and Chaikin Oscillator to the data.
        
        Args:
            data (pd.DataFrame): The DataFrame containing the stock data.
            ticker (str): The ticker symbol (e.g., 'AAPL').
        
        Returns:
            pd.DataFrame: The DataFrame with Fibonacci and Chaikin Oscillator columns added.
        """
        try:
            # Step 1: Calculate Fibonacci retracement levels
            rolling_max = data[f'Close_{ticker}'].rolling(window=14, min_periods=1).max()
            data[f'Fibonacci_38_{ticker}'] = rolling_max * 0.382
            data[f'Fibonacci_50_{ticker}'] = rolling_max * 0.5
            data[f'Fibonacci_61_{ticker}'] = rolling_max * 0.618
            self.logger.debug(f"Fibonacci levels after calculation:\n{data[[f'Fibonacci_38_{ticker}', f'Fibonacci_50_{ticker}', f'Fibonacci_61_{ticker}']].head()}")

            # Step 2: Calculate Chaikin Oscillator
            high_prices = data[f'High_{ticker}'].values.astype(float)
            low_prices = data[f'Low_{ticker}'].values.astype(float)
            close_prices = data[f'Close_{ticker}'].values.astype(float)
            volume = data[f'Volume_{ticker}'].values.astype(float)

            # Check for NaN values in input data
            if np.isnan(high_prices).any() or np.isnan(low_prices).any() or np.isnan(close_prices).any() or np.isnan(volume).any():
                self.logger.warning(f"NaN values found in input data for {ticker}. Forward-filling and backward-filling.")
                high_prices = pd.Series(high_prices).ffill().bfill().values
                low_prices = pd.Series(low_prices).ffill().bfill().values
                close_prices = pd.Series(close_prices).ffill().bfill().values
                volume = pd.Series(volume).ffill().bfill().values

            # Calculate Chaikin Oscillator
            data[f'Chaikin_{ticker}'] = talib.ADOSC(
                high=high_prices,
                low=low_prices,
                close=close_prices,
                volume=volume,
                fastperiod=3,
                slowperiod=10
            )

            # Normalize Chaikin values to a range between 0 and 1
            chaikin_min = np.min(data[f'Chaikin_{ticker}'])
            chaikin_max = np.max(data[f'Chaikin_{ticker}'])
            if chaikin_max != chaikin_min:  # Avoid division by zero
                data[f'Chaikin_{ticker}'] = (data[f'Chaikin_{ticker}'] - chaikin_min) / (chaikin_max - chaikin_min)
            else:
                data[f'Chaikin_{ticker}'] = np.zeros_like(data[f'Chaikin_{ticker}'])  # If all values are the same, set to 0

            # Drop rows with NaN values in the Chaikin Oscillator
            data = data.dropna(subset=[f'Chaikin_{ticker}'])
            self.logger.debug(f"Chaikin_{ticker} values after dropping NaN rows:\n{data[f'Chaikin_{ticker}'].head()}")

            return data

        except Exception as e:
            self.logger.error(f"Error adding Fibonacci and Chaikin Oscillator for {ticker}: {e}")
            return data
    
    def prepare_data(self, ticker):
        try:
            self.logger.info(f"Starting data preparation for ticker: {ticker}")

            # Step 0: Ensure self.data is not None or empty
            if self.data is None or self.data.empty:
                self.logger.error("Input data is None or empty. Check data loading.")
                return None

            # Log the raw data shape and columns
            self.logger.debug(f"Raw data shape: {self.data.shape}")
            self.logger.debug(f"Raw data columns: {self.data.columns}")
            self.logger.debug(f"First 5 rows of raw data:\n{self.data.head()}")

            # Step 1: Ensure required columns are present
            required_columns = [f'Close_{ticker}', f'High_{ticker}', f'Low_{ticker}', f'Volume_{ticker}']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns in data: {missing_columns}")
                return None

            # Step 2: Add technical indicators for the ticker
            self.logger.info(f"Adding technical indicators for {ticker}.")
            self.data = self.add_technical_indicators(self.data, ticker)

            # Step 3: Calculate target variable (percentage change in price)
            self.logger.info("Calculating target variable (percentage change in price).")
            self.data['Target'] = self.data[f'Close_{ticker}'].pct_change().shift(-1)
            self.logger.debug(f"Target column created. First 5 values: {self.data['Target'].head()}")
            self.logger.debug(f"Target column NaN count: {self.data['Target'].isna().sum()}")

            # Step 4: Drop rows with NaN in the target
            self.logger.info("Dropping rows with NaN in the target.")
            self.data = self.data.dropna(subset=['Target'])
            self.logger.debug(f"Data after dropping NaN in Target. Shape: {self.data.shape}")

            # Step 5: Select numeric features
            self.logger.info("Selecting numeric features.")
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            features = self.data[numeric_cols]
            target = self.data['Target'].values
            self.logger.debug(f"Numeric features selected. Shape: {features.shape}")
            self.logger.debug(f"Target values. Shape: {target.shape}")

            # Step 6: Log feature names
            self.feature_names = features.columns.tolist()
            self.logger.debug(f"Feature names: {self.feature_names}")

            # Step 7: Scale the features and target using MinMaxScaler
            self.logger.info("Scaling features and target using MinMaxScaler.")
            self.scaler = MinMaxScaler(feature_range=(0, 1))  # Scale features between 0 and 1
            X_scaled = self.scaler.fit_transform(features)
            self.logger.debug(f"Features scaled. Shape: {X_scaled.shape}")
            self.logger.debug(f"First 5 rows of scaled features:\n{X_scaled[:5]}")

            # Use a separate scaler for the target variable
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))  # Scale target between 0 and 1
            y_scaled = self.target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
            self.logger.debug(f"Target scaled. Shape: {y_scaled.shape}")
            self.logger.debug(f"First 5 values of scaled target: {y_scaled[:5]}")

            # Step 8: Create sequences for Transformer
            self.logger.info("Creating sequences for Transformer.")
            X, y = [], []
            lookback = self.config.get('lookback', 30)
            self.logger.info(f"Creating sequences with lookback: {lookback}")
            for i in range(lookback, len(X_scaled)):
                X.append(X_scaled[i - lookback:i])
                y.append(y_scaled[i])
            X, y = np.array(X), np.array(y)
            self.logger.debug(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")

            # Set input_shape for the model
            self.input_shape = X.shape[1:]
            self.logger.info(f"Input shape set to: {self.input_shape}")

            # Step 9: Split into training, validation, and test sets
            self.logger.info("Splitting data into training, validation, and test sets.")
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
            self.logger.debug(f"First split completed. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            self.logger.debug(f"Temporary data. X_temp shape: {X_temp.shape}, y_temp shape: {y_temp.shape}")

            # Ensure temporary data is not empty
            if len(X_temp) == 0 or len(y_temp) == 0:
                self.logger.error("Temporary data (X_temp, y_temp) is empty. Check data preprocessing.")
                return None

            # Second split: Validation (50% of Temporary) and Test (50% of Temporary)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
            self.logger.debug(f"Second split completed. X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
            self.logger.debug(f"Test data. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # Step 10: Check for NaN or Inf in the data
            if np.isnan(X_train).any() or np.isinf(X_train).any():
                self.logger.error("NaN or Inf values found in X_train. Check data preprocessing.")
                return None
            if np.isnan(y_train).any() or np.isinf(y_train).any():
                self.logger.error("NaN or Inf values found in y_train. Check data preprocessing.")
                return None

            # Ensure all splits are defined
            if y_test is None:
                self.logger.error("y_test is None. Check data splitting logic.")
                return None

            self.logger.info("Data preparation completed successfully.")
            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            self.logger.error(f"Error during data preparation: {e}", exc_info=True)
            return None
        
    def handle_missing_data(self, df):
        """Handle missing data using forward-fill, backward-fill, and KNN imputation."""
        self.logger.info("Handling missing data using forward-fill, backward-fill, and KNN imputation.")

        # Log missing values before handling
        self.logger.debug(f"Missing values before handling:\n{df.isnull().sum()}")

        # Step 1: Drop columns with all NaN values
        cols_with_all_nan = df.columns[df.isna().all()]
        if len(cols_with_all_nan) > 0:
            self.logger.warning(f"Dropping columns with all NaN values: {cols_with_all_nan}")
            df = df.drop(columns=cols_with_all_nan)

        # Step 2: Forward-fill missing values
        df = df.ffill()

        # Step 3: Backward-fill any remaining missing values
        df = df.bfill()

        # Log missing values after forward-fill and backward-fill
        self.logger.debug(f"Missing values after forward-fill and backward-fill:\n{df.isnull().sum()}")

        # Step 4: Apply KNN imputation for all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Apply KNN imputation
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(df[numeric_cols])

            # Replace the original numeric columns with the imputed data
            df[numeric_cols] = imputed_data

        # Log missing values after imputation
        self.logger.debug(f"Missing values after imputation:\n{df.isnull().sum()}")

        return df

    def add_technical_indicators(self, data, ticker):
        """Add essential technical indicators to the data for a specific ticker."""
        logging.info(f"Adding essential technical indicators for {ticker}.")

        # Ensure we're working with a copy of the data
        data = data.copy()

        # Dynamically generate column names for the ticker
        close_col = f'Close_{ticker}'
        high_col = f'High_{ticker}'
        low_col = f'Low_{ticker}'
        volume_col = f'Volume_{ticker}'

        # Extract the required columns
        close_prices = data[close_col].values.astype(float)
        high_prices = data[high_col].values.astype(float)
        low_prices = data[low_col].values.astype(float)
        volume = data[volume_col].values.astype(float)

        # Ensure no NaN values in the input data
        if np.isnan(close_prices).any() or np.isnan(high_prices).any() or np.isnan(low_prices).any() or np.isnan(volume).any():
            logging.warning(f"NaN values found in input data for {ticker}. Forward-filling and backward-filling.")
            close_prices = pd.Series(close_prices).ffill().bfill().values
            high_prices = pd.Series(high_prices).ffill().bfill().values
            low_prices = pd.Series(low_prices).ffill().bfill().values
            volume = pd.Series(volume).ffill().bfill().values

        # Add essential technical indicators
        if len(data) >= 14:  # Minimum required for most indicators
            # Price-Based Indicators
            data = self._add_moving_averages(data, close_prices, ticker)  # Adds EMA_10 and EMA_30
            data = self._add_bollinger_bands(data, close_prices, ticker)  # Adds Bollinger Bands
            data = self._add_vwap(data, close_col, volume_col, ticker)    # Adds VWAP

            # Momentum Indicators
            data = self._add_rsi(data, close_prices, ticker)             # Adds RSI
            data = self._add_macd(data, close_prices, ticker)            # Adds MACD, MACD_Signal, MACD_Hist
            data = self._add_stochastic_oscillator(data, high_prices, low_prices, close_prices, ticker)  # Adds Stochastic

            # Volatility Indicators
            data = self._add_atr(data, high_prices, low_prices, close_prices, ticker)  # Adds ATR
            data = self._add_parabolic_sar(data, high_prices, low_prices, ticker)      # Adds Parabolic SAR

            # Trend Indicators
            data = self._add_ichimoku_cloud(data, close_prices, high_prices, low_prices, ticker)  # Adds Ichimoku Cloud
            data = self._add_adx(data, high_prices, low_prices, close_prices, ticker)             # Adds ADX

            # Other Indicators
            data = self._add_obv(data, close_prices, volume, ticker)     # Adds OBV
            data = self._add_donchian_channels(data, high_prices, low_prices, ticker)  # Adds Donchian Channels
            data = self._add_hma(data, close_prices, ticker)             # Adds HMA
        else:
            logging.warning(f"Insufficient data for {ticker}. Skipping technical indicators.")

        # Handle missing data
        data = self.handle_missing_data(data)

        # Log the columns after adding indicators
        logging.debug(f"Columns after adding technical indicators: {data.columns}")

        return data

    def _add_rsi(self, data, close_prices, ticker):
        """Add RSI indicator."""
        try:
            # Log the close prices used for RSI calculation
            logging.debug(f"Close prices for RSI calculation for {ticker}:\n{close_prices[:20]}")
            logging.debug(f"Number of close prices: {len(close_prices)}")

            # Check if there's enough data for RSI calculation
            if len(close_prices) >= 14:  # RSI requires at least 14 periods
                rsi_values = talib.RSI(close_prices, timeperiod=14)
                
                # Log the RSI values
                logging.debug(f"RSI_{ticker} values:\n{rsi_values[:20]}")
                logging.debug(f"Number of RSI values: {len(rsi_values)}")
                logging.debug(f"NaN values in RSI: {np.isnan(rsi_values).sum()}")

                # Add RSI values to the DataFrame
                data[f'RSI_{ticker}'] = rsi_values
            else:
                logging.warning(f"Insufficient data for RSI_{ticker}. Skipping calculation.")
                data[f'RSI_{ticker}'] = np.nan  # Fill with NaN if insufficient data
        except Exception as e:
            logging.error(f"Error calculating RSI for {ticker}: {e}")
            data[f'RSI_{ticker}'] = np.nan  # Fill with NaN if calculation fails
        
        # Log the final data after adding RSI
        logging.debug(f"Data after adding RSI_{ticker}:\n{data[[f'RSI_{ticker}']].head()}")
        return data

    def _add_moving_averages(self, data, close_prices, ticker):
        """Add EMA indicators (remove SMA)."""
        try:
            logging.debug(f"Close prices for EMA calculation for {ticker}:\n{close_prices[:20]}")
            logging.debug(f"Number of close prices: {len(close_prices)}")

            # Initialize EMA columns with NaN
            data[f'EMA_10_{ticker}'] = np.nan
            data[f'EMA_30_{ticker}'] = np.nan

            # EMA calculations
            if len(close_prices) >= 10:
                ema_10 = talib.EMA(close_prices, timeperiod=10)
                data[f'EMA_10_{ticker}'] = ema_10
                logging.debug(f"EMA_10_{ticker} values:\n{ema_10[:20]}")
            else:
                logging.warning(f"Insufficient data for EMA_10_{ticker}. Skipping calculation.")

            if len(close_prices) >= 30:
                ema_30 = talib.EMA(close_prices, timeperiod=30)
                data[f'EMA_30_{ticker}'] = ema_30
                logging.debug(f"EMA_30_{ticker} values:\n{ema_30[:20]}")
            else:
                logging.warning(f"Insufficient data for EMA_30_{ticker}. Skipping calculation.")

            # Log only the EMA columns
            existing_columns = [col for col in [f'EMA_10_{ticker}', f'EMA_30_{ticker}'] if col in data.columns]
            logging.debug(f"Data after adding EMAs:\n{data[existing_columns].head()}")

        except Exception as e:
            logging.error(f"Error calculating EMAs for {ticker}: {e}")
            data[f'EMA_10_{ticker}'] = np.nan
            data[f'EMA_30_{ticker}'] = np.nan

        return data


    def _add_macd(self, data, close_prices, ticker):
        """Add MACD indicator."""
        try:
            logging.debug(f"Close prices for MACD calculation for {ticker}:\n{close_prices[:20]}")
            if len(close_prices) >= 26:  # MACD requires at least 26 periods
                macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                data[f'MACD_{ticker}'] = macd
                data[f'MACD_Signal_{ticker}'] = macd_signal
                data[f'MACD_Hist_{ticker}'] = macd_hist
                logging.debug(f"MACD_{ticker} values:\n{macd[:20]}")
                logging.debug(f"MACD_Signal_{ticker} values:\n{macd_signal[:20]}")
                logging.debug(f"MACD_Hist_{ticker} values:\n{macd_hist[:20]}")
            else:
                logging.warning(f"Insufficient data for MACD_{ticker}. Skipping calculation.")
                data[f'MACD_{ticker}'] = np.nan
                data[f'MACD_Signal_{ticker}'] = np.nan
                data[f'MACD_Hist_{ticker}'] = np.nan
        except Exception as e:
            logging.error(f"Error calculating MACD for {ticker}: {e}")
            data[f'MACD_{ticker}'] = np.nan
            data[f'MACD_Signal_{ticker}'] = np.nan
            data[f'MACD_Hist_{ticker}'] = np.nan
        return data
    
    def _add_adx(self, data, high_prices, low_prices, close_prices, ticker):
        """Add ADX (Average Directional Index) indicator."""
        try:
            logging.debug(f"High prices for ADX calculation for {ticker}:\n{high_prices[:20]}")
            logging.debug(f"Low prices for ADX calculation for {ticker}:\n{low_prices[:20]}")
            logging.debug(f"Close prices for ADX calculation for {ticker}:\n{close_prices[:20]}")

            if len(high_prices) >= 14:  # ADX requires at least 14 periods
                adx_values = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
                data[f'ADX_{ticker}'] = adx_values
            else:
                logging.warning(f"Insufficient data for ADX_{ticker}. Skipping calculation.")
                data[f'ADX_{ticker}'] = np.nan  # Fill with NaN if insufficient data
        except Exception as e:
            logging.error(f"Error calculating ADX for {ticker}: {e}")
            data[f'ADX_{ticker}'] = np.nan  # Fill with NaN if calculation fails
        return data
    
    def _add_parabolic_sar(self, data, high_prices, low_prices, ticker):
        """Add Parabolic SAR indicator."""
        try:
            logging.debug(f"High prices for Parabolic SAR calculation for {ticker}:\n{high_prices[:20]}")
            logging.debug(f"Low prices for Parabolic SAR calculation for {ticker}:\n{low_prices[:20]}")
            if len(high_prices) >= 1:  # Parabolic SAR requires at least 1 period
                sar_values = talib.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2)
                data[f'Parabolic_SAR_{ticker}'] = sar_values
                logging.debug(f"Parabolic_SAR_{ticker} values:\n{sar_values[:20]}")
            else:
                logging.warning(f"Insufficient data for Parabolic SAR_{ticker}. Skipping calculation.")
                data[f'Parabolic_SAR_{ticker}'] = np.nan
        except Exception as e:
            logging.error(f"Error calculating Parabolic SAR for {ticker}: {e}")
            data[f'Parabolic_SAR_{ticker}'] = np.nan
        return data
    
    def _add_donchian_channels(self, data, high_prices, low_prices, ticker):
        """Add Donchian Channels indicator."""
        try:
            logging.debug(f"High prices for Donchian Channels calculation for {ticker}:\n{high_prices[:20]}")
            logging.debug(f"Low prices for Donchian Channels calculation for {ticker}:\n{low_prices[:20]}")
            window = 20  # Default window size for Donchian Channels
            upper = pd.Series(high_prices).rolling(window=window).max()
            lower = pd.Series(low_prices).rolling(window=window).min()
            data[f'Donchian_Upper_{ticker}'] = upper.values
            data[f'Donchian_Lower_{ticker}'] = lower.values
            logging.debug(f"Donchian_Upper_{ticker} values:\n{upper[:20]}")
            logging.debug(f"Donchian_Lower_{ticker} values:\n{lower[:20]}")
        except Exception as e:
            logging.error(f"Error calculating Donchian Channels for {ticker}: {e}")
            data[f'Donchian_Upper_{ticker}'] = np.nan
            data[f'Donchian_Lower_{ticker}'] = np.nan
        return data
    
    def _add_hma(self, data, close_prices, ticker):
        """Add Hull Moving Average (HMA) indicator."""
        try:
            logging.debug(f"Close prices for HMA calculation for {ticker}:\n{close_prices[:20]}")
            window = 9  # Default window size for HMA
            wma_half = talib.WMA(close_prices, timeperiod=window // 2)
            wma_full = talib.WMA(close_prices, timeperiod=window)
            hma_values = talib.WMA(2 * wma_half - wma_full, timeperiod=int(np.sqrt(window)))
            data[f'HMA_{ticker}'] = hma_values
            logging.debug(f"HMA_{ticker} values:\n{hma_values[:20]}")
        except Exception as e:
            logging.error(f"Error calculating HMA for {ticker}: {e}")
            data[f'HMA_{ticker}'] = np.nan
        return data

    def _add_bollinger_bands(self, data, close_prices, ticker):
        """Add Bollinger Bands indicator."""
        try:
            logging.debug(f"Close prices for Bollinger Bands calculation for {ticker}:\n{close_prices[:20]}")
            logging.debug(f"Number of close prices: {len(close_prices)}")

            if len(close_prices) >= 20:  # Bollinger Bands require at least 20 periods
                upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                data[f'Bollinger_Upper_{ticker}'] = upper
                data[f'Bollinger_Middle_{ticker}'] = middle
                data[f'Bollinger_Lower_{ticker}'] = lower

                # Log the Bollinger Bands values
                logging.debug(f"Bollinger_Upper_{ticker} values:\n{upper[:20]}")
                logging.debug(f"Bollinger_Middle_{ticker} values:\n{middle[:20]}")
                logging.debug(f"Bollinger_Lower_{ticker} values:\n{lower[:20]}")
            else:
                logging.warning(f"Insufficient data for Bollinger Bands_{ticker}. Skipping calculation.")
                data[f'Bollinger_Upper_{ticker}'] = np.nan
                data[f'Bollinger_Middle_{ticker}'] = np.nan
                data[f'Bollinger_Lower_{ticker}'] = np.nan
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands for {ticker}: {e}")
            data[f'Bollinger_Upper_{ticker}'] = np.nan
            data[f'Bollinger_Middle_{ticker}'] = np.nan
            data[f'Bollinger_Lower_{ticker}'] = np.nan

        # Log the final data after adding Bollinger Bands
        logging.debug(f"Data after adding Bollinger Bands:\n{data[[f'Bollinger_Upper_{ticker}', f'Bollinger_Middle_{ticker}', f'Bollinger_Lower_{ticker}']].head()}")
        return data

    def _add_atr(self, data, high_prices, low_prices, close_prices, ticker):
        """Add ATR indicator."""
        try:
            logging.debug(f"High prices for ATR calculation for {ticker}:\n{high_prices[:20]}")
            logging.debug(f"Low prices for ATR calculation for {ticker}:\n{low_prices[:20]}")
            logging.debug(f"Close prices for ATR calculation for {ticker}:\n{close_prices[:20]}")
            logging.debug(f"Number of prices: {len(high_prices)}")

            if len(high_prices) >= 14:  # ATR requires at least 14 periods
                atr_values = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                data[f'ATR_{ticker}'] = atr_values
                logging.debug(f"ATR_{ticker} values:\n{atr_values[:20]}")
            else:
                logging.warning(f"Insufficient data for ATR_{ticker}. Skipping calculation.")
                data[f'ATR_{ticker}'] = np.nan
        except Exception as e:
            logging.error(f"Error calculating ATR for {ticker}: {e}")
            data[f'ATR_{ticker}'] = np.nan

        # Log the final data after adding ATR
        logging.debug(f"Data after adding ATR:\n{data[[f'ATR_{ticker}']].head()}")
        return data

    def _add_obv(self, data, close_prices, volume, ticker):
        """Add OBV indicator."""
        try:
            logging.debug(f"Close prices for OBV calculation for {ticker}:\n{close_prices[:20]}")
            logging.debug(f"Volume for OBV calculation for {ticker}:\n{volume[:20]}")
            logging.debug(f"Number of prices: {len(close_prices)}")

            obv_values = talib.OBV(close_prices, volume)
            data[f'OBV_{ticker}'] = obv_values
            logging.debug(f"OBV_{ticker} values:\n{obv_values[:20]}")
        except Exception as e:
            logging.error(f"Error calculating OBV for {ticker}: {e}")
            data[f'OBV_{ticker}'] = np.nan

        # Log the final data after adding OBV
        logging.debug(f"Data after adding OBV:\n{data[[f'OBV_{ticker}']].head()}")
        return data

    def _add_vwap(self, data, close_col, volume_col, ticker):
        """Add VWAP indicator."""
        try:
            logging.debug(f"Close prices for VWAP calculation for {ticker}:\n{data[close_col][:20]}")
            logging.debug(f"Volume for VWAP calculation for {ticker}:\n{data[volume_col][:20]}")
            logging.debug(f"Number of prices: {len(data[close_col])}")

            vwap_values = (data[close_col] * data[volume_col]).cumsum() / data[volume_col].cumsum()
            data[f'VWAP_{ticker}'] = vwap_values
            logging.debug(f"VWAP_{ticker} values:\n{vwap_values[:20]}")
        except Exception as e:
            logging.error(f"Error calculating VWAP for {ticker}: {e}")
            data[f'VWAP_{ticker}'] = np.nan

        # Log the final data after adding VWAP
        logging.debug(f"Data after adding VWAP:\n{data[[f'VWAP_{ticker}']].head()}")
        return data

    def _add_ichimoku_cloud(self, data, close_prices, high_prices, low_prices, ticker):
        """Add Ichimoku Cloud indicators."""
        try:
            # Log the input data
            logging.debug(f"Close prices for Ichimoku Cloud calculation for {ticker}:\n{close_prices[:20]}")
            logging.debug(f"High prices for Ichimoku Cloud calculation for {ticker}:\n{high_prices[:20]}")
            logging.debug(f"Low prices for Ichimoku Cloud calculation for {ticker}:\n{low_prices[:20]}")
            logging.debug(f"Number of prices: {len(close_prices)}")

            # Tenkan-sen (Conversion Line)
            tenkan_sen = talib.SMA(close_prices, timeperiod=9)
            # Kijun-sen (Base Line)
            kijun_sen = talib.SMA(close_prices, timeperiod=26)
            # Senkou Span A (Leading Span A)
            senkou_span_a = (tenkan_sen + kijun_sen) / 2
            # Senkou Span B (Leading Span B)
            senkou_span_b = talib.SMA(close_prices, timeperiod=52)

            # Add to DataFrame
            data[f'Ichimoku_Tenkan_{ticker}'] = tenkan_sen
            data[f'Ichimoku_Kijun_{ticker}'] = kijun_sen
            data[f'Ichimoku_Senkou_A_{ticker}'] = senkou_span_a
            data[f'Ichimoku_Senkou_B_{ticker}'] = senkou_span_b

            # Log the Ichimoku Cloud values
            logging.debug(f"Ichimoku_Tenkan_{ticker} values:\n{tenkan_sen[:20]}")
            logging.debug(f"Ichimoku_Kijun_{ticker} values:\n{kijun_sen[:20]}")
            logging.debug(f"Ichimoku_Senkou_A_{ticker} values:\n{senkou_span_a[:20]}")
            logging.debug(f"Ichimoku_Senkou_B_{ticker} values:\n{senkou_span_b[:20]}")
        except Exception as e:
            logging.error(f"Error calculating Ichimoku Cloud for {ticker}: {e}")
            data[f'Ichimoku_Tenkan_{ticker}'] = np.nan
            data[f'Ichimoku_Kijun_{ticker}'] = np.nan
            data[f'Ichimoku_Senkou_A_{ticker}'] = np.nan
            data[f'Ichimoku_Senkou_B_{ticker}'] = np.nan

        # Log the final data after adding Ichimoku Cloud
        logging.debug(f"Data after adding Ichimoku Cloud:\n{data[[f'Ichimoku_Tenkan_{ticker}', f'Ichimoku_Kijun_{ticker}', f'Ichimoku_Senkou_A_{ticker}', f'Ichimoku_Senkou_B_{ticker}']].head()}")
        return data

    def _add_stochastic_oscillator(self, data, high_prices, low_prices, close_prices, ticker):
        """Add Stochastic Oscillator indicators."""
        try:
            # Log the input data
            logging.debug(f"High prices for Stochastic Oscillator calculation for {ticker}:\n{high_prices[:20]}")
            logging.debug(f"Low prices for Stochastic Oscillator calculation for {ticker}:\n{low_prices[:20]}")
            logging.debug(f"Close prices for Stochastic Oscillator calculation for {ticker}:\n{close_prices[:20]}")
            logging.debug(f"Number of prices: {len(high_prices)}")

            # Calculate Stochastic Oscillator
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=14, slowk_period=3, slowd_period=3)
            
            # Add to DataFrame
            data[f'Stochastic_SlowK_{ticker}'] = slowk
            data[f'Stochastic_SlowD_{ticker}'] = slowd

            # Log the Stochastic Oscillator values
            logging.debug(f"Stochastic_SlowK_{ticker} values:\n{slowk[:20]}")
            logging.debug(f"Stochastic_SlowD_{ticker} values:\n{slowd[:20]}")
        except Exception as e:
            logging.error(f"Error calculating Stochastic Oscillator for {ticker}: {e}")
            data[f'Stochastic_SlowK_{ticker}'] = np.nan
            data[f'Stochastic_SlowD_{ticker}'] = np.nan

        # Log the final data after adding Stochastic Oscillator
        logging.debug(f"Data after adding Stochastic Oscillator:\n{data[[f'Stochastic_SlowK_{ticker}', f'Stochastic_SlowD_{ticker}']].head()}")
        return data

    def smooth_predictions(self, predictions, window_size=5, polyorder=2):
        """Smooth predictions using Savitzky-Golay filter."""
        return savgol_filter(predictions, window_size, polyorder)
    
    def select_features(self, data, target):
        """Select important features using SHAP values."""
        logging.info("Selecting features for Transformer prediction.")

        # Drop rows with NaN values in the target
        data = data.dropna(subset=[self.close_column])
        target = target[~np.isnan(target)]

        # Ensure data and target have the same length
        if len(data) != len(target):
            logging.error("Data and target length mismatch after dropping NaN values.")
            return None, None

        # Remove columns with constant or all-zero values
        constant_columns = data.columns[data.nunique() <= 1]  # Columns with only one unique value
        if len(constant_columns) > 0:
            logging.warning(f"Removing constant or all-zero columns: {constant_columns}")
            data = data.drop(columns=constant_columns)

        # Check if any features remain
        if data.empty:
            logging.error("No features remaining after removing constant columns.")
            return None, None

        # Debug: Log remaining features
        logging.debug(f"Remaining features after removing constant columns: {data.columns}")

        # Fit the feature selection model
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
        model.fit(data, target)

        # Debug: Log feature importance
        feature_importance = model.feature_importances_
        logging.debug(f"Feature importance: {feature_importance}")
        logging.debug(f"Feature importance scores: {dict(zip(data.columns, feature_importance))}")

        # Select features with a lower threshold
        selector = SelectFromModel(model, prefit=True, threshold='0.5*median')  # Lower threshold
        selector.feature_names_in_ = data.columns  # Add feature names
        data_selected = selector.transform(data)

        # Get the selected feature names
        self.feature_names = data.columns[selector.get_support()].tolist()  # Set feature_names here
        logging.info(f"Selected features: {self.feature_names}")

        # Drop rows with NaN values in the selected features
        data_selected = pd.DataFrame(data_selected, columns=self.feature_names, index=data.index)
        data_selected = data_selected.dropna()

        logging.debug(f"Data after feature selection:\n{data_selected.head()}")
        logging.debug(f"Shape of data_selected: {data_selected.shape}")
        return data_selected.values, selector.get_support()

    def build_transformer_model(self, input_shape, num_features, num_heads=8, dropout_rate=0.3, num_layers=3, ff_dim=256, l2_reg=0.01):
        """Build a simplified Transformer-based model with increased dropout and L2 regularization."""
        try:
            self.logger.info(f"Building Transformer model with input_shape: {input_shape}")

            # Ensure input_shape is valid
            if input_shape is None or len(input_shape) != 2:
                raise ValueError(f"Invalid input_shape: {input_shape}. Expected a tuple of length 2 (sequence_length, num_features).")

            # Input layer
            inputs = Input(shape=input_shape)  # Input shape should be (sequence_length, num_features)
            self.logger.debug(f"Input layer created with shape: {input_shape}")

            # Positional Encoding
            positions = tf.range(start=0, limit=input_shape[0], delta=1)
            positions = tf.expand_dims(positions, axis=-1)  # Shape: (sequence_length, 1)
            positions = tf.cast(positions, dtype=tf.float32)
            positions = Dense(num_features, activation='relu')(positions)  # Shape: (sequence_length, num_features)

            # Expand positions to match input shape (batch_size, sequence_length, num_features)
            positions = tf.expand_dims(positions, axis=0)  # Shape: (1, sequence_length, num_features)

            # Use the custom BatchSizeLayer to get the batch size dynamically
            batch_size = BatchSizeLayer()(inputs)  # Get the batch size dynamically
            self.logger.debug(f"Batch size: {batch_size}")

            # Use the custom TileLayer to tile positions
            positions = TileLayer()([positions, batch_size])  # Shape: (batch_size, sequence_length, num_features)

            x = inputs + positions  # Add positional encoding to inputs
            self.logger.debug(f"Shape after positional encoding: {x.shape}")

            # Stack multiple Transformer layers
            for _ in range(num_layers):
                # Multi-Head Attention Layer
                attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=num_features)(x, x)
                x = LayerNormalization(epsilon=1e-6)(x + attention_output)
                x = Dropout(dropout_rate)(x)  # Increased dropout rate

                # Feed-Forward Network with L2 regularization
                ff_output = Dense(ff_dim, activation='relu', kernel_regularizer=l2(l2_reg))(x)  # L2 regularization
                ff_output = Dense(num_features, kernel_regularizer=l2(l2_reg))(ff_output)  # L2 regularization
                x = LayerNormalization(epsilon=1e-6)(x + ff_output)
                x = Dropout(dropout_rate)(x)  # Increased dropout rate

            # Global Average Pooling to reduce the time dimension
            x = GlobalAveragePooling1D()(x)  # Output shape: (batch_size, num_features)
            self.logger.debug(f"Shape after GlobalAveragePooling1D: {x.shape}")

            # Output Layer with L2 regularization
            outputs = Dense(1, activation='linear', kernel_regularizer=l2(l2_reg))(x)  # L2 regularization

            model = Model(inputs, outputs)
            return model

        except Exception as e:
            self.logger.error(f"Error building Transformer model: {e}", exc_info=True)
            return None

    def build_model(self, hyperparameters):
        try:
            # Extract hyperparameters
            learning_rate = hyperparameters.get('learning_rate', 0.001)
            num_heads = hyperparameters.get('num_heads', 8)
            dropout_rate = hyperparameters.get('dropout_rate', 0.1)
            num_layers = hyperparameters.get('num_layers', 2)
            ff_dim = hyperparameters.get('ff_dim', 256)
            l2_reg = hyperparameters.get('l2_reg', 0.01)  # Add L2 regularization

            # Ensure input_shape is set
            if not hasattr(self, 'input_shape') or self.input_shape is None:
                raise ValueError("input_shape is not set. Call prepare_data() first.")

            # Build the model
            model = self.build_transformer_model(
                input_shape=self.input_shape,
                num_features=len(self.feature_names),
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                num_layers=num_layers,
                ff_dim=ff_dim,
                l2_reg=l2_reg  # Pass L2 regularization
            )

            # Compile the model with gradient clipping
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipvalue=1.0  # Clip gradients to prevent exploding gradients
            )
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            return model

        except Exception as e:
            logger.error(f"Error building model: {e}", exc_info=True)
            return None
        
    def prepare_new_data(self, new_data):
        """
        Prepare new data for online learning.
        """
        try:
            # Add technical indicators to the new data
            new_data = self.add_technical_indicators(new_data, self.close_column.split('_')[-1])
            logging.debug(f"New data after adding indicators:\n{new_data.head()}")

            # Select the relevant features
            features = new_data[self.feature_names]

            # Scale the features using the pre-fitted scaler
            features_scaled = self.scaler.transform(features)
            logging.debug(f"Scaled new data (first 5 rows):\n{features_scaled[:5]}")

            # Add the scaled data to the online buffer
            self.online_buffer.extend(features_scaled)
            logging.info(f"New data added to online buffer. Buffer size: {len(self.online_buffer)}")

            return True
        except Exception as e:
            logging.error(f"Error preparing new data: {e}")
            return False

    def update_model(self):
        """
        Update the model with new data from the online buffer.
        """
        try:
            # Prepare the new data for training
            X_new, y_new = [], []
            for i in range(len(self.online_buffer) - self.config['lookback']):
                X_new.append(self.online_buffer[i:i + self.config['lookback']])
                y_new.append(self.online_buffer[i + self.config['lookback']][0])
            X_new, y_new = np.array(X_new), np.array(y_new)

            # Update the model
            self.best_model.fit(X_new, y_new, epochs=1, batch_size=32, verbose=0)
            logging.info("Model updated with new data.")
            return True
        except Exception as e:
            logging.error(f"Error updating model: {e}")
            return False
    
    def train(self, ticker, epochs=1000, batch_size=16, patience=100, n_trials=50):
        self.logger.info("Preparing data for training.")

        # Step 1: Prepare the data (only do this once)
        if not hasattr(self, '_prepared_data'):
            self._prepared_data = self.prepare_data(ticker=ticker)
            if self._prepared_data is None:
                self.logger.error("Cannot train the model due to empty dataset.")
                return None, None

        # Ensure self._prepared_data is not None and has the correct structure
        if self._prepared_data is None or len(self._prepared_data) != 6:
            self.logger.error("Invalid prepared data. Cannot proceed with training.")
            return None, None

        X_train, X_val, X_test, y_train, y_val, y_test = self._prepared_data

        # Step 2: Check for NaN values in the data
        if np.isnan(X_train).any() or np.isnan(y_train).any() or np.isnan(X_val).any() or np.isnan(y_val).any():
            self.logger.error("NaN values found in training or validation data.")
            return None, None

        # Step 3: Set input shape for the model
        self.input_shape = X_train.shape[1:]  # Ensure input_shape is set correctly
        self.logger.info(f"Input shape for Transformer: {self.input_shape}")

        # Step 4: Tune hyperparameters (only do this once)
        if not hasattr(self, 'best_params'):
            self.best_params = self.tune_hyperparameters(n_trials=n_trials)  # Use the passed n_trials value
            self.logger.info(f"Best hyperparameters: {self.best_params}")

        # Step 5: Build the model with the best hyperparameters
        self.logger.info("Building the model.")
        model = self.build_model(self.best_params)

        # Step 6: Compile the model with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.best_params.get('learning_rate', 0.001),
            clipvalue=1.0  # Add gradient clipping here
        )
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Step 7: Define the learning rate scheduler
        def lr_scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

        # Step 8: Train the model (only do this once)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)  # Add TensorBoard callback
        training_logger = TrainingLogger()  # Create an instance of TrainingLogger

        self.logger.info("Starting model training.")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, lr_callback, NanDetector(), training_logger, tensorboard],  # Add lr_callback
            verbose=1  # Show progress bar
        )

        # Step 9: Save the model (only do this once)
        if model is not None:
            self.save_model(model)  # Use the new save_model method
        else:
            self.logger.error("Model is None. Cannot save.")

        return model, history
    
    def analyze_features(self, features, ticker):
        try:
            # Step 1: Log the IQR of each feature before scaling
            self.logger.debug("IQR of features before scaling:")
            for col in features.columns:
                q1 = features[col].quantile(0.25)
                q3 = features[col].quantile(0.75)
                iqr = q3 - q1
                self.logger.debug(f"{col}: IQR = {iqr}")

            # Step 2: Use XGBoost to calculate feature importance
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
            model.fit(features.drop(columns=[self.close_column]), features[self.close_column])

            # Log feature importance
            feature_importance = model.feature_importances_
            self.logger.debug(f"Feature importance before adjustment:\n{feature_importance}")

            # Step 3: Adjust feature importance for day trading
            importance_weights = {
                f'RSI_{ticker}': 0.15,               # Strong momentum indicator
                f'MACD_{ticker}': 0.2,               # Strong momentum indicator (increased weight)
                f'MACD_Signal_{ticker}': 0.05,       # Less important than MACD line
                f'Bollinger_Width_{ticker}': 0.15,   # Combined upper and lower bands
                f'ATR_{ticker}': 0.15,               # Strong volatility indicator (increased weight)
                f'OBV_{ticker}': 0.1,                # Volume-based indicator
                f'VWAP_{ticker}': 0.15,              # Volume-weighted indicator (increased weight)
                f'Ichimoku_Tenkan_{ticker}': 0.05,   # Trend indicator
                f'Ichimoku_Kijun_{ticker}': 0.05,    # Added for completeness
                f'ADX_{ticker}': 0.1,                # Trend strength indicator
                f'Parabolic_SAR_{ticker}': 0.05,     # Trend direction indicator (reduced weight)
                f'Donchian_Width_{ticker}': 0.05,    # Combined upper and lower channels
                f'HMA_{ticker}': 0.03,               # Smoothed moving average (reduced weight)
            }

            # Apply weights to feature importance scores only for features that exist in numeric_data
            for feature, weight in importance_weights.items():
                if feature in features.columns:
                    feature_index = features.columns.get_loc(feature)
                    if feature_index < len(feature_importance):  # Ensure the index is within bounds
                        feature_importance[feature_index] *= weight
                    else:
                        self.logger.warning(f"Feature index {feature_index} is out of bounds for feature {feature}.")
                else:
                    self.logger.warning(f"Feature {feature} not found in numeric data. Skipping weight adjustment.")

            self.logger.debug(f"Adjusted feature importance for day trading:\n{feature_importance}")

        except Exception as e:
            self.logger.error(f"Error during feature analysis: {e}", exc_info=True)

            # Step 4: Validate feature scaling (optional)
            if hasattr(self, 'scaler'):
                self.logger.debug("Validating feature scaling:")
                scaled_features = self.scaler.transform(features)
                self.logger.debug(f"Scaled features (first 5 rows):\n{scaled_features[:5]}")
                self.logger.debug(f"Scaler mean: {self.scaler.center_}")
                self.logger.debug(f"Scaler scale: {self.scaler.scale_}")

        except Exception as e:
            self.logger.error(f"Error during feature analysis: {e}", exc_info=True)
            
    def train_final_model(self, best_params, epochs=2000, batch_size=16):
        try:
            if best_params is None:
                self.logger.warning("No best hyperparameters found. Using default hyperparameters.")
                best_params = {
                    'learning_rate': 0.001,
                    'num_heads': 8,
                    'dropout_rate': 0.3,  # Increased dropout rate
                    'num_layers': 3,
                    'ff_dim': 256,
                    'batch_size': 16,
                    'l2_reg': 0.01  # Added L2 regularization
                }

            # Step 1: Prepare data (only do this once)
            if not hasattr(self, '_prepared_data'):
                self._prepared_data = self.prepare_data(ticker='AAPL')  # Replace 'AAPL' with the actual ticker
                if self._prepared_data is None:
                    self.logger.error("Data preparation failed. Cannot proceed with training.")
                    return None, None

            # Ensure self._prepared_data is not None and has the correct structure
            if self._prepared_data is None or len(self._prepared_data) != 6:
                self.logger.error("Invalid prepared data. Cannot proceed with training.")
                return None, None

            X_train, X_val, X_test, y_train, y_val, y_test = self._prepared_data

            # Step 2: Analyze features
            self.analyze_features(self.data[self.feature_names], 'AAPL')  # Replace 'AAPL' with the actual ticker

            # Step 3: Build the model with the best hyperparameters
            self.logger.info("Building the model.")
            model = self.build_model(best_params)
            if model is None:
                self.logger.error("Model building failed. Cannot proceed with training.")
                return None, None

            # Step 4: Log the model summary
            model.summary(print_fn=lambda x: self.logger.info(x))

            # Step 5: Train the model (only do this once)
            self.logger.info("Training the model.")
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
            training_logger = TrainingLogger(logger=self.logger)  # Pass the existing logger

            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr, training_logger],  # Add the TrainingLogger
                verbose=1
            )

            # Step 6: Save the model (only do this once)
            self.save_model(model)  # Save the model after training
            self.logger.info("Model saved successfully.")

            return model, history
        except Exception as e:
            self.logger.error(f"Error during final model training: {e}", exc_info=True)
            raise
    
    def save_model(self, model, filename='trading_model_final.keras'):
        """Save the trained model to a file."""
        try:
            model.save(filename)
            self.logger.info(f"Model saved successfully to {filename}.")
        except Exception as e:
            self.logger.error(f"Failed to save the model: {e}", exc_info=True)
            raise  # Re-raise the exception after logging
    
class NanDetector(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if np.isnan(logs['loss']):
            logging.error(f"NaN loss detected at epoch {epoch}.")
            self.model.stop_training = True

def main():
    try:
        # Load config
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Get the list of tickers from the config
        tickers = config.get("tickers", [])
        if not tickers:
            raise ValueError("No tickers found in the config file.")

        # Fetch data for all tickers
        all_data = []
        for ticker in tickers:
            data = fetch_data(ticker, config['start_date'], config['end_date'], config['interval'])
            if not data.empty:
                all_data.append(data)
            else:
                logger.warning(f"No data fetched for {ticker}. Skipping.")

        if not all_data:
            raise ValueError("No data fetched for any ticker. Check the tickers and dates.")

        # Combine data from all tickers
        combined_data = pd.concat(all_data, axis=1)
        logger.debug(f"Combined data columns: {combined_data.columns}")
        logger.debug(f"Combined data head:\n{combined_data.head()}")

        # Initialize the trainer with the combined data
        trainer = TransformerTrainer(combined_data, config, close_column='Close_AAPL')  # Use the first ticker's close column as a placeholder

        # Add technical indicators for each ticker
        for ticker in tickers:
            logger.info(f"Adding technical indicators for {ticker}.")
            combined_data = trainer.add_technical_indicators(combined_data, ticker)

        # Log the combined data after adding technical indicators
        logger.debug(f"Combined data columns after adding indicators: {combined_data.columns}")
        logger.debug(f"Combined data head after adding indicators:\n{combined_data.head()}")

        # Handle missing data
        combined_data = trainer.handle_missing_data(combined_data)

        # Log the final combined data
        logger.debug(f"Final combined data columns: {combined_data.columns}")
        logger.debug(f"Final combined data head:\n{combined_data.head()}")

        # Define the search space for hyperparameters
        search_space = {
            "learning_rate": tune.loguniform(1e-6, 1e-3),
            "num_heads": tune.choice([2, 4, 8]),
            "dropout_rate": tune.uniform(0.1, 0.5),
            "num_layers": tune.choice([2, 3, 4]),
            "ff_dim": tune.choice([128, 256, 512]),
            "batch_size": tune.choice([16, 32, 64]),
            "l2_reg": tune.loguniform(1e-6, 1e-3),
        }

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
            tune.with_parameters(lambda config, checkpoint_dir=None: trainer.train_with_ray_tune(config, checkpoint_dir, data=combined_data)),
            config=search_space,
            num_samples=5,  # Number of trials
            scheduler=scheduler,
            resources_per_trial={"cpu": 2, "gpu": 1},
            storage_path=custom_log_dir,
            name="transformer_tuning",
            trial_dirname_creator=custom_trial_dirname_creator,
        )

        # Fetch the best configuration
        best_config = analysis.get_best_config(metric="loss", mode="min")
        print("Best hyperparameters found: ", best_config)

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