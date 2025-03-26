import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import logging
import talib
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from ray import tune
import os
import sys
from collections import deque

logger = logging.getLogger(__name__)

class TransformerTrainer:
    def __init__(self, data, config, close_column):
        self.data = data
        self.config = config
        self.close_column = close_column
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Scaler for features
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))  # Scaler for target
        self.input_shape = None
        self.feature_names = []
        self.online_buffer = deque(maxlen=1000)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        # Initialize the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add handlers if not already added
        if not self.logger.handlers:
            file_handler = logging.FileHandler('transformer_trainer.log', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.propagate = False

        self.logger.debug("TransformerTrainer initialized successfully.")

    def prepare_data(self, data, lookback=30):
        try:
            self.logger.info("Preparing data for training.")

            if data is None or data.empty:
                self.logger.error("Input data is None or empty. Check data loading.")
                return None

            # Step 1: Normalize column names to lowercase
            data.columns = [col.strip().lower() for col in data.columns]

            # Step 2: Select numeric features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            features = data[numeric_cols]

            # Step 3: Scale the features using the feature scaler
            self.logger.info("Scaling features.")
            X_scaled = self.scaler.fit_transform(features)

            # Step 4: Extract the 6 features for the observation space
            observation_cols = ['close_aapl', 'volatility_aapl', 'rsi_aapl', 'macd_aapl', 'atr_aapl', 'obv_aapl']
            if not all(col in data.columns for col in observation_cols):
                self.logger.error(f"Observation columns not found in data. Available columns: {data.columns}")
                return None

            observation_features = data[observation_cols].values

            # Step 5: Scale the observation features using the target scaler
            self.logger.info("Scaling the observation features.")
            y_scaled = self.target_scaler.fit_transform(observation_features)

            # Step 6: Create sequences for the Transformer model
            self.logger.info("Creating sequences for the Transformer model.")
            X, y = [], []
            for i in range(lookback, len(X_scaled)):
                X.append(X_scaled[i - lookback:i])  # Sequence of features
                y.append(y_scaled[i])  # Target: 6-dimensional observation at the current timestep

            X, y = np.array(X), np.array(y)

            # Log the shapes of the sequences
            self.logger.debug(f"X shape after sequence creation: {X.shape}")
            self.logger.debug(f"y shape after sequence creation: {y.shape}")

            # Step 7: Split into training and validation sets
            self.logger.info("Splitting data into training and validation sets.")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Log the shapes of the prepared data
            self.logger.info(f"X_train shape: {X_train.shape}")
            self.logger.info(f"X_val shape: {X_val.shape}")
            self.logger.info(f"y_train shape: {y_train.shape}")
            self.logger.info(f"y_val shape: {y_val.shape}")

            return X_train, X_val, y_train, y_val, self.scaler, self.target_scaler, features.shape[1]
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return None
    
    def _calculate_volatility(self, data, close_column, window=14):
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
    
    def add_technical_indicators(self, data, ticker):
        """Add essential technical indicators to the data for a specific ticker."""
        try:
            self.logger.info(f"Adding essential technical indicators for {ticker}.")

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
                self.logger.warning(f"NaN values found in input data for {ticker}. Forward-filling and backward-filling.")
                close_prices = pd.Series(close_prices).ffill().bfill().values
                high_prices = pd.Series(high_prices).ffill().bfill().values
                low_prices = pd.Series(low_prices).ffill().bfill().values
                volume = pd.Series(volume).ffill().bfill().values

            # Add essential technical indicators
            if len(data) >= 14:  # Minimum required for most indicators
                # Price-Based Indicators
                self.logger.info("Adding Moving Averages.")
                data = self._add_moving_averages(data, close_prices, ticker)
                self.logger.info("Adding Bollinger Bands.")
                data = self._add_bollinger_bands(data, close_prices, ticker)
                self.logger.info("Adding VWAP.")
                data = self._add_vwap(data, close_col, volume_col, ticker)

                # Momentum Indicators
                self.logger.info("Adding RSI.")
                data = self._add_rsi(data, close_prices, ticker)
                self.logger.info("Adding MACD.")
                data = self._add_macd(data, close_prices, ticker)
                self.logger.info("Adding Stochastic Oscillator.")
                data = self._add_stochastic_oscillator(data, high_prices, low_prices, close_prices, ticker)
                
                # Volatility Indicators
                logger.info("Adding ATR.")
                data = self._add_atr(data, high_prices, low_prices, close_prices, ticker)  # Adds ATR
                logger.info("Adding Parabolic SAR.")
                data = self._add_parabolic_sar(data, high_prices, low_prices, ticker)      # Adds Parabolic SAR
                logger.info("Adding Volatility.")
                data[f'Volatility_{ticker}'] = self._calculate_volatility(data, close_col, window=14)
                
                # Trend Indicators
                self.logger.info("Adding Ichimoku Cloud.")
                data = self._add_ichimoku_cloud(data, close_prices, high_prices, low_prices, ticker)
                self.logger.info("Adding ADX.")
                data = self._add_adx(data, high_prices, low_prices, close_prices, ticker)

                # Other Indicators
                self.logger.info("Adding OBV.")
                data = self._add_obv(data, close_prices, volume, ticker)
                self.logger.info("Adding Donchian Channels.")
                data = self._add_donchian_channels(data, high_prices, low_prices, ticker)
            else:
                self.logger.warning(f"Insufficient data for {ticker}. Skipping technical indicators.")

            # Handle missing data
            data = self.handle_missing_data(data)

            # Log the columns after adding indicators
            self.logger.debug(f"Columns after adding technical indicators: {data.columns}")

            return data
        except Exception as e:
            self.logger.error(f"Error adding technical indicators for {ticker}: {e}")
            return None
    
    def _add_moving_averages(self, data, close_prices, ticker):
        """Add Exponential Moving Averages (EMA) to the data."""
        data[f'EMA_10_{ticker}'] = talib.EMA(close_prices, timeperiod=10)
        data[f'EMA_30_{ticker}'] = talib.EMA(close_prices, timeperiod=30)
        return data

    def _add_bollinger_bands(self, data, close_prices, ticker):
        """Add Bollinger Bands to the data."""
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        data[f'Bollinger_Upper_{ticker}'] = upper
        data[f'Bollinger_Middle_{ticker}'] = middle
        data[f'Bollinger_Lower_{ticker}'] = lower
        return data

    def _add_vwap(self, data, close_col, volume_col, ticker):
        """Add Volume-Weighted Average Price (VWAP) to the data."""
        typical_price = (data[close_col] + data[f'High_{ticker}'] + data[f'Low_{ticker}']) / 3
        cumulative_volume = np.cumsum(data[volume_col])
        cumulative_price_volume = np.cumsum(typical_price * data[volume_col])
        data[f'VWAP_{ticker}'] = cumulative_price_volume / cumulative_volume
        return data

    def _add_rsi(self, data, close_prices, ticker):
        """Add Relative Strength Index (RSI) to the data."""
        data[f'RSI_{ticker}'] = talib.RSI(close_prices, timeperiod=14)
        return data

    def _add_macd(self, data, close_prices, ticker):
        """Add Moving Average Convergence Divergence (MACD) to the data."""
        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        data[f'MACD_{ticker}'] = macd
        data[f'MACD_Signal_{ticker}'] = macd_signal
        data[f'MACD_Hist_{ticker}'] = macd_hist
        return data

    def _add_stochastic_oscillator(self, data, high_prices, low_prices, close_prices, ticker):
        """Add Stochastic Oscillator to the data."""
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        data[f'Stochastic_{ticker}'] = slowk
        return data

    def _add_atr(self, data, high_prices, low_prices, close_prices, ticker):
        """Add Average True Range (ATR) to the data."""
        data[f'ATR_{ticker}'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        return data

    def _add_parabolic_sar(self, data, high_prices, low_prices, ticker):
        """Add Parabolic SAR to the data."""
        data[f'Parabolic_SAR_{ticker}'] = talib.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2)
        return data

    def _add_ichimoku_cloud(self, data, close_prices, high_prices, low_prices, ticker):
        """
        Add Ichimoku Cloud components to the data.
        """
        # Convert numpy arrays to pandas Series for shifting
        close_prices = pd.Series(close_prices)
        high_prices = pd.Series(high_prices)
        low_prices = pd.Series(low_prices)

        # Tenkan-sen (Conversion Line)
        tenkan_sen = (talib.MAX(high_prices, timeperiod=9) + talib.MIN(low_prices, timeperiod=9)) / 2

        # Kijun-sen (Base Line)
        kijun_sen = (talib.MAX(high_prices, timeperiod=26) + talib.MIN(low_prices, timeperiod=26)) / 2

        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)  # Shift forward by 26 periods

        # Senkou Span B (Leading Span B)
        senkou_span_b = ((talib.MAX(high_prices, timeperiod=52) + talib.MIN(low_prices, timeperiod=52)) / 2).shift(26)  # Shift forward by 26 periods

        # Chikou Span (Lagging Span)
        chikou_span = close_prices.shift(-26)  # Shift backward by 26 periods

        # Add Ichimoku components to the DataFrame
        data[f'Ichimoku_Tenkan_{ticker}'] = tenkan_sen
        data[f'Ichimoku_Kijun_{ticker}'] = kijun_sen
        data[f'Ichimoku_Senkou_A_{ticker}'] = senkou_span_a
        data[f'Ichimoku_Senkou_B_{ticker}'] = senkou_span_b
        data[f'Ichimoku_Chikou_{ticker}'] = chikou_span

        # Forward-fill NaN values for Ichimoku components
        data[f'Ichimoku_Tenkan_{ticker}'] = data[f'Ichimoku_Tenkan_{ticker}'].ffill()
        data[f'Ichimoku_Kijun_{ticker}'] = data[f'Ichimoku_Kijun_{ticker}'].ffill()
        data[f'Ichimoku_Senkou_A_{ticker}'] = data[f'Ichimoku_Senkou_A_{ticker}'].ffill()
        data[f'Ichimoku_Senkou_B_{ticker}'] = data[f'Ichimoku_Senkou_B_{ticker}'].ffill()
        data[f'Ichimoku_Chikou_{ticker}'] = data[f'Ichimoku_Chikou_{ticker}'].ffill()

        return data

    def _add_adx(self, data, high_prices, low_prices, close_prices, ticker):
        """Add Average Directional Index (ADX) to the data."""
        data[f'ADX_{ticker}'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        return data

    def _add_obv(self, data, close_prices, volume, ticker):
        """Add On-Balance Volume (OBV) to the data."""
        data[f'OBV_{ticker}'] = talib.OBV(close_prices, volume)
        return data

    def _add_donchian_channels(self, data, high_prices, low_prices, ticker):
        """Add Donchian Channels to the data."""
        upper = talib.MAX(high_prices, timeperiod=20)
        lower = talib.MIN(low_prices, timeperiod=20)
        data[f'Donchian_Upper_{ticker}'] = upper
        data[f'Donchian_Lower_{ticker}'] = lower
        return data

    def handle_missing_data(self, df):
        """Handle missing data using forward-fill, backward-fill, and KNN imputation."""
        try:
            logging.info("Handling missing data using forward-fill, backward-fill, and KNN imputation.")

            # Log missing values before handling
            logging.debug(f"Missing values before handling:\n{df.isnull().sum()}")

            # Step 1: Drop columns with all NaN values
            cols_with_all_nan = df.columns[df.isna().all()]
            if len(cols_with_all_nan) > 0:
                logging.warning(f"Dropping columns with all NaN values: {cols_with_all_nan}")
                df = df.drop(columns=cols_with_all_nan)

            # Step 2: Forward-fill missing values
            df = df.ffill()

            # Step 3: Backward-fill any remaining missing values
            df = df.bfill()

            # Log missing values after forward-fill and backward-fill
            logging.debug(f"Missing values after forward-fill and backward-fill:\n{df.isnull().sum()}")

            # Step 4: Apply KNN imputation for all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Apply KNN imputation
                imputer = KNNImputer(n_neighbors=5)
                imputed_data = imputer.fit_transform(df[numeric_cols])

                # Replace the original numeric columns with the imputed data
                df[numeric_cols] = imputed_data

            # Log missing values after imputation
            logging.debug(f"Missing values after imputation:\n{df.isnull().sum()}")

            return df
        except Exception as e:
            logging.error(f"Error handling missing data: {e}")
            return None
    
    def build_model(self, config, num_features):
        """Build the Transformer-based model."""
        try:
            # Ensure sequence_length is in the config
            sequence_length = config.get("sequence_length", 30)  # Default to 30 if not provided

            # Input layer
            inputs = Input(shape=(sequence_length, num_features))
            
            # Transformer layers
            x = MultiHeadAttention(num_heads=config["num_heads"], key_dim=num_features)(inputs, inputs)
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Dropout(config.get("dropout_rate", 0.1))(x)
            
            # Global average pooling to reduce sequence to a single vector
            x = GlobalAveragePooling1D()(x)
            
            # Output layer (6 features for SAC observation space)
            outputs = Dense(6, activation='linear', kernel_regularizer=l2(config.get("l2_reg", 0.01)))(x)
            
            # Create the model
            model = Model(inputs, outputs)
            return model
        except Exception as e:
            logger.error(f"Error building model: {e}", exc_info=True)
            return None

    def train_with_ray_tune(self, config, checkpoint_dir=None, data=None):
        """
        Train the model with Ray Tune.

        Args:
            config (dict): Hyperparameter configuration.
            checkpoint_dir (str): Directory for saving checkpoints.
            data (pd.DataFrame): The input data.

        Returns:
            dict: A dictionary containing the validation loss.
        """
        try:
            if data is None:
                data = self.data

            # Prepare data for training
            lookback = config.get("lookback", 30)
            prepared_data = self.prepare_data(data, lookback=lookback)
            if prepared_data is None:
                raise ValueError("Data preparation failed.")

            # Unpack the prepared data
            X_train, X_val, y_train, y_val, feature_scaler, target_scaler, num_features = prepared_data

            # Build the model
            model = self.build_model(config, num_features)
            if model is None:
                raise ValueError("Model building failed.")

            # Compile the model
            optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            # Define callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
            ]
            if checkpoint_dir:
                checkpoint_callback = ModelCheckpoint(
                    filepath=os.path.join(checkpoint_dir, "model.keras"),
                    save_best_only=True,
                    monitor="val_loss",
                    mode="min",
                )
                callbacks.append(checkpoint_callback)

            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=config.get("epochs", 100),
                batch_size=config["batch_size"],
                callbacks=callbacks,
                verbose=0
            )

            # Evaluate the model on the validation set
            val_loss = history.history.get('val_loss', [float('inf')])[-1]
            logger.info(f"Validation loss: {val_loss}")

            # Report metrics to Ray Tune
            tune.report({"loss": val_loss})

            return {"loss": val_loss}
        except Exception as e:
            logger.error(f"Error during training with Ray Tune: {e}", exc_info=True)
            return {"loss": float('inf')}