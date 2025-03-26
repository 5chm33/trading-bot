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
from typing import Dict, Any  

logger = logging.getLogger(__name__)

class TransformerTrainer:
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any], close_column: str):
        self.data = data
        self.config = config
        self.close_column = close_column
        
        # Get scaling method from config
        scaling_config = self.config['data']['scaling']
        if scaling_config['price_scaler'] == "MinMax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaling_config['price_scaler'] == "Robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.input_shape = None
        self.feature_names = []
        self.online_buffer = deque(maxlen=1000)

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

    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data using config parameters with comprehensive validation."""
        try:
            # Get parameters from config
            lookback = self.config['model']['transformer']['architecture']['lookback']
            test_size = self.config['data']['validation']['test_size']
            min_data_points = self.config['data']['validation']['min_data_points']
            
            self.logger.info(f"Preparing data with lookback={lookback}, test_size={test_size}")

            if data is None or data.empty:
                self.logger.error("Input data is None or empty")
                return None

            # Validate minimum data length
            if len(data) < min_data_points:
                self.logger.error(f"Data has only {len(data)} points, need at least {min_data_points}")
                return None

            # Get ticker prefix and observation columns from config
            prefix = self.close_column.lower().split('_')[0] + "_"
            observation_cols = [
                f"{prefix}close",
                f"{prefix}volatility",
                f"{prefix}rsi",
                f"{prefix}macd", 
                f"{prefix}atr",
                f"{prefix}obv"
            ]

            # Validate required columns
            missing_obs = [col for col in observation_cols if col not in data.columns]
            if missing_obs:
                available_cols = [col for col in data.columns if col.startswith(prefix)]
                self.logger.error(
                    f"Missing observation columns:\nMissing: {missing_obs}\nAvailable: {available_cols}"
                )
                return None

            # Select and scale features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            features = data[numeric_cols]
            
            # Scale features and targets
            X_scaled = self.scaler.fit_transform(features)
            y_scaled = self.target_scaler.fit_transform(data[observation_cols].values)

            # Create sequences
            X, y = [], []
            for i in range(lookback, len(X_scaled)):
                X.append(X_scaled[i-lookback:i])
                y.append(y_scaled[i])

            X, y = np.array(X), np.array(y)

            # Split data
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            self.logger.info(
                f"Data prepared - X_train: {X_train.shape}, X_val: {X_val.shape}, "
                f"Features: {features.shape[1]}"
            )
            
            return X_train, X_val, y_train, y_val, self.scaler, self.target_scaler, features.shape[1]
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}", exc_info=True)
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
        """Add complete set of technical indicators with NaN handling."""
        try:
            ticker = ticker.lower()
            prefix = f"{ticker}_"
            
            # 1. Validate OHLCV columns exist
            required_cols = [f"{prefix}open", f"{prefix}high", 
                            f"{prefix}low", f"{prefix}close", 
                            f"{prefix}volume"]
            
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing base columns for {ticker}: {missing_cols}")

            # 2. Extract price data
            close_prices = data[f"{prefix}close"].ffill().values.astype(float)
            high_prices = data[f"{prefix}high"].ffill().values.astype(float)
            low_prices = data[f"{prefix}low"].ffill().values.astype(float)
            volume = data[f"{prefix}volume"].ffill().values.astype(float)

            # 3. Add core indicators
            data = self._add_moving_averages(data, close_prices, ticker)
            data = self._add_bollinger_bands(data, close_prices, ticker)
            data = self._add_vwap(data, f"{prefix}close", f"{prefix}volume", ticker)
            
            # 4. Add momentum indicators
            data[f"{prefix}rsi"] = talib.RSI(close_prices, 14)
            macd, macd_signal, _ = talib.MACD(close_prices)
            data[f"{prefix}macd"] = macd
            data[f"{prefix}macd_signal"] = macd_signal
            
            # 5. Add volatility indicators
            data[f"{prefix}atr"] = talib.ATR(high_prices, low_prices, close_prices, 14)
            data[f"{prefix}volatility"] = data[f"{prefix}close"].pct_change().rolling(14).std()
            
            # 6. Add volume indicators
            data[f"{prefix}obv"] = talib.OBV(close_prices, volume)
            
            # 7. Add advanced indicators with NaN handling
            try:
                data = self._add_ichimoku_cloud(data, close_prices, high_prices, low_prices, ticker)
            except Exception as e:
                logger.warning(f"Ichimoku Cloud failed for {ticker}: {str(e)}")
                
            try:
                data = self._add_adx(data, high_prices, low_prices, close_prices, ticker)
            except Exception as e:
                logger.warning(f"ADX failed for {ticker}: {str(e)}")

            return data
            
        except Exception as e:
            logger.error(f"Failed adding indicators for {ticker}: {str(e)}", exc_info=True)
            raise
    
    def _add_moving_averages(self, data: pd.DataFrame, close_prices: np.ndarray, ticker: str) -> pd.DataFrame:
        """Add moving averages using config parameters."""
        try:
            prefix = f"{ticker.lower()}_"
            ma_config = self.config['indicators']['moving_averages']
            
            # Ensure proper input format
            close_prices = np.asarray(close_prices, dtype=np.float64).flatten()
            
            # Add EMAs from config
            for window in ma_config['ema']:
                data[f"{prefix}ema_{window}"] = talib.EMA(close_prices, timeperiod=window)
                
            # Add SMAs from config
            for window in ma_config['sma']:
                data[f"{prefix}sma_{window}"] = talib.SMA(close_prices, timeperiod=window)
                
            return data
            
        except Exception as e:
            self.logger.error(f"Moving averages failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_bollinger_bands(self, data: pd.DataFrame, close_prices: np.ndarray, ticker: str) -> pd.DataFrame:
        """Add Bollinger Bands using config parameters."""
        try:
            prefix = f"{ticker.lower()}_"
            bb_config = self.config['indicators']['bollinger']
            
            close_prices = np.asarray(close_prices, dtype=np.float64).flatten()
            
            upper, middle, lower = talib.BBANDS(
                close_prices,
                timeperiod=bb_config['window'],
                nbdevup=bb_config['std_dev'],
                nbdevdn=bb_config['std_dev']
            )
            
            data[f"{prefix}bb_upper"] = upper
            data[f"{prefix}bb_middle"] = middle 
            data[f"{prefix}bb_lower"] = lower
            
            return data
            
        except Exception as e:
            self.logger.error(f"Bollinger Bands failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_vwap(self, data, close_col, volume_col, ticker):
        """Calculate Volume-Weighted Average Price."""
        try:
            prefix = f"{ticker.lower()}_"
            
            typical_price = (data[close_col] + 
                        data[f"{prefix}high"] + 
                        data[f"{prefix}low"]) / 3
            cumulative_volume = data[volume_col].cumsum()
            vwap = (typical_price * data[volume_col]).cumsum() / cumulative_volume
            
            data[f"{prefix}vwap"] = vwap
            return data
        except Exception as e:
            self.logger.error(f"VWAP calculation failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_rsi(self, data: pd.DataFrame, close_prices: np.ndarray, ticker: str) -> pd.DataFrame:
        """Add RSI using config parameters."""
        try:
            prefix = f"{ticker.lower()}_"
            rsi_config = self.config['indicators']['rsi']
            
            data[f"{prefix}rsi"] = talib.RSI(
                close_prices,
                timeperiod=rsi_config['window']
            )
            
            # Add overbought/oversold markers
            data[f"{prefix}rsi_overbought"] = rsi_config['overbought']
            data[f"{prefix}rsi_oversold"] = rsi_config['oversold']
            
            return data
            
        except Exception as e:
            self.logger.error(f"RSI failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_macd(self, data, close_prices, ticker):
        """Add Moving Average Convergence Divergence."""
        try:
            prefix = f"{ticker.lower()}_"
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            data[f"{prefix}macd"] = macd
            data[f"{prefix}macd_signal"] = macd_signal
            data[f"{prefix}macd_hist"] = macd_hist
            return data
        except Exception as e:
            self.logger.error(f"MACD calculation failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_stochastic_oscillator(self, data, high_prices, low_prices, close_prices, ticker):
        """Add Stochastic Oscillator."""
        try:
            prefix = f"{ticker.lower()}_"
            slowk, slowd = talib.STOCH(
                high_prices, 
                low_prices, 
                close_prices,
                fastk_period=14,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            data[f"{prefix}stochastic"] = slowk
            return data
        except Exception as e:
            self.logger.error(f"Stochastic Oscillator failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_atr(self, data, high_prices, low_prices, close_prices, ticker):
        """Add Average True Range."""
        try:
            prefix = f"{ticker.lower()}_"
            data[f"{prefix}atr"] = talib.ATR(
                high_prices, 
                low_prices, 
                close_prices, 
                timeperiod=14
            )
            return data
        except Exception as e:
            self.logger.error(f"ATR calculation failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_parabolic_sar(self, data, high_prices, low_prices, ticker):
        """Add Parabolic SAR."""
        try:
            prefix = f"{ticker.lower()}_"
            data[f"{prefix}parabolic_sar"] = talib.SAR(
                high_prices, 
                low_prices, 
                acceleration=0.02, 
                maximum=0.2
            )
            return data
        except Exception as e:
            self.logger.error(f"Parabolic SAR failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_ichimoku_cloud(self, data, close_prices, high_prices, low_prices, ticker):
        """Add Ichimoku Cloud components."""
        try:
            prefix = f"{ticker.lower()}_"
            close_prices = pd.Series(close_prices)
            high_prices = pd.Series(high_prices)
            low_prices = pd.Series(low_prices)

            tenkan_sen = (talib.MAX(high_prices, timeperiod=9) + 
                        talib.MIN(low_prices, timeperiod=9)) / 2
            kijun_sen = (talib.MAX(high_prices, timeperiod=26) + 
                        talib.MIN(low_prices, timeperiod=26)) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            senkou_span_b = ((talib.MAX(high_prices, timeperiod=52) + 
                            talib.MIN(low_prices, timeperiod=52)) / 2).shift(26)
            chikou_span = close_prices.shift(-26)

            data[f"{prefix}ichimoku_tenkan"] = tenkan_sen
            data[f"{prefix}ichimoku_kijun"] = kijun_sen
            data[f"{prefix}ichimoku_senkou_a"] = senkou_span_a
            data[f"{prefix}ichimoku_senkou_b"] = senkou_span_b
            data[f"{prefix}ichimoku_chikou"] = chikou_span

            for col in [f"{prefix}ichimoku_tenkan", f"{prefix}ichimoku_kijun",
                    f"{prefix}ichimoku_senkou_a", f"{prefix}ichimoku_senkou_b",
                    f"{prefix}ichimoku_chikou"]:
                data[col] = data[col].ffill()

            return data
        except Exception as e:
            self.logger.error(f"Ichimoku Cloud failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_adx(self, data, high_prices, low_prices, close_prices, ticker):
        """Add Average Directional Index."""
        try:
            prefix = f"{ticker.lower()}_"
            data[f"{prefix}adx"] = talib.ADX(
                high_prices, 
                low_prices, 
                close_prices, 
                timeperiod=14
            )
            return data
        except Exception as e:
            self.logger.error(f"ADX calculation failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_obv(self, data, close_prices, volume, ticker):
        """Add On-Balance Volume."""
        try:
            prefix = f"{ticker.lower()}_"
            data[f"{prefix}obv"] = talib.OBV(close_prices, volume)
            return data
        except Exception as e:
            self.logger.error(f"OBV calculation failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_donchian_channels(self, data, high_prices, low_prices, ticker):
        """Add Donchian Channels."""
        try:
            prefix = f"{ticker.lower()}_"
            upper = talib.MAX(high_prices, timeperiod=20)
            lower = talib.MIN(low_prices, timeperiod=20)
            data[f"{prefix}donchian_upper"] = upper
            data[f"{prefix}donchian_lower"] = lower
            return data
        except Exception as e:
            self.logger.error(f"Donchian Channels failed for {ticker}: {str(e)}", exc_info=True)
            raise

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
            model_config = self.config['model']['transformer']['architecture']
            reg_config = self.config['model']['transformer']['regularization']
            
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

    def train_with_ray_tune(self, config: Dict[str, Any], checkpoint_dir=None, data=None):
        """Train with Ray Tune using dynamic config parameters."""
        try:
            # Merge ray config with main config
            full_config = {
                **self.config['model']['transformer'],
                **config
            }
            training_config = self.config['model']['transformer']['training']
            
            # Prepare data
            prepared_data = self.prepare_data(data if data else self.data)
            if prepared_data is None:
                raise ValueError("Data preparation failed")
                
            X_train, X_val, y_train, y_val, _, _, num_features = prepared_data

            # Build model
            model = self.build_model(full_config, num_features)
            if model is None:
                raise ValueError("Model building failed")

            # Configure optimizer
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=full_config['learning_rate'],
                clipvalue=self.config['model']['transformer']['regularization']['gradient_clip']
            )
            
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            # Configure callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=training_config['patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=training_config['patience']//2,
                    min_lr=1e-6
                )
            ]
            
            if checkpoint_dir:
                callbacks.append(ModelCheckpoint(
                    filepath=os.path.join(checkpoint_dir, "model.keras"),
                    save_best_only=True,
                    monitor="val_loss"
                ))

            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=training_config['epochs'],
                batch_size=full_config['batch_size'],
                callbacks=callbacks,
                verbose=0
            )

            # Report metrics
            val_loss = history.history['val_loss'][-1]
            tune.report({
                "loss": val_loss,
                "epochs": len(history.history['loss'])
            })
            
            return {"loss": val_loss}
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return {"loss": float('inf')}