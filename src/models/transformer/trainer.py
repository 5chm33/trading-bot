# src/models/transformer/trainer.py
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from src.utils.data_schema import ColumnSchema
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class TransformerTrainer:
    """State-of-the-art Transformer for financial time series"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = RobustScaler()
        
    def prepare_data(self, data: pd.DataFrame) -> Optional[Tuple]:
        """Prepare data for transformer training"""
        try:
            data = self._add_regime_markers(data)
            X, y = self._create_sequences(
                self._scale_features(data),
                self.config['model']['transformer']['architecture']['lookback']
            )
            split_idx = int(len(X) * 0.8)
            return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
        except Exception as e:
            logger.error(f"Data prep failed: {str(e)}", exc_info=True)
            return None

    def build_model(self, num_features: int) -> Model:
        """Build quantile output transformer model"""
        config = self.config['model']['transformer']
        seq_length = config['architecture']['sequence_length']
        
        # Input layer
        inputs = Input(shape=(seq_length, num_features))
        
        # Transformer blocks
        x = inputs
        for i in range(config['architecture']['num_layers']):
            # Self-attention
            attn_output = MultiHeadAttention(
                num_heads=config['architecture']['num_heads'],
                key_dim=num_features // config['architecture']['num_heads'],
                dropout=config['regularization']['dropout_rate']
            )(x, x, attention_mask=self._causal_mask(seq_length))
            
            # Residual connections
            x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_output]))
            
            # Feed forward
            ffn = Sequential([
                Dense(config['architecture']['ff_dim'], activation='gelu',
                    kernel_regularizer=l2(config['regularization']['l2_reg'])),
                Dropout(config['regularization']['dropout_rate']),
                Dense(num_features, kernel_regularizer=l2(config['regularization']['l2_reg']))
            ])(x)
            
            x = LayerNormalization(epsilon=1e-6)(Add()([x, ffn]))
        
        # Output layers
        pooled = GlobalAveragePooling1D()(x)
        outputs = [Dense(6, activation='tanh', name=f'q_{q}')(pooled) 
                  for q in [10, 50, 90]]
        
        return Model(inputs, outputs)

    def train_model(self, model: Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.callbacks.History:
        """Train model with regime-aware weighting"""
        # Sample weights based on regime
        train_regimes = X_train[:, -1, -1]
        sample_weights = np.where(
            np.abs(train_regimes) > 0.5, 1.5,
            np.where(np.abs(train_regimes) > 0.2, 1.2, 1.0))
        
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=self.config['model']['transformer']['training']['learning_rate'],
                weight_decay=1e-6
            ),
            loss='huber_loss',
            metrics=['mae']
        )
        
        return model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            validation_data=(X_val, y_val),
            epochs=self.config['model']['transformer']['training']['epochs'],
            batch_size=self.config['model']['transformer']['training']['batch_size'],
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=2
        )

    # Helper methods
    def _add_regime_markers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime classification"""
        close_col = [c for c in data.columns if c.endswith('_close')][0]
        returns = data[close_col].pct_change()
        volatility = returns.rolling(21).std()
        data['regime'] = np.where(
            returns > 2*volatility, 1,
            np.where(returns < -2*volatility, -1, 0)
        )
        return data

    def _scale_features(self, data: pd.DataFrame) -> np.ndarray:
        """Robust scaling per market regime"""
        numeric_cols = [c for c in data.columns if c != 'regime']
        scaled = np.zeros_like(data[numeric_cols])
        
        for regime in data['regime'].unique():
            mask = (data['regime'] == regime)
            if sum(mask) > 0:
                scaler = RobustScaler()
                scaled[mask] = scaler.fit_transform(data.loc[mask, numeric_cols])
        
        return scaled

    def _create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create time series sequences"""
        n_samples = len(data) - lookback + 1
        sequences = np.lib.stride_tricks.as_strided(
            data,
            shape=(n_samples, lookback, data.shape[1]),
            strides=(data.strides[0], data.strides[0], data.strides[1]))
        
        targets = data[lookback:, 0]  # Predict close price changes
        return sequences.astype(np.float32), targets.astype(np.float32)

    def _causal_mask(self, size: int) -> np.ndarray:
        """Prevent attention to future timesteps"""
        return np.triu(np.ones((size, size)), k=1) * -1e9
    
    def create_tf_dataset(self, X: np.ndarray, y: np.ndarray) -> tf.data.Dataset:
        """Create optimized TF Dataset pipeline"""
        return tf.data.Dataset.from_tensor_slices((X, y)) \
            .shuffle(10000, reshuffle_each_iteration=True) \
            .batch(self.config['model']['transformer']['training']['batch_size']) \
            .prefetch(tf.data.AUTOTUNE) \
            .cache()
    
class TransformerFeatureProcessor:
    """Handles real-time feature transformation using trained transformer"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = RobustScaler()
        self.lookback = config['model']['transformer']['architecture']['lookback']
        self.feature_columns = None
        self.transformer_model = None
        
    def fit(self, data: pd.DataFrame):
        """Initialize with training data"""
        self.feature_columns = [c for c in data.columns if not c.startswith('transformer_')]
        
        # Initialize and train transformer
        trainer = TransformerTrainer(
            data[self.feature_columns],
            self.config,
            close_column=self._get_close_column(data)
        )
        X, y = trainer._create_sequences(
            trainer._scale_features(data),
            self.lookback
        )
        self.transformer_model = trainer.build_model(X.shape[-1])
        trainer.train_model(self.transformer_model, X, y, X[:100], y[:100])  # Small validation set
        
    def transform(self, market_data: np.ndarray) -> np.ndarray:
        """Transform single timestep of market data"""
        if self.transformer_model is None:
            raise RuntimeError("Processor must be fit() first")
            
        # Convert to sequence format
        seq = np.stack([market_data] * self.lookback)  # Repeat current obs for lookback
        seq = torch.FloatTensor(seq).unsqueeze(0)
        
        # Get transformer features
        with torch.no_grad():
            features = self.transformer_model(seq).numpy()[0]
            
        return features
        
    def _get_close_column(self, data: pd.DataFrame) -> str:
        """Find first close price column"""
        for col in data.columns:
            if col.endswith('_close'):
                return col
        raise ValueError("No close price column found in data")
    