import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import (PowerTransformer, 
                                 QuantileTransformer, 
                                 RobustScaler)
from typing import List, Dict, Optional, Any
from collections import defaultdict, deque
import logging 

logger = logging.getLogger(__name__)

class RegimeDetector:
    """SOTA market regime detection using volatility clustering and GMM"""
    
    def __init__(self, 
                 n_regimes: int = 3,
                 lookback_window: int = 21,
                 volatility_thresholds: List[float] = [0.15, 0.3]):
        self.gmm = GaussianMixture(n_components=n_regimes)
        self.lookback = lookback_window
        self.vol_thresholds = sorted(volatility_thresholds)
        self.regime_labels = ['low_vol', 'medium_vol', 'high_vol']
        self._is_fitted = False
        
    def fit(self, returns: np.ndarray):
        """Fit on returns (percentage changes)"""
        returns = pd.Series(returns).dropna()
        volatilities = returns.rolling(self.lookback).std().dropna()
        log_vol = np.log(volatilities + 1e-6).values.reshape(-1, 1)
        self.gmm.fit(log_vol)
        self._is_fitted = True
        return self
        
    def predict(self, returns: np.ndarray) -> str:
        """Predict current regime"""
        if not self._is_fitted:
            raise RuntimeError("RegimeDetector not fitted. Call fit() first.")
            
        recent_returns = pd.Series(returns).dropna()
        if len(recent_returns) < self.lookback:
            return 'medium_vol'  # Default regime
            
        recent_vol = np.log(recent_returns.rolling(self.lookback).std().iloc[-1] + 1e-6)
        regime_idx = self.gmm.predict([[recent_vol]])[0]
        return self.regime_labels[regime_idx]

class RegimeAwareScaler(TransformerMixin, BaseEstimator):
    """Adaptive scaling per market regime with robustness features"""
    
    def __init__(self, 
                 regime_key: str = 'volatility_regime',
                 scaler_type: str = 'robust'):
        """
        Args:
            regime_key: Column/key name for regime labels
            scaler_type: 'power', 'quantile', or 'robust'
        """
        self.regime_key = regime_key
        self.scaler_type = scaler_type
        self.scalers = None
        self.detector = RegimeDetector()
        self._feature_columns = None
        
    def _init_scaler(self):
        if self.scaler_type == 'power':
            return PowerTransformer(method='yeo-johnson')
        elif self.scaler_type == 'quantile':
            return QuantileTransformer(output_distribution='normal')
        else:
            return RobustScaler(quantile_range=(5, 95))
    
    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        returns = X['close'].pct_change().dropna()
        self.detector.fit(returns.values)
        
        # Initialize scalers per regime
        self.scalers = {
            regime: self._init_scaler() 
            for regime in self.detector.regime_labels
        }
        
        # Store feature columns for transform
        self._feature_columns = [col for col in X.columns if col != self.regime_key]
        
        for regime, scaler in self.scalers.items():
            regime_mask = (X[self.regime_key] == regime)
            if regime_mask.any():
                scaler.fit(X.loc[regime_mask, self._feature_columns])
        return self
        
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.scalers is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
            
        current_returns = X['close'].pct_change().dropna().values
        current_regime = self.detector.predict(current_returns)
        
        # Ensure DataFrame structure
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._feature_columns + [self.regime_key])
            
        return self.scalers[current_regime].transform(X[self._feature_columns])
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

class NormalizationManager:
    """Central hub for all normalization needs with enhanced error handling"""

    @staticmethod
    def pre_normalize_columns(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Initial normalization for raw data columns with comprehensive validation
        """
        try:
            if df.empty or not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a non-empty pandas DataFrame")

            prefix = f"{ticker.lower()}_"
            df = df.copy()  # Avoid modifying original DataFrame

            for col in df.columns:
                if not col.startswith(prefix):
                    continue

                # Price/volume columns
                if any(x in col for x in ['close', 'open', 'high', 'low', 'volume']):
                    if df[col].lt(0).any():
                        logger.warning(f"Negative values found in {col}, taking absolute values")
                        df[col] = df[col].abs()
                    df[col] = np.log(df[col].replace(0, 1e-8))

                # RSI normalization
                elif 'rsi' in col:
                    if (df[col] < 0).any() or (df[col] > 100).any():
                        logger.warning(f"RSI values out of expected range [0,100] in {col}")
                    df[col] = (df[col] - 50) / 30  # Scale to ~[-1.67, 1.67]

            return df

        except Exception as e:
            logger.error(f"Normalization failed: {str(e)}", exc_info=True)
            return None

    @staticmethod
    def get_default_ranges() -> Dict[str, Any]:
        """
        Get default normalization ranges with validation

        Returns:
            Dictionary of feature ranges with structure:
            {
                'feature_name': [min_value, max_value],
                ...
            }
        """
        return {
            'default': [0, 1],
            'close': [50, 200],          # Typical stock price range
            'volume': [1_000_000, 100_000_000],  # Typical volume range
            'rsi': [30, 70],            # Standard RSI bounds
            'macd': [-3, 3],            # MACD typical range
            'atr': [1, 20],             # Average True Range
            'volatility': [0, 0.1],      # Daily volatility range (10%)
            'obv': [-1e9, 1e9]          # On-Balance Volume range
        }

    @staticmethod
    def get_feature_range(feature_name: str) -> Optional[list]:
        """
        Get normalization range for a specific feature

        Args:
            feature_name: Name of the feature (e.g., 'close', 'rsi')

        Returns:
            [min, max] range or None if feature not found
        """
        ranges = NormalizationManager.get_default_ranges()
        base_feature = feature_name.split('_')[-1]  # Handle 'aapl_close' -> 'close'
        return ranges.get(base_feature, ranges['default'])

    @staticmethod
    def normalize_value(value: float, feature_name: str) -> Optional[float]:
        """
        Normalize a single value based on feature type

        Args:
            value: Value to normalize
            feature_name: Name of the feature (e.g., 'aapl_close')

        Returns:
            Normalized value or None if error occurs
        """
        try:
            min_val, max_val = NormalizationManager.get_feature_range(feature_name)
            return 2 * ((value - min_val) / (max_val - min_val)) - 1  # Scale to [-1, 1]
        except Exception as e:
            logger.error(f"Normalization failed for {feature_name}: {str(e)}")
            return None

class FeatureScaler:
    def __init__(self, feature_ranges: Dict[str, Any]):
        """
        Normalizes features to [-1, 1] range using configured min/max values

        Args:
            feature_ranges: Dictionary of {feature_name: [min, max]} ranges
        """
        self.feature_ranges = feature_ranges

    def transform(self, feature_name: str, value: float) -> float:
        """Normalize a single feature value to [-1, 1] range"""
        try:
            # Extract base feature name (e.g., 'aapl_close' -> 'close')
            base_feature = feature_name.split('_')[-1]
            min_val, max_val = self.feature_ranges.get(base_feature, self.feature_ranges['default'])

            # Handle division by zero
            if max_val == min_val:
                return 0.0

            # Scale to [-1, 1]
            return 2 * ((value - min_val) / (max_val - min_val)) - 1
        except Exception as e:
            logger.error(f"Feature scaling failed for {feature_name}: {str(e)}")
            return 0.0  # Fail-safe return

class RewardNormalizer:
    def __init__(self, window_size: int = 100):
        self.window = deque(maxlen=window_size)

    def normalize(self, reward: float) -> float:
        """Apply z-score normalization using recent rewards"""
        self.window.append(reward)
        if len(self.window) < 5:  # Minimum samples
            return reward
        return (reward - np.mean(self.window)) / (np.std(self.window) + 1e-8)
