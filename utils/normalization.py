import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from collections import deque
# Set up logger
logger = logging.getLogger(__name__)

class NormalizationManager:
    """Central hub for all normalization needs with enhanced error handling"""

    @staticmethod
    def pre_normalize_columns(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Initial normalization for raw data columns with comprehensive validation

        Args:
            df: Input DataFrame with financial data
            ticker: Ticker symbol (e.g., 'AAPL')

        Returns:
            Normalized DataFrame or None if error occurs
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
