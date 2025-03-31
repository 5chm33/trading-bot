# src/data/add_technical_indicators.py
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional
from src.utils.data_schema import ColumnSchema
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class TechnicalIndicatorGenerator:
    """Centralized technical indicator computation with dynamic registry"""
    
    def __init__(self, config: Dict):
        self.config = config
        self._setup_indicator_registry()
        
    def _setup_indicator_registry(self):
        """Initialize all indicator computation methods"""
        self.registry = {
            'price': self._add_price_indicators,
            'volume': self._add_volume_indicators,
            'momentum': self._add_momentum_indicators,
            'volatility': self._add_volatility_indicators,
            'trend': self._add_trend_indicators
        }
        
        # Config-driven indicator activation
        self.active_indicators = self.config.get('indicators', {
            'price': True,
            'volume': True,
            'momentum': True,
            'volatility': True,
            'trend': True
        })

    def add_all_indicators(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add complete set of technical indicators for a ticker
        """
        prefix = f"{ticker.lower()}_"
        
        try:
            # Validate input columns
            required_cols = [f"{prefix}close"]
            if self.active_indicators.get('volume', True):
                required_cols.append(f"{prefix}volume")
                
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Ensure we're working with a copy and proper data types
            data = data.copy()
            
            # Convert numeric columns to float64 for TA-Lib
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].astype(np.float64)
            
            # Add indicators by category
            for category, method in self.registry.items():
                if self.active_indicators.get(category, True):
                    data = method(data, prefix)
            
            return data.dropna()
            
        except Exception as e:
            logger.error(f"Indicator generation failed for {ticker}: {str(e)}", exc_info=True)
            raise

    def _add_price_indicators(self, data: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add price transformations and moving averages"""
        closes = data[f"{prefix}close"]
        
        # Basic transformations
        data[f"{prefix}log_ret"] = np.log(closes / closes.shift(1))
        data[f"{prefix}cum_ret"] = (1 + data[f"{prefix}log_ret"]).cumprod()
        
        # Configurable moving averages
        ma_config = self.config.get('indicators', {}).get('moving_averages', {})
        for window in ma_config.get('sma', [5, 10, 20]):
            data[f"{prefix}sma_{window}"] = closes.rolling(window).mean()
        for window in ma_config.get('ema', [10, 20, 50]):
            data[f"{prefix}ema_{window}"] = talib.EMA(closes.values.astype(np.float64), timeperiod=window)
        
        return data

    def _add_volume_indicators(self, data: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add volume-based indicators"""
        if f"{prefix}volume" not in data.columns:
            return data
            
        closes = data[f"{prefix}close"]
        volumes = data[f"{prefix}volume"]
        
        # Volume features
        data[f"{prefix}volume_z"] = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
        data[f"{prefix}obv"] = talib.OBV(
            closes.values.astype(np.float64), 
            volumes.values.astype(np.float64))
        
        # Volume moving averages
        windows = self.config.get('indicators', {}).get('volume', {}).get('ma_windows', [20, 50])
        for window in windows:
            data[f"{prefix}volume_ma_{window}"] = volumes.rolling(window).mean()
        
        return data

    def _add_momentum_indicators(self, data: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add momentum oscillators"""
        closes = data[f"{prefix}close"]
        highs = data.get(f"{prefix}high", closes)
        lows = data.get(f"{prefix}low", closes)
        
        # Convert to float64 for TA-Lib
        closes_f64 = closes.values.astype(np.float64)
        highs_f64 = highs.values.astype(np.float64)
        lows_f64 = lows.values.astype(np.float64)
        
        # Core momentum indicators
        data[f"{prefix}rsi"] = talib.RSI(closes_f64, timeperiod=14)
        data[f"{prefix}macd"], _, _ = talib.MACD(closes_f64)
        data[f"{prefix}stoch_k"], _ = talib.STOCH(highs_f64, lows_f64, closes_f64)
        
        # Rate of Change indicators
        windows = self.config.get('indicators', {}).get('momentum', {}).get('roc', [5, 10, 20])
        for window in windows:
            data[f"{prefix}roc_{window}"] = talib.ROC(closes_f64, timeperiod=window)
        
        return data

    def _add_volatility_indicators(self, data: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add volatility measures"""
        closes = data[f"{prefix}close"]
        highs = data.get(f"{prefix}high", closes)
        lows = data.get(f"{prefix}low", closes)
        
        # Convert to float64 for TA-Lib
        closes_f64 = closes.values.astype(np.float64)
        highs_f64 = highs.values.astype(np.float64)
        lows_f64 = lows.values.astype(np.float64)
        
        # Standard volatility
        data[f"{prefix}atr"] = talib.ATR(highs_f64, lows_f64, closes_f64, timeperiod=14)
        upper, _, lower = talib.BBANDS(closes_f64, timeperiod=20)
        data[f"{prefix}bb_width"] = (upper - lower) / closes.mean()
        
        # Historical volatility
        windows = self.config.get('indicators', {}).get('volatility', {}).get('windows', [5, 10, 20])
        for window in windows:
            data[f"{prefix}hist_vol_{window}"] = closes.pct_change().rolling(window).std()
        
        return data

    def _add_trend_indicators(self, data: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add trend detection indicators"""
        closes = data[f"{prefix}close"]
        highs = data.get(f"{prefix}high", closes)
        lows = data.get(f"{prefix}low", closes)
        
        # Convert to float64 for TA-Lib
        closes_f64 = closes.values.astype(np.float64)
        highs_f64 = highs.values.astype(np.float64)
        lows_f64 = lows.values.astype(np.float64)
        
        # Trend strength
        data[f"{prefix}adx"] = talib.ADX(highs_f64, lows_f64, closes_f64, timeperiod=14)
        
        # Moving average crossovers
        fast = self.config.get('indicators', {}).get('trend', {}).get('ma_fast', 10)
        slow = self.config.get('indicators', {}).get('trend', {}).get('ma_slow', 20)
        data[f"{prefix}ma_cross"] = (
            talib.EMA(closes_f64, timeperiod=fast) - 
            talib.EMA(closes_f64, timeperiod=slow))
        
        return data