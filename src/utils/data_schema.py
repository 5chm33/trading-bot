# src/utils/data_schema.py
from typing import Dict, List, Set
import pandas as pd

class ColumnSchema:
    """Enhanced column schema validator with dynamic registry"""
    
    # Core price columns
    OHLCV = ["open", "high", "low", "close", "volume"]
    
    # Technical indicator categories
    INDICATOR_CATEGORIES = {
        'price': ["log_ret", "cum_ret", "sma_*", "ema_*"],
        'volume': ["volume_z", "obv", "volume_ma_*"],
        'momentum': ["rsi", "macd", "stoch_k", "roc_*"],
        'volatility': ["atr", "bb_width", "hist_vol_*"],
        'trend': ["adx", "ma_cross"]
    }
    
    @classmethod
    def get_ticker_columns(cls, ticker: str, include_optional: bool = True) -> Dict[str, str]:
        """Dynamically generate expected columns including wildcards"""
        ticker = ticker.lower()
        columns = {}
        
        # Required OHLCV columns
        columns.update({f"{ticker}_{col}": col for col in cls.OHLCV})
        
        # Optional technical indicators
        if include_optional:
            for category, patterns in cls.INDICATOR_CATEGORIES.items():
                for pattern in patterns:
                    if '*' in pattern:
                        # Handle wildcard patterns (e.g. sma_*)
                        base = pattern.replace('*', '')
                        columns.update({f"{ticker}{base}{i}": f"{base}{i}" 
                                      for i in [5, 10, 20, 50]})  # Common windows
                    else:
                        columns[f"{ticker}_{pattern}"] = pattern
        
        # Special columns (make optional if needed)
        if include_optional:
            columns[f"{ticker}_regime"] = "regime"
        
        return columns
    
    @classmethod
    def validate(cls, df: pd.DataFrame, required_tickers: List[str], 
                strict: bool = False) -> bool:
        """
        Enhanced DataFrame validator with pattern matching
        Args:
            df: DataFrame to validate
            required_tickers: List of tickers to check
            strict: Whether to require all optional indicators
        Returns:
            bool: True if validation passes
        Raises:
            ValueError: With detailed missing columns report
        """
        missing = []
        available = set(df.columns)
        
        for ticker in required_tickers:
            expected = cls.get_ticker_columns(ticker, include_optional=strict)
            
            # Check exact matches first
            for col in expected:
                if col not in available:
                    missing.append(col)
            
            # Pattern-based checks (only in non-strict mode)
            if not strict:
                for pattern in cls._get_wildcard_patterns():
                    base_pattern = f"{ticker}_{pattern.replace('*', '')}"
                    if any(col.startswith(base_pattern) for col in available):
                        # Remove matching patterns from missing list
                        missing = [m for m in missing if not m.startswith(base_pattern)]
        
        if missing:
            raise ValueError(
                "Data validation failed:\n"
                f"Missing ({len(missing)}): {sorted(missing)[:5]}{'...' if len(missing)>5 else ''}\n"
                f"Available: {sorted(available)[:5]}{'...' if len(available)>5 else ''}"
            )
        return True
    
    @classmethod
    def _get_wildcard_patterns(cls) -> Set[str]:
        """Get all wildcard patterns from indicator categories"""
        return {
            pattern for patterns in cls.INDICATOR_CATEGORIES.values()
            for pattern in patterns if '*' in pattern
        }