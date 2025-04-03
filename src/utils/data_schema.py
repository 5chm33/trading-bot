<<<<<<< HEAD
# src/utils/data_schema.py
from typing import Dict, List, Set, Union, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class ValidationMode:
    BACKTEST = "backtest"
    LIVE = "live"

@dataclass
class ValidationConfig:
    mode: str = ValidationMode.BACKTEST
    strict: bool = False
    required_tickers: List[str] = None
    skip_empty: bool = True
    log_level: str = "error"  # 'error', 'warn', or 'silent'

class ColumnSchema:
    """Final production-grade schema validator with complete configuration support"""

    # Core price columns
    OHLCV = ["open", "high", "low", "close", "volume"]
    
    # Extended technical indicators
    INDICATOR_CATEGORIES = {
        'price': ["log_ret", "cum_ret", "sma_*", "ema_*"],
        'volume': ["volume_z", "obv", "volume_ma_*"],
        'momentum': ["rsi", "macd", "stoch_k", "roc_*"],
        'volatility': ["atr", "bb_width", "hist_vol_*"],
        'trend': ["adx", "ma_cross"]
    }

    @classmethod
    def create_config(
        cls,
        tickers: List[str],
        mode: str = ValidationMode.BACKTEST,
        **kwargs
    ) -> ValidationConfig:
        """Create validation configuration with defaults"""
        return ValidationConfig(
            required_tickers=tickers,
            mode=mode,
            **kwargs
        )

    @classmethod
    def from_main_config(cls, config: Dict) -> ValidationConfig:
        """Create from main application config"""
        val_config = config.get("validation", {})
        return cls.create_config(
            tickers=config["tickers"]["primary"],
            mode=val_config.get("mode", ValidationMode.BACKTEST),
            strict=val_config.get("strict", False),
            skip_empty=val_config.get("skip_empty", True),
            log_level=val_config.get("log_level", "error")
        )

    @classmethod
    def validate(
        cls,
        data: Union[pd.DataFrame, Dict[str, Dict]],
        tickers: Union[List[str], Dict, ValidationConfig],
        mode: Optional[str] = None,
        strict: Optional[bool] = None
    ) -> bool:
        """
        Unified validation interface supporting:
        - List of tickers (legacy)
        - Full config dict
        - ValidationConfig object
        
        Args:
            data: Input data (DataFrame or ticker dict)
            tickers: One of:
                - List of ticker strings
                - Main config dictionary
                - ValidationConfig instance
            mode: Optional mode override
            strict: Optional strict flag override
        """
        vconfig = cls._resolve_validation_config(tickers, mode, strict)
        
        # Skip empty data if configured
        if vconfig.skip_empty and isinstance(data, pd.DataFrame) and data.empty:
            return True

        try:
            available = cls._get_available_columns(data, vconfig.mode)
            missing = cls._get_missing_columns(available, vconfig)
            
            if missing:
                raise ValueError(cls._format_error_message(missing, available, vconfig.mode))
            return True
            
        except ValueError as e:
            if vconfig.log_level == "error":
                raise
            if vconfig.log_level == "warn":
                import warnings
                warnings.warn(str(e))
            return False

    @classmethod
    def _resolve_validation_config(
        cls,
        tickers: Union[List[str], Dict, ValidationConfig],
        mode: Optional[str],
        strict: Optional[bool]
    ) -> ValidationConfig:
        """Convert all input types to ValidationConfig"""
        if isinstance(tickers, ValidationConfig):
            config = tickers
            if mode is not None:
                config.mode = mode
            if strict is not None:
                config.strict = strict
            return config
            
        if isinstance(tickers, dict):
            config = cls.from_main_config(tickers)
        else:  # Assume list of tickers
            config = cls.create_config(tickers)
            
        if mode is not None:
            config.mode = mode
        if strict is not None:
            config.strict = strict
            
        return config

    @classmethod
    def _get_available_columns(
        cls,
        data: Union[pd.DataFrame, Dict[str, Dict]],
        mode: str
    ) -> Set[str]:
        """Extract available columns based on data type"""
        if isinstance(data, pd.DataFrame):
            return set(data.columns)
        return {
            f"{ticker.lower()}_{key}"
            for ticker, values in data.items()
            for key in values.keys()
        }

    @classmethod
    def _get_missing_columns(
        cls,
        available: Set[str],
        config: ValidationConfig
    ) -> List[str]:
        """Calculate missing columns based on validation rules"""
        missing = []
        for ticker in config.required_tickers:
            prefix = f"{ticker.lower()}_"
            
            if config.mode == ValidationMode.LIVE:
                if f"{prefix}close" not in available:
                    missing.append(f"{prefix}close")
            else:  # Backtest mode
                required = cls.get_ticker_columns(ticker, config.strict)
                missing.extend(col for col in required if col not in available)
                
                # Handle wildcards in non-strict mode
                if not config.strict:
                    missing = cls._filter_wildcard_matches(missing, available, prefix)
        return missing

    @classmethod
    def _filter_wildcard_matches(
        cls,
        missing: List[str],
        available: Set[str],
        prefix: str
    ) -> List[str]:
        """Remove missing items if wildcard pattern exists"""
        for pattern in cls._get_wildcard_patterns():
            base = f"{prefix}{pattern.replace('*', '')}"
            if any(col.startswith(base) for col in available):
                missing = [m for m in missing if not m.startswith(base)]
        return missing

    @classmethod
    def _format_error_message(
        cls,
        missing: List[str],
        available: Set[str],
        mode: str
    ) -> str:
        """Generate consistent error messages"""
        return (
            f"Validation failed ({mode} mode):\n"
            f"Missing columns ({len(missing)}): {sorted(missing)[:5]}{'...' if len(missing)>5 else ''}\n"
            f"Available columns: {sorted(available)[:5]}{'...' if len(available)>5 else ''}"
        )

    @classmethod
    def get_ticker_columns(
        cls,
        ticker: str,
        include_optional: bool = True
    ) -> Dict[str, str]:
        """Get all expected columns for a ticker"""
        ticker = ticker.lower()
        columns = {f"{ticker}_{col}": col for col in cls.OHLCV}
        
        if include_optional:
            for category, patterns in cls.INDICATOR_CATEGORIES.items():
                for pattern in patterns:
                    if '*' in pattern:
                        base = pattern.replace('*', '')
                        columns.update({
                            f"{ticker}{base}{i}": f"{base}{i}" 
                            for i in [5, 10, 20, 50]  # Common window sizes
                        })
                    else:
                        columns[f"{ticker}_{pattern}"] = pattern
            columns[f"{ticker}_regime"] = "regime"
            
        return columns

    @classmethod
    def _get_wildcard_patterns(cls) -> Set[str]:
        """Get all wildcard patterns from indicator categories"""
        return {
            pattern
            for patterns in cls.INDICATOR_CATEGORIES.values()
            for pattern in patterns 
            if '*' in pattern
        }

    @classmethod
    def get_minimum_required(cls, ticker: str) -> List[str]:
        """Get minimum required columns for live trading"""
        return [f"{ticker.lower()}_close"]
=======
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
>>>>>>> 60870aec3b9ed2c2cb804ceb4f1eeb5c6af9d852
