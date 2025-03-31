# src/data/processors/normalizers.py
from src.utils.normalization import NormalizationManager
from src.utils.logging import setup_logger
import pandas as pd

logger = setup_logger(__name__)

class DatasetNormalizer:
    """Applies normalization pipelines to specific datasets"""
    
    def __init__(self, config: dict):
        self.config = config
        
    def normalize_ticker_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Full normalization pipeline for a single ticker"""
        try:
            # 1. Apply your comprehensive pre-normalization
            df = NormalizationManager.pre_normalize_columns(df, ticker)
            if df is None:
                raise ValueError("Pre-normalization failed")
            
            # 2. Add dataset-specific transformations
            prefix = f"{ticker.lower()}_"
            if f"{prefix}volume" in df.columns:
                df[f"{prefix}volume"] = self._normalize_volume(df[f"{prefix}volume"])
                
            return df
            
        except Exception as e:
            logger.error(f"Dataset normalization failed: {str(e)}")
            raise
            
    def _normalize_volume(self, series: pd.Series) -> pd.Series:
        """Special handling for volume spikes"""
        clipped = series.clip(upper=series.quantile(0.99))
        return clipped / clipped.max()