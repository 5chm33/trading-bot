import yfinance as yf
import pandas as pd
import numpy as np
from utils.custom_logging import setup_logger
from utils.normalization import NormalizationManager
from typing import Optional, List
import logging

logger = setup_logger(__name__)

def fetch_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    """Robust data fetcher with comprehensive error handling"""
    try:
        logger.info(f"Fetching {ticker} data ({interval}) from {start_date} to {end_date}")

        # Download data with retry logic
        data = _download_with_retry(ticker, start_date, end_date, interval)
        if data is None or data.empty:
            return None

        # Standardize columns and ensure numeric types
        data = _standardize_columns(data, ticker)
        if data is None:
            return None

        # Validate required columns
        required_cols = _get_required_columns(ticker)
        if not _validate_columns(data, required_cols):
            return None

        # Apply normalization
        normalized_data = NormalizationManager.pre_normalize_columns(data, ticker)
        if normalized_data is None:
            logger.error("Normalization failed")
            return None

        logger.debug(f"Successfully fetched {len(normalized_data)} rows")
        return normalized_data[required_cols]

    except Exception as e:
        logger.error(f"Failed to fetch {ticker} data: {str(e)}", exc_info=True)
        return None

def _download_with_retry(ticker: str, start_date: str, end_date: str, interval: str,
                        max_retries: int = 2) -> Optional[pd.DataFrame]:
    """Download data with automatic retry and date expansion"""
    for attempt in range(max_retries + 1):
        try:
            current_start = pd.to_datetime(start_date) - pd.Timedelta(days=5*attempt)
            data = yf.download(
                ticker,
                start=current_start,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by='ticker',
                threads=True
            )
            if not data.empty:
                return data
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
    logger.error(f"Failed to download data after {max_retries} retries")
    return None

def _standardize_columns(data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Standardize column names and ensure numeric values"""
    try:
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [f"{col[0].lower()}_{col[1].lower()}" for col in data.columns]
        else:
            column_map = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            }
            data.columns = [f"{ticker.lower()}_{column_map.get(col, col.lower())}"
                          for col in data.columns]

        # Force convert numeric columns
        numeric_cols = [col for col in data.columns if any(x in col for x in
                       ['open', 'high', 'low', 'close', 'volume'])]
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        return data.dropna()
    except Exception as e:
        logger.error(f"Column standardization failed: {str(e)}")
        return None

def _get_required_columns(ticker: str) -> List[str]:
    """Get list of required columns for a ticker"""
    prefix = f"{ticker.lower()}_"
    return [f"{prefix}{col}" for col in ['open', 'high', 'low', 'close', 'volume']]

def _validate_columns(data: pd.DataFrame, required_cols: List[str]) -> bool:
    """Validate all required columns exist and are numeric"""
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False

    non_numeric = [col for col in required_cols if not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric:
        logger.error(f"Non-numeric columns found: {non_numeric}")
        return False

    return True
