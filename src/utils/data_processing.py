import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def process_ticker_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Standard processing pipeline for ticker data
    Returns DataFrame with ticker-prefixed columns
    """
    try:
        if data.empty:
            raise ValueError("Empty DataFrame received")
            
        # 1. Standardize column names (consistent with fetch_data.py)
        if not all(col.startswith(f"{ticker.lower()}_") for col in data.columns):
            data.columns = [f"{ticker.lower()}_{col.lower().replace(f"{ticker.lower()}_", "")}" 
                          for col in data.columns]
        
        # 2. Verify we have required columns
        required_cols = [f"{ticker.lower()}_{col}" for col in ['open', 'high', 'low', 'close', 'volume']]
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # 3. Add basic technical features
        close_col = f"{ticker.lower()}_close"
        data[f"{ticker.lower()}_returns"] = data[close_col].pct_change()
        data[f"{ticker.lower()}_volatility"] = data[close_col].pct_change().rolling(14).std()
        
        # 4. Handle missing values
        data = data.ffill().bfill().dropna()
        
        logger.debug(f"Processed {ticker} data | Shape: {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to process {ticker}: {str(e)}")
        raise