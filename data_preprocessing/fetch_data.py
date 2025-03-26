import yfinance as yf
import pandas as pd
import numpy as np
from utils.custom_logging import setup_logger

logger = setup_logger(__name__)

def clean_column_names(columns):
    """Standardize and clean column names for consistency."""
    cleaned = []
    for col in columns:
        if isinstance(col, tuple):
            # Handle multi-index: ('AAPL', 'Close') -> 'AAPL_Close'
            col = '_'.join(str(c).strip() for c in col if c)
        # Remove special characters and normalize
        col = str(col).replace('(', '').replace(')', '').replace("'", "").replace(',', '_').strip()
        cleaned.append(col)
    return cleaned

# data_preprocessing/fetch_data.py (FINAL)
import yfinance as yf
import pandas as pd
import numpy as np
from utils.custom_logging import setup_logger

logger = setup_logger(__name__)

def fetch_data(ticker, start_date, end_date, interval="1d"):
    """Robust data fetcher with complete column normalization."""
    try:
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
        
        # Download with consistent settings
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by='ticker'
        )
        
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Normalize columns
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
            data.columns = [f"{str(ticker).lower()}_{column_map.get(col, col.lower())}" 
                        for col in data.columns]

        # Validate columns
        required_cols = [
            f"{str(ticker).lower()}_open",
            f"{str(ticker).lower()}_high",
            f"{str(ticker).lower()}_low",
            f"{str(ticker).lower()}_close",
            f"{str(ticker).lower()}_volume"
        ]
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing columns for {ticker}:\n"
                f"Missing: {missing_cols}\n"
                f"Received: {list(data.columns)}"
            )

        logger.debug(f"Fetched data columns: {list(data.columns)}")
        return data[required_cols]  # Return only standardized columns
        
    except Exception as e:
        logger.error(f"Failed fetching {ticker}: {str(e)}", exc_info=True)
        raise ValueError(f"Data fetch failed for {ticker}") from e