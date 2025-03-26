import yfinance as yf
import pandas as pd
import numpy as np
from utils.custom_logging import setup_logger

logger = setup_logger()

def fetch_data(ticker, start_date, end_date, interval="1d"):
    """Fetch and standardize ticker data with robust error handling."""
    try:
        logger.info(f"Fetching {ticker} data from {start_date} to {end_date}")
        
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True
        )
        
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")
            
        # Standardize column names
        col_map = {
            'Open': f"{ticker}_Open",
            'High': f"{ticker}_High",
            'Low': f"{ticker}_Low",
            'Close': f"{ticker}_Close",
            'Adj Close': f"{ticker}_Adj_Close",
            'Volume': f"{ticker}_Volume"
        }
        
        data = data.rename(columns=col_map)[list(col_map.values())]
        data = data.ffill().bfill()  # Handle missing values
        
        logger.debug(f"Successfully fetched {len(data)} rows for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {str(e)}")
        raise

def load_and_preprocess(config):
    """Load and combine data for all tickers with validation."""
    try:
        if not config.get('tickers'):
            raise ValueError("No tickers configured")
            
        all_data = []
        for ticker in config['tickers']:
            logger.info(f"Fetching data for {ticker}")
            data = fetch_data(
                ticker,
                config['train_start_date'],
                config['train_end_date'],
                config['interval']
            )
            if data is not None:
                all_data.append(data)
        
        if not all_data:
            raise ValueError("No data was loaded for any ticker")
            
        # Combine all data while preserving column names
        combined_data = pd.concat(all_data, axis=1)
        logger.info(f"Combined data columns: {combined_data.columns.tolist()}")
        return combined_data
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}", exc_info=True)
        raise