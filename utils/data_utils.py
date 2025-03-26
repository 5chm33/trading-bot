import yfinance as yf
import pandas as pd

def clean_column_names(columns, ticker):
    """Convert columns to consistent ticker_metric format"""
    cleaned = []
    for col in columns:
        if isinstance(col, tuple):
            # For multi-index: ('Close', 'AAPL') -> 'aapl_close'
            parts = [str(c).lower() for c in col if c]
            cleaned.append(f"{parts[1]}_{parts[0]}" if len(parts) > 1 else f"{ticker.lower()}_{parts[0]}")
        else:
            # For single columns: 'Close' -> 'aapl_close'
            cleaned.append(f"{ticker.lower()}_{str(col).lower()}")
    return cleaned

def fetch_ticker_data(ticker, start_date, end_date):
    """Fetch and standardize data for a single ticker"""
    try:
        # Download with auto_adjust=True to avoid split/dividend issues
        data = yf.download(ticker,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=True)
        
        if data.empty:
            print(f"No data returned for {ticker}")
            return None
            
        # Clean column names
        data.columns = clean_column_names(data.columns, ticker)
        
        # Verify we have the expected close column
        close_col = f"{ticker.lower()}_close"
        if close_col not in data.columns:
            print(f"Missing close column {close_col} in data for {ticker}")
            print(f"Available columns: {data.columns.tolist()}")
            return None
        
        # Ensure all required OHLCV columns exist
        required_cols = [
            f"{ticker.lower()}_open",
            f"{ticker.lower()}_high",
            f"{ticker.lower()}_low",
            f"{ticker.lower()}_close",
            f"{ticker.lower()}_volume"
        ]
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Missing required columns for {ticker}: {missing_cols}")
            return None
        
        # Add basic volatility
        data[f"{ticker.lower()}_volatility"] = data[close_col].pct_change().rolling(14).std()
        
        print(f"Successfully processed {ticker} with columns: {data.columns.tolist()}")
        return data.dropna()
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")
        return None