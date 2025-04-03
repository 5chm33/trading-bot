import yfinance as yf
import pandas as pd
from typing import Optional
from utils.logging import setup_logger
from datetime import datetime, timedelta

logger = setup_logger(__name__)

class YahooFinanceFetcher:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def fetch_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with retry logic and validation"""
        for attempt in range(self.max_retries):
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,  # Corrects for splits/dividends
                    threads=True
                )
                
                if data.empty:
                    raise ValueError("Empty DataFrame returned")
                    
                return self._standardize_format(data, ticker)
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Max retries exceeded for {ticker}")
                    return None
                
                # Exponential backoff
                time.sleep(2 ** attempt)

    def _standardize_format(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Convert yfinance data to our standard format"""
        # Convert column names
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        standardized = data.rename(columns=column_map)
        standardized.columns = [f"{ticker.lower()}_{col}" for col in standardized.columns]
        
        # Add volatility
        close_col = f"{ticker.lower()}_close"
        standardized[f"{ticker.lower()}_volatility"] = (
            standardized[close_col].pct_change().rolling(14).std()
        )
        
        return standardized.dropna()

    def fetch_options_chain(self, ticker: str, expiration: str) -> Optional[pd.DataFrame]:
        """Fetch options chain for specific expiration"""
        try:
            stock = yf.Ticker(ticker)
            options = stock.option_chain(expiration)
            
            # Combine puts and calls
            chain = pd.concat([options.calls, options.puts])
            chain['expiration'] = pd.to_datetime(expiration)
            chain['ticker'] = ticker.lower()
            
            return chain
        except Exception as e:
            logger.error(f"Failed to fetch options for {ticker}: {str(e)}")
            return None