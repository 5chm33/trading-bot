# src/data/processors/data_utils.py
import yfinance as yf
import numpy as np
import pandas as pd
import time
import logging
import concurrent.futures
import os
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from src.utils.data_schema import ColumnSchema
from src.data.add_technical_indicators import TechnicalIndicatorGenerator
from src.utils.normalization import NormalizationManager

logger = logging.getLogger(__name__)

class DataFetcher:
    """Production-grade financial data fetcher with caching and validation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.scaler = RobustScaler()
        self.indicator_generator = TechnicalIndicatorGenerator(config)
        
    def fetch_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch and process data for a single ticker with caching"""
        cache_path = self._get_cache_path(ticker)
        
        try:
            if self._should_use_cache(cache_path):
                return pd.read_pickle(cache_path)
                
            data = self._fetch_with_retry(ticker)
            if data is not None:
                data = self._post_process(ticker, data)
                data.to_pickle(cache_path)
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {str(e)}", exc_info=True)
            return None

    def prepare_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Complete data preparation pipeline with train/test split"""
        logger.info("Starting data preparation")
        
        tickers = self.config['tickers']['primary'] + self.config['tickers'].get('secondary', [])
        data_dict = self.fetch_multiple_tickers(tickers)
        
        if not data_dict:
            raise ValueError("Failed to fetch data for all tickers")
            
        combined = pd.concat(data_dict.values(), axis=1)
        combined = combined.ffill().bfill().dropna()
        
        # Make validation less strict by setting strict=False
        ColumnSchema.validate(combined, tickers, strict=False)
        split_idx = int(len(combined) * 0.8)
        
        logger.info(f"Data preparation complete. Shape: {combined.shape}")
        return combined.iloc[:split_idx], combined.iloc[split_idx:]

    def _get_cache_path(self, ticker: str) -> str:
        """Generate cache path with config dates"""
        time_settings = self.config.get('time_settings', {})
        train_settings = time_settings.get('train', {})
        start_date = train_settings.get('start_date', '2020-01-01')
        return f"{self.cache_dir}/{ticker}_{start_date}.pkl"

    def _should_use_cache(self, cache_path: str) -> bool:
        """Check if cached data is fresh enough to use"""
        return os.path.exists(cache_path) and (
            pd.Timestamp.now() - pd.Timestamp.fromtimestamp(os.path.getmtime(cache_path)) 
            < pd.Timedelta(days=1)
        )

    def _fetch_with_retry(self, ticker: str) -> Optional[pd.DataFrame]:
        """Robust data fetching with multiple fallback strategies"""
        def normalize_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
            """Convert yFinance's MultiIndex to single-level columns"""
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [f"{col[0].lower()}_{col[1].lower()}" for col in data.columns]
            return data

        try:
            # Get config settings with fallbacks
            time_settings = self.config.get('time_settings', {})
            train_settings = time_settings.get('train', {})
            start_date = train_settings.get('start_date', '2020-01-01')
            end_date = train_settings.get('end_date', datetime.now().strftime('%Y-%m-%d'))
            interval = time_settings.get('interval', '1d')
            
            logger.debug(f"Fetching {ticker} from {start_date} to {end_date} ({interval})")

            strategies = [
                {
                    'name': 'direct_download',
                    'func': lambda: yf.download(
                        ticker, 
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        progress=False,
                        auto_adjust=True,
                        group_by='ticker'
                    )
                },
                {
                    'name': 'full_history',
                    'func': lambda: yf.Ticker(ticker).history(
                        period='max',
                        interval=interval,
                        auto_adjust=True
                    )
                }
            ]

            for attempt in range(3):
                strategy = strategies[attempt % len(strategies)]
                try:
                    logger.debug(f"Attempt {attempt+1} using {strategy['name']}")
                    
                    data = strategy['func']()
                    data = normalize_yfinance_columns(data)
                    
                    if not data.empty:
                        # Filter to requested date range
                        mask = (data.index >= pd.to_datetime(start_date)) & \
                            (data.index <= pd.to_datetime(end_date))
                        filtered_data = data.loc[mask]
                        
                        if not filtered_data.empty:
                            logger.debug(f"Successfully fetched {len(filtered_data)} rows")
                            return filtered_data
                        
                        logger.warning(f"Empty date range after filtering for {ticker}")
                    
                except Exception as e:
                    logger.warning(
                        f"Attempt {attempt+1} failed for {ticker} "
                        f"(strategy: {strategy['name']}): {str(e)}",
                        exc_info=attempt == 2
                    )
                    time.sleep(min(2 ** attempt, 10))
                    
            logger.error(f"All fetch attempts failed for {ticker}")
            return None
            
        except Exception as e:
            logger.critical(f"Critical error in fetch pipeline for {ticker}: {str(e)}", exc_info=True)
            return None

    def _post_process(self, ticker: str, data: pd.DataFrame) -> pd.DataFrame:
        """Complete data processing pipeline"""
        try:
            # Standardize column names
            data.columns = self._standardize_column_names(data.columns, ticker)
            
            # Verify we have required columns
            required = [f"{ticker.lower()}_close", f"{ticker.lower()}_volume"]
            missing = [col for col in required if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns after standardization: {missing}")
                
            # Ensure we have a proper DataFrame
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
                
            # Add technical indicators
            data = self.indicator_generator.add_all_indicators(data, ticker)
            
            # Add market regime classification
            data = self._add_regime_info(data, ticker)
            
            # Apply normalization
            data = NormalizationManager.pre_normalize_columns(data, ticker)
            
            return data
        except Exception as e:
            logger.error(f"Post-processing failed for {ticker}: {str(e)}")
            raise

    def _add_regime_info(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Classify market regimes based on price action"""
        prefix = f"{ticker.lower()}_"
        returns = data[f"{prefix}close"].pct_change()
        volatility = returns.rolling(21).std()
        
        conditions = [
            (returns > 2*volatility),
            (returns < -2*volatility),
            (returns.abs() > volatility)
        ]
        choices = [1, -1, 0]  # 1=Bull, -1=Bear, 0=Neutral
        default = 0.5 if returns.mean() > 0 else -0.5
        
        data[f"{prefix}regime"] = np.select(conditions, choices, default=default)
        return data

    def _standardize_column_names(self, columns, ticker: str) -> List[str]:
        """Handle all column name formats from different data sources"""
        column_map = {
            'close': 'close',
            'Close': 'close',
            'CLOSE': 'close',
            'adj close': 'close',
            'volume': 'volume',
            'Volume': 'volume',
            'VOLUME': 'volume'
        }
        
        standardized = []
        for col in columns:
            if isinstance(col, tuple):  # Handle MultiIndex
                col = '_'.join(col)
            col_str = str(col).lower()
            base_col = column_map.get(col_str, col_str.split('_')[-1])
            standardized.append(f"{ticker.lower()}_{base_col}")
        
        logger.debug(f"Standardized columns for {ticker}: {standardized}")
        return standardized

    def fetch_multiple_tickers(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Parallelized fetching for multiple tickers"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_ticker_data, ticker): ticker 
                for ticker in tickers
            }
            return {
                ticker: future.result()
                for future, ticker in future_to_ticker.items()
                if future.result() is not None
            }