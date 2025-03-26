# data_preprocessing/add_technical_indicators.py
import talib
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_technical_indicators(data, ticker):
    """Add technical indicators to the data for a single ticker."""
    logger.info(f"Adding technical indicators for {ticker}.")
    
    try:
        # Ensure required columns are present
        required_columns = [f'Close_{ticker}', f'High_{ticker}', f'Low_{ticker}', f'Volume_{ticker}']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in data for {ticker}: {missing_columns}")
            return data
        
        # Extract the required columns
        close_prices = data[f'Close_{ticker}'].values.astype(float)
        high_prices = data[f'High_{ticker}'].values.astype(float)
        low_prices = data[f'Low_{ticker}'].values.astype(float)
        volume = data[f'Volume_{ticker}'].values.astype(float)
        
        # Add RSI
        data[f'RSI_{ticker}'] = talib.RSI(close_prices, timeperiod=14)
        
        # Add MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        data[f'MACD_{ticker}'] = macd
        data[f'MACD_Signal_{ticker}'] = macd_signal
        data[f'MACD_Hist_{ticker}'] = macd_hist
        
        # Add Bollinger Bands
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        data[f'Bollinger_Upper_{ticker}'] = upper
        data[f'Bollinger_Middle_{ticker}'] = middle
        data[f'Bollinger_Lower_{ticker}'] = lower
        
        # Add ATR (Average True Range)
        data[f'ATR_{ticker}'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Add OBV (On-Balance Volume)
        data[f'OBV_{ticker}'] = talib.OBV(close_prices, volume)
        
        logger.info(f"Technical indicators added successfully for {ticker}.")
        return data
    
    except Exception as e:
        logger.error(f"Error adding technical indicators for {ticker}: {e}", exc_info=True)
        return data