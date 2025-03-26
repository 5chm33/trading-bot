# first line: 38
@memory.cache
def add_technical_indicators(data, close_column='Close'):
    """
    Add technical indicators to the data using TA-Lib.
    """
    try:
        # Ensure the 'Close' column is present
        if close_column not in data.columns:
            logging.error(f"'{close_column}' column missing in data.")
            return data

        # Ensure the 'Close' column is a 1D NumPy array
        close_prices = data[close_column].values.flatten()  # Flatten to 1D array
        logging.info(f"Close column shape after flattening: {close_prices.shape}")

        # Moving Averages
        data.loc[:, 'SMA_10'] = talib.SMA(close_prices, timeperiod=10)  # Use .loc to avoid SettingWithCopyWarning
        data.loc[:, 'SMA_30'] = talib.SMA(close_prices, timeperiod=30)
        
        # Relative Strength Index (RSI)
        data.loc[:, 'RSI'] = talib.RSI(close_prices, timeperiod=14)
        
        # Bollinger Bands
        data.loc[:, 'Upper_Band'], data.loc[:, 'Middle_Band'], data.loc[:, 'Lower_Band'] = talib.BBANDS(
            close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # MACD
        data.loc[:, 'MACD'], data.loc[:, 'MACD_Signal'], data.loc[:, 'MACD_Hist'] = talib.MACD(
            close_prices, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Drop rows with NaN values (created by indicators)
        data.dropna(inplace=True)
        
        logging.info(f"Technical indicators added to data.")
        return data
    except Exception as e:
        logging.error(f"Error adding technical indicators: {e}")
        raise
