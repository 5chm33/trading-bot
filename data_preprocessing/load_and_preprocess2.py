import yaml
import logging
from data_preprocessing.fetch_data import fetch_data
from data_preprocessing.combine_data import combine_data

logger = logging.getLogger(__name__)

def load_and_preprocess(config):
    """Load and preprocess the data."""
    try:
        logger.info("Fetching data for all tickers.")
        all_data = []
        for ticker in config['tickers']:
            logger.info(f"Fetching data for ticker: {ticker}")
            data = fetch_data(ticker, config['start_date'], config['end_date'], config['interval'])
            if not data.empty:
                all_data.append(data)
                logger.info(f"Data fetched for {ticker}. Shape: {data.shape}")
            else:
                logger.warning(f"No data fetched for ticker: {ticker}")

        logger.info("Combining data from all tickers.")
        combined_data = combine_data(all_data, config)
        if combined_data.empty:
            logger.error("Combined data is empty. Check data fetching and preprocessing.")
            return None
        else:
            logger.info(f"Combined data shape: {combined_data.shape}")
            return combined_data
    except Exception as e:
        logger.error(f"Error in load_and_preprocess: {e}", exc_info=True)
        return None