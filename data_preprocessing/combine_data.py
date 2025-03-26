# data_preprocessing/combine_data.py
import pandas as pd
import logging
import sys
import os

# Add the 'models' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from transformer_trainer import TransformerTrainer  # Import the TransformerTrainer class

logger = logging.getLogger(__name__)

def flatten_columns(data):
    """
    Flatten MultiIndex columns into a single level.
    """
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
    return data

def remove_duplicated_columns(data):
    """
    Remove duplicated columns from the DataFrame.
    """
    # Transpose the DataFrame to check for duplicated rows (original columns)
    data_t = data.T
    # Drop duplicates and transpose back
    data_t = data_t.drop_duplicates()
    return data_t.T

def combine_data(all_data, config):
    """
    Combine data from multiple tickers and handle missing values.
    Also, flatten MultiIndex columns properly and remove duplicated columns.
    """
    # Combine data from all tickers
    combined_data = pd.concat(all_data, axis=1)

    # Log the combined data before flattening
    logger.debug(f"Combined data before flattening:\n{combined_data.head()}")
    logger.debug(f"Combined data columns before flattening:\n{combined_data.columns}")

    # Flatten MultiIndex columns
    combined_data = flatten_columns(combined_data)

    # Log the combined data after flattening
    logger.debug(f"Combined data after flattening:\n{combined_data.head()}")
    logger.debug(f"Combined data columns after flattening:\n{combined_data.columns}")

    # Remove duplicated columns
    combined_data = remove_duplicated_columns(combined_data)

    # Log the combined data after removing duplicates
    logger.debug(f"Combined data after removing duplicates:\n{combined_data.head()}")
    logger.debug(f"Combined data columns after removing duplicates:\n{combined_data.columns}")

    # Initialize the trainer to use its handle_missing_data method
    trainer = TransformerTrainer(combined_data, config, close_column='Close_AAPL')  # Use the first ticker's close column as a placeholder

    # Handle missing data using the trainer's method
    combined_data = trainer.handle_missing_data(combined_data)

    # Log the combined data after handling missing data
    logger.debug(f"Combined data after handling missing data:\n{combined_data.head()}")
    logger.debug(f"Combined data columns after handling missing data:\n{combined_data.columns}")

    # Drop columns with all NaN values
    combined_data = combined_data.dropna(axis=1, how='all')

    # Log the DataFrame shape and columns after dropping NaN columns
    logger.debug(f"DataFrame shape after dropping NaN columns: {combined_data.shape}")
    logger.debug(f"Columns after dropping NaN columns:\n{combined_data.columns}")

    # Check if DataFrame is empty
    if combined_data.empty:
        logger.error("Combined DataFrame is empty. Check the input data.")
        return combined_data

    # Log the final DataFrame
    logger.debug(f"Final DataFrame:\n{combined_data}")
    logger.debug(f"DataFrame columns:\n{combined_data.columns}")

    return combined_data