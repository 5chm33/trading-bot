# data_preprocessing/prepare_data.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

# data_preprocessing/prepare_data.py
def prepare_data(data, sequence_length=30, test_size=0.2):
    """
    Prepare data for training by creating sequences and splitting into features and targets.
    """
    try:
        logger.info("Preparing data for training.")
        
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty.")
        
        # Select numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        features = data[numeric_cols]
        
        # Log the selected features
        logger.debug(f"Selected features: {features.columns}")
        
        # Extract the target column (e.g., 'Close_AAPL')
        target_col = [col for col in features.columns if "Close" in col][0]
        target = features[target_col].values
        
        # Scale the features and target
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        scaled_features = feature_scaler.fit_transform(features)
        scaled_target = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences for Transformer
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i - sequence_length:i])
            y.append(scaled_target[i])
        
        X, y = np.array(X), np.array(y)
        
        # Log the shape of the sequences
        logger.debug(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        logger.info(f"Data prepared. X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
        return X_train, X_val, y_train, y_val, feature_scaler, target_scaler
    
    except Exception as e:
        logger.error(f"Error preparing data: {e}", exc_info=True)
        raise