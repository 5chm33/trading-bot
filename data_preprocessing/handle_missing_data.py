import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import logging

logger = logging.getLogger(__name__)

def handle_missing_data(df):
    """Handle missing data using forward-fill, backward-fill, and KNN imputation."""
    logger.info("Handling missing data.")
    
    # Forward-fill and backward-fill
    df = df.ffill().bfill()
    
    # Apply KNN imputation for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df