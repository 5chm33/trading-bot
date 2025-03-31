# test_data_pipeline.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import logging
from src.pipeline.training.train_rl import _fetch_and_process_data, prepare_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_pipeline():
    """Test the complete data pipeline"""
    test_config = {
        'tickers': {
            'primary': ['AAPL', 'MSFT'],  # Known working tickers
            'secondary': []
        },
        'time_settings': {
            'train': {
                'start_date': '2023-01-01',
                'end_date': '2023-01-10'
            },
            'interval': '1d'
        }
    }
    
    print("\n=== Testing with REAL data ===")
    try:
        train_real, eval_real = prepare_data(test_config)
        print("Success! Columns:", [c for c in train_real.columns if '_close' in c])
    except Exception as e:
        print(f"Real data failed: {str(e)}")
    
    print("\n=== Testing with SYNTHETIC data ===")
    test_config['tickers']['primary'] = ['INVALID1', 'INVALID2']
    try:
        train_synth, eval_synth = prepare_data(test_config)
        print("Success! Columns:", [c for c in train_synth.columns if '_close' in c])
    except Exception as e:
        print(f"Synthetic data failed: {str(e)}")

if __name__ == "__main__":
    test_data_pipeline()