<<<<<<< HEAD
# tests/unit/test_data_pipeline.py
import sys
from pathlib import Path
import pytest
import pandas as pd
import logging

# Add project root to Python path (CRITICAL FIX)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Now import from src
from src.pipeline.training.train_rl import _fetch_and_process_data, prepare_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def test_config():
    return {
        'tickers': {
            'primary': ['AAPL', 'MSFT'],
            'secondary': []
        },
        'time_settings': {
            'train': {
                'start_date': '2023-01-01',
                'end_date': '2023-01-10'
            },
            'interval': '1d'
        },
        'testing': {
            'use_synthetic': False
        }
    }

def test_real_data_pipeline(test_config):
    """Test with real market data"""
    test_config['testing']['use_synthetic'] = False
    train, eval = prepare_data(test_config)
    
    # Basic validation checks
    assert isinstance(train, pd.DataFrame)
    assert isinstance(eval, pd.DataFrame)
    assert len(train) > 0
    assert 'aapl_close' in train.columns
    assert 'msft_close' in train.columns

def test_synthetic_data_pipeline(test_config):
    """Test with synthetic data fallback"""
    test_config.update({
        'testing': {
            'use_synthetic': True,
            'synthetic_seed': 42
        },
        'tickers': {
            'primary': ['INVALID1', 'INVALID2'],
            'secondary': []
        }
    })
    
    train, eval = prepare_data(test_config)
    
    # Validate synthetic data structure
    assert isinstance(train, pd.DataFrame)
    assert isinstance(eval, pd.DataFrame)
    assert 'invalid1_close' in train.columns
    assert 'invalid2_close' in train.columns
=======
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
>>>>>>> 60870aec3b9ed2c2cb804ceb4f1eeb5c6af9d852
