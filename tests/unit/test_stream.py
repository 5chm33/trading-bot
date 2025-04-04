# tests/test_stream.py
import os
import sys
import logging
from pathlib import Path
import joblib
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockMetrics:
    """Enhanced mock metrics for testing"""
    def record_component_init(self, component: str, success: bool):
        logger.info(f"Component {component} initialized: {'Success' if success else 'Failed'}")
    
    def record_price_update(self, ticker: str):
        logger.info(f"Price update recorded for {ticker}")
    
    def record_ml_features(self, features: dict):
        logger.info(f"ML features recorded - Regime: {features.get('regime')}")
    
    def record_broker_error(self, error_type: str, method: str):
        logger.error(f"Broker error: {error_type} in {method}")

def validate_models(config: dict) -> bool:
    """Validate that required ML models exist"""
    required_models = {
        'regime_classifier': config['ml_models']['regime_classifier']['path'],
        'anomaly_detector': config['ml_models']['anomaly_detector']['path']
    }
    
    all_valid = True
    for name, path in required_models.items():
        try:
            model = joblib.load(path)
            logger.info(f"Successfully loaded {name} model")
        except Exception as e:
            logger.error(f"Failed to load {name} model: {str(e)}")
            all_valid = False
    
    return all_valid

def test_stream():
    # Sample config - replace with your actual config loader
    config = {
        'tickers': {
            'primary': ['AAPL']
        },
        'options': {
            'expirations': ['240405'],
            'strike_rules': {
                'min_strike': 15000,
                'max_strike': 20000,
                'step': 500,
                'num_strikes': 5
            },
            'risk_free_rate': 0.05
        },
        'ml_models': {
            'regime_classifier': {
                'path': 'models/regime_classifier.joblib'
            },
            'anomaly_detector': {
                'path': 'models/anomaly_detector.joblib'
            }
        },
        'options_analytics': {
            'volatility_surface': {
                'update_freq': 60
            }
        }
    }
    
    # Validate models first
    if not validate_models(config):
        logger.error("Required ML models are missing. Please run train_models.py first.")
        return
    
    metrics = MockMetrics()
    
    try:
        from src.brokers.alpaca.options import AlpacaOptionsStreamer
        streamer = AlpacaOptionsStreamer(config, metrics)
        
        logger.info("Starting options stream (Ctrl+C to stop)...")
        start_time = time.time()
        streamer.start_stream()
        
    except KeyboardInterrupt:
        logger.info("Stream stopped by user")
    except Exception as e:
        logger.error(f"Stream failed: {str(e)}", exc_info=True)
    finally:
        if 'streamer' in locals():
            try:
                streamer.shutdown()
            except Exception as e:
                logger.error(f"Shutdown failed: {str(e)}")

if __name__ == "__main__":
    test_stream()