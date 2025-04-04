# src/ml/train_models.py
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.exceptions import NotFittedError
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_config(config: dict) -> bool:
    """Validate the configuration dictionary"""
    required_keys = {
        'ml_models': ['regime_classifier', 'anomaly_detector'],
        'data_paths': ['regime_data', 'anomaly_data']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            logger.error(f"Missing config section: {section}")
            return False
        for key in keys:
            if key not in config[section]:
                logger.error(f"Missing config key: {section}.{key}")
                return False
    return True

def train_regime_classifier(config: dict) -> Pipeline:
    """Train and save regime classifier model"""
    try:
        data_path = Path(config['data_paths']['regime_data'])
        df = pd.read_csv(data_path)
        
        # Add label encoding
        le = LabelEncoder()
        labels = le.fit_transform(df['regime_label'])
        
        # Save the label encoder for later use
        label_encoder_path = Path('models/label_encoder.joblib')
        label_encoder_path.parent.mkdir(exist_ok=True)
        joblib.dump(le, label_encoder_path)
        
        features = df[config['ml_models']['regime_classifier']['features']]
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                n_estimators=config['ml_models']['regime_classifier']['hyperparameters']['n_estimators'],
                max_depth=config['ml_models']['regime_classifier']['hyperparameters']['max_depth'],
                learning_rate=config['ml_models']['regime_classifier']['hyperparameters']['learning_rate'],
                objective='multi:softprob',
                random_state=42
            ))
        ])
        
        model.fit(features, labels)
        
        model_path = Path(config['ml_models']['regime_classifier']['path'])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Successfully saved regime classifier to {model_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to train regime classifier: {str(e)}")
        raise

def train_anomaly_detector(config: dict) -> dict:
    """Train and save anomaly detection models with robust validation and adaptive parameters"""
    try:
        # Validate configuration
        if 'anomaly_detector' not in config['ml_models']:
            raise KeyError("Missing 'anomaly_detector' configuration in ml_models")
        
        # Path handling and validation
        data_path = Path(config['data_paths']['anomaly_data'])
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
        if data_path.stat().st_size == 0:
            raise ValueError("Training data file is empty")

        # Data loading and validation
        df = pd.read_csv(data_path)
        if len(df) < 5:  # Minimum reasonable sample size
            logger.warning(f"Very small dataset ({len(df)} samples). Anomaly detection may be unreliable.")
        
        required_features = config['ml_models']['anomaly_detector']['features']
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        # Feature validation
        features = df[required_features]
        if features.isnull().values.any():
            raise ValueError("Training data contains null values")
        
        # Adaptive parameter tuning
        n_samples = len(features)
        min_samples = max(5, int(n_samples * 0.1))  # At least 5 or 10% of samples
        n_neighbors = min(
            config['ml_models']['anomaly_detector'].get('n_neighbors', 20),
            n_samples - 1
        )
        contamination = min(
            config['ml_models']['anomaly_detector'].get('contamination', 0.1),
            0.5  # Absolute maximum
        )

        logger.info(f"Training anomaly detector with {n_samples} samples. Parameters: "
                   f"n_neighbors={n_neighbors}, contamination={contamination}")

        # Model training
        models = {
            'isolation': IsolationForest(
                contamination=contamination,
                random_state=42,
                verbose=1  # Add verbosity for training feedback
            ),
            'lof': LocalOutlierFactor(
                n_neighbors=n_neighbors,
                novelty=True,
                metric='euclidean'
            )
        }
        
        # Fit models with progress feedback
        logger.info("Fitting Isolation Forest...")
        models['isolation'].fit(features)
        
        logger.info("Fitting LOF model...")
        models['lof'].fit(features)  # Note: LOF doesn't actually fit during this step
        
        # Model validation
        try:
            test_preds = models['isolation'].predict(features[:1])  # Test prediction
            if len(np.unique(test_preds)) == 1 and n_samples > 10:
                logger.warning("Model may not be detecting anomalies effectively")
        except Exception as e:
            logger.warning(f"Model validation check failed: {str(e)}")

        # Save models
        model_path = Path(config['ml_models']['anomaly_detector']['path'])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'models': models,
            'metadata': {
                'training_date': datetime.now().isoformat(),
                'n_samples': n_samples,
                'features': required_features,
                'parameters': {
                    'n_neighbors': n_neighbors,
                    'contamination': contamination
                }
            }
        }
        
        joblib.dump(save_data, model_path)
        logger.info(f"Successfully saved anomaly detector to {model_path}")
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to train anomaly detector: {str(e)}", exc_info=True)
        raise RuntimeError("Anomaly detector training failed") from e

if __name__ == "__main__":
    config = {
        'ml_models': {
            'regime_classifier': {
                'path': 'models/regime_classifier.joblib',
                'features': ['volatility_30d', 'iv_rank', 'skew_30d', 'trend_strength', 'atr_14d', 'rsi_14'],
                'hyperparameters': {
                    'n_estimators': 200,
                    'max_depth': 5,
                    'learning_rate': 0.1
                }
            },
            'anomaly_detector': {
                'path': 'models/anomaly_detector.joblib',
                'features': ['probability', 'max_loss', 'expected_value', 'iv_rank', 'days_to_expiry'],
                'contamination': 0.01,
                'n_neighbors': 3,
                'min_samples': 5       
            }
        },
        'data_paths': {
            'regime_data': 'data/historical/regime_data.csv',
            'anomaly_data': 'data/historical/trade_data.csv'
        }
    }
    
    if not validate_config(config):
        sys.exit(1)
    
    try:
        logger.info("Starting model training...")
        train_regime_classifier(config)
        train_anomaly_detector(config)
        logger.info("Model training completed successfully!")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        sys.exit(1)