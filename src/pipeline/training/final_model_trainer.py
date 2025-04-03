<<<<<<< HEAD
import os
import logging
import sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.utils.logging import setup_logger
from src.models.transformer.trainer import TransformerTrainer
import yaml

logger = setup_logger(__name__)

def validate_config(config):
    """Comprehensive config validation matching your structure"""
    required = {
        'tickers': list,
        'time_settings': {
            'train': dict,
            'test': dict
        },
        'model': {
            'transformer': {
                'architecture': dict,
                'training': dict
            }
        }
    }

    def _check(section, requirements):
        for key, typ in requirements.items():
            if key not in section:
                raise ValueError(f"Missing config key: {key}")
            if isinstance(typ, dict):
                _check(section[key], typ)

    _check(config, required)

def load_config(config_path):
    """Load and validate config"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file missing at {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    validate_config(config)
    return config

def main():
    """Complete training pipeline with robust error handling."""
    try:
        # 1. Config Loading & Validation
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
        config = load_config(config_path)

        # Get training dates from config
        train_start = config['time_settings']['train']['start_date']
        train_end = config['time_settings']['train']['end_date']
        interval = config['time_settings']['interval']

        # 2. Process ALL tickers
        processed_tickers = []
        all_data = pd.DataFrame()

        for ticker in config['tickers']:
            ticker_lower = ticker.lower()
            try:
                # Fetch data for each ticker
                ticker_data = fetch_data(ticker, train_start, train_end, interval)

                if not ticker_data.empty:
                    trainer = TransformerTrainer(
                        data=ticker_data,
                        config=config,
                        close_column=f"{ticker_lower}_close"
                    )

                    # Add technical indicators
                    ticker_data = trainer.add_technical_indicators(ticker_data, ticker_lower)
                    all_data = pd.concat([all_data, ticker_data], axis=1)
                    processed_tickers.append(ticker_lower)
                    logger.info(f"Processed {ticker} | Shape: {ticker_data.shape}")

            except Exception as e:
                logger.error(f"Skipping {ticker}: {str(e)}", exc_info=True)

        if not processed_tickers:
            raise ValueError("No tickers processed successfully")

        # 3. Initialize final trainer with complete data
        primary_ticker = processed_tickers[0]
        trainer = TransformerTrainer(
            data=all_data,
            config=config,
            close_column=f"{primary_ticker}_close"
        )

        # 4. Data Preparation
        data = trainer.handle_missing_data(all_data)
        prepared_data = trainer.prepare_data(
            data=data,
            lookback=config['model']['transformer']['architecture']['lookback']
        )
        if not prepared_data:
            raise ValueError("Data preparation failed")

        # 5. Model Training
        model = trainer.build_model(
            config['model']['transformer'],
            prepared_data[6]  # num_features
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config['model']['transformer']['training']['learning_rate']
            ),
            loss='mse'
        )

        history = model.fit(
            prepared_data[0], prepared_data[2],  # X_train, y_train
            validation_data=(prepared_data[1], prepared_data[3]),  # X_val, y_val
            epochs=config['model']['transformer']['training']['epochs'],
            batch_size=config['model']['transformer']['training']['batch_size'],
            callbacks=[
                EarlyStopping(
                    patience=config['model']['transformer']['training']['patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(factor=0.2, patience=5)
            ],
            verbose=1
        )

        # 6. Save Outputs
        save_dir = "model_artifacts"
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, "trained_model.h5"))
        joblib.dump(prepared_data[4], os.path.join(save_dir, "feature_scaler.pkl"))
        joblib.dump(prepared_data[5], os.path.join(save_dir, "target_scaler.pkl"))

        logger.info(
            f"Training completed\n"
            f"Tickers: {processed_tickers}\n"
            f"Final data shape: {data.shape}\n"
            f"Validation loss: {min(history.history['val_loss']):.4f}"
        )

    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
=======
import os
import logging
import sys
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.utils.logging import setup_logger
from src.models.transformer.trainer import TransformerTrainer
import yaml

logger = setup_logger(__name__)

def validate_config(config):
    """Comprehensive config validation matching your structure"""
    required = {
        'tickers': list,
        'time_settings': {
            'train': dict,
            'test': dict
        },
        'model': {
            'transformer': {
                'architecture': dict,
                'training': dict
            }
        }
    }
    
    def _check(section, requirements):
        for key, typ in requirements.items():
            if key not in section:
                raise ValueError(f"Missing config key: {key}")
            if isinstance(typ, dict):
                _check(section[key], typ)

    _check(config, required)

def load_config(config_path):
    """Load and validate config"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file missing at {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    validate_config(config)
    return config

def main():
    """Complete training pipeline with robust error handling."""
    try:
        # 1. Config Loading & Validation
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
        config = load_config(config_path)
        
        # Get training dates from config
        train_start = config['time_settings']['train']['start_date']
        train_end = config['time_settings']['train']['end_date']
        interval = config['time_settings']['interval']

        # 2. Process ALL tickers
        processed_tickers = []
        all_data = pd.DataFrame()
        
        for ticker in config['tickers']:
            ticker_lower = ticker.lower()
            try:
                # Fetch data for each ticker
                ticker_data = fetch_data(ticker, train_start, train_end, interval)
                
                if not ticker_data.empty:
                    trainer = TransformerTrainer(
                        data=ticker_data,
                        config=config,
                        close_column=f"{ticker_lower}_close"
                    )
                    
                    # Add technical indicators
                    ticker_data = trainer.add_technical_indicators(ticker_data, ticker_lower)
                    all_data = pd.concat([all_data, ticker_data], axis=1)
                    processed_tickers.append(ticker_lower)
                    logger.info(f"Processed {ticker} | Shape: {ticker_data.shape}")
                    
            except Exception as e:
                logger.error(f"Skipping {ticker}: {str(e)}", exc_info=True)

        if not processed_tickers:
            raise ValueError("No tickers processed successfully")

        # 3. Initialize final trainer with complete data
        primary_ticker = processed_tickers[0]
        trainer = TransformerTrainer(
            data=all_data,
            config=config,
            close_column=f"{primary_ticker}_close"
        )

        # 4. Data Preparation
        data = trainer.handle_missing_data(all_data)
        prepared_data = trainer.prepare_data(
            data=data,
            lookback=config['model']['transformer']['architecture']['lookback']
        )
        if not prepared_data:
            raise ValueError("Data preparation failed")

        # 5. Model Training
        model = trainer.build_model(
            config['model']['transformer'],
            prepared_data[6]  # num_features
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config['model']['transformer']['training']['learning_rate']
            ),
            loss='mse'
        )

        history = model.fit(
            prepared_data[0], prepared_data[2],  # X_train, y_train
            validation_data=(prepared_data[1], prepared_data[3]),  # X_val, y_val
            epochs=config['model']['transformer']['training']['epochs'],
            batch_size=config['model']['transformer']['training']['batch_size'],
            callbacks=[
                EarlyStopping(
                    patience=config['model']['transformer']['training']['patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(factor=0.2, patience=5)
            ],
            verbose=1
        )

        # 6. Save Outputs
        save_dir = "model_artifacts"
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, "trained_model.h5"))
        joblib.dump(prepared_data[4], os.path.join(save_dir, "feature_scaler.pkl"))
        joblib.dump(prepared_data[5], os.path.join(save_dir, "target_scaler.pkl"))
        
        logger.info(
            f"Training completed\n"
            f"Tickers: {processed_tickers}\n"
            f"Final data shape: {data.shape}\n"
            f"Validation loss: {min(history.history['val_loss']):.4f}"
        )

    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
>>>>>>> 60870aec3b9ed2c2cb804ceb4f1eeb5c6af9d852
