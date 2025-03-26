import logging
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from model_training.build_model import build_model
from data_preprocessing.load_and_preprocess import load_and_preprocess

logger = logging.getLogger(__name__)

def train_final_model(config, best_hyperparams):
    """Train the final model with the best hyperparameters."""
    try:
        # Load and preprocess data
        combined_data = load_and_preprocess(config)
        if combined_data is None:
            logger.error("Data loading and preprocessing failed. Cannot proceed with training.")
            return None, None

        # Initialize the trainer
        logger.info("Initializing the TransformerTrainer.")
        trainer = TransformerTrainer(combined_data, config, close_column='Close_AAPL')

        # Prepare data
        logger.info("Preparing data for training.")
        prepared_data = trainer.prepare_data(ticker='AAPL')
        if prepared_data is None:
            logger.error("prepare_data returned None. Check the method for errors.")
            return None, None
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = prepared_data
            logger.info("prepare_data completed successfully.")
            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"y_train shape: {y_train.shape}")

        # Train the final model
        logger.info("Training the final model with best hyperparameters.")
        final_model = build_model(best_hyperparams)
        history = final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=2000,
            batch_size=best_hyperparams["batch_size"],
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
        )

        return final_model, history
    except Exception as e:
        logger.error(f"Error in train_final_model: {e}", exc_info=True)
        return None, None