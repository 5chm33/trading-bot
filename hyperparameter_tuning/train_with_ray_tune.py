# hyperparameter_tuning/train_with_ray_tune.py
from ray import tune
import logging
import tensorflow as tf
from data_preprocessing.prepare_data import prepare_data
from transformer6 import TransformerTrainer  # Import the TransformerTrainer class
import os

logger = logging.getLogger(__name__)

def train_with_ray_tune(config, checkpoint_dir=None, data=None):
    """Train the model with Ray Tune and save the best model."""
    try:
        # Ensure data is provided
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty.")
        
        # Prepare data for training
        logger.info("Preparing data for training.")
        lookback = config.get("lookback", 30)  # Default to 30 if not provided
        X_train, X_val, y_train, y_val, feature_scaler, target_scaler = prepare_data(data, lookback=lookback)
        
        # Log the shapes of the training and validation sets
        logger.debug(f"X_train shape: {X_train.shape}")
        logger.debug(f"X_val shape: {X_val.shape}")
        logger.debug(f"y_train shape: {y_train.shape}")
        logger.debug(f"y_val shape: {y_val.shape}")
        
        # Initialize the TransformerTrainer
        trainer = TransformerTrainer(data, config, close_column='Close_AAPL')
        
        # Set input_shape for the model
        trainer.input_shape = X_train.shape[1:]  # Set input_shape based on X_train
        
        # Build the model
        logger.info("Building the model.")
        model = trainer.build_model(config)
        if model is None:
            raise ValueError("Model building failed.")
        
        # Log the model summary
        model.summary(print_fn=logger.info)
        
        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6
        )
        
        # Checkpoint Callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "checkpoint.h5"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=0
        )
        
        # Train the model
        logger.info("Training the model.")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.get("epochs", 100),
            batch_size=config["batch_size"],
            callbacks=[early_stopping, reduce_lr, checkpoint_callback],
            verbose=0
        )
        
        # Evaluate the model on the validation set
        val_loss = history.history.get('val_loss', [float('inf')])[-1]
        logger.info(f"Validation loss: {val_loss}")
        
        # Save the best model
        model_save_path = os.path.abspath("model_save_path/best_model.h5")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Ensure directory exists
        if os.path.exists(model_save_path):
            logger.info(f"Deleting old model at {model_save_path}")
            os.remove(model_save_path)

        model.save(model_save_path)
        logger.info(f"Saved new model to {model_save_path}")
        
        # Report metrics to Ray Tune
        tune.report({"loss": val_loss})  # Correctly report the loss as a dictionary
    
    except Exception as e:
        logger.error(f"Error during training with Ray Tune: {e}", exc_info=True)
        raise
    finally:
        # Ensure Ray Tune receives a loss value even if an error occurs
        tune.report({"loss": val_loss if 'val_loss' in locals() else float('inf')})