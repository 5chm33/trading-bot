import logging
import joblib

logger = logging.getLogger(__name__)

def save_model_and_scalers(model, scaler, target_scaler):
    """Save the final model and scalers."""
    try:
        logger.info("Saving the final model and scalers.")
        model.save('final_trading_model.keras')
        joblib.dump(scaler, 'final_scaler.pkl')
        joblib.dump(target_scaler, 'final_target_scaler.pkl')
        logger.info("Model and scalers saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save the model or scalers: {e}", exc_info=True)