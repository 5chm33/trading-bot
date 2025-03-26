from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.regularizers import l2
import logging

logger = logging.getLogger(__name__)

def build_model(config):
    """Build the Transformer-based model."""
    try:
        # Ensure sequence_length and num_features are in the config
        sequence_length = config.get("sequence_length", 30)  # Default to 30 if not provided
        num_features = config.get("num_features", 6)  # Default to 6 if not provided
        
        inputs = Input(shape=(sequence_length, num_features))
        
        # Transformer layers
        x = MultiHeadAttention(num_heads=config["num_heads"], key_dim=num_features)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(config.get("dropout_rate", 0.1))(x)
        
        # Output layer
        outputs = Dense(1, activation='linear', kernel_regularizer=l2(config.get("l2_reg", 0.01)))(x)
        
        model = Model(inputs, outputs)
        return model
    except Exception as e:
        logger.error(f"Error building model: {e}", exc_info=True)
        return None