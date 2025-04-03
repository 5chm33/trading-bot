<<<<<<< HEAD
# config/search_space.py
from ray import tune

def get_search_space():
    """Define a broader search space for hyperparameter tuning."""
    search_space = {
        "learning_rate": tune.loguniform(1e-6, 1e-3),  # Wider range
        "num_heads": tune.choice([2, 4, 8]),           # More options
        "dropout_rate": tune.uniform(0.1, 0.5),        # Wider range
        "num_layers": tune.choice([1, 2, 3]),          # More options
        "ff_dim": tune.choice([128, 256, 512]),        # More options
        "batch_size": tune.choice([32, 64, 128,]),       # More options
        "l2_reg": tune.loguniform(1e-4, 1e-2),         # Wider range
        "lookback": tune.choice([30]),         # More options
        "sequence_length": tune.choice([30]),  # More options
    }
    return search_space
=======
# config/search_space.py
from ray import tune

def get_search_space():
    """Define a broader search space for hyperparameter tuning."""
    search_space = {
        "learning_rate": tune.loguniform(1e-6, 1e-3),  # Wider range
        "num_heads": tune.choice([2, 4, 8]),           # More options
        "dropout_rate": tune.uniform(0.1, 0.5),        # Wider range
        "num_layers": tune.choice([1, 2, 3]),          # More options
        "ff_dim": tune.choice([128, 256, 512]),        # More options
        "batch_size": tune.choice([32, 64, 128,]),       # More options
        "l2_reg": tune.loguniform(1e-4, 1e-2),         # Wider range
        "lookback": tune.choice([30]),         # More options
        "sequence_length": tune.choice([30]),  # More options
    }
    return search_space
>>>>>>> 60870aec3b9ed2c2cb804ceb4f1eeb5c6af9d852
