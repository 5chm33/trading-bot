from ray import tune
from ray.tune.search.sample import Categorical

def get_search_space():
    """Define a broader search space for hyperparameter tuning."""
    # Define common values for lookback and sequence_length
    common_sequence_values = [30, 60, 90]

    # Define the search space
    search_space = {
        "learning_rate": tune.loguniform(1e-6, 1e-3),  # Wider range
        "num_heads": tune.choice([2, 4, 8]),           # More options
        "dropout_rate": tune.uniform(0.1, 0.5),        # Wider range
        "num_layers": tune.choice([1, 2, 3]),          # More options
        "ff_dim": tune.choice([128, 256, 512]),        # More options
        "batch_size": tune.choice([16, 32, 64]),       # More options
        "l2_reg": tune.loguniform(1e-4, 1e-2),         # Wider range
        "lookback": tune.sample_from(lambda spec: spec.config.sequence_length),  # Same as sequence_length
        "sequence_length": tune.choice(common_sequence_values),  # Common values
    }
    return search_space