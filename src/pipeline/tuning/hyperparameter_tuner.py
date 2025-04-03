import os
import json
import yaml
from ray import tune
from ray.tune import Trial
from ray.tune.schedulers import ASHAScheduler
import tensorflow as tf
from utils import setup_logger
from models.transformer_trainer import TransformerTrainer

logger = setup_logger(__name__)

class HyperparameterOptimizer:
    def __init__(self, config_path="config/config.yaml"):
        self.config = self._load_config(config_path)
        self.trainer = None

    def _load_config(self, path):
        """Safe config loader with validation"""
        with open(path) as f:
            config = yaml.safe_load(f)

        required_keys = ['tickers', 'model', 'time_settings']
        if not all(k in config for k in required_keys):
            raise ValueError(f"Config missing required keys: {required_keys}")
        return config

    def prepare_data(self):
        """Centralized data preparation"""
        logger.info("Preparing training data...")
        self.trainer = TransformerTrainer(None, self.config, close_column=None)
        data = self.trainer.process_all_tickers()  # Should be implemented in TransformerTrainer
        return self.trainer.prepare_data(data)

    def train_with_ray(self, config, checkpoint_dir=None):
        """Ray Tune compatible training function"""
        try:
            merged_config = {**self.config['model']['transformer'], **config}
            return self.trainer.train_with_ray_tune(merged_config)
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"loss": float('inf')}

    def optimize(self):
        """Main optimization pipeline"""
        try:
            # 1. Data Preparation
            prepared_data = self.prepare_data()
            if not prepared_data:
                raise ValueError("Data preparation failed")

            # 2. Bayesian Optimization (Initial Search)
            bayes_params = self._run_bayesian_optimization(*prepared_data[:4])
            logger.info(f"Bayesian Optimization results: {bayes_params}")

            # 3. Ray Tune Fine-Tuning
            best_params = self._run_ray_tune(bayes_params)

            # 4. Save Results
            with open("best_params.json", "w") as f:
                json.dump(best_params, f, indent=4)
            logger.info(f"Final best parameters saved to best_params.json")

            return best_params

        except Exception as e:
            logger.critical(f"Optimization failed: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    optimizer = HyperparameterOptimizer()
    optimizer.optimize()
