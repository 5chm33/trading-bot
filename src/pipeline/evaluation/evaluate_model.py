import os
import logging
import joblib
import numpy as np
import pandas as pd
import yaml
import json
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.custom_logging import setup_logger
from models.transformer_trainer import TransformerTrainer

logger = setup_logger(__name__)

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.lookback = config['model']['transformer']['lookback']
        self.required_suffixes = ['close', 'volatility', 'rsi', 'macd', 'atr', 'obv']

    def load_artifacts(self):
        """Load model and scalers with validation"""
        artifacts = {
            'model': SAC.load("final_model.zip"),
            'feature_scaler': joblib.load("model_artifacts/feature_scaler.pkl"),
            'target_scaler': joblib.load("model_artifacts/target_scaler.pkl")
        }
        return artifacts

    def prepare_test_data(self):
        """Fetch and prepare data for all tickers"""
        all_data = []
        for ticker in self.config['tickers']:
            try:
                trainer = TransformerTrainer(pd.DataFrame(), self.config, f"{ticker.lower()}_close")
                data = trainer.process_single_ticker(ticker)
                if data is not None:
                    all_data.append(data)
            except Exception as e:
                logger.error(f"Skipping {ticker}: {str(e)}")

        if not all_data:
            raise ValueError("No valid test data available")
        return pd.concat(all_data, axis=1).dropna()

    def evaluate_ticker(self, ticker_data, artifacts):
        """Core evaluation logic for one ticker"""
        # Feature extraction and scaling
        features = artifacts['feature_scaler'].transform(ticker_data)
        target_cols = [f"{ticker_data.columns[0].split('_')[0]}_{s}" for s in self.required_suffixes]
        target = ticker_data[target_cols].values

        # Sequence preparation
        X, y = [], []
        for i in range(self.lookback, len(features)):
            X.append(features[i-self.lookback:i])
            y.append(target[i])
        X, y = np.array(X), np.array(y)

        # Prediction and post-processing
        y_pred = artifacts['model'].predict(X[:, -1, -6:])[0][:, 0]  # Get first action dimension
        y_pred = self._postprocess_predictions(y_pred, y, artifacts['target_scaler'])

        # Calculate metrics
        returns = np.diff(y[:, 0]) / y[:-1, 0]  # Close price returns
        return {
            'price_metrics': self._calculate_price_metrics(y[:, 0], y_pred),
            'risk_metrics': self._calculate_risk_metrics(returns),
            'plots': self._generate_plots(y[:, 0], y_pred, ticker_data.index[-len(y):])
        }

    def _postprocess_predictions(self, y_pred, y, target_scaler):
        """Apply scaling and smoothing"""
        y_pred_padded = np.zeros((len(y_pred), 6))
        y_pred_padded[:, 0] = y_pred
        y_pred = target_scaler.inverse_transform(y_pred_padded)[:, 0]

        # Kalman Filter smoothing
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=y_pred[0],
            initial_state_covariance=1.0,
            observation_covariance=0.96,
            transition_covariance=0.00135
        )
        return kf.filter(y_pred)[0]

    def _calculate_price_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'Directional_Accuracy': self._directional_accuracy(y_true, y_pred)
        }

    def _calculate_risk_metrics(self, returns):
        """Calculate financial metrics"""
        return {
            'Sharpe_Ratio': self._sharpe_ratio(returns),
            'Sortino_Ratio': self._sortino_ratio(returns),
            'Cumulative_Returns': np.cumprod(1 + returns)[-1] - 1
        }

    def _generate_plots(self, y_true, y_pred, dates):
        """Generate evaluation plots"""
        plt.figure(figsize=(12, 6))
        plt.plot(dates, y_true, label='Actual')
        plt.plot(dates, y_pred, label='Predicted', linestyle='--')
        plt.legend()
        plt.close()
        return plt.gcf()

    @staticmethod
    def _directional_accuracy(y_true, y_pred):
        """Calculate directional accuracy"""
        true_changes = np.sign(y_true[1:] - y_true[:-1])
        pred_changes = np.sign(y_pred[1:] - y_pred[:-1])
        return np.mean(true_changes == pred_changes) * 100

    @staticmethod
    def _sharpe_ratio(returns, risk_free=0.02):
        """Calculate annualized Sharpe ratio"""
        excess = returns - risk_free/252
        return np.mean(excess) / np.std(excess) * np.sqrt(252)

    @staticmethod
    def _sortino_ratio(returns, risk_free=0.02):
        """Calculate annualized Sortino ratio"""
        excess = returns - risk_free/252
        downside = excess[excess < 0]
        return np.mean(excess) / np.std(downside) * np.sqrt(252)

def main():
    try:
        # Load config
        with open(os.path.join("config", "config.yaml")) as f:
            config = yaml.safe_load(f)

        evaluator = Evaluator(config)
        artifacts = evaluator.load_artifacts()
        test_data = evaluator.prepare_test_data()

        # Evaluate all tickers
        results = {}
        for ticker in config['tickers']:
            ticker_cols = [c for c in test_data.columns if c.startswith(f"{ticker.lower()}_")]
            if ticker_cols:
                try:
                    results[ticker] = evaluator.evaluate_ticker(
                        test_data[ticker_cols], artifacts
                    )
                except Exception as e:
                    logger.error(f"Failed {ticker}: {str(e)}")
                    results[ticker] = {'error': str(e)}

        # Save and display results
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=4)

        logger.info("Evaluation completed successfully")
        return results

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
