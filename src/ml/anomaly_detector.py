from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

class TradeAnomalyDetector:
    """
    Detects statistically anomalous trades using:
    - Isolation Forests
    - Local Outlier Factor
    """
    
    def __init__(self):
        self.models = {
            'isolation': IsolationForest(contamination=0.05, random_state=42),
            'lof': LocalOutlierFactor(n_neighbors=20, novelty=True)
        }
        
    def train(self, historical_trades: list):
        """Train on historical trade performance data"""
        features = self._extract_features(historical_trades)
        for model in self.models.values():
            model.fit(features)
            
    def detect(self, new_trades: list) -> list:
        """Filter out anomalous trades"""
        features = self._extract_features(new_trades)
        scores = {}
        
        for name, model in self.models.items():
            scores[name] = model.predict(features)
            
        # Only keep trades that pass both filters
        mask = (scores['isolation'] == 1) & (scores['lof'] == 1)
        return [trade for trade, keep in zip(new_trades, mask) if keep]
        
    def _extract_features(self, trades: list) -> np.ndarray:
        """Convert trades to feature vectors"""
        return np.array([
            [
                trade['probability'],
                trade['max_loss'],
                trade['expected_value'],
                trade['iv_rank'],
                trade['days_to_expiry']
            ] for trade in trades
        ])