import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class RegimeClassifier:
    """
    ML model to classify market regimes (trending/mean-reverting/volatile)
    Uses both technical features and options market data
    """
    
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                objective='multi:softprob'
            ))
        ])
        self.classes = ['trending_up', 'trending_down', 'mean_reverting', 'high_vol']
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train on historical data"""
        self.model.fit(X, y)
        joblib.dump(self.model, 'models/regime_classifier.joblib')
        
    def predict(self, features: list) -> str:
        """Predict current market regime"""
        model = joblib.load('models/regime_classifier.joblib')
        probas = model.predict_proba([features])[0]
        return self.classes[np.argmax(probas)]
    
    @staticmethod
    def create_features(
        volatility: float,
        iv_rank: float,
        term_structure: dict,
        price_action: dict
    ) -> list:
        """Create feature vector for prediction"""
        return [
            volatility,
            iv_rank,
            term_structure['skew'],
            term_structure['curve_value'],
            price_action['trend_strength'],
            price_action['atr'],
            price_action['rsi']
        ]