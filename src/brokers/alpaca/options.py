# src/brokers/alpaca/options.py
import os
import logging
import joblib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream, OptionDataStream
from alpaca.data.models import Quote
from alpaca.data.models.option import OptionQuote
from alpaca.data.requests import OptionChainRequest
from src.monitoring.trading_metrics import TradeMetrics
from src.ml.regime_classifier import RegimeClassifier
from src.ml.anomaly_detector import TradeAnomalyDetector
from py_vollib.black_scholes.greeks import analytical
import numpy as np

logger = logging.getLogger(__name__)

class AlpacaOptionsStreamer:
    """ML-enhanced options trading streamer with real-time analytics"""

    def __init__(self, config: Dict, metrics: TradeMetrics):
        self.config = config
        self.metrics = metrics
        self.underlying_price = None
        self.symbols = []
        self.volatility_surface = {}
        self.current_regime = None
        self.term_structure = {}
        self._last_vol_update = datetime.min
        
        # Initialize ML components
        self.regime_classifier = RegimeClassifier()
        self.anomaly_detector = TradeAnomalyDetector()
        self._load_ml_models()
        
        # Initialize clients
        auth = {
            'api_key': os.getenv('ALPACA_KEY'),
            'secret_key': os.getenv('ALPACA_SECRET')
        }
        self.hist_client = StockHistoricalDataClient(**auth)
        self.stock_stream = StockDataStream(**auth)
        self.options_stream = OptionDataStream(**auth)
        
        # Setup streams
        self._setup_underlying_stream()
        self._generate_option_chain()
        
    def _load_ml_models(self):
        """Load pre-trained ML models with validation"""
        try:
            model_dir = self.config['ml_models'].get('model_dir', 'models/')
            self.regime_classifier = joblib.load(f"{model_dir}regime_classifier.joblib")
            self.anomaly_detector = joblib.load(f"{model_dir}anomaly_detector.joblib")
            self.metrics.record_component_init('ml_models', True)
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ML models: {str(e)}")
            self.metrics.record_component_init('ml_models', False)
            raise RuntimeError("ML model loading failed") from e

    def _setup_underlying_stream(self):
        """Start streaming the underlying stock price"""
        underlying = self.config['tickers']['primary'][0]
        
        def price_handler(quote: Quote):
            self.underlying_price = quote.bid_price
            self.metrics.record_price_update(underlying)
            logger.debug(f"Underlying {underlying} price: ${self.underlying_price:.2f}")
            
        self.stock_stream.subscribe_quotes(price_handler, underlying)
        self.stock_stream.run()

    def _generate_option_chain(self) -> None:
        """Generate option symbols with automatic retry"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.underlying_price is None:
                    raise ValueError("Underlying price not available")
                
                expirations = self._get_available_expirations()
                strikes = self._calculate_strikes()
                
                symbol = self.config['tickers']['primary'][0]
                self.symbols = [
                    f"{symbol}{exp}{typ}{strike:08d}"
                    for exp in expirations
                    for typ in ['C', 'P']
                    for strike in strikes
                ]
                logger.info(f"Generated option chain with {len(self.symbols)} contracts")
                return
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Option chain generation failed after {max_retries} attempts")
                    self.metrics.record_broker_error(
                        error_type=type(e).__name__,
                        method='symbol_generation'
                    )
                    raise
                time.sleep(2 ** attempt)

    def _get_available_expirations(self) -> List[str]:
        """Get expirations with fallback to config defaults"""
        try:
            chain = self.hist_client.get_option_chain(
                OptionChainRequest(
                    symbol=self.config['tickers']['primary'][0],
                    expiration_from=datetime.now(),
                    expiration_to=datetime.now() + timedelta(days=45))
            )
            expirations = sorted({d.strftime("%y%m%d") for d in chain.expirations})
            return expirations[:self.config['options'].get('max_expirations', 4)]
        except Exception as e:
            logger.warning(f"Using config expirations (API failed: {str(e)})")
            return self.config['options'].get('expirations', [
                (datetime.now() + timedelta(days=d)).strftime("%y%m%d")
                for d in [7, 14, 30, 45]
            ])

    def _calculate_strikes(self) -> List[int]:
        """Generate strikes with configurable step sizes"""
        price = self.underlying_price
        step_config = self.config['options'].get('strike_step', {})
        step = (
            step_config.get('below_100', 5) if price < 100 
            else step_config.get('above_100', 10)
        )
        center = round(price / step) * step
        num_strikes = self.config['options'].get('num_strikes', 10)
        
        return [
            int(center + i * step)
            for i in range(-num_strikes, num_strikes + 1)
            if (center + i * step) > 0
        ]

    def start_stream(self, handler: Optional[Callable] = None):
        """Start streaming with enhanced error handling"""
        if not self.symbols:
            self._generate_option_chain()
            
        try:
            logger.info(f"Starting options stream for {len(self.symbols)} contracts")
            self.options_stream.subscribe_quotes(
                handler or self._ml_enhanced_handler,
                *self.symbols
            )
            self.options_stream.run()
        except Exception as e:
            self.metrics.record_broker_error(
                error_type=type(e).__name__,
                method='start_stream'
            )
            logger.critical(f"Stream start failed: {str(e)}")
            raise

    def _ml_enhanced_handler(self, quote: OptionQuote):
        """Integrated ML processing pipeline"""
        try:
            # Update market state
            current_time = datetime.now()
            if (current_time - self._last_vol_update).total_seconds() > \
               self.config['options_analytics'].get('volatility_update_freq', 60):
                self._update_volatility_surface(quote)
                self._last_vol_update = current_time
                
            self._update_term_structure()
            
            # Classify regime
            features = self.regime_classifier.create_features(
                volatility=self._calculate_volatility(),
                iv_rank=self._calculate_iv_rank(),
                term_structure=self.term_structure,
                price_action=self._get_price_action()
            )
            self.current_regime = self.regime_classifier.predict(features)
            
            # Calculate and validate Greeks
            greeks = self._validate_greeks(self._calculate_greeks(quote))
            
            # Package ML features
            ml_data = {
                'regime': self.current_regime,
                'volatility_surface': self.volatility_surface,
                'greeks': greeks,
                'anomaly_score': self._calculate_anomaly_score(quote),
                'timestamp': current_time.isoformat()
            }
            
            # Pass to downstream systems
            self._dispatch_ml_data(ml_data)
            
        except Exception as e:
            logger.error(f"ML processing failed: {str(e)}", exc_info=True)
            self.metrics.record_broker_error(
                error_type=type(e).__name__,
                method='ml_handler'
            )

    def _validate_greeks(self, greeks: Dict) -> Dict:
        """Ensure greeks values are within reasonable bounds"""
        valid_greeks = {}
        bounds = self.config['options_analytics'].get('greeks_bounds', {
            'delta': (-1.5, 1.5),
            'gamma': (0, 0.2),
            'theta': (-0.1, 0),
            'vega': (0, 0.5)
        })
        
        for greek, value in greeks.items():
            if bounds.get(greek):
                valid_greeks[greek] = np.clip(value, *bounds[greek])
            else:
                valid_greeks[greek] = value
        return valid_greeks

    def _dispatch_ml_data(self, ml_data: Dict):
        """Send ML data to all registered consumers"""
        self.metrics.record_ml_features(ml_data)
        
        # Optional strategy engine integration
        if hasattr(self, 'strategy_engine'):
            try:
                self.strategy_engine.update_ml_state(ml_data)
            except Exception as e:
                logger.error(f"Strategy engine update failed: {str(e)}")

    def _calculate_greeks(self, quote: OptionQuote) -> Dict[str, float]:
        """Calculate Black-Scholes Greeks with configurable parameters"""
        if self.underlying_price is None:
            raise ValueError("Underlying price not available")
            
        S = self.underlying_price
        K = float(quote.symbol[-8:])/1000
        T = max(1/365, self._calculate_dte(quote.symbol[6:12])/365)  # Minimum 1 day
        r = self.config['options_analytics'].get('risk_free_rate', 0.05)
        sigma = self.volatility_surface.get(
            quote.symbol[:6], 
            self.config['options_analytics'].get('default_volatility', 0.25)
        )
        
        flag = 'c' if 'C' in quote.symbol else 'p'
        
        return {
            'delta': analytical.delta(flag, S, K, T, r, sigma),
            'gamma': analytical.gamma(flag, S, K, T, r, sigma),
            'theta': analytical.theta(flag, S, K, T, r, sigma),
            'vega': analytical.vega(flag, S, K, T, r, sigma)
        }

    def _calculate_dte(self, expiration: str) -> float:
        """Calculate days to expiration with validation"""
        try:
            exp_date = datetime.strptime(expiration, "%y%m%d")
            return max(0, (exp_date - datetime.now()).days)  # Never negative
        except ValueError:
            logger.warning(f"Invalid expiration format: {expiration}")
            return 0

    def shutdown(self):
        """Cleanly shutdown all streams"""
        try:
            self.stock_stream.stop()
            self.options_stream.stop()
            logger.info("All streams stopped successfully")
        except Exception as e:
            logger.error(f"Stream shutdown failed: {str(e)}")
            raise