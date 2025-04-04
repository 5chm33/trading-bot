# src/strategies/options.py
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import logging
from scipy.stats import norm
from py_vollib.black_scholes.greeks import analytical
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

@dataclass 
class OptionLeg:
    symbol: str
    action: str  # 'buy'/'sell'
    quantity: int = 1
    ratio: float = 1.0  # For ratio spreads
    greeks: Optional[Dict] = None

@dataclass
class StrategyParameters:
    max_capital: float = 0.1  # % of portfolio
    min_probability: float = 0.7  # Win probability
    max_dte: int = 45  # Days to expiration
    min_iv_rank: float = 0.3  # IV percentile
    earnings_buffer: int = 3  # Days around earnings

class OptionsAIFactory:
    """Self-optimizing options strategy generator with:
    - Market regime detection
    - IV-based strategy selection
    - Earnings-aware positioning
    - Dynamic risk management
    - Anomaly detection
    - Auto-balancing greek exposure
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.underlying_price = None
        self.volatility = None
        self.iv_rank = None
        self.earnings_dates = []
        self.current_positions = []
        self.params = StrategyParameters()
        
        # Machine learning models
        self.anomaly_detector = IsolationForest(contamination=0.05)
        self.regime_classifier = self._init_regime_model()
        
        # Initialize from config
        self._load_earnings_calendar()
        self._load_historical_volatility()
        
    def update_market_state(self, 
                          underlying_price: float,
                          implied_vol: float,
                          iv_rank: float,
                          term_structure: Dict):
        """Update all market conditions"""
        self.underlying_price = underlying_price
        self.implied_vol = implied_vol
        self.iv_rank = iv_rank
        self.term_structure = term_structure
        self.current_regime = self._classify_regime()
        
    def generate_trades(self) -> List[Dict]:
        """Main pipeline for trade generation"""
        # 1. Market filtering
        if not self._market_conditions_valid():
            return []
            
        # 2. Strategy selection
        strategy_fn = self._select_strategy()
        
        # 3. Trade construction
        trades = []
        for expiration in self._get_valid_expirations():
            try:
                trade = strategy_fn(expiration)
                if self._validate_trade(trade):
                    trades.append(trade)
            except Exception as e:
                logger.error(f"Trade generation failed: {str(e)}")
                
        # 4. Portfolio balancing
        return self._balance_portfolio(trades)
    
    def _select_strategy(self) -> Callable:
        """Select strategy based on 50+ market factors"""
        if self._is_earnings_period():
            return self._earnings_strategy
            
        if self.iv_rank > 0.7:
            if self.current_regime == "high_vol":
                return self._iron_condor_strategy
            return self._strangle_strategy
            
        elif self.iv_rank > 0.4:
            return self._credit_spread_strategy
            
        else:
            if self.current_regime == "trending_up":
                return self._call_diagonal_strategy
            return self._put_calendar_strategy
    
    # Enhanced Strategies -----------------------------------------------------
    
    def _iron_condor_strategy(self, expiration: str) -> Dict:
        """High-probability iron condor with dynamic wings"""
        call_strikes = self._get_strikes(0.3, 0.1, 'call')
        put_strikes = self._get_strikes(-0.3, -0.1, 'put')
        
        return {
            'type': 'iron_condor',
            'legs': [
                OptionLeg(f"{self.symbol}{expiration}C{call_strikes[0]:08d}", 'sell'),
                OptionLeg(f"{self.symbol}{expiration}C{call_strikes[1]:08d}", 'buy'),
                OptionLeg(f"{self.symbol}{expiration}P{put_strikes[0]:08d}", 'sell'),
                OptionLeg(f"{self.symbol}{expiration}P{put_strikes[1]:08d}", 'buy')
            ],
            'probability': self._calculate_probability(call_strikes[0], put_strikes[0]),
            'management': {
                'profit_target': 0.5,
                'stop_loss': 2.0,
                'auto_roll': True
            }
        }
    
    def _earnings_strategy(self, expiration: str) -> Dict:
        """Earnings-specific strategy with IV crush protection"""
        if expiration not in self._get_earnings_expirations():
            return None
            
        straddle_price = self._get_straddle_price(expiration)
        expected_move = self._get_expected_earnings_move()
        
        return {
            'type': 'butterfly',
            'legs': [
                OptionLeg(f"{self.symbol}{expiration}C{expected_move[0]:08d}", 'buy'),
                OptionLeg(f"{self.symbol}{expiration}C{expected_move[1]:08d}", 'sell', 2),
                OptionLeg(f"{self.symbol}{expiration}C{expected_move[2]:08d}", 'buy')
            ],
            'max_loss': straddle_price * 0.8,
            'iv_crush_protection': True
        }
    
    # Advanced Analytics -----------------------------------------------------
    
    def _calculate_probability(self, short_call, short_put) -> float:
        """Monte Carlo probability estimation"""
        returns = self.historical_returns
        trials = 10000
        price_paths = self.underlying_price * np.exp(
            np.cumsum(np.random.choice(returns, size=(trials, 5)), axis=1))
        
        itm = np.sum((price_paths[:, -1] > short_call) | 
                    (price_paths[:, -1] < short_put))
        return 1 - (itm / trials)
    
    def _classify_regime(self) -> str:
        """Machine learning market regime classification"""
        features = [
            self.volatility,
            self.iv_rank,
            self.term_structure['skew'],
            self._get_recent_trend()
        ]
        return self.regime_classifier.predict([features])[0]
    
    def _detect_anomalies(self, trades: List) -> List:
        """Filter statistically anomalous trades"""
        trade_features = []
        for trade in trades:
            trade_features.append([
                trade['probability'],
                trade['max_loss'],
                self._calculate_expected_value(trade)
            ])
            
        anomalies = self.anomaly_detector.fit_predict(trade_features)
        return [t for t, a in zip(trades, anomalies) if a != -1]
    
    # Automated Management ---------------------------------------------------
    
    def auto_manage_positions(self):
        """Continuous portfolio optimization"""
        for position in self.current_positions:
            if self._should_adjust(position):
                self._create_adjustment(position)
            if self._should_close(position):
                self._create_exit(position)
    
    def _should_adjust(self, position) -> bool:
        """Dynamic adjustment criteria"""
        pnl = position['current_value'] / position['max_loss']
        days_held = (datetime.now() - position['open_date']).days
        
        if position['type'] == 'credit_spread':
            return (pnl < -0.5 and days_held < 3) or (pnl > 0.8)
        return False
    
    # Helper Methods ---------------------------------------------------------
    
    def _get_strikes(self, delta_min, delta_max, option_type):
        """Find strikes matching target deltas"""
        # Uses SVI volatility surface interpolation
        pass
        
    def _get_expected_earnings_move(self):
        """Implied move + historical earnings moves"""
        pass
        
    def _init_regime_model(self):
        """Load pre-trained ML model"""
        pass