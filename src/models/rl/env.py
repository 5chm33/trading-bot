# src/models/rl/env.py
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace
from typing import Dict, Any, Tuple
from collections import deque
import logging
from prometheus_client import Gauge, Counter
from src.utils.data_schema import ColumnSchema
from src.utils.monitoring import TradingMonitor

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """Production-ready Trading Environment for RL"""
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        self.config = config
        self.tickers = config['tickers']['primary'] + config['tickers'].get('secondary', [])
        
        # Validate and preprocess data
        ColumnSchema.validate(data, self.tickers)
        self.data = self._preprocess_data(data.copy())
        
        # Initialize state
        self.portfolio_value = float(config['trading']['initial_balance'])
        self._peak_portfolio = self.portfolio_value
        self.positions = np.zeros(len(self.tickers))
        self.returns = deque(maxlen=21)
        self.current_step = 0
        
        # Initialize components
        self._init_spaces()
        self._init_metrics()
        self.monitor = TradingMonitor(config)
        
        logger.info(f"TradingEnv initialized for {len(self.tickers)} assets")

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for trading"""
        # Ensure all required columns exist
        for ticker in self.tickers:
            prefix = f"{ticker.lower()}_"
            if f"{prefix}close" not in data.columns:
                raise ValueError(f"Missing close price for {ticker}")
        
        return data.dropna()

    def _init_spaces(self):
        """Initialize observation and action spaces"""
        self.observation_space = DictSpace({
            'market_data': Box(low=-np.inf, high=np.inf, 
                             shape=(len(self.data.columns),), dtype=np.float32),
            'portfolio': Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        })
        
        self.action_space = Box(
            low=-1.0, high=1.0, 
            shape=(len(self.tickers),), 
            dtype=np.float32
        )

    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        self.metrics = {
            'portfolio': Gauge('portfolio_value', 'Current value', ['ticker']),
            'positions': Gauge('position_size', 'Current position', ['ticker']),
            'trades': Counter('trades_executed', 'Trade count', ['ticker', 'type']),
            'sharpe': Gauge('sharpe_ratio', 'Rolling Sharpe ratio'),
            'drawdown': Gauge('max_drawdown', 'Current drawdown')
        }

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        """Execute one trading step"""
        prev_value = self.portfolio_value
        
        # Execute trades
        self.positions = np.clip(action, -1, 1)
        price_changes = self._get_price_changes()
        self.portfolio_value *= (1 + np.sum(price_changes * self.positions))
        self._peak_portfolio = max(self._peak_portfolio, self.portfolio_value)
        
        # Calculate metrics
        self.returns.append((self.portfolio_value - prev_value) / prev_value)
        reward = self._calculate_reward()
        
        # Prepare outputs
        observation = self._get_observation()
        terminated = self.current_step >= len(self.data) - 1
        info = {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'positions': dict(zip(self.tickers, self.positions))
        }
        
        # Update state
        self._update_metrics(action)
        self.current_step += 1
        
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        """Reset environment state"""
        self.current_step = 0
        self.portfolio_value = float(self.config['trading']['initial_balance'])
        self._peak_portfolio = self.portfolio_value
        self.positions = np.zeros(len(self.tickers))
        self.returns = deque(maxlen=21)
        
        obs = self._get_observation()
        return obs, {}

    # Helper methods
    def _get_price_changes(self) -> np.ndarray:
        """Calculate normalized price changes"""
        changes = np.zeros(len(self.tickers))
        for i, ticker in enumerate(self.tickers):
            close_col = f"{ticker.lower()}_close"
            current = self.data.iloc[self.current_step][close_col]
            prev = self.data.iloc[self.current_step-1][close_col] if self.current_step > 0 else current
            changes[i] = (current - prev) / (prev + 1e-9)
        return changes

    def _calculate_reward(self) -> float:
        """Calculate regime-aware trading reward"""
        ret = (self.portfolio_value - self._last_portfolio_value) / self._last_portfolio_value
        sharpe = self._calculate_sharpe()
        drawdown = self._current_drawdown()
        
        reward = (
            self.config['reward']['components']['returns'] * ret +
            self.config['reward']['components']['sharpe'] * sharpe +
            self.config['reward']['components']['drawdown'] * drawdown
        )
        return float(np.clip(reward, *self.config['reward']['scaling']['clip']))

    def _calculate_sharpe(self, window: int = 21) -> float:
        """Calculate rolling Sharpe ratio"""
        if len(self.returns) < 2:
            return 0.0
        returns = np.array(list(self.returns)[-window:])
        return float(np.mean(returns) / (np.std(returns) + 1e-9))

    def _current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        return float((self._peak_portfolio - self.portfolio_value) / (self._peak_portfolio + 1e-9))

    def _update_metrics(self, action: np.ndarray):
        """Update monitoring metrics"""
        for i, ticker in enumerate(self.tickers):
            self.metrics['positions'].labels(ticker=ticker).set(float(action[i]))
            if abs(action[i] - self.positions[i]) > 0.05:
                trade_type = 'long' if action[i] > self.positions[i] else 'short'
                self.metrics['trades'].labels(ticker=ticker, type=trade_type).inc()
        
        self.metrics['sharpe'].set(self._calculate_sharpe())
        self.metrics['drawdown'].set(self._current_drawdown())
        self.positions = action.copy()

    def _get_observation(self) -> dict:
        """Get current environment observation"""
        return {
            'market_data': self.data.iloc[self.current_step].values.astype(np.float32),
            'portfolio': np.array([
                self.portfolio_value,
                np.sum(np.abs(self.positions)),
                self.current_step / len(self.data)
            ], dtype=np.float32)
        }