import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace
from typing import Dict, Any, Tuple, List, Optional
from collections import deque
from utils.custom_logging import setup_logger, log_with_context, log_execution_time
from utils.normalization import FeatureScaler, RewardNormalizer
from utils.monitoring import TradingMonitor
import logging
from collections import deque

logger = setup_logger(__name__)

class TradingEnv(gym.Env):
    """Enhanced trading environment with production-grade logging"""

    @log_execution_time(logger)
    def __init__(self, data: pd.DataFrame, tickers: List[str], config: Dict[str, Any]):
        super().__init__()
        self.data = data.astype({c: np.float32 for c in data.select_dtypes(np.number).columns})
        self.tickers = tickers
        self.config = config
        self.features_per_ticker = len([c for c in data.columns if c.startswith(tickers[0].lower() + '_')])
        self._initialize_components()
        log_with_context(logger, logging.INFO, "TradingEnv initialized",
                       num_tickers=len(tickers),
                       features_per_ticker=self.features_per_ticker,
                       initial_balance=self.initial_balance)
        self.reward_history = deque(maxlen=1000)
        self.monitor = TradingMonitor(port=8000) if config.get('monitoring', {}).get('enabled', False) else None

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state - Gymnasium compatible"""
        super().reset(seed=seed)

        # Reset all tracking variables
        self.current_step = 0
        self.positions = np.zeros(len(self.tickers), dtype=np.float32)
        self.portfolio_value = self.initial_balance
        self._peak_portfolio = self.initial_balance
        self.returns = deque(maxlen=252)
        self.prev_prices = None

        # Get initial state
        observation = self._get_state()
        info = {
            'step': self.current_step,
            'portfolio_value': float(self.portfolio_value),
            'positions': {t: float(p) for t, p in zip(self.tickers, self.positions)}}

        log_with_context(
            logger=logger,
            level=logging.DEBUG,
            msg="Environment reset complete",  # Added required msg parameter
            context={
                'portfolio_value': float(self.portfolio_value),
                'positions': info['positions']
            }
        )
        return observation, info

    def _initialize_components(self):
        """Consolidated initialization"""
        self._configure_spaces()
        self._setup_normalizers()
        self._setup_trading_params()
        self._init_state_vars()

    def _configure_spaces(self):
        """Configure observation and action spaces"""
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.tickers),),
            dtype=np.float32
        )

        self.observation_space = DictSpace({
            'market_data': Box(
                low=-np.inf,
                high=np.inf,
                shape=(len(self.data.columns),),
                dtype=np.float32
            ),
            'portfolio': Box(
                low=0,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            )
        })

    @log_execution_time(logger=logger, level=logging.DEBUG)
    def _setup_normalizers(self):
        """Initialize all normalization components"""
        norm = self.config['normalization']
        self.feature_scaler = FeatureScaler(norm['features'])
        self.reward_normalizer = RewardNormalizer(norm['rewards']['window'])

    def _setup_trading_params(self):
        """Configure trading parameters with validation"""
        trading = self.config['trading']
        self.transaction_cost = float(trading['transaction_cost'])
        self.slippage = float(trading['slippage'])
        self.initial_balance = float(trading['initial_balance'])
        self.leverage = float(trading['position_limits']['leverage'])

    def _init_state_vars(self):
        """Initialize tracking variables"""
        self.current_step = 0
        self.n_steps = len(self.data) - 1
        self.positions = np.zeros(len(self.tickers), dtype=np.float32)
        self.portfolio_value = self.initial_balance
        self._peak_portfolio = self.initial_balance
        self.returns = deque(maxlen=252)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute trading step with enhanced logging"""
        current_prices, tech_features = self._get_current_market_data()

        reward, info = self._process_trade(action, current_prices, tech_features)
        next_state = self._get_state()

        self._log_step_metrics(info)
        self.reward_history.append(reward)
        if self.monitor:
            self.monitor.update(self)

        terminated = self.current_step >= self.n_steps
        truncated = False

        return next_state, reward, terminated, truncated, info

    def _get_current_market_data(self) -> Tuple[np.ndarray, list]:
        """Get current market prices and technical features"""
        current_prices = []
        tech_features = []

        for t in self.tickers:
            prefix = f"{t.lower()}_"
            row = self.data.iloc[self.current_step]

            price = float(row[f"{prefix}close"])
            if price <= 0:
                logger.warning(f"Invalid price for {t} at step {self.current_step}")
                price = self.prev_prices[self.tickers.index(t)] if self.prev_prices else 0.01

            current_prices.append(price)

            tech_features.extend([
                float(row[f"{prefix}rsi"]),
                float(row[f"{prefix}macd"]),
                float(row[f"{prefix}atr"]),
                float(row[f"{prefix}obv"]),
                float(row[f"{prefix}close_20ma"]),
                float(row[f"{prefix}volume_20ma"])
            ])

        self.prev_prices = np.array(current_prices, dtype=np.float32)
        return self.prev_prices, tech_features

    def _process_trade(self, action, current_prices, tech_features) -> Tuple[float, dict]:
        """Core trade execution logic"""
        self.positions = np.clip(action, -0.5, 0.5)

        if self.prev_prices is None:
            return 0.0, self._init_info_dict(current_prices, tech_features)

        returns = self._calculate_returns(current_prices)
        reward = self._calculate_reward(tech_features, returns)

        # Update portfolio value
        self._last_portfolio_value = self.portfolio_value
        self.portfolio_value *= (1 + np.sum(returns * self.positions))
        self._peak_portfolio = max(self._peak_portfolio, self.portfolio_value)
        self.returns.append(np.sum(returns * self.positions))

        return reward, self._create_trade_info(current_prices, tech_features, returns)

    def _calculate_returns(self, current_prices: np.ndarray) -> np.ndarray:
        """Calculate returns based on price changes"""
        if self.prev_prices is None:
            return np.zeros(len(self.tickers))
        return (current_prices - self.prev_prices) / self.prev_prices

    @property
    def _last_portfolio_value(self) -> float:
        """Get last portfolio value with default"""
        return getattr(self, '_last_pv', self.initial_balance)

    @_last_portfolio_value.setter
    def _last_portfolio_value(self, value: float):
        """Set last portfolio value"""
        self._last_pv = value

    def _calculate_reward(self, tech_features: list, returns: np.ndarray) -> float:
        """Calculate trading reward with technical adjustments
        Args:
            tech_features: List of technical indicators
            returns: Array of returns for each ticker
        Returns:
            Calculated reward value
        """
        try:
            tech_matrix = np.array(tech_features).reshape(len(self.tickers), 6)
            rsi_values = tech_matrix[:, 0]
            macd_values = tech_matrix[:, 1]

            # Calculate portfolio return
            portfolio_return = np.sum(returns * self.positions)

            # Technical adjustments with clamping
            rsi_adjustment = np.clip(np.mean([max(0, rsi - 65)/30 for rsi in rsi_values]), -1, 1)
            macd_signal = np.clip(np.mean([1 if macd > 0 else -1 for macd in macd_values]), -1, 1)

            reward = (
                0.7 * portfolio_return * (1 + 0.3 * macd_signal) * (1 - 0.2 * rsi_adjustment) +
                0.2 * self._calculate_sharpe() +
                0.1 * (-0.1 * np.max(np.abs(self.positions)) -
                0.1 * self._current_drawdown()
            ))

            # Clip reward to configured range
            reward_clip = self.config.get('reward_clip', [-1, 1])
            return float(np.clip(reward, reward_clip[0], reward_clip[1]))

        except Exception as e:
            logger.error(f"Reward calculation failed: {str(e)}")
            return 0.0

    def _init_info_dict(self, current_prices, tech_features) -> dict:
        """Initialize info dictionary for first step"""
        return {
            'current_prices': current_prices,
            'current_tech': tech_features,
            'portfolio_value': float(self.portfolio_value),
            'positions': {t: float(p) for t, p in zip(self.tickers, self.positions)}
        }

    def _create_trade_info(self, current_prices, tech_features, returns) -> dict:
        """Create trade information dictionary"""
        info = self._init_info_dict(current_prices, tech_features)
        info.update({
            'returns': returns,
            'sharpe': self._calculate_sharpe(),
            'sortino': self._calculate_sortino(),
            'drawdown': self._current_drawdown()
        })
        return info

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio of recent returns"""
        if len(self.returns) < 2:
            return 0.0
        return float(np.mean(self.returns) / (np.std(self.returns) + 1e-8))

    def _calculate_sortino(self) -> float:
        """Calculate Sortino ratio of recent returns"""
        if len(self.returns) < 2:
            return 0.0
        downside_returns = [r for r in self.returns if r < 0]
        if not downside_returns:
            return 0.0
        return float(np.mean(self.returns) / (np.std(downside_returns) + 1e-8))

    def _current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        return float((self._peak_portfolio - self.portfolio_value) / self._peak_portfolio)

    @log_execution_time(logger=logger, level=logging.DEBUG)
    def log_portfolio_update(self):
        """Structured portfolio logging for dashboards"""
        log_with_context(
            logger=logger,
            level=logging.INFO,
            msg="PortfolioState",  # Added required msg parameter
            context={
                "step": self.current_step,
                "value": float(self.portfolio_value),
                "positions": {t: float(p) for t, p in zip(self.tickers, self.positions)},
                "metrics": {
                    "sharpe": self._calculate_sharpe(),
                    "sortino": self._calculate_sortino(),
                    "drawdown": self._current_drawdown()
                }
            }
        )

    def _log_step_metrics(self, info: dict):
        """Log key metrics at configured intervals"""
        if self.current_step % self.config['logging'].get('portfolio_log_freq', 100) == 0:
            self.log_portfolio_update()
            log_with_context(
                logger=logger,
                level=logging.DEBUG,
                msg="Market Observation",  # Added required msg parameter
                current_tech=info['current_tech']
            )

    def _get_state(self) -> dict:
        """Get current environment state"""
        market_data = self.data.iloc[self.current_step].values.astype(np.float32)
        portfolio_state = np.array([
            self.portfolio_value,
            np.sum(self.positions),
            self.current_step / self.n_steps
        ], dtype=np.float32)

        return {
            'market_data': market_data,
            'portfolio': portfolio_state
        }

    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate"""
        terminated = self.current_step >= self.n_steps
        truncated = False
        return terminated, truncated

# ===== Dashboard Integration =====
"""
Grafana Dashboard Setup:
1. Install Loki and Promtail for log collection
2. Add this to your docker-compose.yml:

  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"

  promtail:
    image: grafana/promtail:latest
    volumes:
      - ./logs:/var/log/trading
      - ./promtail-config.yml:/etc/promtail/config.yml

3. Create promtail-config.yml:
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
- job_name: trading
  static_configs:
  - targets:
      - localhost
    labels:
      job: trading_bot
      __path__: /var/log/trading/*.log

4. Grafana Dashboard JSON available here:
   https://gist.github.com/ai-trading/... (see below)
"""
