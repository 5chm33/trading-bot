import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Box
from utils.custom_logging import setup_logger
from typing import Dict, Any

logger = setup_logger(__name__)

class TradingEnv(Env):
    def __init__(self, data: pd.DataFrame, tickers: list, config: Dict[str, Any]):
        """Initialize with dynamic config."""
        super().__init__()
        self.data = data
        self.tickers = tickers
        self.config = config
        trading_config = config['trading']
        
        # Set trading parameters from config
        self.transaction_cost = trading_config['transaction_cost']
        self.slippage = trading_config['slippage'] 
        self.risk_free_rate = trading_config['risk_free_rate']
        self.epsilon = 1e-10
        
        # Initialize state tracking
        self.current_step = 0
        self.n_steps = len(data) - 1
        self.positions = np.zeros(len(tickers))
        self.portfolio_value = trading_config['initial_balance']
        self.returns = []
        
        # Dynamic action space from config
        self.action_space = Box(
            low=trading_config['position_limits']['min'],
            high=trading_config['position_limits']['max'],
            shape=(len(tickers),),
            dtype=np.float32
        )
        
        # Observation space (6 features per ticker)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(6 * len(tickers),),
            dtype=np.float32
        )
        
        logger.info(f"Initialized env for {len(tickers)} tickers with config: {trading_config}")

    def reset(self):
        """Reset environment with config-based initial balance."""
        self.current_step = 0
        self.positions = np.zeros(len(self.tickers))
        self.portfolio_value = self.config['trading']['initial_balance']
        self.returns = []
        
        if len(self.data) <= self.current_step:
            raise ValueError(f"Insufficient data (has {len(self.data)} rows)")
            
        logger.debug("Environment reset")
        return self._get_state()

    def _get_state(self):
        """Get state with config-based feature validation."""
        state = []
        for ticker in self.tickers:
            prefix = f"{ticker.lower()}_"
            required_cols = [
                f"{prefix}close", f"{prefix}volatility",
                f"{prefix}rsi", f"{prefix}macd",
                f"{prefix}atr", f"{prefix}obv"
            ]
            
            missing = [col for col in required_cols if col not in self.data.columns]
            if missing:
                raise ValueError(f"Missing columns for {ticker}: {missing}")
                
            ticker_data = self.data.iloc[self.current_step][required_cols].values
            state.extend(ticker_data)
            
        return np.array(state)

    def step(self, action):
        try:
            self.current_step += 1
            done = self.current_step >= self.n_steps
            
            # Clip actions to configured limits
            action = np.clip(
                np.nan_to_num(action, nan=0.0),
                self.config['trading']['position_limits']['min'],
                self.config['trading']['position_limits']['max']
            )
            
            # Calculate returns
            ticker_returns = self._calculate_ticker_returns()
            portfolio_return = self._calculate_portfolio_return(ticker_returns)
            self._update_portfolio(portfolio_return)
            
            # Get reward from config-based calculation
            reward = self._calculate_reward()
            
            return self._get_state(), reward, done, {}
            
        except Exception as e:
            logger.error(f"Step error: {str(e)}", exc_info=True)
            return self._get_state(), 0.0, True, {"error": str(e)}

    def _calculate_reward(self):
        """Calculate reward using config-based weights."""
        if len(self.returns) < 2:
            return 0.0
            
        reward_config = self.config['reward']
        
        # Sharpe component
        returns = np.clip(self.returns, -0.1, 0.1)
        excess_returns = returns - self.risk_free_rate
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10)
        sharpe = np.clip(sharpe, -10, 10)
        
        # Growth component
        growth = np.log(self.portfolio_value + 1e-10)
        growth = np.clip(growth, -10, 10)
        
        # Combined reward
        combined = (
            reward_config['components']['sharpe_weight'] * sharpe +
            reward_config['components']['growth_weight'] * growth
        )
        
        return np.clip(combined, reward_config['clipping']['min'], reward_config['clipping']['max'])