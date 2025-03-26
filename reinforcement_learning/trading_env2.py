import numpy as np
from gym import Env
from gym.spaces import Box
import logging
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

class TradingEnv(Env):
    def __init__(self, data, close_column, volatility_column, transaction_cost=0.001, slippage=0.0005, risk_free_rate=0.02):
        super(TradingEnv, self).__init__()
        self.data = data
        self.close_column = close_column
        self.volatility_column = volatility_column
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        self.current_step = 0
        self.n_steps = len(data) - 1
        self.position = 0
        self.portfolio_value = 1.0
        self.returns = []

        # Define action and observation space
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(6,), dtype=np.float32)  # Updated to (6,)

    def reset(self):
        """Reset the environment to the initial state."""
        self.current_step = 0
        self.position = 0
        self.portfolio_value = 1.0
        self.returns = []
        return self._get_state()

    def _get_state(self):
        """Get the current state of the environment."""
        state = self.data.iloc[self.current_step].values
        return state

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.n_steps

        # Clip action to [-1, 1]
        action = np.clip(action, -1, 1)

        # Update position
        self.position = action

        # Get the current and previous close prices
        current_price = self.data.iloc[self.current_step][self.close_column]
        previous_price = self.data.iloc[self.current_step - 1][self.close_column]

        # Calculate price change with slippage
        price_change = (current_price - previous_price)
        price_change *= (1 + np.random.normal(0, self.slippage))

        # Calculate raw profit (scaled by position size)
        raw_profit = price_change * self.position

        # Subtract transaction cost
        transaction_cost = self.transaction_cost * abs(self.position)
        net_profit = raw_profit - transaction_cost

        # Update portfolio value (with clipping to avoid overflow)
        self.portfolio_value = np.clip(self.portfolio_value * (1 + net_profit), 1e-6, 1e6)

        # Track returns for Sharpe Ratio calculation
        self.returns.append(net_profit)

        # Calculate reward (Sharpe Ratio)
        reward = self._calculate_sharpe_ratio()

        # Get the next state
        observation = self._get_state()

        # Log key variables for debugging
        logger.debug(f"Step: {self.current_step}, Action: {action}, Portfolio Value: {self.portfolio_value}, Reward: {reward}, Done: {done}")

        return observation, reward, done, {}

    def _calculate_sharpe_ratio(self):
        """Calculate the Sharpe Ratio based on historical returns."""
        if len(self.returns) < 2:
            return 0.0

        returns = np.array(self.returns)
        excess_returns = returns - self.risk_free_rate
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)

        if std_excess_return < 1e-10:
            return 0.0

        return mean_excess_return / std_excess_return