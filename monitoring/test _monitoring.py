import time
from trading_env import TradingEnv
import pandas as pd
import numpy as np

# Mock data
data = pd.DataFrame({
    'AAPL_close': np.random.uniform(150, 200, 1000),
    'AAPL_volume': np.random.randint(1e6, 1e7, 1000),
    'AAPL_rsi': np.random.uniform(30, 70, 1000)
})

env = TradingEnv(data, ['AAPL'], config={
    'monitoring': {'enabled': True},
    'trading': {'initial_balance': 10000}
})

for _ in range(100):
    action = np.random.uniform(-0.5, 0.5, 1)
    env.step(action)
    time.sleep(0.1)  # Simulate real-time trading
