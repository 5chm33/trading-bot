from prometheus_client import Gauge, Counter, REGISTRY, start_http_server
import logging
from typing import Optional
import time
import os

logger = logging.getLogger(__name__)

class TradingMonitor:
    _instance = None

    def __new__(cls, port: int = 8000, log_file: str = "metrics.log"):
        if cls._instance is None:
            cls._instance = super(TradingMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, port: int = 8000, log_file: str = "metrics.log"):
        if not self._initialized:
            self.port = port
            self.log_file = log_file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self._init_metrics()
            self._initialized = True

    def _init_metrics(self):
        """Initialize metrics with dual logging"""
        # Clear existing Prometheus metrics
        for metric in list(REGISTRY._collector_to_names.keys()):
            REGISTRY.unregister(metric)

        # Prometheus metrics
        self.portfolio_gauge = Gauge('portfolio_value', 'Current portfolio value', ['ticker'])
        self.sharpe_gauge = Gauge('sharpe_ratio', 'Current Sharpe ratio')
        self.trade_counter = Counter('trades_executed', 'Total trades executed', ['action'])
        self.position_gauge = Gauge('position_size', 'Current position size', ['ticker'])

        # File logging header
        with open(self.log_file, 'w') as f:
            f.write("timestamp,metric,value,ticker\n")

        logger.info(f"Metrics initialized (HTTP: {self.port}, File: {self.log_file})")
        start_http_server(self.port)

    def update(self, env) -> None:
        """Dual update to Prometheus and log file"""
        try:
            timestamp = int(time.time())

            # Update Prometheus metrics
            for i, ticker in enumerate(env.tickers):
                self.portfolio_gauge.labels(ticker=ticker).set(env.portfolio_value)
                self.position_gauge.labels(ticker=ticker).set(env.positions[i])

            self.sharpe_gauge.set(env._calculate_sharpe())
            self.trade_counter.labels(action='execute').inc()

            # Append to log file
            with open(self.log_file, 'a') as f:
                for i, ticker in enumerate(env.tickers):
                    f.write(f"{timestamp},portfolio_value,{env.portfolio_value},{ticker}\n")
                    f.write(f"{timestamp},position_size,{env.positions[i]},{ticker}\n")
                f.write(f"{timestamp},sharpe_ratio,{env._calculate_sharpe()},system\n")
                f.write(f"{timestamp},trades_executed,{env.current_step},system\n")

        except Exception as e:
            logger.error(f"Metrics update failed: {str(e)}", exc_info=True)
