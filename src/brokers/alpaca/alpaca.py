import os
import time
from typing import Optional, Dict, Any, List
from src.monitoring.trading_metrics import TradeMetrics
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.trading.models import Order
from alpaca.data.models import Quote
from src.utils.logging import setup_logger
from prometheus_client import Counter, Gauge

logger = setup_logger(__name__)

class AlpacaTrader:
    """Live trading-only client with core functionality:
    - Connection management
    - Order execution
    - Market data streaming
    - Basic metrics
    """

    def __init__(self, metrics: TradeMetrics, config: Dict[str, Any]):
        self.metrics = metrics
        self.config = config
        self.primary_ticker = config['tickers']['primary'][0]
        self.max_order_size = config['brokers']['alpaca']['max_order_size']
        self.order_timeout = config['brokers']['alpaca']['order_timeout']
        self.paper = config['brokers']['alpaca']['paper']
        
        self._connect()
        
        # Initialize account metrics
        if self.metrics:
            self.metrics.record_portfolio_value(
                ticker=self.primary_ticker,
                value=self.get_account_balance(),
                currency='USD'
            )
            self.metrics.system_component_init.labels(
                component='broker',
                status='success'
            ).inc()

    def _connect(self) -> None:
        """Secure connection with retry logic"""
        for attempt in range(3):
            try:
                self.client = TradingClient(
                    os.getenv('ALPACA_KEY'),
                    os.getenv('ALPACA_SECRET'),
                    paper=self.paper
                )
                self.stream = StockDataStream(
                    os.getenv('ALPACA_KEY'),
                    os.getenv('ALPACA_SECRET')
                )
                logger.info(f"Connected to Alpaca {'PAPER' if self.paper else 'LIVE'} trading")
                return
            except Exception as e:
                if attempt == 2:
                    logger.critical("Alpaca connection failed after 3 attempts")
                    raise ConnectionError(f"Alpaca connection failed: {str(e)}")
                time.sleep(2 ** attempt)

    def _setup_metrics(self) -> None:
        """Initialize core metrics"""
        self.account_balance = Gauge('alpaca_account_balance', 'Current account balance')
        self.position_size = Gauge('alpaca_position_size', 'Current position size', ['symbol'])
        
        try:
            account = self.client.get_account()
            self.account_balance.set(float(account.cash))
            for position in self.client.get_all_positions():
                self.position_size.labels(symbol=position.symbol).set(float(position.qty))
        except Exception as e:
            logger.warning(f"Couldn't initialize metrics: {str(e)}")

    def submit_order(self, symbol: str, notional: float, side: str, **kwargs) -> Optional[Order]:
        """Submit order with full metrics instrumentation"""
        start_time = time.time()
        
        try:
            # Validate and submit
            if notional > self.max_order_size:
                raise ValueError(f"Order size {notional} exceeds limit {self.max_order_size}")
            
            order = self.client.submit_order(
                symbol=symbol,
                notional=notional,
                side=side,
                **kwargs
            )
            
            # Record metrics
            if self.metrics:
                self.metrics.record_trade(
                    ticker=symbol,
                    direction=side,
                    status='pending'
                )
                
                # Wait for execution
                filled_order = self._confirm_order(order.id)
                latency = time.time() - start_time
                
                self.metrics.record_trade(
                    ticker=symbol,
                    direction=side,
                    status='filled'
                )
                self.metrics.trading_rollback_latency_seconds.observe(latency)
                
                # Update portfolio metrics
                self.metrics.record_portfolio_value(
                    ticker=self.primary_ticker,
                    value=self.get_account_balance()
                )
                
            return filled_order
            
        except Exception as e:
            if self.metrics:
                self.metrics.record_trade(
                    ticker=symbol,
                    direction=side,
                    status='failed'
                )
                self.metrics.system_errors_total.labels(
                    error_type=type(e).__name__,
                    component='broker'
                ).inc()
            logger.error(f"Order failed: {str(e)}")
            raise
    
    def _confirm_order(self, order_id: str) -> Order:
        """Order confirmation with metrics"""
        start_time = time.time()
        try:
            order = self.client.get_order(order_id)
            if order.status == 'filled':
                if self.metrics:
                    self.metrics.trading_rollback_latency_seconds.observe(
                        time.time() - start_time
                    )
                return order
            raise TimeoutError("Order not filled")
        except Exception as e:
            if self.metrics:
                self.metrics.trading_rollback_failures_total.labels(
                    ticker=order.symbol if 'order' in locals() else 'unknown',
                    reason=type(e).__name__
                ).inc()
            raise
        
    def stream_quotes(self, symbols: List[str], handler: callable) -> None:
        """Real-time market data streaming"""
        self.stream.subscribe_quotes(handler, *symbols)
        self.stream.run()

    def get_account_balance(self) -> float:
        """Get current buying power"""
        return float(self.client.get_account().cash)