# src/brokers/alpaca/paper.py
import os
import time
import re
from typing import Dict, Any, List, Optional, Callable, override
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading.models import Order
from alpaca.data.models import Quote
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.common.exceptions import APIError
from dataclasses import dataclass
import logging

from src.utils.logging import setup_logger
from src.monitoring.trading_metrics import TradeMetrics
from src.brokers.interfaces import BrokerInterface

logger = setup_logger(__name__)

@dataclass
class PositionInfo:
    symbol: str
    qty: float
    market_value: float
    current_price: float
    side: str

class AlpacaPaperBroker(BrokerInterface):
    """Paper trading broker with enhanced simulation features:
    - Realistic order fills with configurable latency
    - Price slippage simulation
    - Account balance tracking
    - Comprehensive metrics
    """
    
    def __init__(self, config: Dict[str, Any], metrics: Optional[TradeMetrics] = None):
        """Initialize with paper trading specific settings"""
        self._init_environment()
        self.config = self._validate_config(config)
        self.primary_ticker = config['tickers']['primary'][0]
        self.metrics = metrics if metrics else TradeMetrics()  # Initialize if not provided
        self._setup_state()
        self._initialize_clients()
        logger.info(f"Paper broker ready for {self.config['tickers']['primary'][0]}")

    def _init_environment(self):
        """Setup paper trading environment variables"""
        os.environ['APCA_API_SECRET_VERSION'] = '2'
        if not load_dotenv():
            logger.warning("Using system environment variables")

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set paper trading defaults"""
        required_sections = ['brokers', 'tickers', 'trading']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing config section: {section}")

        # Set paper trading defaults
        config['brokers']['alpaca'].setdefault('max_order_size', 10000)
        config['brokers']['alpaca'].setdefault('order_timeout', 30)
        config['brokers']['alpaca'].setdefault('simulated_fill_delay', 1.5)
        
        if not config['tickers'].get('primary'):
            raise ValueError("Must specify primary tickers")

        return config

    def _setup_state(self):
        """Initialize tracking state"""
        self._price_cache = {}
        self._price_cache_ttl = 5  # seconds
        self._last_price_update = 0
        self._active_orders = {}

    def _validate_credentials(self):
        """Validate paper trading credentials"""
        self.api_key = os.getenv('ALPACA_KEY', '').strip()
        self.api_secret = os.getenv('ALPACA_SECRET', '').strip()

        if not re.match(r'^PK[A-Z0-9]{18}$', self.api_key):
            raise ValueError("Invalid paper trading key format")
        if len(self.api_secret) not in {32, 40}:
            raise ValueError("Invalid secret key length")

    def _initialize_clients(self):
        """Initialize Alpaca clients with retry logic"""
        self._validate_credentials()
        
        for attempt in range(3):
            try:
                self.client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                    paper=True,
                    url_override='https://paper-api.alpaca.markets'
                )
                
                self.data_client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.api_secret
                )
                
                self.stream = StockDataStream(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                    url_override='wss://paper-api.alpaca.markets/stream'
                )
                
                self._verify_connection()
                self._initialize_metrics()
                return
                
            except APIError as e:
                if attempt == 2:
                    raise PermissionError(f"API rejected credentials: {e.message}") from e
                time.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 2:
                    raise ConnectionError("Failed to connect after 3 attempts") from e
                time.sleep(2 ** attempt)

    def _verify_connection(self):
        """Verify initial connection and log account info"""
        account = self.client.get_account()
        logger.info(
            f"Paper Trading Account:\n"
            f"ID: {account.account_number}\n"
            f"Buying Power: ${float(account.cash):,.2f}\n"
            f"Equity: ${float(account.equity):,.2f}"
        )

    def _initialize_metrics(self) -> None:
        """Initialize all paper trading metrics"""
        if self.metrics:  # Only if metrics is available
            self.metrics.record_broker_init(success=True)
            for ticker in self.config['tickers']['primary']:
                self.metrics.record_trade(
                    ticker=ticker,
                    direction='buy',
                    status='initialized'
                )
                self.metrics.record_trade(
                    ticker=ticker,
                    direction='sell',
                    status='initialized'
                )
            
    @override        
    def cancel_all_orders(self) -> None:
        """Cancel all open orders in the paper trading account"""
        try:
            self.client.cancel_orders()
            logger.info("Successfully cancelled all open orders")
            self.metrics.record_order_cancellation()
        except Exception as e:
            logger.error(f"Failed to cancel orders: {str(e)}")
            self.metrics.record_broker_error("cancel_orders")
            raise RuntimeError(f"Failed to cancel orders: {str(e)}") from e
    @override
    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for executing trades (converts notional to qty if needed)"""
        if 'notional' in action:
            logger.warning("'notional' parameter is deprecated, use 'qty' instead")
            action['qty'] = action.pop('notional')
        return self.execute_order(action)
    
    # Core Trading Methods
    def execute_order(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order with paper trading simulations"""
        try:
            self._validate_order_action(action)
            
            # Apply paper trading adjustments
            action = self._apply_paper_adjustments(action)
            
            # Submit and confirm order
            order = self.client.submit_order(**action)
            filled_order = self._confirm_order(order.id)
            
            # Record metrics and return results
            return self._process_order_result(filled_order)
            
        except APIError as e:
            self._handle_api_error('execute_order', e)
            raise
        except Exception as e:
            logger.error(f"Order failed: {str(e)}", exc_info=True)
            self.metrics.record_broker_error("execute_order", type(e).__name__)
            raise

    def _validate_order_action(self, action: Dict[str, Any]):
        """Validate order parameters"""
        required = ['symbol', 'qty', 'side']
        if not all(k in action for k in required):
            raise ValueError(f"Missing required fields: {required}")
            
        if float(action['qty']) > self.config['brokers']['alpaca']['max_order_size']:
            raise ValueError("Order size exceeds limit")

    def _apply_paper_adjustments(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Add paper trading specific parameters"""
        return {
            **action,
            'type': action.get('type', 'limit'),
            'time_in_force': 'day',
            'limit_price': self._get_safe_limit_price(
                action['symbol'],
                action['side']
            ),
            'client_order_id': f"paper_{int(time.time())}"
        }

    def _process_order_result(self, order: Order) -> Dict[str, Any]:
        """Convert order to result dict and record metrics"""
        result = {
            'status': order.status,
            'order_id': order.id,
            'symbol': order.symbol,
            'filled_qty': float(order.filled_qty),
            'filled_price': float(order.filled_avg_price),
            'timestamp': order.filled_at.isoformat()
        }
        
        self.metrics.TRADE_SUCCESS.labels(
            ticker=order.symbol,
            side=order.side
        ).inc()
        
        return result

    # Price and Market Data
    def get_current_prices(self) -> Dict[str, Dict[str, float]]:
        """Get cached prices with automatic refresh"""
        try:
            if time.time() - self._last_price_update < self._price_cache_ttl:
                return self._price_cache.copy()
                
            request = StockLatestQuoteRequest(
                symbol_or_symbols=self.config['tickers']['primary']
            )
            quotes = self.data_client.get_stock_latest_quote(request)
            
            self._price_cache = {
                symbol: {
                    'close': float(quote.bid_price),
                    'bid': float(quote.bid_price),
                    'ask': float(quote.ask_price),
                    'spread': float(quote.ask_price - quote.bid_price),
                    'timestamp': quote.timestamp.isoformat()
                }
                for symbol, quote in quotes.items()
            }
            self._last_price_update = time.time()
            
            if self.metrics:
                self.metrics.record_price_update()
                
            return self._price_cache.copy()
            
        except Exception as e:
            if self.metrics:
                self.metrics.record_broker_error(
                    error_type=type(e).__name__,
                    method='get_current_prices'
                )
            raise

    # Account Management
    def get_account_balance(self) -> float:
        """Get current buying power with metrics"""
        try:
            account = self.client.get_account()
            balance = float(account.cash)
            if self.metrics:
                self.metrics.record_portfolio_value(
                    ticker=self.primary_ticker,
                    value=balance
                )
            return balance
        except Exception as e:
            if self.metrics:
                self.metrics.record_broker_error(
                    error_type=type(e).__name__,
                    method='get_account_balance'
                )
            raise

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions with enhanced formatting"""
        try:
            positions = self.client.get_all_positions()
            return {
                p.symbol: PositionInfo(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    market_value=float(p.market_value),
                    current_price=float(p.current_price),
                    side='long' if float(p.qty) > 0 else 'short'
                )
                for p in positions
            }
        except Exception as e:
            self._handle_api_error('get_positions', e)
            raise

    # Enhanced Paper Trading Features
    def _get_safe_limit_price(self, symbol: str, side: str) -> float:
        """Calculate realistic limit price with simulated spread"""
        prices = self.get_current_prices()
        if symbol not in prices:
            raise ValueError(f"No price data for {symbol}")
            
        quote = prices[symbol]
        spread = quote['ask'] - quote['bid']
        
        # Simulate realistic fills
        if side == 'buy':
            return round(quote['ask'] * 1.005, 2)
        return round(quote['bid'] * 0.995, 2)

    def _confirm_order(self, order_id: str, timeout: int = None) -> Order:
        """Enhanced order confirmation with paper trading simulations"""
        timeout = timeout or self.config['brokers']['alpaca']['order_timeout']
        start = time.time()
        last_status = None
        
        while time.time() - start < timeout:
            try:
                order = self.client.get_order(order_id)
                
                if order.status != last_status:
                    logger.info(f"Order {order_id} status: {order.status}")
                    last_status = order.status
                
                if order.status == 'filled':
                    logger.info(f"Filled {order.filled_qty} shares @ {order.filled_avg_price}")
                    return order
                elif order.status == 'rejected':
                    raise ValueError(f"Order rejected: {getattr(order, 'reason', 'No reason given')}")
                    
                time.sleep(self.config['brokers']['alpaca']['simulated_fill_delay'])
                
            except APIError as e:
                logger.warning(f"Order check error: {str(e)}")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Order check failed: {str(e)}")
                time.sleep(2)
        
        raise TimeoutError(f"Order {order_id} not filled in {timeout}s")

    # Error Handling
    def _handle_api_error(self, method: str, error: Exception):
        """Standard API error handling"""
        error_type = type(error).__name__
        logger.error(f"API Error in {method}: {str(error)}")
        self.metrics.record_broker_error(error_type, method)
        
        if isinstance(error, APIError) and error.status_code == 401:
            self.reconnect()

    def reconnect(self):
        """Reconnect with full initialization"""
        try:
            logger.info("Reconnecting to paper trading API...")
            self._initialize_clients()
            self.metrics.record_reconnection(success=True)
        except Exception as e:
            logger.critical(f"Reconnection failed: {str(e)}")
            self.metrics.record_reconnection(success=False)
            raise

    # Stream Methods
    def stream_market_data(self, handler: Callable):
        """Start real-time data streaming"""
        symbols = self.config['tickers']['primary']
        if not symbols:
            raise ValueError("No symbols configured")
            
        try:
            self.stream.subscribe_quotes(handler, *symbols)
            self.stream.run()
        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            raise