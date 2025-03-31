import os
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class AlpacaTrader:
    def __init__(self, paper=True):
        self.client = TradingClient(
            os.getenv('ALPACA_KEY'),
            os.getenv('ALPACA_SECRET'),
            paper=paper
        )
        self.stream = StockDataStream(
            os.getenv('ALPACA_KEY'),
            os.getenv('ALPACA_SECRET')
        )

    def submit_order(self, symbol, qty, side, order_type="market", time_in_force="day"):
        """Universal order handler"""
        try:
            order = self.client.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,  # 'buy'/'sell'
                type=order_type,
                time_in_force=time_in_force
            )
            logger.info(f"Order submitted: {order.id}")
            return order
        except Exception as e:
            logger.error(f"Order failed: {str(e)}")
            raise

    def stream_quotes(self, symbols, handler):
        """Real-time price streaming"""
        self.stream.subscribe_quotes(handler, *symbols)
        self.stream.run()

# Example Usage:
# trader = AlpacaTrader(paper=True)
# trader.submit_order("AAPL", 10, "buy")