import os
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream, OptionDataStream
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class AlpacaOptionsTrader:
    def __init__(self, paper=True):
        self.trading_client = TradingClient(
            os.getenv('ALPACA_KEY'),
            os.getenv('ALPACA_SECRET'),
            paper=paper
        )
        self.option_stream = OptionDataStream(os.getenv('ALPACA_KEY'))
        
    def stream_options_data(self, symbols):
        """Subscribe to real-time options data"""
        async def handle_data(data):
            print(f"Options data received: {data}")
            
        self.option_stream.subscribe_options(handle_data, *symbols)
        self.option_stream.run()