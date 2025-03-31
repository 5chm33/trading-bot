# utils/portfolio.py
class PortfolioTracker:
    def __init__(self, initial_balance: float):
        self.assets = {}
        self.balance = initial_balance
        self._history = []

    @log_execution_time(logger)
    def update(self, positions: Dict[str, float], prices: Dict[str, float]):
        """Log portfolio snapshot"""
        self.assets = {
            symbol: {
                'position': pos,
                'value': pos * prices[symbol],
                'price': prices[symbol]
            }
            for symbol, pos in positions.items()
        }

        log_with_context(
            logger=logger,
            level=logging.INFO,
            message="Portfolio Snapshot",
            context={
                "total_value": self.total_value,
                "leverage": self.calculate_leverage(),
                "diversification": self.calculate_diversification()
            }
        )
