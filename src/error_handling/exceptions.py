class PartialFillError(Exception):
    """Raised when order partially fills beyond tolerance"""
    def __init__(self, symbol: str, filled_qty: float, total_qty: float):
        self.symbol = symbol
        self.filled_qty = filled_qty
        self.total_qty = total_qty
        message = f"Partial fill on {symbol}: {filled_qty}/{total_qty}"
        super().__init__(message)

class RollbackFailedError(Exception):
    """Raised when rollback attempt fails"""
    pass