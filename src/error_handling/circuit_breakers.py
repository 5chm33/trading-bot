from src.models.rl.safety_monitor import SafetyMonitor  # Existing import
from src.error_handling.rollback_manager import RollbackManager

class TradingCircuitBreaker(SafetyMonitor):
    """Extends your existing safety system with rollback awareness"""
    
    def __init__(self, config):
        super().__init__(config)
        self.rollback = RollbackManager()
        
    def check_rollback_conditions(self):
        """Auto-triggers rollbacks for dangerous states"""
        positions = self.get_positions()
        for symbol, pos in positions.items():
            if pos['unrealized_pnl'] < -self.max_loss_per_trade:
                self.rollback.rollback_position(symbol)
                
    def rollback_position(self, symbol):
        """Full position reversal"""
        pos = self.get_position(symbol)
        inverse_side = 'sell' if pos['side'] == 'long' else 'buy'
        self.broker.execute({
            'symbol': symbol,
            'qty': abs(pos['qty']),
            'side': inverse_side,
            'type': 'market'
        })