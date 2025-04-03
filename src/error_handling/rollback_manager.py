import sqlite3
from datetime import datetime
from typing import Dict, Optional
from src.error_handling.exceptions import RollbackFailedError

class RollbackManager:
    """Atomic trade reversal system with journaling"""
    
    def __init__(self, db_path: str = "order_history.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        
    def _init_db(self):
        """Creates audit trail database"""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            filled_price REAL,
            original_order TEXT,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
    
    def log_order(self, order: Dict):
        """Records order attempts"""
        self.conn.execute("""
        INSERT INTO orders (id, symbol, side, qty, original_order, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """, [
            order['id'],
            order['symbol'],
            order['side'],
            order['qty'],
            str(order),
            'ATTEMPTED'
        ])
        self.conn.commit()
    
    def rollback_order(self, order_id: str) -> bool:
        """Returns True if rollback succeeds, False if not needed"""
        order = self.conn.execute(
            "SELECT * FROM orders WHERE id = ?", 
            (order_id,)
        ).fetchone()
        
        if not order or order['status'] != 'ATTEMPTED':
            return False
            
        unfilled_qty = float(order['qty']) - float(order.get('filled_qty', 0))
        
        # Skip if already fully filled
        if unfilled_qty <= 0:
            return True
            
        try:
            inverse = {
                'symbol': order['symbol'],
                'side': 'sell' if order['side'] == 'buy' else 'buy',
                'qty': unfilled_qty,
                'type': 'market'
            }
            self._execute_via_broker(inverse)
            self._update_order_status(order_id, 'ROLLED_BACK')
            return True
        except Exception as e:
            self._update_order_status(order_id, 'ROLLBACK_FAILED')
            raise RollbackFailedError(f"Failed to rollback {order_id}: {str(e)}")