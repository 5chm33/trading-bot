# Quick connectivity test (add this temporarily to executor.py)
def test_connectivity(self):
    try:
        balance = self.broker.get_account_balance()
        prices = self.broker.get_current_prices()
        logger.info(f"✅ Connectivity OK | Balance: ${balance:,.2f} | Latest Prices: {prices}")
        return True
    except Exception as e:
        logger.error(f"❌ Connection failed: {str(e)}")
        return False

# Add this right after __init__ in PaperTradingExecutor
self.test_connectivity()