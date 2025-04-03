try:
    from src.brokers.alpaca.paper import AlpacaPaperBroker
    print("✅ All imports work!")
    print("Broker class:", AlpacaPaperBroker)
except ImportError as e:
    print("❌ Import failed:", e)
input("Press Enter to exit...")