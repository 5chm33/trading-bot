import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Adjust based on your test location
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.error_handling.exceptions import PartialFillError
    print("✅ Import successful!")
    print("Version:", PartialFillError.__module__)
except ImportError as e:
    print("❌ Import failed:", e)
    print("Python path:")
    for p in sys.path:
        print(" -", p)
input("Press Enter to exit...")