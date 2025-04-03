import sys
from pathlib import Path

# Add this at the VERY TOP (before any other imports)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now your regular imports
from src.pipeline.paper_trading.executor import main

if __name__ == "__main__":
    main()