<<<<<<< HEAD
# tests/unit/test_regime_detection.py
import pytest
import pandas as pd
import numpy as np
from src.data.processors.data_utils import _detect_market_regimes

def test_regime_detection():
    # Create test data with clear bull/bear periods
    prices = pd.Series(np.concatenate([
        np.linspace(100, 200, 50),  # Bull
        np.linspace(200, 150, 50),   # Bear
        np.linspace(150, 160, 50)    # Neutral
    ]))

    regimes = _detect_market_regimes(pd.DataFrame({'close': prices}))

    # Verify we get expected regime patterns
    assert np.all(regimes[:50] == 1)  # Bull
    assert np.all(regimes[50:100] == -1)  # Bear
    assert np.all(regimes[100:] == 0)  # Neutral
=======
# tests/unit/test_regime_detection.py
import pytest
import pandas as pd
import numpy as np
from src.data.processors.data_utils import _detect_market_regimes

def test_regime_detection():
    # Create test data with clear bull/bear periods
    prices = pd.Series(np.concatenate([
        np.linspace(100, 200, 50),  # Bull
        np.linspace(200, 150, 50),   # Bear
        np.linspace(150, 160, 50)    # Neutral
    ]))
    
    regimes = _detect_market_regimes(pd.DataFrame({'close': prices}))
    
    # Verify we get expected regime patterns
    assert np.all(regimes[:50] == 1)  # Bull
    assert np.all(regimes[50:100] == -1)  # Bear
    assert np.all(regimes[100:] == 0)  # Neutral
>>>>>>> 60870aec3b9ed2c2cb804ceb4f1eeb5c6af9d852
