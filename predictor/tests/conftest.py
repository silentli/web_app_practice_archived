import sys
from unittest.mock import MagicMock

# Mock pycaret and its submodules
sys.modules['pycaret'] = MagicMock()
sys.modules['pycaret.regression'] = MagicMock()
