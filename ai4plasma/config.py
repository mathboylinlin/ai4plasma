"""Global configuration settings for AI4Plasma.

This module provides centralized configuration management for the AI4Plasma
project. It defines global singleton instances for precision control and
device management used throughout the framework.

Config Global Variables
-----------------------
REAL : Real
    Singleton instance for managing floating-point precision across the framework.
    Provides consistent data type conversion between numpy and PyTorch tensors.
    Default precision is set to 32-bit floating point.

DEVICE : Device
    Singleton instance for managing computation device placement (CPU/GPU).
    Handles automatic device detection, assignment, and tensor movement.
    Default device is CPU unless GPU is explicitly configured.

Config Usage
------------
Import these global configuration objects to ensure consistent precision
and device placement across all AI4Plasma modules:

>>> from ai4plasma.config import REAL, DEVICE
>>> 
>>> # Use REAL for precision-aware tensor creation
>>> x = REAL.to_torch(numpy_array)
>>> 
>>> # Use DEVICE for device placement
>>> model = MyModel().to(DEVICE())
>>> tensor = torch.zeros(10).to(DEVICE())
"""

from ai4plasma.utils.math import Real
from ai4plasma.utils.device import Device

# Global singleton for floating-point precision management (32-bit by default)
REAL = Real(precision=32)

# Global singleton for computation device management (CPU by default)
DEVICE = Device()
