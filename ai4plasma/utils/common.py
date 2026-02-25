"""Common utilities for AI4Plasma.

Common Classes
--------------
- `Timer`: Simple timer for measuring elapsed time
"""

import random
import torch
import numpy as np
from datetime import datetime

from ai4plasma.config import DEVICE


Boltz_k = 1.380649e-23         # Boltzmann constant (J/K)
Elec = 1.602176634e-19         # Elementary charge (C)
Epsilon_0 = 8.8541878128e-12   # Vacuum electric permittivity


def set_seed(seed=2020):
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Seed value used for Python, NumPy, and PyTorch RNGs.
        Default is 2020.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def numpy2torch(x, require_grad=True):
    """Convert a NumPy array to a PyTorch tensor on the configured device.

    Parameters
    ----------
    x : numpy.ndarray
        Input NumPy array.
    require_grad : bool, optional
        If True, set ``requires_grad`` for the returned tensor. Default is True.

    Returns
    -------
    torch.Tensor
        Converted tensor on the device from ``DEVICE()``.
    """
    X = torch.from_numpy(x).requires_grad_(require_grad).to(DEVICE())
    
    return X


def list2torch(x_list, require_grad=True):
    """Convert a list of NumPy arrays to PyTorch tensors on the configured device.

    Parameters
    ----------
    x_list : list of numpy.ndarray
        List of input NumPy arrays.
    require_grad : bool, optional
        If True, set ``requires_grad`` for each returned tensor. Default is True.

    Returns
    -------
    list of torch.Tensor
        List of converted tensors on the device from ``DEVICE()``.
    """
    X_list = [torch.from_numpy(x).requires_grad_(require_grad).to(DEVICE()) for x in x_list]
    
    return X_list


def print_runing_time(t):
    """Print elapsed time in a human-readable unit.

    Parameters
    ----------
    t : float
        Time value in seconds.
    """
    if t <= 60:
        print('Time consumption: %f s' % (t))
    elif t <= 3600:
        print('Time consumption: %f min' % (t/60))
    else:
        print('Time consumption: %f h' % (t/3600))


class Timer:
    """Simple timer for measuring elapsed wall-clock time."""
    
    def __init__(self) -> None:
        """Initialize the timer with the current start time."""
        self.timer_start = datetime.now()
    
    def current(self, print_required=True):
        """Return current time and optionally print elapsed duration.

        Parameters
        ----------
        print_required : bool, optional
            If True, print the elapsed time since initialization.
            Default is True.

        Returns
        -------
        datetime
            Current timestamp.
        """
        timer_current = datetime.now()

        if print_required:
            print_runing_time((timer_current - self.timer_start).seconds)

        return timer_current