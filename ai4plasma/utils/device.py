"""Device management utilities for GPU/CPU selection in AI4Plasma.

This module provides device abstraction and management utilities for seamless
GPU/CPU computation in the AI4Plasma framework. It handles device selection,
validation, and provides a unified interface for device configuration.

Device Functions
----------------
- `check_gpu()`: Check if GPU is available on the system.
- `select_gpu_by_id()`: Select a specific GPU by its ID.
- `torch_device()`: Create a PyTorch device object.

Device Classes
--------------
- `Device`: Centralized device management class for GPU/CPU selection and handling.
"""

import torch

def check_gpu(print_required=False):
    """Check if GPU is available.
    
    Detects GPU availability on the current system using PyTorch.
    
    Parameters
    ----------
    print_required : bool, optional
        If True, prints GPU availability status to console. Defaults to False.
    
    Returns
    -------
    bool
        True if GPU is available, False otherwise.
    """
    if torch.cuda.is_available():
        has_gpu = True
        if print_required:
            print('GPU is available!')
    else:
        has_gpu = False
        if print_required:
            print('GPU is not available!')

    return has_gpu


def select_gpu_by_id(gpu_id=0):
    """Select GPU by its ID.
    
    Sets the active GPU using torch.cuda.set_device(). Validates GPU availability
    and ID validity before selection.
    
    Parameters
    ----------
    gpu_id : int, optional
        GPU ID to select. Must be a non-negative integer and less than the total
        number of available GPUs. Defaults to 0.
    
    Raises
    ------
    ValueError
        If gpu_id is not an integer, negative, exceeds available GPU count,
        or if GPU is not available on the system.
    """
    if not isinstance(gpu_id, int):
        raise ValueError("gpu_id must be an integer.")
    
    if gpu_id < 0:
        raise ValueError(f"Invalid gpu_id {gpu_id}. gpu_id must be an integer greater than or equal to zero.")
    
    elif gpu_id >= torch.cuda.device_count():
        raise ValueError(f"Invalid gpu_id {gpu_id}. Only {torch.cuda.device_count()} GPUs are available.")
    
    else:
        if not torch.cuda.is_available():
            raise ValueError("GPU is not available.")
        
    torch.cuda.set_device(gpu_id)
    

def torch_device(device_id=-1):
    """Create and return a PyTorch device object.
    
    Generates a torch.device object configured for either GPU or CPU based on
    the provided device_id. Performs validation of device_id before creation.
    
    Parameters
    ----------
    device_id : int, optional
        Device ID to use. If >= 0, selects the GPU with that ID. If < 0,
        uses CPU. Defaults to -1 (CPU).
    
    Returns
    -------
    torch.device
        A PyTorch device object configured for the specified device.
    
    Raises
    ------
    ValueError
        If device_id is not an integer, if a GPU device_id is specified but GPU
        is not available, or if device_id exceeds the number of available GPUs.
    """
    if not isinstance(device_id, int):
        raise ValueError("device_id must be an integer.")
    
    if device_id >= 0:
        if not torch.cuda.is_available():
            raise ValueError("GPU is not available, but a GPU device_id was specified.")
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid device_id {device_id}. Only {torch.cuda.device_count()} GPUs are available.")
    
    return torch.device(f"cuda:{device_id}" if device_id >= 0 else "cpu")


class Device:
    """Centralized device management for GPU/CPU selection and handling.
    
    This class provides an abstraction layer for device management, allowing
    flexible switching between GPU and CPU devices with validation. It maintains
    device state and provides a callable interface for dynamic device retrieval.
    
    Attributes
    ----------
    device_id : int
        The current device ID. Negative values indicate CPU, non-negative values
        indicate GPU ID.
    device : torch.device
        The PyTorch device object for the current device.
    """

    def __init__(self, device_id=-1) -> None:
        """Initialize Device class with specified device ID.
        
        Sets up device management with the specified device ID, creating the
        underlying torch.device object through validation.
        
        Parameters
        ----------
        device_id : int, optional
            Device ID to select. If < 0, uses CPU. Non-negative values select
            the GPU with that ID. Defaults to -1 (CPU).
        
        Raises
        ------
        ValueError
            If device_id is invalid or GPU device_id is specified but GPU
            is not available.
        """
        self.device_id = None
        self.device = None
        self.set_device(device_id)

    def __call__(self, device_id=None):
        """Retrieve current device or set a new device.
        
        Provides callable interface allowing the Device object to be used as a
        function for both device switching and retrieval.
        
        Parameters
        ----------
        device_id : int, optional
            Device ID to set. If None, returns the current device without changing it.
            If provided, updates the device to the specified ID. Defaults to None.
        
        Returns
        -------
        torch.device
            The current (or newly set) PyTorch device object.
        """
        if device_id is not None:
            self.set_device(device_id)

        return self.device
    
    def __str__(self) -> str:
        """Return string representation of Device object.
        
        Provides a human-readable string representation showing the device type
        and ID (for GPU).
        
        Returns
        -------
        str
            String representation in format 'Device(cpu)' for CPU or
            'Device(cuda:{device_id})' for GPU.
        """
        if self.device_id < 0:
            return "Device(cpu)"
        else:
            return f"Device(cuda:{self.device_id})"

    def set_device(self, device_id=-1) -> None:
        """Set the device based on provided device ID.
        
        Updates the internal device configuration to the specified device_id.
        Creates a new torch.device object and updates device_id attribute.
        
        Parameters
        ----------
        device_id : int, optional
            Device ID to select. If < 0, uses CPU. Non-negative values select
            the GPU with that ID. Defaults to -1 (CPU).
        
        Raises
        ------
        ValueError
            If device_id is invalid or GPU device_id is specified but GPU
            is not available.
        """
        self.device = torch_device(device_id)
        self.device_id = device_id



