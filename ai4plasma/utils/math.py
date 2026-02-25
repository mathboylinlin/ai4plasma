"""Mathematical utilities and tools for neural network computations in AI4Plasma.

This module provides a collection of mathematical functions and utility classes for
neural network training and inference in the AI4Plasma project. It includes error
metrics computation, automatic differentiation utilities, and floating-point precision
management.

Math Functions
--------------
- `calc_l2_err()`: Compute L2 error between true and predicted values
- `calc_relative_l2_err()`: Compute relative L2 error for normalized comparison
- `df_dX()`: Calculate tensor derivatives using automatic differentiation

Math Classes
------------
- `Real`: Floating-point precision manager for cross-library compatibility
"""

import torch
import numpy as np


def calc_l2_err(x_true, x_pred):
    """Calculate L2 error between true and predicted values.
    
    Computes the L2 (Euclidean) norm of the difference between true and
    predicted values, normalized by the number of elements. This metric
    provides a scale-dependent measure of prediction accuracy.
    
    Parameters
    ----------
    x_true : numpy.ndarray or torch.Tensor
        True values with shape (N,) or (N, ...).
    x_pred : numpy.ndarray or torch.Tensor
        Predicted values with same shape as x_true.
    
    Returns
    -------
    float or torch.Tensor
        L2 error normalized by number of elements.
    
    Raises
    ------
    TypeError
        If inputs are not both numpy arrays or both torch tensors.
    """
    if isinstance(x_true, np.ndarray) and isinstance(x_pred, np.ndarray):
        l2_err = np.linalg.norm(x_true - x_pred, 2) / len(x_true)
    elif isinstance(x_true, torch.Tensor) and isinstance(x_pred, torch.Tensor):
        l2_err = torch.norm(x_true - x_pred, 2) / len(x_true)
    else:
        raise TypeError("Inputs must be either both numpy.ndarray or both torch.Tensor")
    
    return l2_err


def calc_relative_l2_err(x_true, x_pred):
    """Calculate relative L2 error between true and predicted values.
    
    Computes the ratio of the L2 norm of the error to the L2 norm of the
    true values. This provides a scale-independent error metric suitable
    for comparing predictions across different value ranges.
    
    Parameters
    ----------
    x_true : numpy.ndarray or torch.Tensor
        True values with shape (N,) or (N, ...). Must have non-zero norm.
    x_pred : numpy.ndarray or torch.Tensor
        Predicted values with same shape as x_true.
    
    Returns
    -------
    float or torch.Tensor
        Relative L2 error as a dimensionless quantity.
    
    Raises
    ------
    TypeError
        If inputs are not both numpy arrays or both torch tensors.
    """
    if isinstance(x_true, np.ndarray) and isinstance(x_pred, np.ndarray):
        l2_err = np.linalg.norm(x_true - x_pred, 2) / np.linalg.norm(x_true, 2)
    elif isinstance(x_true, torch.Tensor) and isinstance(x_pred, torch.Tensor):
        l2_err = torch.norm(x_true - x_pred, 2) / torch.norm(x_true, 2)
    else:
        raise TypeError("Inputs must be either both numpy.ndarray or both torch.Tensor")
    
    return l2_err


def df_dX(f, X, retain_graph=True, create_graph=True):
    """Calculate derivatives df/dX using automatic differentiation.
    
    Computes the gradient of a scalar function f with respect to input tensor X
    using PyTorch's automatic differentiation. Useful for computing Jacobian
    matrices and derivatives in neural network loss functions.
    
    Parameters
    ----------
    f : torch.Tensor
        Function output tensor, typically of shape (N, 1) or scalar.
        Must require gradients (requires_grad=True).
    X : torch.Tensor
        Input tensor with respect to which derivative is computed,
        shape (N, M) or similar.
    retain_graph : bool, optional
        Whether to retain the computation graph after backward pass.
        Default is True, required for multiple backward passes or
        subsequent operations.
    create_graph : bool, optional
        Whether to create a graph for computing higher-order derivatives.
        Default is True, enabling computation of second derivatives if needed.
    
    Returns
    -------
    torch.Tensor
        Gradient of f with respect to X. Shape depends on f and X shapes.
    """
    grad = torch.autograd.grad(f, X, 
                                grad_outputs=torch.ones_like(f), 
                                retain_graph=retain_graph, 
                                create_graph=create_graph)[0]
    return grad


class Real:
    """Floating-point precision manager for cross-library compatibility.
    
    Manages floating-point precision (16, 32, 64-bit) for both NumPy and PyTorch,
    ensuring consistent data types across different libraries and enabling easy
    switching between different precision levels.
    
    Attributes
    ----------
    precision : int
        Current floating-point precision (16, 32, or 64).
    real : dict
        Dictionary mapping library names ('numpy', 'torch') to their respective
        floating-point types.
    """
    
    def __init__(self, precision=32) -> None:
        """Initialize Real class with specified floating-point precision.
        
        Parameters
        ----------
        precision : int, optional
            Floating-point precision: 16 (half), 32 (single), or 64 (double).
            Default is 32.
        
        Raises
        ------
        Note
            Invalid precision values default to 32-bit.
        """
        self.precision = precision
        self.real = None
        self.set_float_precision(precision)

    def __call__(self, tag=None):
        """Retrieve floating-point type for specified library.
        
        Parameters
        ----------
        tag : str, optional
            Library name: 'numpy' or 'torch'. Default is 'numpy'.
        
        Returns
        -------
        type
            Floating-point type for the specified library.
        """
        tag = 'numpy' if tag is None else tag

        return self.real[tag]
    
    def __str__(self) -> str:
        """Return a string representation of the Real object.

        Returns
        -------
        str
            String in format 'Float{precision}', e.g., 'Float32', 'Float64'.
        """
        return f'Float{self.precision}'

    def set_float_precision(self, precision=32) -> None:
        """Set floating-point precision for both NumPy and PyTorch.
        
        Updates the floating-point types for both libraries to match the
        specified precision level.
        
        Parameters
        ----------
        precision : int, optional
            Precision value: 16 (half), 32 (single), or 64 (double).
            Default is 32. Invalid values default to 32.
        """
        if precision == 16:
            self.real = {
                'numpy': np.float16,
                'torch': torch.float16
            }
        elif precision == 64:
            self.real = {
                'numpy': np.float64,
                'torch': torch.float64
            }
        else: # default
            self.real = {
                'numpy': np.float32,
                'torch': torch.float32
            }

    def set_torch_dtype(self, precision=32) -> None:
        """Set default floating-point data type for PyTorch globally.
        
        Configures PyTorch's default tensor dtype to use the specified
        floating-point precision for all subsequently created tensors
        without explicit dtype specification.
        
        Parameters
        ----------
        precision : int, optional
            Precision value: 16 (half), 32 (single), or 64 (double).
            Default is 32. Invalid values default to 32.
        """
        if precision == 16:
            torch.set_default_dtype(torch.float16)
        elif precision == 64:
            torch.set_default_dtype(torch.float64)
        else: # default
            torch.set_default_dtype(torch.float32)