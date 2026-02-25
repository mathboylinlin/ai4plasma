"""Geometry classes and sampling utilities for Physics-Informed Neural Networks (PINN).

This module provides comprehensive geometry classes and sampling utilities for
Physics-Informed Neural Networks (PINNs), supporting flexible domain and boundary
sampling for 1D, 2D, and time-dependent problems.

Core Geometry Classes
---------------------
- `GeoTime`: Temporal domain [ts, te]
- `Geo1D`: Spatial domain [xl, xu]
- `Geo1DTime`: Space-time domain for 1D problems
- `GeoPoly2D`: Polygonal domain in 2D
- `GeoRect2D`: Rectangular domain in 2D
- `GeoPoly2DTime`: Space-time domain for 2D problems
"""

import numpy as np
import torch
from shapely import Polygon, bounds, contains_xy
from typing import List, Tuple, Union, Optional
from enum import Enum

from ai4plasma.utils.common import numpy2torch, list2torch
from ai4plasma.config import REAL


class SamplingMode(Enum):
    """Enumeration for sampling strategies to avoid magic strings.
    
    Attributes
    ----------
    UNIFORM : str
        Evenly spaced sampling with uniform grid spacing.
    RANDOM : str
        Random sampling with uniform distribution.
    LHS : str
        Latin Hypercube Sampling (reserved for future implementation).
    """
    UNIFORM = 'uniform'
    RANDOM = 'random'
    LHS = 'lhs'  # Reserved for future implementation


class Geometry:
    """Abstract base class for geometric domains in PINN problems.
    
    This class defines the interface that all geometry subclasses must implement,
    including domain creation, interior sampling, and boundary sampling. It provides
    a consistent API for handling various geometric shapes in physics simulations.
    
    Interface Requirements
    ----------------------
    Subclasses must implement:
    - create_domain(): Define the geometric domain
    - sample_domain(): Generate sampling points inside the domain
    - sample_boundary(): Generate sampling points on the boundary
    """
    
    def __init__(self) -> None:
        """Initialize base geometry with uninitialized state."""
        self._is_initialized = False
    
    def create_domain(self, *args, **kwargs):
        """Create and initialize the geometric domain.
        
        This abstract method must be overridden by subclasses.
        
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement create_domain()")
    
    def sample_domain(self, N: int, mode: Union[str, SamplingMode] = SamplingMode.UNIFORM,
                     to_tensor: bool = True, require_grad: bool = True, **kwargs):
        """Sample points inside the geometric domain.
        
        Abstract method. Must be implemented by subclasses.
        
        Parameters
        ----------
        N : int
            Number of sampling points.
        mode : Union[str, SamplingMode], default=SamplingMode.UNIFORM
            Sampling strategy.
        to_tensor : bool, default=True
            If True, return PyTorch tensor.
        require_grad : bool, default=True
            If True, enable gradient computation.
        **kwargs : dict
            Additional parameters.
        
        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            Sampled points.
        
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement sample_domain()")
    
    def sample_boundary(self, to_tensor: bool = True, require_grad: bool = True, **kwargs):
        """Sample points on the geometric boundary.
        
        Abstract method. Must be implemented by subclasses.
        
        Parameters
        ----------
        to_tensor : bool, default=True
            If True, return PyTorch tensor.
        require_grad : bool, default=True
            If True, enable gradient computation.
        **kwargs : dict
            Additional parameters.
        
        Returns
        -------
        Union[torch.Tensor, np.ndarray, list]
            Boundary points.
        
        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement sample_boundary()")
    
    def _validate_sampling_params(self, N: int, mode: Union[str, SamplingMode]):
        """Validate common sampling parameters.
        
        Parameters
        ----------
        N : int
            Number of samples.
        mode : Union[str, SamplingMode]
            Sampling mode.
        
        Returns
        -------
        SamplingMode
            Validated mode as enum.
        
        Raises
        ------
        ValueError
            If N <= 0 or mode is invalid.
        """
        if N <= 0:
            raise ValueError(f"Number of samples N must be positive, got {N}")
        
        # Convert string mode to enum if needed
        if isinstance(mode, str):
            try:
                mode = SamplingMode(mode.lower())
            except ValueError:
                valid_modes = [m.value for m in SamplingMode]
                raise ValueError(f"Invalid sampling mode '{mode}'. Valid options: {valid_modes}")
        
        return mode
    
    def _convert_output(self, data: Union[np.ndarray, List[np.ndarray]], 
                       to_tensor: bool, require_grad: bool):
        """Convert output data to appropriate format (tensor or array).
        
        Parameters
        ----------
        data : Union[np.ndarray, List[np.ndarray]]
            Input data.
        to_tensor : bool
            If True, convert to tensor.
        require_grad : bool
            If True, enable gradients.
        
        Returns
        -------
        Union[torch.Tensor, np.ndarray, list]
            Converted data.
        """
        if to_tensor:
            if isinstance(data, list):
                return list2torch(data, require_grad=require_grad)
            else:
                return numpy2torch(data, require_grad=require_grad)
        return data


class GeoTime(Geometry):
    """One-dimensional temporal domain for time-dependent problems.
    
    This class represents a time interval [ts, te] and provides methods to sample
    time points within this interval. It supports both uniform and random sampling,
    and can optionally include boundary time points.
    
    Attributes
    ----------
    ts : float
        Start time of the temporal domain.
    te : float
        End time of the temporal domain.
    """
    
    def __init__(self, ts: float, te: float) -> None:
        """Initialize the temporal domain.
        
        Parameters
        ----------
        ts : float
            Start time of the domain.
        te : float
            End time of the domain.
        
        Raises
        ------
        ValueError
            If ts >= te (invalid time interval).
        """
        super().__init__()
        
        if ts >= te:
            raise ValueError(f"Start time ({ts}) must be less than end time ({te})")
        
        self.create_domain(ts, te)
        self._is_initialized = True
    
    def create_domain(self, ts: float, te: float):
        """Create the temporal domain with specified time bounds.
        
        Parameters
        ----------
        ts : float
            Start time for the domain.
        te : float
            End time for the domain.
        """
        self.ts = float(ts)
        self.te = float(te)
    
    def sample_domain(self, N: int, mode: Union[str, SamplingMode] = SamplingMode.UNIFORM,
                     include_boundary: bool = False, to_tensor: bool = True, 
                     require_grad: bool = True) -> Union[np.ndarray, 'torch.Tensor']:
        """Sample time points within the temporal domain.
        
        Generates N time points in the interval [ts, te] using the specified
        sampling strategy. Points are returned in ascending order.
        
        Parameters
        ----------
        N : int
            Number of time points to sample.
        mode : Union[str, SamplingMode], default=SamplingMode.UNIFORM
            Sampling strategy ('uniform' or 'random').
        include_boundary : bool, default=False
            If True, explicitly include ts and te in samples.
        to_tensor : bool, default=True
            If True, return PyTorch tensor.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Array or tensor of shape (N, 1) containing sampled time points.
        """
        # Validate parameters
        mode = self._validate_sampling_params(N, mode)
        
        # Generate samples in normalized interval [0, 1]
        if mode == SamplingMode.RANDOM:
            t = np.random.rand(N, 1).astype(REAL())
            t = np.sort(t, axis=0)  # Sort for chronological order
        else:  # UNIFORM
            if include_boundary:
                t = np.linspace(0, 1, N, endpoint=True, dtype=REAL()).reshape(-1, 1)
            else:
                # Exclude boundaries by taking interior points
                t = np.linspace(0, 1, N + 1, endpoint=False, dtype=REAL())[1:].reshape(-1, 1)
        
        # Map from [0, 1] to [ts, te]
        t = self.ts + t * (self.te - self.ts)
        
        return self._convert_output(t, to_tensor, require_grad)
    
    def sample_boundary(self, to_tensor: bool = True, 
                       require_grad: bool = True) -> List[Union[np.ndarray, 'torch.Tensor']]:
        """Sample the temporal boundary points (start and end times).
        
        Parameters
        ----------
        to_tensor : bool, default=True
            If True, return PyTorch tensors.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List containing two elements [ts_point, te_point], each with shape (1, 1).
        """
        t = [np.array([[self.ts]], dtype=REAL()),
             np.array([[self.te]], dtype=REAL())]
        
        return self._convert_output(t, to_tensor, require_grad)
    

    def sample_space_time(self, x: np.ndarray, t: np.ndarray, 
                         to_tensor: bool = True, require_grad: bool = True) -> Union[np.ndarray, 'torch.Tensor']:
        """Combine spatial and temporal sampling points into space-time coordinates.
        
        Creates a Cartesian product of spatial points x and temporal points t.
        The output is organized such that for each time point, all spatial points
        are listed sequentially.
        
        Parameters
        ----------
        x : np.ndarray
            Spatial coordinates with shape (N, d).
        t : np.ndarray
            Temporal coordinates with shape (M, 1).
        to_tensor : bool, default=True
            If True, return PyTorch tensor.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        Union[np.ndarray,torch.Tensor]
            Space-time coordinates with shape (N*M, d+1).
        """
        x_size, t_size = x.shape[0], t.shape[0]
        
        # Tile spatial points for each time point
        xx = np.tile(x, (t_size, 1))
        
        # Repeat each time point for all spatial points
        tt = np.repeat(t, x_size, axis=0)
        
        # Concatenate spatial and temporal coordinates
        xt = np.concatenate((xx, tt), axis=1)

        return self._convert_output(xt, to_tensor, require_grad)
    

    def sample_space_time_boundary(self, x: np.ndarray, xb_list: List[np.ndarray], 
                                  t: np.ndarray, tb_list: List[np.ndarray],
                                  to_tensor: bool = True, require_grad: bool = True) -> Tuple:
        """Generate space-time boundary sampling points for PINN boundary conditions.
        
        Creates three types of boundary point collections:
        1. Spatial boundaries across all time points
        2. Initial condition points at t=ts
        3. Final condition points at t=te
        
        Parameters
        ----------
        x : np.ndarray
            Interior spatial points with shape (N, d).
        xb_list : List[np.ndarray]
            List of spatial boundary arrays, e.g., [x_left, x_right].
        t : np.ndarray
            Temporal points with shape (M, 1).
        tb_list : List[np.ndarray]
            Boundary times [t_start, t_end].
        to_tensor : bool, default=True
            If True, return PyTorch tensors.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        Tuple[List, Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
            Tuple of (xb, xt0, xt1) where xb is list of boundary arrays,
            xt0 is initial condition points, xt1 is final condition points.
        """
        # Initial time boundary: all spatial points at t=ts
        xt0 = self.sample_space_time(x, tb_list[0], to_tensor=to_tensor, require_grad=require_grad)
        
        # Final time boundary: all spatial points at t=te
        xt1 = self.sample_space_time(x, tb_list[1], to_tensor=to_tensor, require_grad=require_grad)

        # Spatial boundaries across all time points
        xb = [self.sample_space_time(xb, t, to_tensor=to_tensor, require_grad=require_grad) 
              for xb in xb_list]

        return xb, xt0, xt1
    
    
    def sample_space_time_list(self, x: np.ndarray, t_list: List[float],
                              to_tensor: bool = True, require_grad: bool = True) -> List[Union[np.ndarray, 'torch.Tensor']]:
        """Generate space-time snapshots at specific time instances.
        
        Creates spatial snapshots at discrete time points in t_list.
        Useful for evaluating solutions at specific observation times.
        
        Parameters
        ----------
        x : np.ndarray
            Spatial points with shape (N, d).
        t_list : List[float]
            List of specific time values for snapshots.
        to_tensor : bool, default=True
            If True, return PyTorch tensors.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of space-time coordinate arrays, one per time in t_list.
            Each element has shape (N, d+1).
        """
        xt = [self.sample_space_time(x, np.array([[t]], dtype=REAL()), 
                                     to_tensor=to_tensor, require_grad=require_grad) 
              for t in t_list]

        return xt
    

class Geo1D(Geometry):
    """One-dimensional spatial domain (line segment) for 1D problems.
    
    This class represents a 1D interval [xl, xu] and provides methods to sample
    spatial points within this interval. It's fundamental for 1D steady-state
    problems or spatial discretization in 1D time-dependent problems.
    
    Attributes
    ----------
    xl : float
        Lower bound of the spatial domain.
    xu : float
        Upper bound of the spatial domain.
    """
    
    def __init__(self, points: List[float]) -> None:
        """Initialize the 1D spatial domain.
        
        Parameters
        ----------
        points : List[float]
            Two-element list [xl, xu] defining the interval bounds.
        
        Raises
        ------
        ValueError
            If xl >= xu (invalid interval).
        """
        super().__init__()
        
        if points[0] >= points[1]:
            raise ValueError(f"Lower bound ({points[0]}) must be less than upper bound ({points[1]})")
        
        self.create_domain(points)
        self._is_initialized = True

    def create_domain(self, points: List[float]):
        """Create the 1D spatial domain with specified bounds.
        
        Parameters
        ----------
        points : List[float]
            Two-element list [xl, xu] defining the interval.
        """
        self.xl, self.xu = float(points[0]), float(points[1])

    def sample_domain(self, N: int, mode: Union[str, SamplingMode] = SamplingMode.UNIFORM,
                     include_boundary: bool = False, to_tensor: bool = True, 
                     require_grad: bool = True) -> Union[np.ndarray, 'torch.Tensor']:
        """Sample spatial points within the 1D domain.
        
        Generates N spatial points in the interval [xl, xu] using the specified
        sampling strategy. Points are returned in ascending order.
        
        Parameters
        ----------
        N : int
            Number of spatial points to sample.
        mode : Union[str, SamplingMode], default=SamplingMode.UNIFORM
            Sampling strategy ('uniform' or 'random').
        include_boundary : bool, default=False
            If True, explicitly include xl and xu.
        to_tensor : bool, default=True
            If True, return PyTorch tensor.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Array or tensor of shape (N, 1) containing sampled points.
        """
        # Validate parameters
        mode = self._validate_sampling_params(N, mode)
        
        # Generate samples in normalized interval [0, 1]
        if mode == SamplingMode.RANDOM:
            x = np.random.rand(N, 1).astype(REAL())
            x = np.sort(x, axis=0)  # Sort for ascending order
        else:  # UNIFORM
            if include_boundary:
                x = np.linspace(0, 1, N, endpoint=True, dtype=REAL()).reshape(-1, 1)
            else:
                # Exclude boundaries by taking interior points
                x = np.linspace(0, 1, N + 1, endpoint=False, dtype=REAL())[1:].reshape(-1, 1)
        
        # Map from [0, 1] to [xl, xu]
        x = self.xl + x * (self.xu - self.xl)

        return self._convert_output(x, to_tensor, require_grad)
    
    def sample_boundary(self, to_tensor: bool = True, 
                       require_grad: bool = True) -> List[Union[np.ndarray, 'torch.Tensor']]:
        """Sample the spatial boundary points (left and right endpoints).
        
        Parameters
        ----------
        to_tensor : bool, default=True
            If True, return PyTorch tensors.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List containing two elements [x_left, x_right], each (1, 1).
        """
        x = [np.array([[self.xl]], dtype=REAL()),
             np.array([[self.xu]], dtype=REAL())]

        return self._convert_output(x, to_tensor, require_grad)


class Geo1DTime(Geometry):
    """Composite 1D space-time domain for time-dependent 1D problems.
    
    This class combines a 1D spatial domain (Geo1D) and a temporal domain (GeoTime)
    to handle time-dependent problems. It delegates spatial and temporal sampling
    to respective component geometries.
    
    Applications
    -------------
    Suitable for solving time-dependent 1D PDEs:
    - Heat equation: du/dt = alpha * d2u/dx2
    - Wave equation: d2u/dt2 = c2 * d2u/dx2
    - Advection-diffusion: du/dt + v*du/dx = D*d2u/dx2
    
    Attributes
    ----------
    geo_space : Geo1D
        Spatial domain component [xl, xu].
    geo_time : GeoTime
        Temporal domain component [ts, te].
    """
    
    def __init__(self, points: List[float], ts: float, te: float) -> None:
        """Initialize the 1D space-time composite geometry.
        
        Parameters
        ----------
        points : List[float]
            Two-element list [xl, xu] for spatial domain bounds.
        ts : float
            Start time of temporal domain.
        te : float
            End time of temporal domain.
        
        Raises
        ------
        ValueError
            If spatial or temporal bounds are invalid.
        """
        super().__init__()

        self.geo_space = Geo1D(points)
        self.geo_time = GeoTime(ts, te)
        self._is_initialized = True

    def sample_domain(self, N: int, mode: Union[str, SamplingMode] = SamplingMode.UNIFORM,
                     include_boundary: bool = False, to_tensor: bool = True, 
                     require_grad: bool = True) -> Union[np.ndarray, 'torch.Tensor']:
        """Sample spatial points only (delegates to geo_space).
        
        Note: Samples only the spatial domain. For space-time sampling,
        use sample_all_domain() instead.
        
        Parameters
        ----------
        N : int
            Number of spatial samples.
        mode : Union[str, SamplingMode], default=SamplingMode.UNIFORM
            Sampling strategy ('uniform' or 'random').
        include_boundary : bool, default=False
            If True, include spatial boundaries.
        to_tensor : bool, default=True
            If True, return PyTorch tensor.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Spatial points with shape (N, 1).
        """
        return self.geo_space.sample_domain(N, mode, include_boundary, to_tensor, require_grad)
    
    def sample_space_time_list(self, x: np.ndarray, t_list: List[float],
                              to_tensor: bool = True, require_grad: bool = True) -> List[Union[np.ndarray, 'torch.Tensor']]:
        """Generate spatial snapshots at specific time instances (delegates to geo_time).
        
        Parameters
        ----------
        x : np.ndarray
            Spatial points with shape (N, 1).
        t_list : List[float]
            List of specific time values.
        to_tensor : bool, default=True
            If True, return PyTorch tensors.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of space-time coordinate arrays.
        """
        return self.geo_time.sample_space_time_list(x, t_list, to_tensor, require_grad)

    def sample_all_domain(self, Nx: int, Nt: int, 
                         mode: List[Union[str, SamplingMode]] = ['uniform', 'uniform'],
                         include_boundary: bool = False,
                         to_tensor: bool = True, require_grad: bool = True) -> Tuple:
        """Sample all domains including interior and boundaries for PINN training.
        
        Primary method for generating training data for time-dependent 1D PDE problems.
        Generates:
        1. Interior space-time points for PDE residual
        2. Spatial boundaries across time for boundary conditions
        3. Initial time boundary for initial conditions
        4. Final time boundary for terminal conditions
        
        Parameters
        ----------
        Nx : int
            Number of spatial sampling points.
        Nt : int
            Number of temporal sampling points.
        mode : List[Union[str, SamplingMode]], default=['uniform', 'uniform']
            List of sampling modes [space_mode, time_mode].
        include_boundary : bool, default=False
            If True, include temporal boundaries in t samples.
        to_tensor : bool, default=True
            If True, return as PyTorch tensors.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        Tuple[Union[np.ndarray, torch.Tensor], Tuple]
            Tuple of (xt, (xb, xbt0, xbt1)) where:
            - xt: Interior space-time points, shape (Nx*Nt, 2)
            - xb: List of [left_boundary, right_boundary] across time
            - xbt0: Initial condition points at t=ts, shape (Nx, 2)
            - xbt1: Final condition points at t=te, shape (Nx, 2)
        """
        # Sample spatial domain
        x = self.geo_space.sample_domain(Nx, mode[0], include_boundary, to_tensor=False)
        
        # Sample temporal domain
        t = self.geo_time.sample_domain(Nt, mode[1], include_boundary, to_tensor=False)

        # Combine into space-time interior points
        xt = self.geo_time.sample_space_time(x, t, to_tensor=to_tensor, require_grad=require_grad)

        # Sample spatial boundaries
        xb_list = self.geo_space.sample_boundary(to_tensor=False)
        
        # Sample temporal boundaries
        tb_list = self.geo_time.sample_boundary(to_tensor=False)

        # Combine boundaries
        xb, xbt0, xbt1 = self.geo_time.sample_space_time_boundary(x, xb_list, t, tb_list,
                                                                  to_tensor=to_tensor, 
                                                                  require_grad=require_grad)

        return xt, (xb, xbt0, xbt1)
    

    
class GeoPoly2DTime(Geometry):
    """Composite 2D space-time domain for time-dependent 2D problems.
    
    This class combines a 2D polygonal spatial domain (GeoPoly2D) and a temporal
    domain (GeoTime) to handle time-dependent problems in complex 2D geometries.
    
    Applications
    -------------
    - 2D heat conduction in complex geometries
    - Fluid flow in irregular domains
    - Electromagnetic field evolution
    - Reaction-diffusion systems
    
    Attributes
    ----------
    geo_space : GeoPoly2D
        2D polygonal spatial domain.
    geo_time : GeoTime
        Temporal domain component [ts, te].
    """
    
    def __init__(self, points: np.ndarray, ts: float, te: float) -> None:
        """Initialize the 2D space-time composite geometry.
        
        Parameters
        ----------
        points : np.ndarray
            Array of polygon vertices with shape (n_vertices, 2).
            Vertices should be ordered consistently.
        ts : float
            Start time of temporal domain.
        te : float
            End time of temporal domain.
        
        Raises
        ------
        ValueError
            If temporal bounds are invalid or polygon is degenerate.
        """
        super().__init__()

        self.geo_space = GeoPoly2D(points)
        self.geo_time = GeoTime(ts, te)
        self._is_initialized = True

    def sample_all_domain(self, Nx: int, Nt: int, Nb_list: List[int],
                         mode: List[Union[str, SamplingMode]] = ['uniform', 'uniform', 'uniform'],
                         include_boundary: bool = False,
                         to_tensor: bool = True, require_grad: bool = True) -> Tuple:
        """Sample all domains including interior and boundaries for PINN training.
        
        Generates comprehensive training data for time-dependent 2D PDE problems.
        
        Parameters
        ----------
        Nx : int
            Number of spatial sampling points in the interior.
        Nt : int
            Number of temporal sampling points.
        Nb_list : List[int]
            List specifying number of samples per polygon edge.
            Must have length equal to number of polygon vertices.
        mode : List[Union[str, SamplingMode]], default=['uniform', 'uniform', 'uniform']
            List of sampling modes [space_mode, time_mode, boundary_mode].
        include_boundary : bool, default=False
            If True, include temporal boundaries in t samples.
        to_tensor : bool, default=True
            If True, return as PyTorch tensors.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        Tuple[Union[np.ndarray, torch.Tensor], Tuple]
            Tuple of (xt, (xb, xbt0, xbt1)) where:
            - xt: Interior space-time points, shape (Nx*Nt, 3)
            - xb: List of boundary segments across time
            - xbt0: Initial condition points at t=ts, shape (Nx, 3)
            - xbt1: Final condition points at t=te, shape (Nx, 3)
        """
        # Sample 2D spatial domain (using rejection sampling)
        x = self.geo_space.sample_domain(Nx, mode[0], to_tensor=False)
        
        # Sample temporal domain
        t = self.geo_time.sample_domain(Nt, mode[1], include_boundary, to_tensor=False)

        # Combine into space-time interior points
        xt = self.geo_time.sample_space_time(x, t, to_tensor=to_tensor, require_grad=require_grad)

        # Sample spatial boundaries (one array per polygon edge)
        xb_list = self.geo_space.sample_boundary(Nb_list, mode=mode[2], to_tensor=False)
        
        # Sample temporal boundaries
        tb_list = self.geo_time.sample_boundary(to_tensor=False)

        # Combine boundaries
        xb, xbt0, xbt1 = self.geo_time.sample_space_time_boundary(x, xb_list, t, tb_list,
                                                                  to_tensor=to_tensor, 
                                                                  require_grad=require_grad)

        return xt, (xb, xbt0, xbt1)
       
    

class GeoPoly2D(Geometry):
    """Two-dimensional polygonal domain for 2D spatial problems.
    
    This class represents an arbitrary polygon defined by its vertices and uses the
    Shapely library for geometric operations. It supports sampling interior points
    using rejection sampling and boundary points on each polygon edge.
    
    Polygon Properties
    ------------------
    The polygon can be convex or non-convex, but should be:
    - Simple (edges don't cross each other)
    - Non-degenerate (has non-zero area)
    - Properly oriented (vertices in consistent order)
    
    Attributes
    ----------
    points : np.ndarray
        Array of polygon vertices, shape (n_vertices, 2).
    points_num : int
        Number of vertices.
    polygon : Polygon
        Shapely polygon object for geometric operations.
    bound : tuple
        Bounding box (xmin, ymin, xmax, ymax).
    """
    
    def __init__(self, points: np.ndarray) -> None:
        """Initialize the 2D polygonal geometry.
        
        Parameters
        ----------
        points : np.ndarray
            Array of polygon vertices with shape (n_vertices, 2).
            Vertices should be ordered consistently.
        
        Raises
        ------
        ValueError
            If polygon is degenerate or invalid.
        """
        super().__init__()
        self.create_domain(points)
        self._is_initialized = True

    def get_bounding_box(self, geo: Polygon) -> Tuple[float, float, float, float]:
        """Get the axis-aligned bounding box of a geometry.
        
        Parameters
        ----------
        geo : Polygon
            Shapely geometry object.
        
        Returns
        -------
        Tuple[float, float, float, float]
            Tuple of (xmin, ymin, xmax, ymax) defining the bounding box.
        """
        return bounds(geo)

    def create_domain(self, points: np.ndarray):
        """Create the 2D polygonal domain from vertices.
        
        Parameters
        ----------
        points : np.ndarray
            Array of polygon vertices with shape (n_vertices, 2).
        """
        self.points = points.astype(REAL())
        self.points_num = points.shape[0]
        self.polygon = Polygon(points)
        self.bound = bounds(self.polygon)
        
        # Validate polygon
        if not self.polygon.is_valid:
            raise ValueError("Invalid polygon: edges may intersect or polygon may be degenerate")
        if self.polygon.area == 0:
            raise ValueError("Degenerate polygon: area is zero")

    def sample_domain(self, N: int, mode: Union[str, SamplingMode] = '',
                     to_tensor: bool = True, require_grad: bool = True) -> Union[np.ndarray, 'torch.Tensor']:
        """Sample interior points within the polygonal domain using rejection sampling.
        
        Uses rejection sampling: randomly generates points in the bounding box
        and accepts only those inside the polygon.
        
        Parameters
        ----------
        N : int
            Number of interior points to sample.
        mode : Union[str, SamplingMode], default=''
            Sampling mode (currently unused, reserved for future).
        to_tensor : bool, default=True
            If True, return PyTorch tensor.
        require_grad : bool, default=True
            If True and to_tensor=True, enable gradient computation.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Array or tensor of shape (N, 2) containing sampled points.
        """
        if N <= 0:
            raise ValueError(f"Number of samples N must be positive, got {N}")
        
        xmin, ymin, xmax, ymax = self.bound
        x = np.zeros((N, 2), dtype=REAL())
        counter = 0
        
        # Rejection sampling loop
        while counter < N:
            # Generate random point in bounding box
            xy = np.random.rand(1, 2).astype(REAL())
            xy[0, 0] = xmin + (xmax - xmin) * xy[0, 0]
            xy[0, 1] = ymin + (ymax - ymin) * xy[0, 1]
            
            # Accept if inside polygon
            if contains_xy(self.polygon, xy[0, 0], xy[0, 1]):
                x[counter, :] = xy
                counter += 1

        return self._convert_output(x, to_tensor, require_grad)
    
    def sample_boundary(self, N_list: List[int], mode: Union[str, SamplingMode] = SamplingMode.UNIFORM,
                       to_tensor: bool = True, require_grad: bool = True) -> List[Union[np.ndarray, 'torch.Tensor']]:
        """Sample points on the boundary edges of the polygon.
        
        Generates points along each edge with specified distribution.
        Each edge is sampled independently.
        
        Parameters
        ----------
        N_list : List[int]
            Number of points per edge. Must match polygon vertices count.
        mode : Union[str, SamplingMode], default=SamplingMode.UNIFORM
            Sampling strategy ('uniform' or 'random').
        to_tensor : bool, default=True
            If True, return PyTorch tensors.
        require_grad : bool, default=True
            If True, enable gradient computation.
        
        Returns
        -------
        List[Union[np.ndarray, torch.Tensor]]
            List of boundary arrays, one per edge, shape (N_i, 2).
        
        Raises
        ------
        ValueError
            If N_list length doesn't match polygon edges.
        """
        # Validate input
        if len(N_list) != self.points_num:
            raise ValueError(f"N_list length ({len(N_list)}) must match number of edges ({self.points_num})")
        
        # Convert mode to enum if needed
        if isinstance(mode, str) and mode:
            try:
                mode = SamplingMode(mode.lower())
            except ValueError:
                mode = SamplingMode.UNIFORM

        xy_list = []
        for i in range(self.points_num):
            N = N_list[i]
            
            if N <= 0:
                raise ValueError(f"Number of samples for edge {i} must be positive, got {N}")
            
            # Get edge endpoints
            point1 = self.points[i, :]
            point2 = self.points[(i + 1) % self.points_num, :]

            if mode == SamplingMode.RANDOM:
                # Random sampling along edge
                t = np.random.rand(N, 1).astype(REAL())
                xy = np.zeros((N, 2), dtype=REAL())
                xy[:, 0:1] = point1[0] + t * (point2[0] - point1[0])
                xy[:, 1:2] = point1[1] + t * (point2[1] - point1[1])
            else:  # UNIFORM
                # Uniform sampling along edge (exclude endpoint)
                xy_x = np.linspace(point1[0], point2[0], N, endpoint=False, dtype=REAL()).reshape(-1, 1)
                xy_y = np.linspace(point1[1], point2[1], N, endpoint=False, dtype=REAL()).reshape(-1, 1)
                xy = np.concatenate((xy_x, xy_y), axis=1)
            
            xy_list.append(xy)

        return self._convert_output(xy_list, to_tensor, require_grad)
    

class GeoRect2D(Geometry):
    """
    Two-dimensional rectangular domain for 2D spatial problems.
    
    This class represents an axis-aligned rectangle defined by its bounds in x and y directions.
    It provides efficient sampling methods for both interior (domain) and boundary points,
    with support for uniform grid sampling and random sampling strategies.
    
    The rectangle is defined by [xmin, xmax] × [ymin, ymax] and supports:
    - Uniform grid sampling for structured collocation points
    - Random sampling for stochastic training approaches
    - Boundary sampling with controllable density on each edge
    
    Attributes:
        xmin (float): Minimum x-coordinate
        xmax (float): Maximum x-coordinate
        ymin (float): Minimum y-coordinate
        ymax (float): Maximum y-coordinate
        width (float): Rectangle width (xmax - xmin)
        height (float): Rectangle height (ymax - ymin)
        area (float): Rectangle area (width × height)
    """
    
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        """Initialize the 2D rectangular geometry.
        
        Parameters
        ----------
        xmin : float
            Minimum x-coordinate.
        xmax : float
            Maximum x-coordinate.
        ymin : float
            Minimum y-coordinate.
        ymax : float
            Maximum y-coordinate.
        
        Raises
        ------
        ValueError
            If xmin >= xmax or ymin >= ymax.
        """
        super().__init__()
        self.create_domain(xmin, xmax, ymin, ymax)
        self._is_initialized = True

    def create_domain(self, xmin: float, xmax: float, ymin: float, ymax: float):
        """Create the 2D rectangular domain from bounds.
        
        Parameters
        ----------
        xmin : float
            Minimum x-coordinate.
        xmax : float
            Maximum x-coordinate.
        ymin : float
            Minimum y-coordinate.
        ymax : float
            Maximum y-coordinate.
        
        Raises
        ------
        ValueError
            If bounds are invalid (min >= max in any dimension).
        """
        if xmin >= xmax:
            raise ValueError(f"Invalid x-bounds: xmin ({xmin}) must be less than xmax ({xmax})")
        if ymin >= ymax:
            raise ValueError(f"Invalid y-bounds: ymin ({ymin}) must be less than ymax ({ymax})")
        
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        
        # Compute derived properties
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        self.area = self.width * self.height

    def sample_domain(self, N: Union[int, Tuple[int, int], List[int]],
                     mode: Union[str, SamplingMode] = SamplingMode.RANDOM,
                     to_tensor: bool = True, require_grad: bool = True) -> Union[np.ndarray, 'torch.Tensor']:
        """Sample interior points within the rectangular domain.
        
        Supports two strategies:
        1. UNIFORM: Nx × Ny grid
        2. RANDOM: N uniformly random points
        
        Parameters
        ----------
        N : Union[int, Tuple[int, int], List[int]]
            Grid resolution or number of samples.
            - int + uniform: N×N grid
            - (Nx, Ny) + uniform: Nx×Ny grid
            - int + random: N random points
        mode : Union[str, SamplingMode], default=SamplingMode.RANDOM
            Sampling strategy ('uniform' or 'random').
        to_tensor : bool, default=True
            If True, return PyTorch tensor.
        require_grad : bool, default=True
            If True, enable gradient computation.
        
        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Sampled points with shape (M, 2) where:
            - M = Nx*Ny for uniform mode
            - M = N for random mode
        
        Raises
        ------
        ValueError
            If N is invalid or mode unsupported.
        """
        # Normalize and validate N based on mode
        if isinstance(mode, str):
            try:
                mode = SamplingMode(mode.lower())
            except ValueError:
                valid_modes = [m.value for m in SamplingMode]
                raise ValueError(f"Invalid sampling mode '{mode}'. Valid options: {valid_modes}")

        if mode == SamplingMode.UNIFORM:
            if isinstance(N, (tuple, list)):
                if len(N) != 2:
                    raise ValueError(f"For uniform mode, N must be an int or (Nx, Ny); got length {len(N)}")
                Nx, Ny = int(N[0]), int(N[1])
            elif isinstance(N, int):
                Nx = Ny = N
            else:
                raise ValueError(f"For uniform mode, N must be int or tuple/list of two ints, got {type(N)}")

            if Nx <= 0 or Ny <= 0:
                raise ValueError(f"Grid dimensions must be positive, got Nx={Nx}, Ny={Ny}")

            # Create uniform grid with separate resolutions on width/height
            x = np.linspace(self.xmin, self.xmax, Nx, dtype=REAL())
            y = np.linspace(self.ymin, self.ymax, Ny, dtype=REAL())
            X, Y = np.meshgrid(x, y)
            xy = np.column_stack((X.ravel(), Y.ravel()))

        elif mode == SamplingMode.RANDOM:
            if not isinstance(N, int):
                raise ValueError("For random mode, N must be an int (number of points)")
            if N <= 0:
                raise ValueError(f"Number of samples N must be positive, got {N}")

            # Generate N random points
            xy = np.random.rand(N, 2).astype(REAL())
            xy[:, 0] = self.xmin + self.width * xy[:, 0]
            xy[:, 1] = self.ymin + self.height * xy[:, 1]

        else:
            raise ValueError(f"Unsupported sampling mode: {mode}")

        return self._convert_output(xy, to_tensor, require_grad)
    
    def sample_boundary(self, N_list: Union[List[int], int] = None, 
                       mode: Union[str, SamplingMode] = SamplingMode.UNIFORM,
                       to_tensor: bool = True, require_grad: bool = True) -> List[Union[np.ndarray, 'torch.Tensor']]:
        """Sample points on the four rectangle boundary edges.
        
        The four edges are ordered as: left, right, bottom, top.
        Each edge is sampled independently with uniform or random strategy.
        
        Edge Definitions
        ================
        - **Left** (Edge 0):   x = x_min, y ∈ [y_min, y_max]
        - **Right** (Edge 1):  x = x_max, y ∈ [y_min, y_max]
        - **Bottom** (Edge 2): y = y_min, x ∈ [x_min, x_max]
        - **Top** (Edge 3):    y = y_max, x ∈ [x_min, x_max]
        
        Parameters
        ----------
        N_list : Union[List[int], int], default=None
            Samples per edge [N_left, N_right, N_bottom, N_top].
            - int: Same N on all 4 edges
            - List[4]: Individual N per edge
            - None: Defaults to 50 per edge
        mode : Union[str, SamplingMode], default=SamplingMode.UNIFORM
            Strategy: 'uniform' (evenly spaced) or 'random' (uniformly distributed).
        to_tensor : bool, default=True
            If True, return PyTorch tensors.
        require_grad : bool, default=True
            If True, enable gradient computation.
        
        Returns
        -------
        List[np.ndarray] or List[torch.Tensor]
            Four arrays [left, right, bottom, top] each with shape (N_i, 2).
        
        Raises
        ------
        ValueError
            If N_list length ≠ 4 or values ≤ 0.
        """
        # Handle default and single-value N_list
        if N_list is None:
            N_list = [50, 50, 50, 50]
        elif isinstance(N_list, int):
            if N_list <= 0:
                raise ValueError(f"Number of samples must be positive, got {N_list}")
            N_list = [N_list] * 4
        elif len(N_list) != 4:
            raise ValueError(f"N_list must have exactly 4 elements (one per edge), got {len(N_list)}")
        
        # Validate all N values
        for i, N in enumerate(N_list):
            if N <= 0:
                raise ValueError(f"Number of samples for edge {i} must be positive, got {N}")
        
        # Convert mode to enum if needed
        if isinstance(mode, str):
            try:
                mode = SamplingMode(mode.lower())
            except ValueError:
                valid_modes = [m.value for m in SamplingMode]
                raise ValueError(f"Invalid sampling mode '{mode}'. Valid options: {valid_modes}")
        
        xy_list = []
        
        # Edge 0: Left boundary (x = xmin)
        N = N_list[0]
        if mode == SamplingMode.RANDOM:
            y = self.ymin + self.height * np.random.rand(N).astype(REAL())
        else:  # UNIFORM
            y = np.linspace(self.ymin, self.ymax, N, endpoint=False, dtype=REAL())
        x = np.full(N, self.xmin, dtype=REAL())
        xy_left = np.column_stack((x, y))
        xy_list.append(xy_left)
        
        # Edge 1: Right boundary (x = xmax)
        N = N_list[1]
        if mode == SamplingMode.RANDOM:
            y = self.ymin + self.height * np.random.rand(N).astype(REAL())
        else:  # UNIFORM
            y = np.linspace(self.ymin, self.ymax, N, endpoint=False, dtype=REAL())
        x = np.full(N, self.xmax, dtype=REAL())
        xy_right = np.column_stack((x, y))
        xy_list.append(xy_right)
        
        # Edge 2: Bottom boundary (y = ymin)
        N = N_list[2]
        if mode == SamplingMode.RANDOM:
            x = self.xmin + self.width * np.random.rand(N).astype(REAL())
        else:  # UNIFORM
            x = np.linspace(self.xmin, self.xmax, N, endpoint=False, dtype=REAL())
        y = np.full(N, self.ymin, dtype=REAL())
        xy_bottom = np.column_stack((x, y))
        xy_list.append(xy_bottom)
        
        # Edge 3: Top boundary (y = ymax)
        N = N_list[3]
        if mode == SamplingMode.RANDOM:
            x = self.xmin + self.width * np.random.rand(N).astype(REAL())
        else:  # UNIFORM
            x = np.linspace(self.xmin, self.xmax, N, endpoint=False, dtype=REAL())
        y = np.full(N, self.ymax, dtype=REAL())
        xy_top = np.column_stack((x, y))
        xy_list.append(xy_top)

        return self._convert_output(xy_list, to_tensor, require_grad)


