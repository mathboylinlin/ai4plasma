"""Advanced Physics-Informed Neural Networks (PINNs) for solving PDEs and multi-physics problems.

This module implements a flexible and extensible framework for training Physics-Informed
Neural Networks (PINNs) to solve complex partial differential equations (PDEs) and coupled
multi-physics problems. PINNs embed physics knowledge directly into neural networks through
residual-based loss functions, enabling accurate solutions without requiring large amounts
of labeled training data.

PINN Classes
------------
- `EquationTerm`: Encapsulates a single physics constraint with residual function.
- `VisualizationCallback`: Abstract base for custom visualization during training.
- `PINN`: Main physics-informed neural network model class.
"""

import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from ai4plasma.core.model import BaseModel
from ai4plasma.config import DEVICE


class EquationTerm:
    '''Represents a single physics equation term with its residual function and weight.
    
    This class encapsulates one component of the physics loss function in a PINN model.
    For example, a PDE problem might have separate equation terms for:
    - Interior domain residual (PDE satisfaction)
    - Boundary condition residuals
    - Initial condition residuals
    - Additional constraints or regularization terms
    
    Each term has:
    - A residual function that evaluates how well the network satisfies the equation
    - A weight factor that controls its contribution to the total loss
    - Associated data points where the residual is evaluated
    - Optional DataLoader for batch training on large datasets
    
    Attributes:
    -----------
    name : str
        Unique identifier for this equation term (e.g., 'domain', 'boundary_left', 
        'initial_condition'). Used for logging and loss tracking.
    residual_fn : Callable
        Function with signature: residual_fn(network, data) -> torch.Tensor
        Computes the residual (error) at given data points. Should return a tensor
        where ideally all values are close to zero when the physics is satisfied.
    weight : float
        Multiplicative weight for this loss term. Higher weights make the optimizer
        prioritize satisfying this equation. Typical range: 0.1 to 100.
    data : torch.Tensor
        Input data points where the residual is evaluated. Shape depends on problem
        dimensionality: (N, d) for d-dimensional problems with N points.
    dataloader : DataLoader or None
        PyTorch DataLoader for batch training. Created via create_dataloader() when
        batch training is enabled. None for full-batch training.
    '''
    
    def __init__(self, name: str, residual_fn: Callable, weight: float = 1.0, data: torch.Tensor = None):
        '''
        Initialize an equation term.
        
        Parameters:
        -----------
        name : str
            Unique name for this equation term
        residual_fn : Callable
            Residual function: (network, data) -> residual_tensor
        weight : float, optional
            Weight factor for this loss term. Default: 1.0
        data : torch.Tensor, optional
            Input data for residual evaluation. Default: None
        '''
        self.name = name
        self.residual_fn = residual_fn
        self.weight = weight
        self.data = data
        self.dataloader = None  # For batched training
    
    def compute_residual(self, network, batch_data: torch.Tensor = None):
        '''
        Compute residual using the neural network.
        
        This method evaluates the residual function at the specified data points.
        The residual represents how well the network solution satisfies the physics
        equation at those points. Ideally, residuals should be close to zero.
        
        Parameters:
        -----------
        network : nn.Module
            The neural network model (PINN solution)
        batch_data : torch.Tensor, optional
            Batch of data points for residual computation. If None, uses self.data.
            This allows for batch training where different batches are used in
            different iterations.
        
        Returns:
        --------
        torch.Tensor
            Residual values at the evaluation points. Shape depends on the residual
            function but typically (N,) or (N, output_dim) for N points.
        '''
        if batch_data is not None:
            return self.residual_fn(network, batch_data)
        else:
            return self.residual_fn(network, self.data)
    
    def update_weight(self, new_weight: float):
        '''
        Update the weight of this equation term.
        
        This allows dynamic adjustment of loss weights during training, which can
        be useful for:
        - Curriculum learning (gradually emphasizing different terms)
        - Adaptive weighting based on loss magnitudes
        - Manual tuning during training
        
        Parameters:
        -----------
        new_weight : float
            New weight value for this term
        '''
        self.weight = new_weight
    
    def update_data(self, new_data: torch.Tensor):
        '''
        Update the data points for this equation term.
        
        Useful for:
        - Adaptive sampling (resampling points in regions with high error)
        - Time-dependent problems (updating temporal points)
        - Progressive training (starting with coarse then fine grids)
        
        Parameters:
        -----------
        new_data : torch.Tensor
            New input data tensor
        '''
        self.data = new_data
        self.dataloader = None  # Reset dataloader when data is updated
    
    def create_dataloader(self, batch_size: int, shuffle: bool = False, drop_last: bool = False):
        '''
        Create a PyTorch DataLoader for batched training on this equation term.
        
        For large datasets, batch training is more memory-efficient and can lead
        to better generalization. This method wraps the data tensor in a DataLoader
        that provides automatic batching and optional shuffling.
        
        Parameters:
        -----------
        batch_size : int
            Number of samples per batch. Smaller batches use less memory but may
            be noisier. Typical values: 32, 64, 128, 256, 512.
        shuffle : bool, optional
            Whether to shuffle the data at each epoch. Default: False.
            Shuffling can improve convergence but changes the order of samples.
        drop_last : bool, optional
            Whether to drop the last incomplete batch if the dataset size is not
            divisible by batch_size. Default: False.
        
        Returns:
        --------
        DataLoader or None
            PyTorch DataLoader for iterating over batches, or None if no data
            is available.
        '''
        if self.data is None:
            return None
        
        # Handle different data shapes
        if isinstance(self.data, torch.Tensor):
            if self.data.dim() == 1:
                # 1D data, reshape to (N, 1)
                data = self.data.view(-1, 1)
            else:
                data = self.data
        else:
            data = torch.tensor(self.data)
        
        dataset = TensorDataset(data)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        return self.dataloader
    
    def get_dataloader(self):
        '''
        Retrieve the current DataLoader for this equation term.
        
        Returns:
        --------
        DataLoader or None
            The previously created DataLoader, or None if create_dataloader()
            has not been called yet or if the data was updated.
        '''
        return self.dataloader


class VisualizationCallback:
    '''
    Base class for custom visualization callbacks executed during PINN training.
    
    Visualization callbacks allow you to create and log custom plots or figures during
    training without modifying the core PINN training loop. This is useful for:
    - Monitoring solution evolution over time
    - Comparing predictions with analytical solutions
    - Visualizing residuals and errors
    - Creating animations of training progress
    - Tracking problem-specific metrics
    
    The callback is executed at regular intervals (every log_freq epochs) and can
    generate matplotlib figures that are automatically logged to TensorBoard.
    
    Subclasses must implement the visualize() method which receives:
    - The current network state
    - Current epoch number
    - TensorBoard writer
    - Additional kwargs from the training loop (e.g., loss_dict, total_loss)
    
    Attributes:
    -----------
    name : str
        Unique identifier for this callback. Used in TensorBoard logging paths
        and console output. Example: '1D_Solution', '2D_Heatmap'
    log_freq : int
        Frequency of visualization (every N epochs). Higher values reduce overhead
        but provide less frequent feedback. Typical range: 10-100.
    '''
    
    def __init__(self, name: str, log_freq: int = 10):
        '''
        Initialize the visualization callback.
        
        Parameters:
        -----------
        name : str
            Unique name for this callback. Will appear in TensorBoard as
            'Visualization/{name}/{plot_name}'
        log_freq : int, optional
            Execute visualization every N epochs. Default: 10.
            Set to 0 to disable the callback.
        '''
        self.name = name
        self.log_freq = log_freq
    
    @abstractmethod
    def visualize(self, network, epoch: int, writer: SummaryWriter, **kwargs) -> Dict[str, plt.Figure]:
        '''
        Perform custom visualization and return matplotlib figures.
        
        This method is called automatically during training at the specified frequency.
        It should create one or more matplotlib figures showing relevant information
        about the current state of training.
        
        Parameters:
        -----------
        network : nn.Module
            The neural network being trained. Set to eval() mode before inference
            if you don't want dropout/batchnorm to affect visualization.
        epoch : int
            Current epoch number (1-indexed). Useful for labeling plots.
        writer : SummaryWriter
            TensorBoard writer instance. Can be used for additional custom logging
            if needed, though figures are logged automatically.
        kwargs : dict
            Additional arguments passed from the training loop, which may include:
            - 'loss_dict': Dict mapping equation names to their loss values
            - 'total_loss': Total weighted loss value
            - Any custom kwargs passed to train() via visualization_kwargs parameter
        
        Returns:
        --------
        Dict[str, plt.Figure]
            Dictionary mapping plot names to matplotlib Figure objects.
            Each figure will be logged to TensorBoard at the path:
            'Visualization/{callback_name}/{plot_name}'
            
            Example return values:
            {'comparison': fig1, 'error_heatmap': fig2, 'residuals': fig3}
            
            Return None or empty dict if no visualization should be logged.
        '''
        pass


class PINN(BaseModel, ABC):
    '''
    Advanced Physics-Informed Neural Network (PINN) base class for solving PDEs and multi-physics problems.
    
    This class implements a flexible framework for training neural networks to solve
    partial differential equations (PDEs) by incorporating physics constraints directly
    into the loss function. Unlike traditional supervised learning, PINNs learn from:
    1. Governing equations (PDE residuals in the domain)
    2. Boundary conditions (BCs)
    3. Initial conditions (ICs) for time-dependent problems
    4. Optional observational data
    
    Attributes:
    -----------
    network : nn.Module
        Neural network that approximates the solution u(x,t,...)
    equation_terms : Dict[str, EquationTerm]
        Dictionary of physics equation terms indexed by name
    writer : SummaryWriter or None
        TensorBoard writer for logging (None if not configured)
    optimizer : torch.optim.Optimizer or None
        Optimizer for training
    loss_func : nn.Module
        Loss function (typically MSELoss for PINNs)
    start_epoch : int
        Starting epoch for training (non-zero if resumed)
    training_history : Dict
        Historical record of losses and epochs
    adaptive_weights : bool
        Whether to use adaptive loss weighting
    weight_update_freq : int
        Frequency of weight updates (if adaptive)
    visualization_callbacks : Dict[str, VisualizationCallback]
        Registered visualization callbacks
    '''
    
    def __init__(self, network):
        '''
        Initialize the PINN model with a neural network.
        
        This constructor sets up the core data structures for managing physics equations,
        training state, and visualization. Subclasses should call super().__init__(network)
        before adding equation terms.
        
        Parameters:
        -----------
        network : nn.Module
            PyTorch neural network that will approximate the PDE solution.
            The network architecture should be appropriate for the problem:
            - Input dim = problem dimensionality (e.g., 2 for 2D, 3 for 2D+time)
            - Output dim = number of solution components
            - Hidden layers: typically 3-8 layers with 20-100 neurons each
            
            Example architectures:
            - 1D steady: [1, 50, 50, 50, 1]
            - 2D steady: [2, 100, 100, 100, 1]
            - 1D+time: [2, 50, 50, 50, 50, 1]
        '''
        super().__init__(network)
        
        # Equation management
        self.equation_terms: Dict[str, EquationTerm] = {}
        
        # Training state
        self.writer = None
        # self.checkpoint_dir = None
        self.start_epoch = 0
        self.training_history = {'loss': [], 'epoch': []}
        
        # Loss weighting strategy
        self.adaptive_weights = False
        self.weight_update_freq = 10
        
        # Visualization callbacks
        self.visualization_callbacks: Dict[str, VisualizationCallback] = {}
        
        # Initialization - define equations in subclass
        self._define_loss_terms()
    
    @abstractmethod
    def _define_loss_terms(self):
        '''
        Define all physics equations and loss terms for this PINN problem.
        
        This abstract method MUST be implemented by all PINN subclasses. It is called
        automatically during __init__ to register the physics equations that will be
        used during training. Each equation term represents one component of the loss
        function, such as:
        - PDE residuals in the interior domain
        - Boundary condition residuals
        - Initial condition residuals  
        - Data-fitting terms
        - Regularization constraints
        
        Parameters:
        -----------
        None (method does not take parameters beyond self)
        
        Returns:
        --------
        None (method registers equations through add_equation calls)
        '''
        pass
    
    def add_equation(self, name: str, residual_fn: Callable, weight: float = 1.0, data: torch.Tensor = None):
        '''
        Add a physics equation term to the model for loss calculation during training.
        
        Registers a new equation term with the PINN model. Each equation represents
        one component of the multi-objective loss function being minimized during training.
        Typical equations include domain PDEs, boundary conditions, initial conditions,
        and data constraints.
        
        Parameters:
        -----------
        name : str
            Unique identifier for this equation term. Used for:
            - Loss tracking and logging
            - Weight management and adjustment
            - Identifying equations in get_equation_info()
            - Accessing individual loss contributions
            
            Should be descriptive (e.g., 'domain_pde', 'bc_left', 'initial', 'data_fit')
            to make training logs readable.
        
        residual_fn : Callable
            Function that computes the residual (error) at given points.
            
            Signature: residual_fn(network: nn.Module, data: torch.Tensor) -> torch.Tensor
            
            Parameters:
            - network: The neural network (nn.Module) being trained
            - data: Input tensor of evaluation points, shape (N, d)
            
            Returns:
            - Residual tensor, typically shape (N,) or (N, output_dim)
            - Residual should be ~0 at points satisfying the equation
            - Must maintain computational graph for backpropagation
            
            Implementation Tips:
            - Use torch.autograd.grad() to compute derivatives
            - Set create_graph=True to enable second derivatives
            - All operations should be differentiable
            - Return residual (not loss)
        
        weight : float, optional
            Loss weight for this equation term in the total loss. Default: 1.0
            
            Interpretation:
            - weight = 1.0: default/unit contribution
            - weight > 1.0: emphasize this constraint
            - weight << 1.0: de-emphasize relative to other terms
            - weight = 0.0: effectively disabled
            
            Typical values:
            - Domain PDE: 1.0-5.0
            - Essential BC: 10.0-100.0
            - Natural BC: 1.0-10.0
            - Initial conditions: 5.0-10.0
            - Data approximation: 0.01-1.0
            
            Can be adjusted later via set_equation_weight() or enable_adaptive_weights().
        
        data : torch.Tensor, optional
            Input data tensor where the equation is evaluated. Default: None
            
            Shape: (N, d) where:
            - N: number of evaluation points
            - d: input dimensionality (problem dependent)
            
            Examples:
            - 1D domain: shape (1000, 1)
            - 2D domain: shape (10000, 2)
            - 2D + time: shape (5000, 3) for (x, y, t)
            - Boundary on 2D: shape (100, 2)
            
            Can be None initially and set later via set_equation_data().
            For batched training, data is used to create DataLoaders automatically.
        
        Returns:
        --------
        None
            The equation is registered internally and accessible via get_equation(name).
        
        Raises:
        -------
        None
            No explicit error checking at registration; errors occur if:
            - residual_fn is not callable
            - data incompatible with network input dimension
            - name already exists (overwrites silently)
        '''
        self.equation_terms[name] = EquationTerm(name, residual_fn, weight, data)
    
    def remove_equation(self, name: str):
        '''
        Remove an equation term from the model by name.
        
        This method allows dynamic removal of physics equations that were previously
        added to the PINN model. Useful for removing constraints during different 
        training phases or switching between problem configurations.
        
        Parameters:
        -----------
        name : str
            Unique name of the equation term to remove.
            Must be a name that was previously added via add_equation().
        '''
        if name in self.equation_terms:
            del self.equation_terms[name]
    
    def get_equation(self, name: str) -> Optional[EquationTerm]:
        '''
        Retrieve an equation term by its unique name.
        
        This method provides access to a specific equation term object, useful for:
        - Inspecting equation properties and current weights
        - Modifying equation data or functions programmatically
        - Computing residuals for analysis
        - Debugging equation configuration
        
        Parameters:
        -----------
        name : str
            Unique identifier of the equation term to retrieve.
            Should match the name used in add_equation() call.
        
        Returns:
        --------
        EquationTerm or None
            The requested EquationTerm object if it exists, None otherwise.
            The EquationTerm contains:
            - name: identifier string
            - residual_fn: function for computing residuals
            - weight: current weight in the loss function
            - data: input data points
            - dataloader: optional batched data loader
        '''
        return self.equation_terms.get(name, None)
    
    def set_equation_weight(self, name: str, weight: float):
        '''
        Update the weight (loss contribution) of a specific equation term.
        
        The weight controls how much this particular equation contributes to the total
        loss function during training. This is critical for:
        - Balancing multi-physics problems with competing objectives
        - Emphasizing important constraints (e.g., boundary conditions)
        - Curriculum learning (gradually changing weights during training)
        - Handling multi-scale problems with disparate magnitudes
        
        Parameters:
        -----------
        name : str
            Unique name of the equation term whose weight to update.
            Must be a name that was previously added via add_equation().
        weight : float
            New weight value for this equation term. Interpretation:
            - weight > 0: emphasizes this equation in the total loss
            - weight = 1.0: default/baseline influence
            - weight >> 1.0: strongly enforces the constraint (typical: 5-100)
            - weight << 1.0: reduces constraint importance (typical: 0.01-0.1)
            - weight = 0: effectively disables the equation
        
        Raises:
        -------
        ValueError
            If the equation name is not found in the model.
            Message: 'Equation term "{name}" not found'
        '''
        if name in self.equation_terms:
            self.equation_terms[name].update_weight(weight)
        else:
            raise ValueError(f'Equation term "{name}" not found')
    
    def set_equation_data(self, name: str, data: torch.Tensor):
        '''
        Update the evaluation points (data) for a specific equation term.
        
        This method allows changing which points are used to evaluate a particular
        equation's residual. Essential for:
        - Adaptive mesh refinement / point resampling in high-error regions
        - Progressive training (coarse to fine grids)
        - Time-stepping problems (updating temporal points)
        - Importance sampling (focusing on difficult regions)
        - Dynamically adding new training data during training
        
        Parameters:
        -----------
        name : str
            Unique name of the equation term whose data to update.
            Must be a name that was previously added via add_equation().
        data : torch.Tensor
            New input data tensor for this equation term.
            Shape should be (N, d) where N is number of points and d is dimensionality.
            Example shapes:
            - 1D problem: (1000, 1)
            - 2D problem: (10000, 2)
            - 2D+time problem: (5000, 3) for (x, y, t) coordinates
        
        Raises:
        -------
        ValueError
            If the equation name is not found in the model.
            Message: 'Equation term "{name}" not found'
        '''
        if name in self.equation_terms:
            self.equation_terms[name].update_data(data)
        else:
            raise ValueError(f'Equation term "{name}" not found')
    
    def set_all_equation_weights(self, weights: Dict[str, float]):
        '''
        Set weights for multiple equation terms at once.
        
        This convenience method allows updating all equation weights in a single call,
        rather than calling set_equation_weight() multiple times. Useful for:
        - Switching between different loss configurations
        - Curriculum learning schedules (gradually changing all weights)
        - Rebalancing the loss function for different training phases
        - Implementing weight scheduling algorithms
        
        Parameters:
        -----------
        weights : Dict[str, float]
            Dictionary mapping equation term names to their new weight values.
            Format: {'equation_name': weight_value, ...}
            Example: {'domain': 1.0, 'boundary': 20.0, 'initial': 5.0}
        
        Raises:
        -------
        ValueError
            If any equation name in the dictionary is not found in the model.
            The error is raised by the internal set_equation_weight() call.
        '''
        for name, weight in weights.items():
            self.set_equation_weight(name, weight)
    
    def enable_adaptive_weights(self, enable: bool = True, update_freq: int = 10):
        '''
        Enable or disable adaptive weight adjustment during training.
        
        Adaptive weighting automatically balances different physics equations by
        adjusting their loss weights based on current loss magnitudes. This helps
        prevent one term from dominating the total loss and ensures all constraints
        are satisfied. Particularly useful for:
        - Multi-physics problems with equations of different scales
        - Problems where loss magnitudes vary significantly between terms
        - Avoiding manual tuning of loss weights
        - Curriculum learning where priorities change during training
        
        The adaptive weight for each term is computed as:
            new_weight = current_weight * (average_loss / current_term_loss)
        
        This inverse scaling ensures that equations with larger residuals get
        higher weights to encourage the optimizer to satisfy them better.
        
        Parameters:
        -----------
        enable : bool, optional
            Whether to enable adaptive weighting. Default: True
            - True: Enable adaptive weight adjustments
            - False: Disable adaptive weighting (use fixed weights)
        update_freq : int, optional
            Frequency of weight updates in epochs. Default: 10
            - The weights are updated every N epochs
            - Larger values = less frequent updates (more stable training)
            - Smaller values = more responsive to loss changes
        '''
        self.adaptive_weights = enable
        self.weight_update_freq = update_freq
        status = 'enabled' if enable else 'disabled'
        print(f'Adaptive weighting {status} (update frequency: {update_freq} epochs)')
    
    def _compute_adaptive_weights(self, loss_dict: Dict[str, torch.Tensor]):
        '''
        Compute and update adaptive weights based on current loss magnitudes.
        
        This internal method implements the adaptive weighting strategy. It computes
        a scaling factor for each loss term based on its magnitude relative to the
        average, then applies this scaling to the current weights. This helps ensure
        that all physics equations contribute roughly equally to the total loss,
        preventing any single term from dominating optimization.
        
        The scaling factor for each term is computed as:
            scale_i = average_loss / (current_loss_i + epsilon)
        
        where average_loss is the mean of all loss magnitudes and epsilon is a small
        regularization term to avoid division by zero.
        
        Parameters:
        -----------
        loss_dict : Dict[str, torch.Tensor]
            Dictionary mapping equation names to their current loss values.
            Values should be scalar tensors (each represents accumulated loss for one term).
            Format: {'domain': loss_tensor, 'boundary': loss_tensor, ...}
        '''
        if not self.adaptive_weights or len(loss_dict) < 2:
            return
        
        # Compute average of individual loss magnitudes
        avg_loss = torch.mean(torch.stack(list(loss_dict.values())))
        
        # Scale weights inversely proportional to loss magnitude
        for name, loss in loss_dict.items():
            if loss.item() > 0:
                scaling_factor = avg_loss / (loss + 1e-8)
                current_weight = self.equation_terms[name].weight
                new_weight = current_weight * scaling_factor.item()
                self.equation_terms[name].update_weight(new_weight)
    
    def calc_loss(self, weights_override: Dict[str, float] = None, batch_data: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        '''
        Calculate total loss as weighted sum of individual physics equations.
        
        This is the core loss computation function for PINN training. It evaluates all
        registered equation residuals, computes individual losses, and returns both the
        total weighted loss and a breakdown by equation.
        
        The total loss is computed as:
            L_total = Σᵢ wᵢ * L(residualᵢ)
        
        where wᵢ is the weight for equation i, and L() is the loss function (typically MSE).
        
        Parameters:
        -----------
        weights_override : Dict[str, float], optional
            Temporary weight overrides for specific equation terms. Default: None
            - Used to temporarily change weights without modifying the model state
            - Only affects this loss calculation; permanent weights unchanged
            - Example: {'domain': 0.5, 'boundary': 10.0}
            - Terms not in dictionary use their current model weights
        batch_data : Dict[str, torch.Tensor], optional
            Batch data for each equation term (for batched training). Default: None
            - Maps equation names to their batch data tensors
            - Used when batch processing large datasets
            - If None, uses full dataset for each equation term
            - Example: {'domain': batch_tensor, 'boundary': batch_tensor}
        
        Returns:
        --------
        total_loss : torch.Tensor
            Scalar tensor representing weighted sum of all losses.
            This is what gets backpropagated during training.
            
        loss_dict : Dict[str, torch.Tensor]
            Dictionary mapping equation names to their individual loss values.
            Each value is a scalar tensor (unweighted individual loss).
            Example: {'domain': 0.01, 'boundary': 0.05, 'initial': 0.02}
        
        Raises:
        -------
        RuntimeError
            If no equations have been defined (empty equation_terms dict).
            Message: 'No equations defined. Implement _define_loss_terms() in subclass.'
        '''
        if len(self.equation_terms) == 0:
            raise RuntimeError('No equations defined. Implement _define_loss_terms() in subclass.')
        
        loss_dict = {}
        weighted_losses = []
        
        # Compute residual and loss for each equation term
        for name, eq_term in self.equation_terms.items():
            # Use batch data if provided, otherwise use stored data
            term_batch_data = batch_data.get(name) if batch_data is not None else None
            residual = eq_term.compute_residual(self.network, batch_data=term_batch_data)
            individual_loss = self.loss_func(residual, torch.zeros_like(residual))
            loss_dict[name] = individual_loss
            
            # Get weight (use override if provided)
            if weights_override and name in weights_override:
                weight = weights_override[name]
            else:
                weight = eq_term.weight
            
            weighted_losses.append(weight * individual_loss)
        
        total_loss = sum(weighted_losses)
        
        return total_loss, loss_dict
    
    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        '''
        Make predictions using the trained network.
        
        Evaluates the neural network at given input points to generate predictions.
        The network operates in evaluation mode (no dropout/batchnorm effects) and
        without gradient computation for efficiency.
        
        Parameters:
        -----------
        input_data : torch.Tensor
            Input data for prediction. Shape depends on problem dimensionality:
            - 1D problem: (N, 1) where N is number of points
            - 2D problem: (N, 2) for spatial coordinates
            - 2D+time: (N, 3) for (x, y, t) coordinates
            
            Input should be the same dimensionality as training data.
        
        Returns:
        --------
        torch.Tensor
            Network output (predictions) at input points.
            Shape typically (N, output_dim) where output_dim depends on problem.
            
            Examples:
            - For single scalar field: shape (N, 1)
            - For vector field: shape (N, vector_dim)
            - Tensor is returned on the same device as the network
        '''
        self.network.eval()
        with torch.no_grad():
            output = self.network(input_data)
        return output
    
    def compute_residual(self, name: str, input_data: torch.Tensor = None) -> torch.Tensor:
        '''
        Compute residual for a specific equation term.
        
        Evaluates how well the network satisfies a particular physics equation at
        specified points. Residual values represent the error in the PDE or constraint
        at those points. Ideally, residuals should be close to zero for a well-trained
        PINN.
        
        Parameters:
        -----------
        name : str
            Name of the equation term whose residual to compute.
            Must be a registered equation name (from add_equation).
        input_data : torch.Tensor, optional
            Input points where residual is evaluated. Default: None
            - If None: uses the stored data for this equation term
            - If provided: temporarily uses this data (does not modify model)
            - Should have same dimensionality as training data
        
        Returns:
        --------
        torch.Tensor
            Residual values at the evaluation points.
            Shape depends on the residual_fn implementation, typically:
            - (N,) or (N, 1) for scalar output
            - (N, m) for vector output with m components
        
        Raises:
        -------
        ValueError
            If the equation term name is not found in the model.
            Message: 'Equation term "{name}" not found'
        '''
        eq_term = self.get_equation(name)
        if eq_term is None:
            raise ValueError(f'Equation term "{name}" not found')
        
        if input_data is not None:
            original_data = eq_term.data
            eq_term.update_data(input_data)
        
        self.network.eval()
        with torch.no_grad():
            residual = eq_term.compute_residual(self.network)
        
        if input_data is not None:
            eq_term.update_data(original_data)
        
        return residual
    
    def get_equation_info(self) -> Dict:
        '''
        Get comprehensive information about all defined equations.
        
        This method returns a summary of all registered equation terms, including
        their weights and associated data shapes. Useful for:
        - Inspecting model configuration before training
        - Debugging weight imbalances
        - Verifying data shapes and problem dimensions
        - Monitoring multi-physics problem setup
        
        Returns:
        --------
        Dict
            Dictionary with equation names as keys and info dicts as values.
        '''
        info = {}
        for name, eq_term in self.equation_terms.items():
            info[name] = {
                'weight': eq_term.weight,
                'data_shape': eq_term.data.shape if eq_term.data is not None else None
            }
        return info
    
    def get_training_history(self) -> Dict:
        '''
        Retrieve the complete training history (losses and epochs).
        
        Returns a copy of the internal training history, useful for:
        - Analyzing convergence behavior after training
        - Creating custom plots of loss over time
        - Comparing different training runs
        - Detecting training anomalies or divergence
        - Implementing early stopping or custom training logic
        
        Returns:
        --------
        Dict
            Copy of training history with the following keys:
            - 'loss': List[float] - Total weighted loss values, one per epoch
            - 'epoch': List[int] - Epoch numbers (typically 1, 2, 3, ...)
        '''
        return self.training_history.copy()
    
    def create_optimizer(self, optimizer_name: str = 'Adam', lr: float = 1e-4, **kwargs):
        '''
        Create and set a optimizer for the network.
        
        This method allows setting up an optimizer before calling train(), which enables
        users to create a learning rate scheduler using the optimizer before training starts.
        
        Parameters:
        -----------
        optimizer_name : str, optional
            Name of the optimizer class from torch.optim. Default: 'Adam'
            Common options: 'Adam', 'SGD', 'LBFGS', 'RMSprop', 'AdamW', 'Adamax'
        lr : float, optional
            Learning rate. Default: 1e-4
        kwargs : dict
            Additional parameters specific to the optimizer
            Examples:
            - For 'Adam': weight_decay, betas=(0.9, 0.999), eps, amsgrad
            - For 'SGD': momentum, weight_decay, nesterov
            - For 'LBFGS': max_iter, max_eval, line_search_fn
        
        Returns:
        --------
        torch.optim.Optimizer
            The created optimizer instance (also stored in self.optimizer)
        
        Raises:
        -------
        ValueError
            If the specified optimizer name is not found in torch.optim
        '''
        opt_cls = getattr(torch.optim, optimizer_name, None)
        if opt_cls is None:
            raise ValueError(f'Unknown optimizer name: {optimizer_name}. '
                           f'Available: Adam, SGD, LBFGS, RMSprop, AdamW, Adamax, etc.')
        
        optimizer = opt_cls(self.network.parameters(), lr=lr, **kwargs)
        self.set_optimizer(optimizer)
        print(f'Default optimizer created: {optimizer_name}(lr={lr}, {kwargs})')
        return optimizer
    
    def create_lr_scheduler(self, scheduler_name: str, **kwargs):
        '''
        Create and set a learning rate scheduler for the current optimizer.
        
        This method creates a learning rate scheduler that will be used during training.
        It requires an optimizer to be already set (via create_default_optimizer() or train()).
        
        Parameters:
        -----------
        scheduler_name : str
            Name of the scheduler class from torch.optim.lr_scheduler
            Common options:
            - 'StepLR': Decay LR by gamma every step_size epochs
            - 'ExponentialLR': Decay LR exponentially with gamma each epoch
            - 'CosineAnnealingLR': Annealing with cosine function
            - 'ReduceLROnPlateau': Reduce LR when metric plateaus
            - 'CyclicLR': Cyclically vary learning rate
            - 'LambdaLR': Apply custom function to LR
        kwargs : dict
            Parameters specific to the scheduler
            Examples:
            - 'StepLR': step_size (int), gamma (float, default=0.1)
            - 'ExponentialLR': gamma (float)
            - 'CosineAnnealingLR': T_max (int), eta_min (float, default=0)
            - 'ReduceLROnPlateau': mode ('min'/'max'), factor, patience, threshold, etc.
        
        Returns:
        --------
        torch.optim.lr_scheduler._LRScheduler
            The created scheduler instance
        
        Raises:
        -------
        RuntimeError
            If no optimizer has been set yet
        ValueError
            If the specified scheduler name is not found
        '''
        if self.optimizer is None:
            raise RuntimeError('Optimizer must be set before creating lr_scheduler. '
                             'Call create_default_optimizer() or set_optimizer() first.')
        
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name, None)
        if scheduler_cls is None:
            raise ValueError(f'Unknown scheduler name: {scheduler_name}. '
                           f'Available: StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, '
                           f'CyclicLR, LambdaLR, etc.')
        
        scheduler = scheduler_cls(self.optimizer, **kwargs)
        self.set_lr_scheduler(scheduler)
        print(f'Learning rate scheduler created: {scheduler_name}({kwargs})')
        return scheduler
    
    def register_visualization_callback(self, callback: VisualizationCallback):
        '''
        Register a visualization callback to be executed during training.
        
        Visualization callbacks enable real-time plotting and analysis during training
        without modifying the core training loop. Useful for:
        - Monitoring solution evolution over training
        - Comparing predicted vs analytical solutions
        - Visualizing residuals and error distributions
        - Creating animations of training progress
        - Tracking problem-specific metrics
        
        The callback's visualize() method is called periodically (every log_freq epochs)
        and the returned figures are automatically logged to TensorBoard.
        
        Parameters:
        -----------
        callback : VisualizationCallback
            Visualization callback instance. Must be a subclass of VisualizationCallback
            and implement the visualize() method.
            The callback should define:
            - name: unique identifier for the callback
            - log_freq: execution frequency (every N epochs)
            - visualize(): method that creates and returns matplotlib figures
        '''
        self.visualization_callbacks[callback.name] = callback
        print(f'Visualization callback "{callback.name}" registered with frequency {callback.log_freq}')
    
    def _execute_visualization_callbacks(self, epoch: int, **kwargs):
        '''
        Execute registered visualization callbacks.
        
        This internal method is called from the training loop to execute visualization
        callbacks at the appropriate frequency. It handles:
        - Checking if each callback should run at this epoch
        - Calling the callback's visualize() method
        - Logging returned figures to TensorBoard
        - Closing figures to free memory
        - Error handling to prevent callback failures from breaking training
        
        Parameters:
        -----------
        epoch : int
            Current epoch number (0-indexed). Used to determine if callback should
            execute based on its log_freq setting.
        kwargs : dict
            Additional arguments passed to callback visualize() methods.
            Typically includes:
            - 'loss_dict': Individual loss values for each equation term
            - 'total_loss': Total weighted loss for the epoch
            - Any custom kwargs passed to train() via visualization_kwargs parameter
        '''
        if self.writer is None:
            return
        
        for name, callback in self.visualization_callbacks.items():
            if callback.log_freq > 0 and (epoch + 1) % callback.log_freq == 0:
                try:
                    figures = callback.visualize(self.network, epoch + 1, self.writer, **kwargs)
                    
                    # Log figures to tensorboard
                    if figures is not None and isinstance(figures, dict):
                        for plot_name, fig in figures.items():
                            if isinstance(fig, plt.Figure):
                                # Add matplotlib figure directly to tensorboard
                                self.writer.add_figure(f'Visualization/{name}/{plot_name}', fig, epoch + 1)
                                # Close figure to free memory
                                plt.close(fig)
                        # Flush writer to ensure logging
                        self.writer.flush()
                except Exception as e:
                    print(f'Warning: Visualization callback "{name}" failed with error: {str(e)}')
        
        self.writer.flush()
    
    
    @staticmethod
    def plot_1d_comparison(x_data: np.ndarray, y_pred: np.ndarray, 
                          y_true: np.ndarray = None, y_ref: np.ndarray = None,
                          title: str = '1D Comparison', xlabel: str = 'x', 
                          ylabel: str = 'y') -> plt.Figure:
        '''
        Create a 1D comparison plot of predictions vs ground truth/reference.
        
        Generates a line plot comparing predicted and reference solutions along a 1D
        domain. Useful for visualizing solution accuracy in 1D problems.
        
        Parameters:
        -----------
        x_data : np.ndarray
            x-coordinates (domain points). Shape: (N,)
        y_pred : np.ndarray
            Predicted solution values. Shape: (N,)
        y_true : np.ndarray, optional
            Ground truth/analytical solution. Default: None
            If provided, it is plotted as a dashed red line.
        y_ref : np.ndarray, optional
            Reference/alternative solution. Default: None
            If provided, it is plotted as a dash-dot green line.
        title : str, optional
            Plot title. Default: '1D Comparison'
        xlabel : str, optional
            x-axis label. Default: 'x'
        ylabel : str, optional
            y-axis label. Default: 'y'
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure object that can be displayed, saved, or logged.
        '''
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(x_data, y_pred, 'b-', linewidth=2, label='Prediction')
        if y_true is not None:
            ax.plot(x_data, y_true, 'r--', linewidth=2, label='Ground Truth')
        if y_ref is not None:
            ax.plot(x_data, y_ref, 'g-.', linewidth=2, label='Reference')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_2d_heatmap(data: np.ndarray, title: str = '2D Heatmap', 
                       xlabel: str = 'x', ylabel: str = 'y',
                       cbar_label: str = 'value') -> plt.Figure:
        '''
        Create a 2D heatmap (contour plot) visualization.
        
        Generates a 2D false-color image showing spatial field values. Commonly used
        for visualizing:
        - Solution fields in 2D spatial domains
        - Error distributions over 2D regions
        - Residual magnitude maps
        
        Parameters:
        -----------
        data : np.ndarray
            2D array of values to visualize. Shape: (M, N)
            Each element represents the field value at that grid point.
        title : str, optional
            Plot title. Default: '2D Heatmap'
        xlabel : str, optional
            x-axis label. Default: 'x'
        ylabel : str, optional
            y-axis label. Default: 'y'
        cbar_label : str, optional
            Colorbar label. Default: 'value'
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure object with colorbar for value scale.
        '''
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label, fontsize=12)
        
        return fig
    
    @staticmethod
    def plot_2d_comparison(data_pred: np.ndarray, data_true: np.ndarray = None,
                          title_pred: str = 'Prediction', title_true: str = 'Ground Truth',
                          cbar_label: str = 'value', figsize=(16, 6)) -> plt.Figure:
        '''
        Create side-by-side 2D comparison plots (predicted vs ground truth).
        
        Generates a figure with 1 or 2 subplots to compare 2D field predictions
        with analytical or numerical solutions side-by-side for easy comparison.
        
        Parameters:
        -----------
        data_pred : np.ndarray
            Predicted 2D field data. Shape: (M, N)
        data_true : np.ndarray, optional
            Ground truth/analytical 2D field data. Default: None
            - If provided, creates 2-panel figure
            - If None, creates 1-panel figure showing only prediction
        title_pred : str, optional
            Title for prediction subplot. Default: 'Prediction'
        title_true : str, optional
            Title for ground truth subplot. Default: 'Ground Truth'
        cbar_label : str, optional
            Colorbar label for both subplots. Default: 'value'
        figsize : tuple, optional
            Figure size (width, height) in inches. Default: (16, 6)
            - For 2-panel: (16, 6) is typical
            - For 1-panel: (8, 6) 
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with 1 or 2 subplots and colorbars.
        '''
        if data_true is None:
            n_plots = 1
            figsize = (8, 6)
        else:
            n_plots = 2
        
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        im0 = axes[0].imshow(data_pred, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(title_pred, fontsize=12, fontweight='bold')
        cbar0 = fig.colorbar(im0, ax=axes[0])
        cbar0.set_label(cbar_label, fontsize=10)
        
        if data_true is not None:
            im1 = axes[1].imshow(data_true, aspect='auto', origin='lower', cmap='viridis')
            axes[1].set_title(title_true, fontsize=12, fontweight='bold')
            cbar1 = fig.colorbar(im1, ax=axes[1])
            cbar1.set_label(cbar_label, fontsize=10)
        
        return fig
    
    @staticmethod
    def plot_error_heatmap(data_pred: np.ndarray, data_true: np.ndarray,
                          title: str = 'Absolute Error',
                          cbar_label: str = 'error') -> plt.Figure:
        '''
        Create an error heatmap showing absolute difference between predictions and truth.
        
        Visualizes the spatial distribution of prediction errors
        using a false-color heatmap.
        
        Parameters:
        -----------
        data_pred : np.ndarray
            Predicted 2D field data. Shape: (M, N)
        data_true : np.ndarray
            Ground truth 2D field data. Shape: (M, N)
            Must have same shape as data_pred.
        title : str, optional
            Plot title. Default: 'Absolute Error'
        cbar_label : str, optional
            Colorbar label. Default: 'error'
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with error heatmap and colorbar.
        '''
        error = np.abs(data_pred - data_true)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(error, aspect='auto', origin='lower', cmap='hot')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label, fontsize=12)
        
        return fig
    
    @staticmethod
    def plot_residual_distribution(residuals: np.ndarray, 
                                   title: str = 'Residual Distribution',
                                   xlabel: str = 'Residual', ylabel: str = 'Frequency') -> plt.Figure:
        '''
        Create a histogram visualization of residual distribution.
        
        Parameters:
        -----------
        residuals : np.ndarray
            Array of residual values, typically flattened. Shape: (N,) or can be any shape
            (will be flattened internally).
            Values represent how well each equation is satisfied at evaluation points.
        title : str, optional
            Plot title. Default: 'Residual Distribution'
        xlabel : str, optional
            x-axis label. Default: 'Residual'
        ylabel : str, optional
            y-axis label. Default: 'Frequency'
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with histogram and statistical annotations.
        '''
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(residuals.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax.text(0.7, 0.95, f'Mean: {mean_res:.2e}\nStd: {std_res:.2e}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return fig
    
    def train(self, num_epochs,
              optimizer=None,
              optimizer_cfg: Dict = None,
              lr: float = 1e-4,
              lr_scheduler=None,
              weights_override: Dict[str, float] = None,
              print_loss: bool = True,
              print_loss_freq: int = 1,
              tensorboard_logdir: str = None,
              save_final_model: bool = False,
              final_model_path: str = None,
              checkpoint_dir: str = None,
              checkpoint_freq: int = 1,
              resume_from: str = None,
              batch_size: Optional[int] = None,
              shuffle_batches: bool = False,
              visualization_kwargs: Dict = None):
        '''
        Train the PINN model with advanced options and optional batch processing.

        Parameters:
        -----------
        - num_epochs (int): number of epochs to train
        - optimizer (torch.optim.Optimizer): pre-built optimizer instance (optional)
        - optimizer_cfg (dict): configuration to build optimizer. Example:
              {'name': 'Adam', 'params': {'lr': 1e-4, 'weight_decay': 0}}
        - lr (float): default learning rate if optimizer is not specified (default: 1e-4)
        - lr_scheduler: optional pre-built learning rate scheduler instance.
              Note: Can also be created via create_lr_scheduler() before calling train()
        - weights_override (dict): temporarily override loss weights for specific terms
        - print_loss (bool): whether to print loss each epoch
        - print_loss_freq (int): print loss every N epochs
        - tensorboard_logdir (str): path for tensorboard logs
        - save_final_model (bool): whether to save final model
        - final_model_path (str): path to save final model
        - checkpoint_dir (str): directory for epoch checkpoints
        - checkpoint_freq (int): save checkpoint every N epochs
        - resume_from (str): path to checkpoint to resume from
        - batch_size (int): batch size for training data loading. If None (default), loads all data at once.
        - shuffle_batches (bool): whether to shuffle batches during training
        - visualization_kwargs (dict): additional arguments to pass to visualization callbacks
        
        Optimizer Setup Approaches (in order of priority):
        --------------------------------------------------
        1. Pre-set via create_optimizer(): Most flexible approach
           >>> pinn.create_default_optimizer('Adam', lr=1e-3)
           >>> pinn.train(num_epochs=1000)
        
        2. Pre-set via set_optimizer(): Direct optimizer assignment
           >>> opt = torch.optim.SGD(pinn.network.parameters(), lr=0.01, momentum=0.9)
           >>> pinn.set_optimizer(opt)
           >>> pinn.train(num_epochs=1000)
        
        3. Pass optimizer parameter to train()
           >>> pinn.train(num_epochs=1000, optimizer=custom_optimizer)
        
        4. Pass optimizer_cfg parameter to train()
           >>> pinn.train(num_epochs=1000, optimizer_cfg={'name': 'SGD', 'params': {'lr': 0.01}})
        
        5. Default: Adam with specified lr
           >>> pinn.train(num_epochs=1000, lr=1e-3)
        
        Learning Rate Scheduler Setup Approaches:
        -----------------------------------------
        1. Pre-created via create_lr_scheduler(): Recommended approach
           >>> pinn.create_optimizer('Adam', lr=1e-3)
           >>> pinn.create_lr_scheduler('StepLR', step_size=500, gamma=0.5)
           >>> pinn.train(num_epochs=2000)
        
        2. Pass pre-built scheduler to train()
           >>> pinn.create_optimizer('Adam', lr=1e-3)
           >>> scheduler = torch.optim.lr_scheduler.StepLR(pinn.optimizer, step_size=500, gamma=0.5)
           >>> pinn.train(num_epochs=2000, lr_scheduler=scheduler)
        '''
        # Validate equations are defined
        if len(self.equation_terms) == 0:
            raise RuntimeError('No equations defined. Call _define_loss_terms() first.')
        
        # Set default loss function
        if self.loss_func is None:
            self.set_loss_func(nn.MSELoss())

        # Setup Tensorboard
        if tensorboard_logdir is not None and tensorboard_logdir != "":
            os.makedirs(tensorboard_logdir, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_logdir)
            print(f'Tensorboard writer created at: {tensorboard_logdir}')
            
            # Log equation information
            eq_info = self.get_equation_info()
            for name, info in eq_info.items():
                print(f'  Equation: {name}, Weight: {info["weight"]:.4f}')

        # Setup optimizer
        # Priority: pre-built optimizer > optimizer parameter > optimizer_cfg > default Adam
        if self.optimizer is None:  # Only setup if not already set via create_optimizer()
            if optimizer is not None:
                self.set_optimizer(optimizer)
            elif optimizer_cfg is not None and isinstance(optimizer_cfg, dict):
                name = optimizer_cfg.get('name', 'Adam')
                params = optimizer_cfg.get('params', {})
                if 'lr' not in params and lr is not None:
                    params['lr'] = lr

                opt_cls = getattr(torch.optim, name, None)
                if opt_cls is None:
                    raise ValueError(f'Unknown optimizer name: {name}')
                self.set_optimizer(opt_cls(self.network.parameters(), **params))
            else:
                # Default optimizer
                self.set_optimizer(torch.optim.Adam(self.network.parameters(), lr=lr))
        else:
            # Optimizer already set, just override if explicitly provided
            if optimizer is not None:
                self.set_optimizer(optimizer)
        
        # Setup learning rate scheduler
        # Priority: pre-created scheduler > lr_scheduler parameter > lr_scheduler_cfg
        if lr_scheduler is not None:
            # Use provided scheduler
            self.set_lr_scheduler(lr_scheduler)
            print(f'Using provided learning rate scheduler')
        elif self.lr_scheduler is None:
            # Create scheduler if not already created via create_lr_scheduler()
            # This allows backward compatibility where lr_scheduler_cfg can be passed to train()
            pass  # lr_scheduler remains None if not provided

        # Setup checkpoints
        if checkpoint_dir is not None and checkpoint_dir != "":
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir

        # Resume if needed
        self.start_epoch = 0
        if resume_from is not None and resume_from != "":
            if os.path.exists(resume_from):
                ckpt = torch.load(resume_from, map_location=DEVICE())
                if 'model' in ckpt:
                    self.network.load_state_dict(ckpt['model'])
                if 'optimizer' in ckpt and self.optimizer is not None:
                    try:
                        self.optimizer.load_state_dict(ckpt['optimizer'])
                    except Exception:
                        print('Warning: Failed to load optimizer state (incompatible)')
                if 'epoch' in ckpt:
                    self.start_epoch = ckpt['epoch'] + 1
                print(f'Resumed from checkpoint {resume_from} at epoch {self.start_epoch}')
            else:
                raise FileNotFoundError(f'Resume checkpoint not found: {resume_from}')

        # Setup batch loading if batch_size is specified
        dataloaders = {}
        if batch_size is not None and batch_size > 0:
            print(f'Batch loading enabled with batch_size={batch_size}, shuffle={shuffle_batches}')
            for name, eq_term in self.equation_terms.items():
                dataloader = eq_term.create_dataloader(batch_size, shuffle=shuffle_batches)
                if dataloader is not None:
                    dataloaders[name] = dataloader
                    num_batches = len(dataloader)
                    print(f'  {name}: {num_batches} batches')
                else:
                    print(f'  {name}: no data, skipping batch loading')
        
        # Training loop
        total_epochs = self.start_epoch + num_epochs
        print(f'Starting training from epoch {self.start_epoch} to {total_epochs}')
        
        # Prepare visualization kwargs
        if visualization_kwargs is None:
            visualization_kwargs = {}
        
        for epoch in range(self.start_epoch, total_epochs):
            # Check if using batch loading
            if dataloaders and len(dataloaders) > 0:
                # Batched training
                self._train_epoch_batched(
                    epoch, total_epochs, dataloaders, weights_override, 
                    print_loss, print_loss_freq, self.lr_scheduler, visualization_kwargs
                )
            else:
                # Non-batched training (original behavior)
                self._train_epoch(
                    epoch, total_epochs, weights_override, 
                    print_loss, print_loss_freq, self.lr_scheduler, visualization_kwargs
                )

            # Checkpointing
            if checkpoint_dir is not None and checkpoint_freq > 0 and ((epoch+1) % checkpoint_freq == 0):
                ckpt_file = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'model': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss_history': self.training_history
                }, ckpt_file)

        # Final save
        if save_final_model:
            if final_model_path is None or final_model_path == "":
                if checkpoint_dir is not None:
                    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
                else:
                    final_model_path = 'final_model.pth'

            torch.save({
                'model': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': total_epochs - 1,
                'loss_history': self.training_history
            }, final_model_path)
            print(f'Final model saved to: {final_model_path}')

        # Close tensorboard writer
        if self.writer is not None:
            self.writer.close()
            print('Tensorboard writer closed')
        
        print('Training completed!')

    def _train_epoch(self, epoch: int, total_epochs: int, 
                     weights_override: Dict[str, float],
                     print_loss: bool, print_loss_freq: int,
                     lr_scheduler, visualization_kwargs: Dict):
        '''
        Train for one epoch without batch loading (original behavior).
        
        Parameters:
        - epoch (int): Current epoch number
        - total_epochs (int): Total number of epochs
        - weights_override (Dict): Weight overrides
        - print_loss (bool): Whether to print loss
        - print_loss_freq (int): Print frequency
        - lr_scheduler: Learning rate scheduler
        - visualization_kwargs (Dict): Visualization arguments
        '''
        # Compute loss
        total_loss, loss_dict = self.calc_loss(weights_override=weights_override)

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer.step()

        # Adaptive weight adjustment
        if self.adaptive_weights and (epoch + 1) % self.weight_update_freq == 0:
            self._compute_adaptive_weights(loss_dict)

        # Learning rate scheduling
        if lr_scheduler is not None:
            try:
                lr_scheduler.step()
            except Exception:
                lr_scheduler(epoch)

        # Logging
        if print_loss and print_loss_freq > 0 and (epoch+1) % print_loss_freq == 0:
            loss_str = f'Epoch [{epoch+1}/{total_epochs}], Total Loss: {total_loss.item():g}'
            for name, loss in loss_dict.items():
                loss_str += f', {name}: {loss.item():g}'
            print(loss_str)

        # Tensorboard logging
        if self.writer is not None:
            self.writer.add_scalar('Loss/total', total_loss.item(), epoch+1)
            for name, loss in loss_dict.items():
                self.writer.add_scalar(f'Loss/{name}', loss.item(), epoch+1)
            self.writer.flush()
        
        # Execute visualization callbacks
        self._execute_visualization_callbacks(epoch, loss_dict=loss_dict, total_loss=total_loss, **visualization_kwargs)

        # Training history
        self.training_history['loss'].append(total_loss.item())
        self.training_history['epoch'].append(epoch + 1)

    def _train_epoch_batched(self, epoch: int, total_epochs: int,
                             dataloaders: Dict[str, DataLoader],
                             weights_override: Dict[str, float],
                             print_loss: bool, print_loss_freq: int,
                             lr_scheduler, visualization_kwargs: Dict):
        '''
        Train for one epoch with batch loading.
        
        Parameters:
        - epoch (int): Current epoch number
        - total_epochs (int): Total number of epochs
        - dataloaders (Dict): Dictionary mapping equation names to DataLoaders
        - weights_override (Dict): Weight overrides
        - print_loss (bool): Whether to print loss
        - print_loss_freq (int): Print frequency
        - lr_scheduler: Learning rate scheduler
        - visualization_kwargs (Dict): Visualization arguments
        '''
        # Determine the number of batches (use the max across all dataloaders)
        num_batches = max(len(dl) for dl in dataloaders.values()) if dataloaders else 1
        
        epoch_loss = 0.0
        epoch_loss_dict = {name: 0.0 for name in dataloaders.keys()}
        
        # Create iterators that cycle if needed
        iterators = {name: iter(dl) for name, dl in dataloaders.items()}
        
        for batch_idx in range(num_batches):
            # Prepare batch data
            batch_data = {}
            for name, iterator in iterators.items():
                try:
                    batch = next(iterator)
                except StopIteration:
                    # Restart iterator if exhausted
                    iterator = iter(dataloaders[name])
                    iterators[name] = iterator
                    batch = next(iterator)
                
                # Extract data from batch (TensorDataset returns tuple)
                if isinstance(batch, (tuple, list)) and len(batch) > 0:
                    batch_data[name] = batch[0].to(DEVICE())
                else:
                    batch_data[name] = batch.to(DEVICE())
            
            # Compute loss for this batch
            total_loss, loss_dict = self.calc_loss(weights_override=weights_override, batch_data=batch_data)

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.optimizer.step()

            # Accumulate losses
            epoch_loss += total_loss.item()
            for name, loss in loss_dict.items():
                epoch_loss_dict[name] += loss.item()

        # Average losses over batches
        epoch_loss /= num_batches
        for name in epoch_loss_dict:
            epoch_loss_dict[name] /= num_batches

        # Adaptive weight adjustment
        if self.adaptive_weights and (epoch + 1) % self.weight_update_freq == 0:
            self._compute_adaptive_weights(epoch_loss_dict)

        # Learning rate scheduling
        if lr_scheduler is not None:
            try:
                lr_scheduler.step()
            except Exception:
                lr_scheduler(epoch)

        # Logging
        if print_loss and print_loss_freq > 0 and (epoch+1) % print_loss_freq == 0:
            loss_str = f'Epoch [{epoch+1}/{total_epochs}], Total Loss: {epoch_loss:g}'
            for name, loss in epoch_loss_dict.items():
                loss_str += f', {name}: {loss:g}'
            print(loss_str)

        # Tensorboard logging
        if self.writer is not None:
            self.writer.add_scalar('Loss/total', epoch_loss, epoch+1)
            for name, loss in epoch_loss_dict.items():
                self.writer.add_scalar(f'Loss/{name}', loss, epoch+1)
            self.writer.flush()
        
        # Execute visualization callbacks
        self._execute_visualization_callbacks(epoch, loss_dict=epoch_loss_dict, total_loss=epoch_loss, **visualization_kwargs)

        # Training history
        self.training_history['loss'].append(epoch_loss)
        self.training_history['epoch'].append(epoch + 1)





