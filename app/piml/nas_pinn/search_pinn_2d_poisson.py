"""2D Neural Architecture Search for Physics-Informed Neural Networks (NAS-PINN).

This module demonstrates the application of NAS-PINN to solve a 2D Poisson equation
using automatic architecture search. It provides example implementations of PINN models
with architecture search capability and comprehensive visualization tools for monitoring
training progress and solution quality.

References
----------
[1] Y. Wang, L. Zhong, "NAS-PINN: Neural architecture search-guided physics-informed neural
    network for solving PDEs," Journal of Computational Physics, vol. 496, p. 112603, 2024.
"""

import sys
sys.path.append('.')

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Callable, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ai4plasma.piml.geo import GeoPoly2D
from ai4plasma.piml.pinn import PINN, VisualizationCallback, EquationTerm
from ai4plasma.core.network import RelaxFNN
from ai4plasma.utils.math import df_dX
from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import set_seed, numpy2torch, Timer
from ai4plasma.config import DEVICE, REAL
from ai4plasma.piml.nas_pinn import NasPINN


class Example2DNasPINN(PINN):
    """
    2D Physics-Informed Neural Network for Poisson equation with NAS capability.
    
    This class implements a PINN model for solving the 2D Poisson equation on a unit square
    domain with Dirichlet boundary conditions. It extends the base PINN class with support
    for neural architecture search through separate loss term definitions for architecture
    search and standard training phases.
    
    Attributes
    ----------
    geo : GeoPoly2D
        Geometry object representing the unit square domain [0, 1] × [0, 1]
    n_domain : int
        Number of interior collocation points (per dimension for tensor product grid)
    n_boundary_list : list of int
        Number of collocation points for each boundary edge
    n_domain_archi : int
        Number of interior collocation points for architecture search phase
    n_boundary_list_archi : list of int
        Number of boundary collocation points for architecture search phase
    equation_terms_archi : dict
        Dictionary storing loss terms specific to architecture search phase
    
    Problem Definition
    ------------------
    **PDE (Poisson equation)**:
    
      ∂²u/∂x² + ∂²u/∂y² = -2π²cos(πx)cos(πy)
    
    **Domain**: Ω = [0, 1] × [0, 1] (unit square)
    
    **Boundary Conditions**: Dirichlet BCs (u = cos(πx)cos(πy) on all boundaries)
    
    **Analytical Solution**: u(x,y) = cos(πx)cos(πy)
    
    Loss Function Structure
    -----------------------
    **Standard Training Loss**:
    
      L_total = Σ w_i * L_i where w_pde=1.0, w_bc=1.0
    
    **Architecture Search Loss**:
    
      Uses separate point sets for efficiency.
    """
    
    def __init__(self,
                 network: RelaxFNN,
                 n_domain: int = 50,
                 n_boundary_list: list = [50, 50, 50, 50],
                 n_domain_archi: int = 20,
                 n_boundary_list_archi: list = [20, 20, 20, 20],
                ):
        """
        Initialize the 2D PINN model for Poisson equation with NAS.
        
        Sets up the geometry domain, collocation point sampling, and loss term structure
        for both standard PINN training and architecture search phases.
        
        Parameters
        ----------
        network : RelaxFNN
            Relaxed feed-forward neural network for solution approximation.
            Expected: 2 inputs (x, y) → 1 output (u).
        n_domain : int, default=50
            Number of interior collocation points per dimension for training.
        n_boundary_list : list of int, default=[50, 50, 50, 50]
            Number of points for each boundary edge.
        n_domain_archi : int, default=20
            Number of interior points for architecture search (typically 40-50% of n_domain).
        n_boundary_list_archi : list of int, default=[20, 20, 20, 20]
            Number of boundary points for architecture search.
        """
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self.geo = GeoPoly2D(points)

        self.n_domain = n_domain
        self.n_boundary_list = n_boundary_list
        self.n_domain_archi = n_domain_archi
        self.n_boundary_list_archi = n_boundary_list_archi

        self.equation_terms_archi: Dict[str, EquationTerm] = {}

        super().__init__(network)

        self._define_loss_terms_archi()
        

    @staticmethod    
    def _pde_residual(net: nn.Module, xy: torch.Tensor) -> torch.Tensor:
        """
        Compute the PDE residual for the Poisson equation.
        
        Computes ∇²u - f(x,y) using automatic differentiation through the network.
        For the Poisson equation ∂²u/∂x² + ∂²u/∂y² = -2π²cos(πx)cos(πy),
        the residual is:
        
          R_pde = ∂²u/∂x² + ∂²u/∂y² + 2π²cos(πx)cos(πy)
        
        Parameters
        ----------
        net : nn.Module
            Neural network computing u = net(xy).
        xy : torch.Tensor, shape (N, 2)
            Interior collocation points.
        
        Returns
        -------
        residual : torch.Tensor, shape (N, 1)
            PDE residual at each collocation point.
        """
        u = net(xy)
        
        # First derivatives
        u_xy = df_dX(u, xy)  # shape: (N, 2)
        u_x = u_xy[:, 0:1]  # ∂u/∂x
        u_y = u_xy[:, 1:2]  # ∂u/∂y
        
        # Second derivatives
        u_xx = df_dX(u_x, xy)[:, 0:1]  # ∂²u/∂x²
        u_yy = df_dX(u_y, xy)[:, 1:2]  # ∂²u/∂y²
        
        # Source term: f(x,y) = -2π²cos(πx)cos(πy)
        x, y = xy[:, 0:1], xy[:, 1:2]
        f = -2.0 * np.pi**2 * torch.cos(np.pi*x) * torch.cos(np.pi*y)
        
        # PDE residual
        residual = u_xx + u_yy - f
        
        return residual
    
    @staticmethod
    def _bc_residual(net: nn.Module, xy: torch.Tensor) -> torch.Tensor:
        """
        Compute the boundary condition residual.
        
        For Dirichlet BC u = cos(πx)cos(πy) on boundaries, the residual is:
        
          R_bc = u(x,y) - cos(πx)cos(πy)
        
        Parameters
        ----------
        net : nn.Module
            Neural network computing u = net(xy).
        xy : torch.Tensor, shape (N, 2)
            Boundary collocation points.
        
        Returns
        -------
        residual : torch.Tensor, shape (N, 1)
            Boundary condition residual.
        """
        u = net(xy)

        x = xy[:, 0]
        y = xy[:, 1]
        b = torch.cos(np.pi*x)*torch.cos(np.pi*y)

        return u - b  # BC: u = cos(πx)cos(πy) on boundaries
    
    @staticmethod
    def _pde_residual_archi(net: nn.Module, xy: torch.Tensor) -> torch.Tensor:
        """
        Compute the PDE residual for architecture search phase.
        
        Uses a simplified residual formulation for efficient architecture updates:
        
          R_archi = u(x,y) - cos(πx)cos(πy)
        
        This variant encourages learning the analytical solution directly,
        providing a different optimization landscape for architecture search.

        For more complex problems, this could be a weighted summation of the
        PDE residual and a provided reference solution.
        
        Parameters
        ----------
        net : nn.Module
            Neural network computing u = net(xy).
        xy : torch.Tensor, shape (N, 2)
            Interior collocation points.
        
        Returns
        -------
        residual : torch.Tensor, shape (N, 1)
            Modified residual for architecture search.
        """
        u = net(xy)
        pi = np.pi

        x = xy[:, 0]
        y = xy[:, 1]

        f = torch.cos(pi*x)*torch.cos(pi*y)

        return u - f
    
    def _define_loss_terms(self):
        """
        Define standard loss terms for PINN training on the Poisson problem.
        
        Sets up loss terms for domain (PDE residual) and boundary conditions.
        Uses collocation point sets specified during initialization.
        
        Loss Term Structure
        -------------------
        - **Domain**: Interior PDE residual at n_domain×n_domain points, weight=1.0
        - **Boundary1-4**: BC residuals at each edge, weight=1.0 each
        
        Collocation Point Distribution
        --------------------------------
        - Interior: n_domain × n_domain points covering unit square uniformly
        - Boundary: Four edges with n_boundary_list[i] points per edge
        
        Weight Justification
        --------------------
        - PDE weight (1.0): Baseline importance
        - BC weight (1.0): Equal weight ensures balanced training of PDE and BC
        """
        
        xy_domain = self.geo.sample_domain(self.n_domain)
        xy_bc = self.geo.sample_boundary(self.n_boundary_list)
        
        # Add loss terms to PINN
        self.add_equation('Domain', self._pde_residual, weight=1.0, data=xy_domain)
        self.add_equation('Boundary1', self._bc_residual, weight=1.0, data=xy_bc[0])
        self.add_equation('Boundary2', self._bc_residual, weight=1.0, data=xy_bc[1])
        self.add_equation('Boundary3', self._bc_residual, weight=1.0, data=xy_bc[2])
        self.add_equation('Boundary4', self._bc_residual, weight=1.0, data=xy_bc[3])

    def _define_loss_terms_archi(self):
        """
        Define loss terms for the architecture search phase.
        
        Sets up loss terms with reduced collocation point sets for efficient
        architecture parameter updates. Uses separate equation_terms_archi dictionary.
        """
        xy_domain = self.geo.sample_domain(self.n_domain_archi)
        xy_bc = self.geo.sample_boundary(self.n_boundary_list_archi)
        
        # Add architecture search loss term
        self.add_equation_archi('Domain_archi', self._pde_residual_archi, weight=1.0, data=xy_domain)
        self.add_equation_archi('Boundary1_archi', self._bc_residual, weight=1.0, data=xy_bc[0])
        self.add_equation_archi('Boundary2_archi', self._bc_residual, weight=1.0, data=xy_bc[1])
        self.add_equation_archi('Boundary3_archi', self._bc_residual, weight=1.0, data=xy_bc[2])
        self.add_equation_archi('Boundary4_archi', self._bc_residual, weight=1.0, data=xy_bc[3])

    def add_equation_archi(self, name: str, residual_fn: Callable, weight: float = 1.0, data: torch.Tensor = None):
        """
        Add a loss term to the architecture search loss function.
        
        Registers a residual function and collocation points as a loss term
        for architecture search. Multiple terms combined in calc_loss_archi().
        
        Parameters
        ----------
        name : str
            Unique identifier for loss term (e.g., 'Domain_archi').
        residual_fn : callable
            Function computing residual: residual_fn(network, data) → tensor(N,1).
        weight : float, default=1.0
            Scaling factor for this loss term.
        data : torch.Tensor, optional
            Collocation points, shape (N, 2).
        """
        self.equation_terms_archi[name] = EquationTerm(name, residual_fn, weight, data)
    
    def set_loss_func_archi(self, loss_func: Callable):
        """
        Set the loss function for architecture search phase.
        
        Parameters
        ----------
        loss_func : callable
            Loss criterion mapping residuals to scalar values.
            Typically nn.MSELoss() for least-squares minimization.
        """
        self.loss_func_archi = loss_func
    
    def calc_loss_archi(self, weights_override: Dict[str, float] = None, batch_data: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the total loss for the architecture search phase.
        
        Evaluates all architecture search loss terms, applies weights, and returns
        the weighted sum for architecture parameter updates.
        
        Parameters
        ----------
        weights_override : dict, optional
            Override weights for specific terms {term_name: weight}.
        batch_data : dict, optional
            Alternative collocation point data {term_name: data_tensor}.
        
        Returns
        -------
        total_loss : torch.Tensor
            Scalar tensor representing total weighted architecture loss.
        loss_dict : dict
            Individual loss term values {term_name: loss_value}.
        """
        if len(self.equation_terms_archi) == 0:
            raise RuntimeError("No architecture search loss terms defined. Please call _define_loss_terms_archi() to set up the loss terms.")
        
        loss_dict = {}
        weighted_losses = []

        # Compute residual and loss for each equation term
        for name, eq_term in self.equation_terms_archi.items():
            # Use batch data if provided, otherwise use stored data
            term_batch_data = batch_data.get(name) if batch_data is not None else None
            residual = eq_term.compute_residual(self.network, term_batch_data)
            individual_loss = self.loss_func_archi(residual, torch.zeros_like(residual))
            loss_dict[name] = individual_loss

            # Get weight (use ovverride if provided)
            if weights_override and name in weights_override:
                weight = weights_override[name]
            else:  
                weight = eq_term.weight

            weighted_losses.append(weight * individual_loss)

        total_loss = sum(weighted_losses)

        return total_loss, loss_dict


class Example2DVisCallback(VisualizationCallback):
    """
    Visualization callback for 2D PINN training with TensorBoard logging.
    
    Generates comprehensive visualizations of the PINN solution during training
    and stores history for post-training animation generation.
    
    Attributes
    ----------
    x_domain : np.ndarray, shape (Nx,)
        1D array of x coordinates for evaluation grid
    y_domain : np.ndarray, shape (Ny,)
        1D array of y coordinates for evaluation grid
    u_true_fn : callable, optional
        Function computing analytical solution u_true_fn(X, Y)
    history : dict
        Training history: 'epoch', 'loss', 'u_pred'
    """
    
    def __init__(self, x_domain: np.ndarray, y_domain: np.ndarray,
                 u_true_fn=None, log_freq: int = 100, history_freq: int = None):
        """
        Initialize the 2D visualization callback for PINN training monitoring.
        
        Sets up evaluation grid and history tracking for real-time visualization.
        
        Parameters
        ----------
        x_domain : np.ndarray, shape (Nx,)
            1D array of x coordinates for evaluation grid.
        y_domain : np.ndarray, shape (Ny,)
            1D array of y coordinates for evaluation grid.
        u_true_fn : callable, optional
            Analytical solution function u_true_fn(X, Y) → U.
            If None, skips comparison visualizations.
        log_freq : int, default=100
            Logging frequency for TensorBoard (every N epochs).
        history_freq : int, optional
            Recording frequency for snapshots (every N epochs).
            Defaults to log_freq. Higher values reduce memory usage.
        """
        super().__init__(name='2D_Solution', log_freq=log_freq)
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.u_true_fn = u_true_fn
        self.history_freq = history_freq if history_freq is not None else log_freq
        
        # Training history for animation
        self.history = {
            'epoch': [],
            'loss': [],
            'u_pred': []
        }
        
        # Create meshgrid for evaluation
        self.X, self.Y = np.meshgrid(x_domain, y_domain)
        self.XY = np.column_stack([self.X.ravel(), self.Y.ravel()])
        
    def visualize(self, network: nn.Module, epoch: int, writer: SummaryWriter, 
                  total_loss: float = None, **kwargs):
        """
        Generate and log 2D visualizations of the PINN solution.
        
        Evaluates network on grid, generates comparative visualizations,
        and logs to TensorBoard. Also records snapshots for animation.
        
        Parameters
        ----------
        network : nn.Module
            Neural network model for solution prediction.
        epoch : int
            Current training epoch number.
        writer : SummaryWriter
            TensorBoard writer for logging figures.
        total_loss : torch.Tensor, optional
            Total training loss for history tracking.
        **kwargs : dict
            Additional arguments (unused).
        
        Returns
        -------
        figures : dict
            Generated matplotlib figures:
            - 'heatmap': Solution field heatmap
            - 'contour': Contour plot
            - 'comparison': Prediction vs analytical (if available)
            - 'error': Absolute error map (if available)
        """
        network.eval()
        
        # Evaluate network on 2D grid
        with torch.no_grad():
            xy_tensor = numpy2torch(self.XY, require_grad=False)
            u_pred = network(xy_tensor).cpu().numpy().reshape(self.X.shape)
        
        # Compute analytical solution if available
        u_true = None
        if self.u_true_fn is not None:
            u_true = self.u_true_fn(self.X, self.Y)
        
        # Record history for animation
        if epoch % self.history_freq == 0:
            self.history['epoch'].append(epoch)
            self.history['loss'].append(total_loss.item() if total_loss is not None else 0.0)
            self.history['u_pred'].append(u_pred.copy())
        
        # Create visualizations
        figures = {}
        
        # 1. Heatmap of prediction
        fig_heatmap = self._plot_heatmap(u_pred, epoch)
        figures['heatmap'] = fig_heatmap
        
        # 2. Contour plot
        fig_contour = self._plot_contour(u_pred, epoch)
        figures['contour'] = fig_contour
        
        # 3. Comparison and error (if analytical solution available)
        if u_true is not None:
            fig_comparison = self._plot_comparison(u_pred, u_true, epoch)
            figures['comparison'] = fig_comparison
            
            fig_error = self._plot_error(u_pred, u_true, epoch)
            figures['error'] = fig_error
        
        return figures
    
    def _plot_heatmap(self, u: np.ndarray, epoch: int):
        """Create heatmap visualization of the solution field."""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(u, extent=[0, 1, 0, 1], origin='lower', 
                      cmap='viridis', aspect='auto')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Solution u(x,y) - Epoch {epoch}', fontsize=14)
        plt.colorbar(im, ax=ax, label='u')
        plt.tight_layout()
        return fig
    
    def _plot_contour(self, u: np.ndarray, epoch: int):
        """Create contour plot of the solution field."""
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(self.X, self.Y, u, levels=20, cmap='viridis')
        ax.contour(self.X, self.Y, u, levels=10, colors='black', 
                  linewidths=0.5, alpha=0.4)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Contour Plot - Epoch {epoch}', fontsize=14)
        plt.colorbar(contour, ax=ax, label='u')
        plt.tight_layout()
        return fig
    
    def _plot_comparison(self, u_pred: np.ndarray, u_true: np.ndarray, epoch: int):
        """Create side-by-side comparison of prediction and analytical solution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Prediction
        im1 = axes[0].imshow(u_pred, extent=[0, 1, 0, 1], origin='lower',
                            cmap='viridis', aspect='auto')
        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel('y', fontsize=12)
        axes[0].set_title(f'Prediction - Epoch {epoch}', fontsize=12)
        plt.colorbar(im1, ax=axes[0], label='u')
        
        # Analytical solution
        im2 = axes[1].imshow(u_true, extent=[0, 1, 0, 1], origin='lower',
                            cmap='viridis', aspect='auto')
        axes[1].set_xlabel('x', fontsize=12)
        axes[1].set_ylabel('y', fontsize=12)
        axes[1].set_title('Analytical Solution', fontsize=12)
        plt.colorbar(im2, ax=axes[1], label='u')
        
        plt.tight_layout()
        return fig
    
    def _plot_error(self, u_pred: np.ndarray, u_true: np.ndarray, epoch: int):
        """Create error heatmap showing absolute error."""
        error = np.abs(u_pred - u_true)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(error, extent=[0, 1, 0, 1], origin='lower',
                      cmap='hot', aspect='auto')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Absolute Error - Epoch {epoch}', fontsize=14)
        plt.colorbar(im, ax=ax, label='|u_pred - u_true|')
        plt.tight_layout()
        return fig
    

def analytical_solution(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the analytical solution of the 2D Poisson equation.
    
    Returns the exact solution u(x,y) = cos(πx)cos(πy) to:
    
      ∂²u/∂x² + ∂²u/∂y² = -2π²cos(πx)cos(πy)
    
    with Dirichlet BC u = cos(πx)cos(πy) on ∂Ω.
    
    Parameters
    ----------
    X : np.ndarray, shape (Ny, Nx)
        2D meshgrid of x coordinates.
    Y : np.ndarray, shape (Ny, Nx)
        2D meshgrid of y coordinates.
    
    Returns
    -------
    u : np.ndarray, shape (Ny, Nx)
        Analytical solution field.
    """
    return np.cos(np.pi * X) * np.cos(np.pi * Y)


def plot_final_comparison(pinn: PINN, x_eval: np.ndarray, y_eval: np.ndarray,
                          u_true_fn=None, save_path: str = None):
    """
    Create and save comprehensive comparison visualization after training.
    
    Generates multi-panel figure comparing PINN predictions with analytical solution
    (if available) for final validation.
    
    Parameters
    ----------
    pinn : PINN
        Trained PINN model.
    x_eval : np.ndarray, shape (Nx,)
        1D array of x coordinates for evaluation grid.
    y_eval : np.ndarray, shape (Ny,)
        1D array of y coordinates for evaluation grid.
    u_true_fn : callable, optional
        Function computing analytical solution u_true_fn(X, Y) → U.
    save_path : str, optional
        File path to save the figure.
    """
    pinn.network.eval()
    
    # Create meshgrid
    X, Y = np.meshgrid(x_eval, y_eval)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    
    # Prediction
    with torch.no_grad():
        xy_tensor = numpy2torch(XY, require_grad=False)
        u_pred = pinn.network(xy_tensor).cpu().numpy().reshape(X.shape)
    
    # Analytical solution
    u_true = u_true_fn(X, Y) if u_true_fn is not None else None
    
    # Create figure
    if u_true is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Prediction heatmap
        im1 = axes[0, 0].imshow(u_pred, extent=[0, 1, 0, 1], origin='lower',
                               cmap='viridis', aspect='auto')
        axes[0, 0].set_xlabel('x', fontsize=12)
        axes[0, 0].set_ylabel('y', fontsize=12)
        axes[0, 0].set_title('PINN Prediction', fontsize=14)
        plt.colorbar(im1, ax=axes[0, 0], label='u')
        
        # Prediction contour
        contour1 = axes[0, 1].contourf(X, Y, u_pred, levels=20, cmap='viridis')
        axes[0, 1].contour(X, Y, u_pred, levels=10, colors='black',
                          linewidths=0.5, alpha=0.4)
        axes[0, 1].set_xlabel('x', fontsize=12)
        axes[0, 1].set_ylabel('y', fontsize=12)
        axes[0, 1].set_title('Contour Plot', fontsize=14)
        plt.colorbar(contour1, ax=axes[0, 1], label='u')
        
        # Analytical solution
        im2 = axes[1, 0].imshow(u_true, extent=[0, 1, 0, 1], origin='lower',
                               cmap='viridis', aspect='auto')
        axes[1, 0].set_xlabel('x', fontsize=12)
        axes[1, 0].set_ylabel('y', fontsize=12)
        axes[1, 0].set_title('Analytical Solution', fontsize=14)
        plt.colorbar(im2, ax=axes[1, 0], label='u')
        
        # Error
        error = np.abs(u_pred - u_true)
        im3 = axes[1, 1].imshow(error, extent=[0, 1, 0, 1], origin='lower',
                               cmap='hot', aspect='auto')
        axes[1, 1].set_xlabel('x', fontsize=12)
        axes[1, 1].set_ylabel('y', fontsize=12)
        axes[1, 1].set_title(f'Absolute Error (Max: {error.max():.2e})', fontsize=14)
        plt.colorbar(im3, ax=axes[1, 1], label='|u_pred - u_true|')
        
        # Add metrics
        l2_error = np.sqrt(np.mean(error**2))
        max_error = np.max(error)
        fig.suptitle(f'Final Comparison | L2 Error: {l2_error:.2e} | Max Error: {max_error:.2e}',
                    fontsize=16, y=0.995)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prediction heatmap
        im1 = axes[0].imshow(u_pred, extent=[0, 1, 0, 1], origin='lower',
                            cmap='viridis', aspect='auto')
        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel('y', fontsize=12)
        axes[0].set_title('PINN Prediction', fontsize=14)
        plt.colorbar(im1, ax=axes[0], label='u')
        
        # Prediction contour
        contour1 = axes[1].contourf(X, Y, u_pred, levels=20, cmap='viridis')
        axes[1].contour(X, Y, u_pred, levels=10, colors='black',
                       linewidths=0.5, alpha=0.4)
        axes[1].set_xlabel('x', fontsize=12)
        axes[1].set_ylabel('y', fontsize=12)
        axes[1].set_title('Contour Plot', fontsize=14)
        plt.colorbar(contour1, ax=axes[1], label='u')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Final comparison saved to {save_path}")
    
    # plt.show()


def create_training_animation(callback: Example2DVisCallback, 
                              u_true_fn=None,
                              save_path: str = 'training_animation.gif',
                              fps: int = 5):
    """
    Create animated GIF showing PINN training evolution over epochs.
    
    Generates animated visualization with solution field, loss curve, and error
    evolution throughout training.
    
    Parameters
    ----------
    callback : Example2DVisCallback
        Visualization callback containing training history.
    u_true_fn : callable, optional
        Analytical solution function for error computation.
    save_path : str, default='training_animation.gif'
        File path where the GIF is saved.
    fps : int, default=5
        Animation frame rate (frames per second).
    """
    if not callback.history['epoch']:
        print("No history data available for animation.")
        return
    
    epochs = callback.history['epoch']
    losses = callback.history['loss']
    u_preds = callback.history['u_pred']
    
    # Compute analytical solution if available
    u_true = None
    if u_true_fn is not None:
        u_true = u_true_fn(callback.X, callback.Y)
    
    # Determine subplot layout
    n_plots = 3 if u_true is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    
    # Initialize plots
    im1 = axes[0].imshow(u_preds[0], extent=[0, 1, 0, 1], origin='lower',
                        cmap='viridis', aspect='auto', vmin=-1, vmax=1)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Solution u(x,y)')
    plt.colorbar(im1, ax=axes[0], label='u')
    plt.tight_layout()
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    line, = axes[1].plot([], [], 'b-', linewidth=2)
    axes[1].set_xlim(0, max(epochs))
    axes[1].set_ylim(min(losses)*0.5, max(losses)*2)
    
    if u_true is not None:
        error_0 = np.abs(u_preds[0] - u_true)
        im3 = axes[2].imshow(error_0, extent=[0, 1, 0, 1], origin='lower',
                           cmap='hot', aspect='auto', vmin=0, vmax=np.max(error_0)*1.5)
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].set_title('Absolute Error')
        plt.colorbar(im3, ax=axes[2], label='|u_pred - u_true|')
    
    def update(frame):
        """Update function for animation."""
        # Update solution heatmap
        im1.set_array(u_preds[frame])
        axes[0].set_title(f'Solution u(x,y) - Epoch {epochs[frame]}')
        
        # Update loss curve
        line.set_data(epochs[:frame+1], losses[:frame+1])
        
        # Update error heatmap
        if u_true is not None:
            error = np.abs(u_preds[frame] - u_true)
            im3.set_array(error)
            axes[2].set_title(f'Absolute Error (Max: {error.max():.2e})')
        
        return [im1, line] + ([im3] if u_true is not None else [])
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(epochs), 
                        interval=1000//fps, blit=True, repeat=True)
    
    # Save animation
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    print(f"Training animation saved to {save_path}")
    
    plt.close(fig)

    

if __name__ == '__main__':
    """
    Main execution workflow demonstrating 2D NAS-PINN architecture search.
    """
    
    ## Step 1: Setup and Configuration ##
    # Fix random seed for reproducibility
    set_seed(2022)

    # Configure device (GPU if available, otherwise CPU)
    if check_gpu(print_required=True):
        DEVICE.set_device(0)  # Using CUDA device 0
    else:
        DEVICE.set_device(-1) # Using CPU
    print(DEVICE)

    # Start performance timer
    my_timer = Timer()

    ## Step 2: Model Definition ##  
    print("=" * 70)
    print("2D NAS-PINN Example: Solving Poisson Equation ∇²u = f(x,y)")
    print("=" * 70)
    print(f"Device: {DEVICE()}")
    print(f"PDE: ∂²u/∂x² + ∂²u/∂y² = -2π²cos(πx)cos(πy)")
    print(f"Domain: [0, 1] × [0, 1]")
    print(f"BCs: u = cos(πx)cos(πy) on all boundaries")
    print(f"Analytical solution: u(x,y) = cos(πx)cos(πy)")
    print("=" * 70)
    
    # ========== Network Architecture ==========
    network = RelaxFNN(layers=5,
                          C_in_list=[2, 70, 70, 70, 70, 70],
                          neuron_list=[0, 30, 50, 70],
                          )
    
    # ========== NAS-PINN Initialization ==========
    n_domain = 500
    n_boundary_list = [25, 25, 25, 25]
    n_domain_archi = 1000
    n_boundary_list_archi = [50, 50, 50, 50]
    
    pinn = Example2DNasPINN(network,
                             n_domain=n_domain,
                             n_boundary_list=n_boundary_list,
                             n_domain_archi=n_domain_archi,
                             n_boundary_list_archi=n_boundary_list_archi)
    pinn.set_loss_func_archi(nn.functional.mse_loss)
    pinn.set_loss_func(nn.functional.smooth_l1_loss)
    
    # ========== Visualization Callback ==========
    x_eval = np.linspace(0, 1, 100, dtype=REAL())
    y_eval = np.linspace(0, 1, 100, dtype=REAL())
    
    log_freq = 100
    history_freq = 100  # Save every 100 epochs (50 snapshots for 5000 epochs)
    
    vis_callback = Example2DVisCallback(
        x_eval, y_eval,
        u_true_fn=analytical_solution,
        log_freq=log_freq,
        history_freq=history_freq
    )
    pinn.register_visualization_callback(vis_callback)
    
    print(f"\nVisualization:")
    print(f"  Evaluation grid: {len(x_eval)}×{len(y_eval)} = {len(x_eval)*len(y_eval)}")
    print(f"  Log frequency: {log_freq} epochs")
    print(f"  History frequency: {history_freq} epochs")
    
    # ========== Training ==========
    nas_pinn = NasPINN(pinn_model=pinn)
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    
    nas_pinn.search(outer_epochs=10000,
                    inner_epochs=10,
                    print_freq=10,
                    tensorboard_logdir='app/piml/nas_pinn/runs',
                    log_freq=log_freq,
                    checkpoint_dir='app/piml/nas_pinn/models',
                    checkpoint_freq=100,
                    final_model_path='app/piml/nas_pinn/results/final_model.pth',
                    )

    ## Print total running time ##
    print("\n>>> Total running time:")
    my_timer.current()


    ## After search, you should do a vanilla pinn training with the best architecture found, and then visualize the results.
    ## If the found architecture contains skip connections like '0+50', you need to manually modify the network architecture
    ## to include the skip connection in the specified layers. The threshold of retraining skip connections can be modified in
    ## the `relaxed_FNN` class `searched_neuron` method.