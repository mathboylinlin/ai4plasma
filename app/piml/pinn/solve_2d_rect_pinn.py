###
## Author: Linlin Zhong
## Email: linlin@seu.edu.cn
## Created: Jan 17, 2026
## Updated: Jan 17, 2026
## Description: 2D Physics-Informed Neural Network (PINN) example for solving
##              Poisson equation with Dirichlet boundary conditions.
##              PDE: ∇²u = -8π²sin(2πx)sin(2πy) on domain [0, 1] × [0, 1]
##              BCs: u = 0 on all boundaries
##              Analytical solution: u(x,y) = sin(2πx)sin(2πy)
##
## Features:
##   - 2D domain sampling and visualization
##   - Custom visualization callback with heatmap and contour plots
##   - Post-training visualization: comparison with analytical solution
##   - Animated training process showing 2D solution evolution
##   - Memory-efficient history recording with configurable frequency
##   - TensorBoard integration for real-time monitoring
##
###

import sys
sys.path.append('.')

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ai4plasma.piml.geo import GeoRect2D
from ai4plasma.piml.pinn import PINN, VisualizationCallback
from ai4plasma.core.network import FNN
from ai4plasma.utils.math import df_dX
from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import set_seed, numpy2torch, Timer
from ai4plasma.config import DEVICE, REAL


class Example2DPINN(PINN):
    """
    2D PINN implementation for solving Poisson equation ∇²u = f(x,y).
    
    This example demonstrates:
      - How to define 2D PDE residual using automatic differentiation
      - How to set up boundary conditions on all four edges
      - How to handle 2D domain sampling and collocation points
      - How to weight different loss terms (domain vs boundary)
      - How to visualize 2D solutions with heatmaps and contours
    
    Problem Setup:
      - PDE: ∂²u/∂x² + ∂²u/∂y² = -8π²sin(2πx)sin(2πy)
      - Domain: Ω = [0, 1] × [0, 1]
      - BCs: u(x,y) = 0 on ∂Ω (all boundaries)
      - Analytical solution: u(x,y) = sin(2πx)sin(2πy)
    
    The loss function is:
      L = w_pde * L_pde + w_bc * L_bc
    where:
      - L_pde: MSE of PDE residual at interior collocation points
      - L_bc: MSE of boundary condition residual at boundary points
    """
    
    def __init__(self, network: nn.Module, n_domain: list = [50, 50], n_boundary: list = [50, 50, 50, 50]):
        """
        Initialize the 2D PINN model.
        
        Parameters:
        -----------
        network : nn.Module
            Neural network for approximating the solution u(x,y)
            Input: 2D coordinates [x, y]
            Output: scalar field u
        n_domain : list of int, optional (default=[50, 50])
            Number of collocation points along each axis in the domain
            Total interior points = n_domain[0] × n_domain[1]
        n_boundary : list of int, optional (default=[50, 50, 50, 50])
            Number of points along each boundary edge
            Total boundary points = sum(n_boundary)
        
        Notes:
        ------
        - Higher n_domain improves PDE residual accuracy but increases computation
        - Boundary points should be sufficient to enforce BCs accurately
        - Typical values: n_domain=30-100, n_boundary=30-100
        """
        self.geo = GeoRect2D(0.0, 1.0, 0.0, 1.0)

        self.n_domain = n_domain
        self.n_boundary = n_boundary

        super().__init__(network)
        

    @staticmethod    
    def _pde_residual(net: nn.Module, xy: torch.Tensor) -> torch.Tensor:
        """
        Compute the PDE residual: ∇²u - f(x,y).
        
        For the Poisson equation ∇²u = f, the residual is:
          R_pde = ∂²u/∂x² + ∂²u/∂y² - f(x,y)
        
        where f(x,y) = -8π²sin(2πx)sin(2πy) for this example.
        
        Parameters:
        -----------
        net : nn.Module
            Neural network model
        xy : torch.Tensor
            Input coordinates of shape (N, 2) where N is number of points
            xy[:, 0] = x coordinates, xy[:, 1] = y coordinates
        
        Returns:
        --------
        residual : torch.Tensor
            PDE residual of shape (N, 1)
        
        Implementation Details:
        -----------------------
        Uses automatic differentiation to compute second derivatives:
          1. u = net(xy)
          2. u_x = ∂u/∂x, u_y = ∂u/∂y (first derivatives)
          3. u_xx = ∂²u/∂x², u_yy = ∂²u/∂y² (second derivatives)
          4. residual = u_xx + u_yy - f(x,y)
        """
        u = net(xy)
        
        # First derivatives
        u_xy = df_dX(u, xy)  # shape: (N, 2)
        u_x = u_xy[:, 0:1]  # ∂u/∂x
        u_y = u_xy[:, 1:2]  # ∂u/∂y
        
        # Second derivatives
        u_xx = df_dX(u_x, xy)[:, 0:1]  # ∂²u/∂x²
        u_yy = df_dX(u_y, xy)[:, 1:2]  # ∂²u/∂y²
        
        # Source term: f(x,y) = -8π²sin(2πx)sin(2πy)
        x, y = xy[:, 0:1], xy[:, 1:2]
        f = -8.0 * np.pi**2 * torch.sin(2*np.pi*x) * torch.sin(2*np.pi*y)
        
        # PDE residual
        residual = u_xx + u_yy - f
        
        return residual
    
    @staticmethod
    def _bc_residual(net: nn.Module, xy: torch.Tensor) -> torch.Tensor:
        """
        Compute the boundary condition residual.
        
        For Dirichlet BC u = 0 on boundaries, the residual is simply:
          R_bc = u(x,y) - 0 = u(x,y)
        
        Parameters:
        -----------
        net : nn.Module
            Neural network model
        xy : torch.Tensor
            Boundary coordinates of shape (N, 2)
        
        Returns:
        --------
        residual : torch.Tensor
            BC residual of shape (N, 1)
        """
        u = net(xy)
        return u  # BC: u = 0 on boundaries
    
    def _define_loss_terms(self):
        """
        Define loss terms for the 2D Poisson problem.
        
        This method sets up two loss terms:
          1. Domain loss: PDE residual in the interior (Ω)
          2. Boundary loss: BC residual on all boundaries (∂Ω)
        
        Domain Sampling Strategy:
        -------------------------
        - Interior points: Currently only support random sampling on [0, 1] × [0, 1]
        - Boundary points: Four edges (top, bottom, left, right)
        
        Loss Weights:
        -------------
        - w_pde = 1.0: Weight for PDE residual
        - w_bc = 10.0: Weight for boundary conditions (higher to enforce BCs)
        
        Memory Consideration:
        ---------------------
        Total collocation points = n_domain² + 4×n_boundary
        For n_domain=50, n_boundary=50: 2500 + 200 = 2700 points
        """
        # Interior domain points (uniform grid)
        xy_domain = self.geo.sample_domain(self.n_domain, mode='uniform')
        
        # Boundary points (four edges)
        xy_bc = self.geo.sample_boundary(self.n_boundary, mode='uniform')
        
        # Add loss terms to PINN
        self.add_equation('Domain', self._pde_residual, weight=1.0, data=xy_domain)
        self.add_equation('Boundary1', self._bc_residual, weight=10.0, data=xy_bc[0])
        self.add_equation('Boundary2', self._bc_residual, weight=10.0, data=xy_bc[1])
        self.add_equation('Boundary3', self._bc_residual, weight=10.0, data=xy_bc[2])
        self.add_equation('Boundary4', self._bc_residual, weight=10.0, data=xy_bc[3])


class Example2DVisCallback(VisualizationCallback):
    """
    Custom visualization callback for 2D PINN training.
    
    This callback generates and logs visualizations to TensorBoard:
      - Heatmap: 2D solution field u(x,y)
      - Contour: Contour lines of the solution
      - Comparison: Side-by-side prediction vs analytical solution
      - Error: Absolute error |u_pred - u_true|
    
    The callback also stores training history for creating animations:
      - epoch: List of epoch numbers
      - loss: List of total loss values
      - u_pred: List of predicted solution fields
    
    Memory Consideration:
    ---------------------
    For a 100×100 grid with 10000 epochs and history_freq=100:
      - Stored snapshots: 100
      - Memory per snapshot: 100×100×4 bytes = 40 KB
      - Total memory: 100 × 40 KB = 4 MB (manageable)
    
    For larger grids or longer training:
      - Increase history_freq to reduce memory usage
      - history_freq=200 reduces memory by 50%
      - Consider downsampling the grid for visualization
    """
    
    def __init__(self, x_domain: np.ndarray, y_domain: np.ndarray,
                 u_true_fn=None, log_freq: int = 100, history_freq: int = None):
        """
        Initialize the 2D visualization callback.
        
        Parameters:
        -----------
        x_domain : np.ndarray
            1D array of x coordinates for evaluation, shape (Nx,)
        y_domain : np.ndarray
            1D array of y coordinates for evaluation, shape (Ny,)
        u_true_fn : callable, optional
            Function to compute analytical solution: u_true_fn(X, Y) -> U
            where X, Y are 2D meshgrids and U is the solution field
        log_freq : int, optional (default=100)
            Frequency for logging to TensorBoard (every N epochs)
        history_freq : int, optional (default=log_freq)
            Frequency for recording history (every N epochs)
            Set higher than log_freq to reduce memory usage
        
        Example:
        --------
        >>> x = np.linspace(0, 1, 100)
        >>> y = np.linspace(0, 1, 100)
        >>> callback = Example2DVisCallback(x, y, u_true_fn=analytical_solution,
        ...                                  log_freq=100, history_freq=200)
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
        Generate and log 2D visualizations.
        
        This method:
          1. Evaluates the network on a 2D grid
          2. Creates heatmap and contour plots
          3. Compares with analytical solution if available
          4. Logs to TensorBoard
          5. Records history for animation
        
        Parameters:
        -----------
        network : nn.Module
            Trained neural network
        epoch : int
            Current epoch number
        writer : SummaryWriter
            TensorBoard writer for logging
        total_loss : float, optional
            Total loss value for history tracking
        **kwargs : dict
            Additional arguments (not used)
        
        Returns:
        --------
        figures : dict
            Dictionary of matplotlib figures {'name': figure}
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
    
    For the problem ∇²u = -8π²sin(2πx)sin(2πy) with u=0 on boundaries,
    the analytical solution is:
      u(x,y) = sin(2πx)sin(2πy)
    
    Parameters:
    -----------
    X : np.ndarray
        2D meshgrid of x coordinates, shape (Ny, Nx)
    Y : np.ndarray
        2D meshgrid of y coordinates, shape (Ny, Nx)
    
    Returns:
    --------
    u : np.ndarray
        Analytical solution field, shape (Ny, Nx)
    """
    return np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)


def plot_final_comparison(pinn: PINN, x_eval: np.ndarray, y_eval: np.ndarray,
                          u_true_fn=None, save_path: str = None):
    """
    Plot final comparison after training completes.
    
    This function creates a comprehensive visualization with:
      - Top row: Prediction heatmap and contour plot
      - Bottom row: Analytical solution and absolute error
    
    Parameters:
    -----------
    pinn : PINN
        Trained PINN model
    x_eval : np.ndarray
        1D array of x coordinates for evaluation
    y_eval : np.ndarray
        1D array of y coordinates for evaluation
    u_true_fn : callable, optional
        Function to compute analytical solution
    save_path : str, optional
        Path to save the figure (if provided)
    
    Example:
    --------
    >>> x = np.linspace(0, 1, 100)
    >>> y = np.linspace(0, 1, 100)
    >>> plot_final_comparison(pinn, x, y, analytical_solution,
    ...                       save_path='results/final_comparison.png')
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
    Create animated GIF showing the training evolution of the 2D solution.
    
    The animation displays:
      - Left: Solution field u(x,y) as heatmap
      - Middle: Loss curve over training epochs
      - Right: Error field (if analytical solution available)
    
    Parameters:
    -----------
    callback : Example2DVisCallback
        Visualization callback containing training history
    u_true_fn : callable, optional
        Function to compute analytical solution for error calculation
    save_path : str, optional (default='training_animation.gif')
        Path to save the animation GIF
    fps : int, optional (default=5)
        Frames per second for the animation
    
    Memory Consideration:
    ---------------------
    - Animation size depends on grid resolution and number of frames
    - For 100×100 grid with 100 frames: ~10-20 MB GIF file
    - Consider reducing fps or frame count for large animations
    
    Example:
    --------
    >>> create_training_animation(callback, analytical_solution,
    ...                           save_path='results/training.gif', fps=10)
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
    Main execution workflow for 2D PINN example.
    
    Workflow:
    ---------
    1. Setup: Check GPU, set seed, create directories
    2. Network: Define FNN architecture (2 inputs → 1 output)
    3. PINN: Initialize Example2DPINN with collocation points
    4. Callback: Register visualization callback for monitoring
    5. Training: Train PINN with Adam optimizer and loss logging
    6. Visualization: Plot final comparison and create animation
    7. Save: Store model checkpoint and results
    
    Configuration:
    --------------
    - n_domain: 2500 (2500 interior points)
    - n_boundary_list: [50, 50, 50, 50] (200 boundary points)
    - epochs: 5000
    - learning_rate: 1e-3
    - network: [2, 64, 64, 64, 1]
    - log_freq: 100 (every 100 epochs)
    - history_freq: 100 (save every 100 epochs for animation)
    
    Memory Usage Estimate:
    ----------------------
    - Collocation points: ~0.1 MB
    - Network parameters: ~0.5 MB
    - History (50 snapshots): ~2 MB
    - Total: ~3 MB (very lightweight)
    
    Output Files:
    -------------
    - Logs: runs/2d_rect_pinn/
    - Model: models/2d_rect_pinn/final_model.pth
    - Plot: results/2d_rect_pinn/final_comparison.png
    - Animation: results/2d_rect_pinn/training_animation.gif
    """
    
    ## Step 1: Setup and Configuration ##
    # Fix random seed for reproducibility
    set_seed(2026)

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
    print("2D PINN Example: Solving Poisson Equation ∇²u = f(x,y)")
    print("=" * 70)
    print(f"Device: {DEVICE()}")
    print(f"PDE: ∂²u/∂x² + ∂²u/∂y² = -8π²sin(2πx)sin(2πy)")
    print(f"Domain: [0, 1] × [0, 1]")
    print(f"BCs: u = 0 on all boundaries")
    print(f"Analytical solution: u(x,y) = sin(2πx)sin(2πy)")
    print("=" * 70)
    
    # ========== Network Architecture ==========
    network = FNN(
        layers=[2, 64, 64, 64, 1],  # 2 inputs (x,y) → 1 output (u)
        act_fun=nn.Tanh()
    )
    
    print(f"\nNetwork Architecture:")
    print(f"  Layers: {[2, 64, 64, 64, 1]}")
    print(f"  Activation: Tanh")
    print(f"  Parameters: {sum(p.numel() for p in network.parameters())}")
    
    # ========== PINN Initialization ==========
    n_domain = [50, 50]  # 2500 interior points
    n_boundary = [50, 50, 50, 50]  # 200 boundary points
    
    pinn = Example2DPINN(network, n_domain=n_domain, n_boundary=n_boundary)
    pinn.set_loss_func(nn.MSELoss())
    
    # ========== Visualization Callback ==========
    x_eval = np.linspace(0, 1, 100, endpoint=True, dtype=REAL())
    y_eval = np.linspace(0, 1, 100, endpoint=True,  dtype=REAL())
    
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
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70)
    
    pinn.train(
        num_epochs=5000,
        lr=1e-3,
        print_loss=True,
        print_loss_freq=100,
        tensorboard_logdir='./app/piml/pinn/runs/2d_rect_pinn',
        save_final_model=True,
        checkpoint_dir='./app/piml/pinn/models/2d_rect_pinn',
        checkpoint_freq=100
    )
    
    # ========== Post-Training Visualization ==========
    print("\n" + "=" * 70)
    print("Generating Post-Training Visualizations...")
    print("=" * 70)

    # Create results directory if not exists
    results_dir = './app/piml/pinn/results/2d_rect_pinn'
    os.makedirs(results_dir, exist_ok=True)
    
    # Final comparison plot
    plot_final_comparison(
        pinn, x_eval, y_eval,
        u_true_fn=analytical_solution,
        save_path=f'{results_dir}/final_comparison.png'
    )
    
    # Training animation
    create_training_animation(
        vis_callback,
        u_true_fn=analytical_solution,
        save_path=f'{results_dir}/training_animation.gif',
        fps=5
    )
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    
    # Compute final error metrics
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    XY_eval = np.column_stack([X_eval.ravel(), Y_eval.ravel()])
    
    pinn.network.eval()
    with torch.no_grad():
        xy_tensor = numpy2torch(XY_eval, require_grad=False)
        u_pred_final = pinn.network(xy_tensor).cpu().numpy().reshape(X_eval.shape)
    
    u_true_final = analytical_solution(X_eval, Y_eval)
    error_final = np.abs(u_pred_final - u_true_final)
    
    l2_error = np.sqrt(np.mean(error_final**2))
    max_error = np.max(error_final)
    rel_l2_error = l2_error / np.sqrt(np.mean(u_true_final**2))
    
    print(f"Final Error Metrics:")
    print(f"  L2 Error: {l2_error:.6e}")
    print(f"  Max Error: {max_error:.6e}")
    print(f"  Relative L2 Error: {rel_l2_error:.6e}")
    print(f"\nOutput Files:")
    print(f"  Model: ./app/piml/pinn/results/2d_pinn/2d_pinn_final.pth")
    print(f"  Comparison: ./app/piml/pinn/results/2d_pinn/final_comparison.png")
    print(f"  Animation: ./app/piml/pinn/results/2d_pinn/training_animation.gif")
    print(f"  TensorBoard logs: ./app/piml/pinn/runs/2d_pinn/")
    print("\nTo view TensorBoard logs:")
    print("  tensorboard --logdir ./app/piml/pinn/runs/2d_pinn")
    print("=" * 70)


    ## Print total running time ##
    print("\n>>> Total running time:")
    my_timer.current()
