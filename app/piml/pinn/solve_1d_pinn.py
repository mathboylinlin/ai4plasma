###
## Author: Linlin Zhong
## Email: linlin@seu.edu.cn
## Created: Dec 04, 2021
## Updated: Jan 17, 2026
## Description: 1D Physics-Informed Neural Network (PINN) example for solving
##              second-order ODE with Dirichlet boundary conditions.
##              PDE: d²u/dx² = -sin(x) on domain [0, 1]
##              BCs: u(0) = 0, u(1) = 0
##              Analytical solution: u(x) = sin(x) - x*sin(1)
##
## Features:
##   - Custom visualization callback with training history tracking
##   - Post-training visualization: final comparison plot
##   - Animated training process showing solution evolution
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

from ai4plasma.piml.pinn import PINN, VisualizationCallback
from ai4plasma.piml.geo import Geo1D
from ai4plasma.core.network import FNN
from ai4plasma.utils.math import df_dX
from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import set_seed, numpy2torch, Timer
from ai4plasma.config import DEVICE


class Example1DPINN(PINN):
    """
    1D PINN implementation for solving d²u/dx² = -sin(x) with Dirichlet BCs.
    
    This example demonstrates:
      - How to define PDE residual using automatic differentiation
      - How to set up boundary conditions as separate loss terms
      - How to use geometry classes for domain sampling
      - How to weight different loss terms (domain vs boundary)
    
    Mathematical Problem:
      PDE: d²u/dx² + sin(x) = 0,  x ∈ (0, 1)
      BC1: u(0) = 0
      BC2: u(1) = 0
      Analytical: u(x) = sin(x) - x*sin(1)
    
    Loss Function:
      L = w_pde * ||d²u/dx² + sin(x)||² + w_bc * (||u(0)||² + ||u(1)||²)
      where w_pde = 1.0, w_bc = 10.0 (boundary conditions weighted higher)
    """
    
    def __init__(self, network):
        """
        Initialize the 1D PINN model.
        
        Parameters:
        -----------
        network : nn.Module
            Neural network architecture (e.g., FNN) for approximating the solution
        """
        # Define spatial domain geometry
        self.geo = Geo1D([0.0, 1.0])
        super().__init__(network)

    @staticmethod
    def _pde_residual(network, x):
        """
        Compute the PDE residual: d²u/dx² + sin(x).
        
        This is the physics equation that must be satisfied in the domain.
        The residual should be zero when the network approximates the true solution.
        
        Parameters:
        -----------
        network : nn.Module
            Neural network that approximates u(x)
        x : torch.Tensor
            Input spatial coordinates with shape (N, 1)
        
        Returns:
        --------
        torch.Tensor
            PDE residual values with shape (N, 1)
        
        Implementation:
        ---------------
        Uses automatic differentiation to compute:
          u_x = du/dx (first derivative)
          u_xx = d²u/dx² (second derivative)
        Then returns: u_xx + sin(x)
        """
        u = network(x)
        u_x = df_dX(u, x)
        u_xx = df_dX(u_x, x)
        return u_xx + torch.sin(x)
    
    @staticmethod
    def _bc_residual(network, x):
        """
        Compute the boundary condition residual: u(x) - 0.
        
        Both boundaries have zero Dirichlet conditions: u(0) = 0, u(1) = 0.
        The residual should be zero at boundary points.
        
        Parameters:
        -----------
        network : nn.Module
            Neural network that approximates u(x)
        x : torch.Tensor
            Boundary point coordinates (either x=0 or x=1)
        
        Returns:
        --------
        torch.Tensor
            Boundary residual (simply the network output at boundary)
        """
        u = network(x)
        return u
    
    def _define_loss_terms(self):
        """
        Define all loss terms for the PINN optimization problem.
        
        This method is called automatically during initialization to set up:
          1. Domain loss: PDE residual over interior collocation points
          2. Boundary losses: BC residuals at x=0 and x=1
        
        Loss Weighting Strategy:
        ------------------------
        - Domain weight = 1.0 (baseline)
        - Boundary weights = 10.0 (higher to ensure BC satisfaction)
        
        The higher boundary weights help enforce boundary conditions more strictly,
        which is often necessary for accurate PINN solutions.
        
        Sampling Strategy:
        ------------------
        - 100 uniform collocation points in domain (0, 1)
        - 2 boundary points at x=0 and x=1
        """
        # Sample domain collocation points
        n_domain = 100
        x_domain = self.geo.sample_domain(n_domain, mode='uniform')
        
        # Sample boundary points
        x_bc = self.geo.sample_boundary()
        x0_bc, x1_bc = x_bc[0], x_bc[1]
        
        # Add equation terms with weights
        self.add_equation('Domain', self._pde_residual, weight=1.0, data=x_domain)
        self.add_equation('Left Boundary', self._bc_residual, weight=10.0, data=x0_bc)
        self.add_equation('Right Boundary', self._bc_residual, weight=10.0, data=x1_bc)


class Example1DVisCallback(VisualizationCallback):
    """
    Custom visualization callback for 1D PINN training.
    
    This callback provides three main functions:
      1. Real-time TensorBoard logging during training
      2. Training history tracking for post-training animation
      3. Solution comparison plots (prediction vs analytical)
    
    Features:
      - Memory-efficient: saves history at configurable intervals
      - Flexible: separate control for TensorBoard and history frequencies
      - Extensible: easy to add additional plots (e.g., residual distributions)
    
    Usage Example:
    --------------
    >>> callback = Example1DVisCallback(
    ...     log_freq=50,        # Log to TensorBoard every 50 epochs
    ...     save_history=True,  # Enable history for animation
    ...     history_freq=100    # Save history every 100 epochs (saves memory)
    ... )
    >>> pinn.register_visualization_callback(callback)
    """
    
    def __init__(self, log_freq: int = 50, save_history: bool = True, history_freq: int = None):
        """
        Initialize the visualization callback.
        
        Parameters:
        -----------
        log_freq : int, default=50
            Frequency (in epochs) for logging visualizations to TensorBoard.
            E.g., log_freq=50 means visualize every 50 epochs.
        
        save_history : bool, default=True
            Whether to save prediction snapshots for creating training animations.
            Set to False if you don't need animations to save memory.
        
        history_freq : int, optional
            Frequency (in epochs) for saving history snapshots.
            If None, defaults to log_freq.
            Use a larger value to reduce memory consumption:
              - history_freq = 50:  saves ~100 frames for 5000 epochs
              - history_freq = 100: saves ~50 frames (50% less memory)
              - history_freq = 200: saves ~25 frames (75% less memory)
        
        Memory Consideration:
        ---------------------
        Each history snapshot stores a full prediction array.
        For 200 evaluation points with float32:
          - 1 snapshot ≈ 0.8 KB
          - 100 snapshots ≈ 80 KB
          - 1000 snapshots ≈ 800 KB
        Adjust history_freq based on your memory constraints and desired animation smoothness.
        """
        super().__init__(name='1D_PINN', log_freq=log_freq)

        # Create evaluation grid for visualization
        geo = Geo1D([0.0, 1.0])
        self.x = geo.sample_domain(200, mode='uniform', include_boundary=True, to_tensor=False)
        
        # Training history tracking for animation
        self.save_history = save_history
        self.history_freq = history_freq if history_freq is not None else log_freq
        self.history = {
            'epochs': [],       # List of epoch numbers
            'predictions': [],  # List of prediction arrays
            'losses': []        # List of total loss values
        }

    def _y_true_fn(self, x):
        """
        Compute analytical solution: u(x) = sin(x) - x*sin(1).
        
        Parameters:
        -----------
        x : np.ndarray
            Spatial coordinates with shape (N,) or (N, 1)
        
        Returns:
        --------
        np.ndarray
            Analytical solution values
        """
        return np.sin(x) - x * np.sin(1)
    
    def visualize(self, network, epoch: int, writer: SummaryWriter, **kwargs):
        """
        Generate visualization plots for the current training epoch.
        
        This method is called automatically by PINN.train() at specified intervals.
        It performs two main tasks:
          1. Creates comparison plots and logs to TensorBoard
          2. Saves prediction snapshots for training animation (if enabled)
        
        Parameters:
        -----------
        network : nn.Module
            The neural network being trained
        epoch : int
            Current training epoch number
        writer : SummaryWriter
            TensorBoard writer for logging figures
        **kwargs : dict
            Additional training information:
              - 'total_loss': Total loss value (torch.Tensor or float)
              - 'loss_dict': Dictionary of individual loss terms
        
        Returns:
        --------
        dict
            Dictionary mapping plot names to matplotlib figures:
              {'comparison': fig_comparison, ...}
            These figures are automatically logged to TensorBoard.
        
        Notes:
        ------
        - Network is set to eval mode during visualization
        - Gradients are disabled for efficient inference
        - History is saved only if (epoch % history_freq == 0)
        """
        network.eval()

        # Generate predictions on evaluation grid
        with torch.no_grad():
            y_pred = network(numpy2torch(self.x, require_grad=False)).cpu().numpy()
        
        # Compute analytical solution for comparison
        y_true = self._y_true_fn(self.x)
        
        # Save history for animation (only at specified frequency)
        if self.save_history and epoch % self.history_freq == 0:
            self.history['epochs'].append(epoch)
            self.history['predictions'].append(y_pred.copy())
            
            # Extract total loss from training info
            total_loss = kwargs.get('total_loss', None)
            if total_loss is not None:
                self.history['losses'].append(total_loss.item())
        
        # Create comparison plot for TensorBoard
        fig_comparison = PINN.plot_1d_comparison(
            self.x, y_pred, y_true=y_true,
            title=f'1D-PINN Solution (Epoch {epoch})',
            xlabel='x', ylabel='Solution'
        )
        
        # Optional: Plot residual distribution (currently commented out)
        # Uncomment below to add residual histogram to TensorBoard
        # residuals = kwargs.get('loss_dict', {}).get('Domain', None)
        # if residuals is not None:
        #     residuals_np = residuals.detach().cpu().numpy()
        #     fig_residual = PINN.plot_residual_distribution(
        #         residuals_np,
        #         title=f'Residual Distribution (Epoch {epoch})'
        #     )
        #     return {'comparison': fig_comparison, 'residual': fig_residual}
        
        return {'comparison': fig_comparison}


def plot_final_comparison(pinn, viz_callback, save_path):
    """
    Generate and save final comparison plot after training completes.
    
    Creates a two-panel figure showing:
      1. Solution comparison: PINN prediction vs analytical solution
      2. Absolute error distribution across the domain
    
    This provides a comprehensive visual assessment of the PINN's accuracy.
    
    Parameters:
    -----------
    pinn : PINN
        Trained PINN model
    viz_callback : Example1DVisCallback
        Visualization callback containing evaluation grid and analytical solution
    save_path : str
        Directory path where the plot will be saved
    
    Output:
    -------
    Saves 'final_comparison.png' in the specified directory with:
      - High resolution (300 DPI) for publication quality
      - Tight bounding box to minimize whitespace
    
    Plot Features:
    --------------
    - Blue solid line: Analytical solution (ground truth)
    - Red dashed line: PINN prediction
    - Green curve: Absolute error |u_pred - u_true|
    - Error statistics displayed in title (max and mean error)
    """
    os.makedirs(save_path, exist_ok=True)
    
    # pinn.network.eval()
    # x = viz_callback.x
    
    # # Generate final predictions
    # with torch.no_grad():
    #     y_pred = pinn.network(numpy2torch(x, require_grad=False)).cpu().numpy()

    x = viz_callback.x
    y_pred = pinn.predict(numpy2torch(x, require_grad=False)).cpu().numpy()
    
    # Compute analytical solution and error
    y_true = viz_callback._y_true_fn(x)
    error = np.abs(y_pred.flatten() - y_true.flatten())
    
    # Create figure with two subplots (stacked vertically)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Subplot 1: Solution comparison
    ax1 = axes[0]
    ax1.plot(x, y_true, 'b-', label='Analytical Solution', linewidth=2)
    ax1.plot(x, y_pred, 'r--', label='PINN Prediction', linewidth=2)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u(x)', fontsize=12)
    ax1.set_title('Final Solution Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Absolute error
    ax2 = axes[1]
    ax2.plot(x, error, 'g-', linewidth=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title(f'Absolute Error (Max: {error.max():.2e}, Mean: {error.mean():.2e})', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure to file
    save_file = os.path.join(save_path, 'final_comparison.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Final comparison plot saved to: {save_file}")
    
    plt.close()


def create_training_animation(viz_callback, save_path):
    """
    Create animated GIF showing the evolution of PINN training.
    
    This function generates a comprehensive animation with three synchronized panels:
      1. Solution evolution: Shows how prediction approaches analytical solution
      2. Point-wise error: Displays absolute error at each spatial location
      3. Loss curve: Tracks total loss decay on logarithmic scale
    
    The animation provides intuitive visualization of:
      - How the network learns the solution progressively
      - Where errors are largest during training
      - Training convergence behavior
    
    Parameters:
    -----------
    viz_callback : Example1DVisCallback
        Callback containing training history with:
          - epochs: List of epoch numbers
          - predictions: List of prediction arrays (N_frames, N_points)
          - losses: List of loss values (N_frames,)
    save_path : str
        Directory where animation will be saved
    
    Output:
    -------
    Saves 'training_animation.gif' in the specified directory.
    
    Animation Settings:
    -------------------
    - Frame rate: 5 FPS (frames per second)
    - Interval: 200 ms between frames
    - Resolution: 100 DPI (balance between quality and file size)
    - Format: GIF (universally compatible, no codec needed)
    
    Memory and Performance:
    -----------------------
    - GIF file size ≈ 1-5 MB depending on number of frames
    - Generation time ≈ 2-10 seconds for typical training runs
    - Frames = len(epochs) = num_epochs / history_freq
    
    Typical file sizes:
      - 25 frames (history_freq=200): ~1 MB
      - 50 frames (history_freq=100): ~2 MB
      - 100 frames (history_freq=50): ~4 MB
    
    Notes:
    ------
    If no history is saved (save_history=False or empty history),
    function prints a message and returns without creating animation.
    """
    # Check if training history is available
    if not viz_callback.save_history or len(viz_callback.history['epochs']) == 0:
        print("No training history available for animation.")
        return
    
    os.makedirs(save_path, exist_ok=True)
    
    # Extract training history data
    x = viz_callback.x
    y_true = viz_callback._y_true_fn(x)
    epochs = viz_callback.history['epochs']
    predictions = viz_callback.history['predictions']
    losses = viz_callback.history['losses']
    
    # Create figure with 2x2 grid layout (without residual histogram)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[:, 0])    # Solution evolution (left, full height)
    ax2 = fig.add_subplot(gs[0, 1])    # Point-wise error (top right)
    ax3 = fig.add_subplot(gs[1, 1])    # Loss curve (bottom right)
    
    # Initialize plot elements for Panel 1: Solution evolution
    line_true, = ax1.plot(x, y_true, 'b-', label='Analytical', linewidth=2)
    line_pred, = ax1.plot([], [], 'r--', label='PINN', linewidth=2)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('u(x)', fontsize=12)
    ax1.set_title('Solution Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(y_true.min() - 0.1, y_true.max() + 0.1)
    
    # Initialize plot elements for Panel 2: Point-wise error
    line_error, = ax2.plot([], [], 'g-', linewidth=2)
    ax2.set_xlabel('x', fontsize=10)
    ax2.set_ylabel('Absolute Error', fontsize=10)
    ax2.set_title('Point-wise Error', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x.min(), x.max())
    
    # Initialize plot elements for Panel 3: Training loss
    line_loss, = ax3.plot([], [], 'purple', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('Loss', fontsize=10)
    ax3.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')  # Logarithmic scale for better loss visualization
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, epochs[-1])
    if losses:
        ax3.set_ylim(min(losses) * 0.5, max(losses) * 2)
        
    
    # Epoch counter text at top center
    epoch_text = fig.text(0.5, 0.95, '', ha='center', fontsize=16, fontweight='bold')
    
    def init():
        """
        Initialize animation by setting all plot elements to empty.
        
        Called once at the start of animation creation.
        """
        line_pred.set_data([], [])
        line_error.set_data([], [])
        line_loss.set_data([], [])
        epoch_text.set_text('')
        return line_pred, line_error, line_loss, epoch_text
    
    def animate(frame):
        """
        Update animation for a specific frame.
        
        This function is called for each frame in the animation sequence.
        It updates all three panels with data from the corresponding training epoch.
        
        Parameters:
        -----------
        frame : int
            Frame index (0 to len(epochs)-1)
        
        Returns:
        --------
        tuple
            Updated plot elements (needed for blit=False animation)
        """
        # Extract data for current frame
        epoch = epochs[frame]
        y_pred = predictions[frame].flatten()
        error = np.abs(y_pred - y_true.flatten())
        
        # Update Panel 1: Solution prediction line
        line_pred.set_data(x, y_pred)
        
        # Update Panel 2: Error line with dynamic y-axis
        line_error.set_data(x, error)
        ax2.set_ylim(0, max(error) * 1.1)  # Auto-scale to fit error range
        
        # Update Panel 3: Loss curve (cumulative up to current frame)
        line_loss.set_data(epochs[:frame+1], losses[:frame+1])
        
        # Update epoch counter
        epoch_text.set_text(f'Epoch: {epoch}')
        
        return line_pred, line_error, line_loss, epoch_text
    
    # Create animation using FuncAnimation
    print("Creating training animation...")
    anim = FuncAnimation(
        fig, 
        animate, 
        init_func=init, 
        frames=len(epochs),
        interval=200,      # 200ms between frames
        blit=False,        # Full redraw (needed for changing axis limits)
        repeat=True        # Loop animation
    )
    
    # Save animation as GIF
    save_file = os.path.join(save_path, 'training_animation.gif')
    print(f"Saving animation to: {save_file}")
    writer = PillowWriter(fps=5)  # 5 frames per second
    anim.save(save_file, writer=writer, dpi=100)
    print(f"Animation saved successfully!")
    
    # Clean up
    plt.close()


if __name__ == "__main__":
    """
    Main execution script for 1D PINN example.
    
    Workflow:
    ---------
    1. Setup: Configure device, random seed, and timer
    2. Model Creation: Define network architecture and PINN instance
    3. Training: Train PINN with visualization callbacks
    4. Post-Processing: Generate final plots and training animation
    
    Output Files:
    -------------
    - ./app/piml/pinn/results/1d_pinn/final_comparison.png
        Final solution comparison and error plot
    
    - ./app/piml/pinn/results/1d_pinn/training_animation.gif
        Animated training evolution
    
    - ./app/piml/pinn/runs/1d_pinn/
        TensorBoard logs for real-time monitoring
        View with: tensorboard --logdir=./app/piml/pinn/runs/1d_pinn
    
    - ./app/piml/pinn/models/1d_pinn/final_model.pth
        Trained model checkpoint
    
    Configuration Summary:
    ----------------------
    - Network: 4-layer FNN [1, 64, 64, 64, 1]
    - Loss: MSE (Mean Squared Error)
    - Optimizer: Adam with lr=1e-3
    - Epochs: 5000
    - Collocation points: 100 uniform in domain
    - Boundary weight: 10x domain weight
    - Visualization: Every 50 epochs to TensorBoard
    - History: Every 100 epochs for animation (memory efficient)
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
    print(" " * 15 + "1D PINN Example: d²u/dx² = -sin(x)")
    print("=" * 70)
    
    # Define neural network architecture
    # Input: 1D coordinate x
    # Hidden: 3 layers with 64 neurons each
    # Output: Solution u(x)
    network = FNN([1, 64, 64, 64, 1])
    pinn = Example1DPINN(network)
    pinn.set_loss_func(nn.MSELoss())
    
    ## Step 3: Visualization Setup ##
    # Register visualization callback with memory-efficient history tracking
    # Strategy for memory management:
    #   - log_freq=50: Update TensorBoard every 50 epochs (moderate frequency)
    #   - history_freq=100: Save snapshots every 100 epochs (reduces memory by 50%)
    #   - Total snapshots = 5000/100 = 50 frames (smooth animation, ~2MB GIF)
    #
    # Adjust history_freq based on your needs:
    #   - Smoother animation: decrease history_freq (more frames, larger file)
    #   - Less memory/smaller file: increase history_freq (fewer frames)
    viz_callback = Example1DVisCallback(
        log_freq=50,          # TensorBoard logging frequency
        save_history=True,    # Enable history for animation
        history_freq=100      # Save history every 100 epochs (memory efficient)
    )
    pinn.register_visualization_callback(viz_callback)
    
    ## Step 4: Training ##
    print("\n>>> Starting training...")
    pinn.train(
        num_epochs=5000,
        lr=1e-3,
        print_loss=True,
        print_loss_freq=50,
        tensorboard_logdir='./app/piml/pinn/runs/1d_pinn',
        save_final_model=True,
        checkpoint_dir='./app/piml/pinn/models/1d_pinn',
        checkpoint_freq=100
    )
    
    print("\n" + "=" * 70)
    print(" " * 20 + "Training Complete!")
    print("=" * 70)
    
    ## Step 5: Post-Training Visualization ##
    print("\n>>> Generating post-training visualizations...")
    
    # Generate final comparison plot (high-resolution PNG)
    print("\n[1/2] Creating final comparison plot...")
    plot_final_comparison(pinn, viz_callback, save_path='./app/piml/pinn/results/1d_pinn')
    
    # Create training animation (GIF showing solution evolution)
    print("\n[2/2] Creating training animation...")
    create_training_animation(viz_callback, save_path='./app/piml/pinn/results/1d_pinn')
    
    ## Step 6: Summary ##
    print("\n" + "=" * 70)
    print(" " * 25 + "All Done!")
    print("=" * 70)
    print("\nResults saved to: ./app/piml/pinn/results/1d_pinn/")
    print("  - final_comparison.png: Final solution comparison")
    print("  - training_animation.gif: Training process animation")
    print("\nTensorboard logs: ./app/piml/pinn/runs/1d_pinn/")
    print("  View with: tensorboard --logdir=./app/piml/pinn/runs/1d_pinn")
    print("\nModel checkpoint: ./app/piml/pinn/models/1d_pinn/")
    print("=" * 70)

    ## Print total running time ##
    print("\n>>> Total running time:")
    my_timer.current()



