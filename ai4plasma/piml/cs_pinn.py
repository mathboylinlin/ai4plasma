"""CS-PINN models for plasma simulations in AI4Plasma.

This module provides specialized Physics-Informed Neural Network (PINN) implementations
and visualization tools for simulating plasma discharge using cubic spline interpolation
and automatic differentiation on PyTorch.

CS-PINN Classes
---------------
The CS-PINN (Coefficient-Subnet Physics-Informed Neural Network) framework integrates:
  
- `StaArc1DModel`: PINN model for steady-state 1D arc plasma.
- `TraArc1DTempModel`: PINN model for transient 1D arc plasma without radial velocity.
- `TraArc1DModel`: PINN model for transient 1D arc plasma with radial velocity.
- `StaArc1DVisCallback`: Visualization callback for steady-state arc plasma simulation.
- `TraArc1DTempVisCallback`: Visualization callback for transient arc plasma simulation.

CS-PINN References
------------------
[1] L. Zhong, B. Wu, and Y. Wang, "Low-temperature plasma simulation based on 
    physics-informed neural networks: Frameworks and preliminary applications," 
    Physics of Fluids, vol. 34, no. 8, p. 087116, 2022.

[2] L. Zhong, Q. Gu, and B. Wu, "Deep learning for thermal plasma simulation: 
    Solving 1-D arc model as an example," Computer Physics Communications, 
    vol. 257, p. 107496, 2020.
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as intp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ai4plasma.core.network import FNN
from ai4plasma.config import REAL
from ai4plasma.utils.common import numpy2torch
from ai4plasma.utils.math import df_dX, calc_relative_l2_err
from ai4plasma.plasma.prop import ArcPropSpline
from ai4plasma.utils.io import img2gif
from .geo import Geo1D, Geo1DTime
from .pinn import PINN, VisualizationCallback


class StaArc1DNet(nn.Module):
    """
    Neural network wrapper for solving 1D stationary arc equation with
    automatic boundary condition enforcement.

    Attributes
    ----------
    network : nn.Module
        Backbone neural network (e.g., FNN) that maps r → N(r)
    R : float
        Normalized arc radius (default: 1.0)
    Tb : float
        Normalized boundary temperature at r = R (default: 0.03)

    Parameters (Constructor)
    ------------------------
    network : nn.Module
        Backbone neural network (e.g., FNN) that maps r → N(r)
    R : float, optional
        Normalized arc radius (default: 1.0)
    Tb : float, optional
        Normalized boundary temperature at r = R (default: 0.03)
    """
    def __init__(self, network, R=1.0, Tb=0.03):
        super(StaArc1DNet, self).__init__()

        self.network = network
        self.R = R # Reduced arc radius
        self.Tb = Tb # Reduced boundary temperature at r=R

    def forward(self, x):
        out = self.network(x)
        out = (x - self.R)*out + self.Tb # Enforce boundary condition at r=R
        return out


def calc_GL_coefs(degree):
    """
    Calculate Gauss-Legendre quadrature coefficients for arc conductance integral.

    Parameters
    ----------
    degree : int
        Degree of Gauss-Legendre quadrature (number of quadrature points)

    Returns
    -------
    Xq : torch.Tensor
        Abscissae (quadrature points) mapped to [0, 1], shape (degree, 1)
    Wq : torch.Tensor
        Quadrature weights normalized to [0, 1], shape (degree, 1)
    """
    quad_x, quad_w = np.polynomial.legendre.leggauss(degree)
    quad_x, quad_w = quad_x.reshape((-1,1)).astype(REAL()), quad_w.reshape((-1,1)).astype(REAL())
    Xq = numpy2torch(quad_x*0.5 + 0.5, require_grad=False)
    Wq = numpy2torch(quad_w, require_grad=False)
    
    return Xq, Wq


def get_Tfunc_from_file(csv_file):
    """
    Load reference temperature profile from CSV file for comparison.

    Parameters
    ----------
    csv_file : str
        Path to CSV file containing reference temperature data.
        The CSV should have columns: 'r(m)' (radius in meters) and
        'T(K)' (temperature in Kelvin).

    Returns
    -------
    function
        A cubic spline interpolation function T_spline(r) that returns
        interpolated temperature at radius r.

    Raises
    ------
    FileNotFoundError
        If csv_file does not exist.
    """

    # Check file existence
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    r_data = df['r(m)'].values.astype(REAL())
    T_data = df['T(K)'].values.astype(REAL())
    T_spline = intp.CubicSpline(r_data, T_data, extrapolate=True)
    
    return T_spline


class StaArc1DModel(PINN):
    """
    PINN model for solving 1D steady-state arc discharge energy equation.

    Implements a Physics-Informed Neural Network specifically designed for arc 
    plasma simulations. Solves the Elenbaas-Heller equation considering Joule 
    heating, thermal conduction, and radiation losses. The model enforces
    automatic boundary conditions through the StaArc1DNet architecture.

    Attributes
    ----------
    R : float
        Arc radius [m]
    I : float
        Arc current [A]
    T_red : float
        Temperature reduction factor for normalization [K]
    Tb : float
        Boundary temperature at r = R [K]
    Xq : torch.Tensor
        Gauss-Legendre quadrature abscissae for arc conductance integral
    Wq : torch.Tensor
        Gauss-Legendre quadrature weights
    geo : Geo1D
        Geometry object for domain and boundary sampling
    prop : ArcPropSpline
        Material properties interpolation object

    Parameters (Constructor)
    ------------------------
    R : float
        Arc radius [m]. Normalized to 1.0 in the network.
    I : float
        Arc current [A]. Used to compute electric field and Joule heating.
    Tb : float, optional
        Boundary temperature at r = R [K] (default: 300.0).
        Set to ambient temperature at the arc boundary.
    T_red : float, optional
        Temperature reduction factor for normalization [K] (default: 1e4).
        Used for non-dimensionalization: T_normalized = T_physical / T_red
    backbone_net : nn.Module, optional
        Backbone neural network that outputs the unscaled network function.
        Default: FNN with architecture [1, 100, 100, 100, 100, 1].
        Will be wrapped in StaArc1DNet for boundary condition enforcement.
    train_data_size : int, optional
        Number of training collocation points in the domain (default: 500).
        Collocation points are sampled according to sample_mode.
    test_data_size : int, optional
        Number of test/evaluation points for visualization (default: 501).
    sample_mode : str, optional
        Collocation point sampling strategy (default: 'uniform').
        Options: 'uniform' (uniform grid), 'lhs' (Latin hypercube), 'random'
    GL_degree : int, optional
        Degree of Gauss-Legendre quadrature for arc conductance integral 
        (default: 100). Higher degree provides better accuracy for integral computation.
    prop : ArcPropSpline, optional
        Arc material properties object (ArcPropSpline instance) for temperature-
        dependent properties like κ, σ, ε_nec. If None, properties must be provided
        externally.
    """
    def __init__(
        self,
        R,
        I,
        Tb=300.0,
        T_red=1e4,
        backbone_net=FNN(layers=[1, 100, 100, 100, 100, 1]),
        train_data_size=500,
        test_data_size=501,
        sample_mode='uniform',
        GL_degree=100,
        prop:ArcPropSpline=None,
    ):
        self.R = R 
        self.I = I 
        self.T_red = T_red
        self.Tb = Tb
        self.train_data_size = train_data_size
        self.test_data_size = test_data_size
        self.sample_mode = sample_mode
        self.GL_degree = GL_degree
        self.Xq, self.Wq = calc_GL_coefs(GL_degree)
        self.prop = prop

        self.geo = Geo1D([0.0, 1.0])
        network = StaArc1DNet(backbone_net, R=1.0, Tb=Tb/T_red)
        super().__init__(network)

        self.set_loss_func(F.smooth_l1_loss)

    
    def _define_loss_terms(self):
        """
        Define physics-informed loss terms for the steady-state arc model.

        This method constructs the complete loss function by defining residuals for:
        1. Energy PDE in the domain (steady-state energy balance)
        2. Boundary condition at r=0 (symmetry: dT/dr = 0)
        """
        def _pde_residual(network, x):
            """
            PDE residual for stationary arc equation in normalized coordinates.
            """
            T = network(x)
            kappa = self.prop.kappa(T.view(-1)*self.T_red).view(-1,1)
            sigma = self.prop.sigma(T.view(-1)*self.T_red).view(-1,1)
            nec = self.prop.nec(T.view(-1)*self.T_red).view(-1,1)

            Tq = network(self.Xq)
            sigma_q = self.prop.sigma(Tq.view(-1)*self.T_red).view(-1,1)
            arc_cond = np.pi*self.R*self.R*torch.sum(self.Wq*self.Xq*sigma_q)

            joule = sigma*(self.I/arc_cond)**2
            radiation = 4*np.pi*nec
            net_energy = (joule - radiation)/self.T_red*self.R*self.R

            T_x = df_dX(T, x)
            T_term = x*kappa*T_x
            T_xx = df_dX(T_term, x)

            func = T_xx + x*net_energy
            return func
        
        def _bc_residual(network, x):
            """
            Boundary condition residual at r=0 (symmetry condition).
            """
            T = network(x)
            T_x = df_dX(T, x)
            return T_x
        
        # Sample domain collocation points
        x_domain = self.geo.sample_domain(self.train_data_size, mode=self.sample_mode)
        
        # Sample boundary points
        x_bc = self.geo.sample_boundary()
        x_bc_left = x_bc[0]
        
        # Add equation terms with weights
        self.add_equation('Domain', _pde_residual, weight=1.0, data=x_domain)
        self.add_equation('Left Boundary', _bc_residual, weight=10.0, data=x_bc_left)


class StaArc1DVisCallback(VisualizationCallback):
    """
    Custom visualization callback for 1D stationary arc PINN training.
    
    This callback provides comprehensive monitoring of the arc model training:
      1. Real-time TensorBoard logging during training
      2. Training history tracking for post-training animation
      3. Temperature distribution evolution and error tracking
      4. Loss convergence monitoring
    """
    
    def __init__(self, model: 'StaArc1DModel', log_freq: int = 50, 
                 save_history: bool = True, history_freq: int = None,
                 T_csv_file: str = None,
                 gif_enabled: bool = False,
                 gif_dir: str = None,
                 gif_freq: int = None,
                 gif_duration_ms: int = 300,
                 gif_cleanup_tmp: bool = True):
        """
        Initialize the arc visualization callback.
        
        Parameters:
        -----------
        model : StaArc1DModel
            The arc model instance, needed to access material properties and parameters
        
        log_freq : int, default=50
            Frequency (in epochs) for logging visualizations to TensorBoard.
            E.g., log_freq=50 means visualize every 50 epochs.
        
        save_history : bool, default=True
            Whether to save prediction snapshots for creating training animations.
            Set to False if you don't need animations to save memory.
        
        history_freq : int, optional
            Frequency (in epochs) for saving history snapshots.
            If None, defaults to log_freq.
            Use a larger value to reduce memory consumption.
        
        T_csv_file : str, optional
            Path to CSV file containing reference temperature data for comparison.
            If provided, the reference temperature profile will be loaded and used for error analysis.
        
        gif_enabled : bool, default=False
            Whether to save per-epoch frames and generate a GIF animation that shows
            both the temperature comparison (T vs T_ref) and the loss curve.

        gif_dir : str, optional (default=None)
            Output directory for the final GIF and final summary plots.
            If None, defaults to the current working directory.
            The temporary frames directory (gif_tmp_dir) is automatically created
            as <gif_dir>/tmp_frames.
            The GIF filename is fixed as 'training_animation.gif' and saved in gif_dir.

        gif_freq : int, optional
            Frequency (in epochs) to save frames for the GIF. If None, uses history_freq.

        gif_duration_ms : int, default=300
            Duration per frame in milliseconds in the final GIF.

        gif_cleanup_tmp : bool, default=True
            Whether to remove the temporary PNG frames after saving the GIF.
        """
        super().__init__(name='CS-PINN_1D_Arc', log_freq=log_freq)
        
        self.model = model
        
        # Create evaluation grid for visualization
        self.x = model.geo.sample_domain(model.test_data_size, mode='uniform', 
                                        include_boundary=True, to_tensor=False)
        
        # Physical parameters for context (from material properties)
        self.T_red = model.T_red
        self.R = model.R
        self.I = model.I
        self.Tb = model.Tb

        # Load reference temperature profile if CSV file is provided
        if T_csv_file is not None:
            self.T_ref_func = get_Tfunc_from_file(T_csv_file)
            self.T_ref = self.T_ref_func(self.x*self.R) # Physical units
        
        # Training history tracking for animation
        self.save_history = save_history
        self.history_freq = history_freq if history_freq is not None else log_freq
        self.history = {
            'epochs': [],           # List of epoch numbers
            'axis': self.x,         # Evaluation grid (fixed), with shape (N,1)
            'T_ref': self.T_ref if T_csv_file is not None else None,  # Reference temperature profile for comparison
            'T': [],                # List of temperature profiles (T_red*T_reduced)
            'losses': [],           # List of total loss values
            'arc_conductance': [],  # List of arc conductance values per epoch
            'integral_powers': [],   # List of integrated Joule heating values
            'T_center': []         # List of center temperatures at r→0
        }

        # GIF configuration/state
        self.gif_enabled = gif_enabled
        # Set gif_dir to current working directory if not provided
        if gif_dir is None:
            self.gif_dir = os.getcwd()
        else:
            self.gif_dir = gif_dir
        # gif_tmp_dir is automatically set as gif_dir/tmp_frames
        self.gif_tmp_dir = os.path.join(self.gif_dir, 'tmp_frames')
        # gif_filename is fixed
        self.gif_filename = 'training_animation.gif'
        self.gif_freq = gif_freq if gif_freq is not None else self.history_freq
        self.gif_duration_ms = gif_duration_ms
        self.gif_cleanup_tmp = gif_cleanup_tmp
        self._gif_frames = []
        if self.gif_enabled:
            os.makedirs(self.gif_dir, exist_ok=True)
            os.makedirs(self.gif_tmp_dir, exist_ok=True)
    

    def _compute_material_properties(self, T_reduced: np.ndarray) -> dict:
        """
        Compute material properties at given temperature profile.
        
        Parameters:
        -----------
        T_reduced : np.ndarray
            Temperature profile in reduced units (divide by T_red to get physical units)
        
        Returns:
        --------
        dict
            Dictionary containing:
              - 'T_physical': Physical temperature (K)
              - 'kappa': Thermal conductivity (W/(m·K))
              - 'sigma': Electrical conductivity (1/(Ω·m))
              - 'nec': Net emission coefficient (W/m³)
              - 'max_T': Maximum temperature
              - 'T_center': Center temperature at r→0
        """
        T_physical = T_reduced * self.T_red
        
        # Ensure material property class is available
        if self.model.prop is None:
            return {
                'T_physical': T_physical,
                'kappa': None,
                'sigma': None,
                'nec': None,
                'max_T': T_physical.max(),
                'T_center': T_physical[0]
            }
        
        # Compute properties using model's material property class
        # T_tensor = torch.tensor(T_physical, dtype=REAL())
       
        
        with torch.no_grad():
            T_tensor = numpy2torch(T_physical, require_grad=False)
            kappa = self.model.prop.kappa(T_tensor).cpu().numpy()
            sigma = self.model.prop.sigma(T_tensor).cpu().numpy()
            nec = self.model.prop.nec(T_tensor).cpu().numpy()
        
        return {
            'T_physical': T_physical,
            'kappa': kappa,
            'sigma': sigma,
            'nec': nec,
            'max_T': T_physical.max(),
            'T_center': T_physical[0]
        }
    
    def _make_figure(self, epoch: int, T_reduced: np.ndarray, props: dict, **kwargs) -> plt.Figure:
        """
        Build the multi-panel matplotlib figure used for both TensorBoard logging
        and GIF frames.

        Parameters:
        - epoch: current epoch number (int)
        - T_reduced: model output in reduced units (np.ndarray)
        - props: material property dictionary from _compute_material_properties()
        - kwargs: optional training info (e.g., total_loss)

        Returns:
        - matplotlib Figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

        # Panel 1: Temperature profile - Prediction vs Reference
        ax1 = fig.add_subplot(gs[0, 0])
        T_physical = T_reduced * self.T_red

        # Plot network prediction
        ax1.plot(self.x, T_physical, 'b-', linewidth=2.5, label='CS-PINN Prediction')

        # Plot reference temperature if available
        if self.T_ref is not None:
            ax1.plot(self.x, self.T_ref, 'r--', linewidth=2.0, label='Reference', alpha=0.8)
            # Compute and display error metrics
            error = np.abs(T_physical.flatten() - self.T_ref.flatten())
            relative_error = error / (self.T_ref.flatten() + 1e-10) * 100
            max_error = error.max()
            mean_error = error.mean()
            max_rel_error = relative_error.max()
            rel_l2_error = calc_relative_l2_err(self.T_ref, T_physical)
            info_text = (f'Epoch {epoch}\n'
                         f'Max T: {props["max_T"]:.0f} K\n'
                         f'Center T: {props["T_center"]:.0f} K\n'
                         f'Max Error: {max_error:.1f} K\n'
                         f'Mean Error: {mean_error:.1f} K\n'
                         f'Max Rel Error: {max_rel_error:.2f}%\n'
                         f'Rel L2 Error: {rel_l2_error:.5g}')
        else:
            ax1.fill_between(self.x.flatten(), T_physical.flatten(), alpha=0.3)
            info_text = (f'Epoch {epoch}\n'
                         f'Max T: {props["max_T"]:.0f} K\n'
                         f'Center T: {props["T_center"]:.0f} K')

        ax1.set_xlabel('Normalized radius r/R', fontsize=11)
        ax1.set_ylabel('Temperature (K)', fontsize=11)
        ax1.set_title('Temperature Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        ax1.text(0.05, 0.10, info_text,
                 transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Panel 2: Material properties (kappa and sigma)
        if props['kappa'] is not None and props['sigma'] is not None:
            ax2 = fig.add_subplot(gs[0, 1])
            ax2_twin = ax2.twinx()
            line1 = ax2.plot(self.x, props['kappa'], 'g-', linewidth=2, label='κ(T) Thermal conductivity')
            line2 = ax2_twin.plot(self.x, props['sigma'], 'orange', linewidth=2, linestyle='--', 
                                  label='σ(T) Electrical conductivity')
            ax2.set_xlabel('Normalized radius r/R', fontsize=11)
            ax2.set_ylabel('κ (W/(m·K))', fontsize=10, color='g')
            ax2_twin.set_ylabel('σ (1/(Ω·m))', fontsize=10, color='orange')
            ax2.set_title('Material Properties', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='y', labelcolor='g')
            ax2_twin.tick_params(axis='y', labelcolor='orange')
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='best', fontsize=9)
        else:
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.text(0.5, 0.5, 'Material properties\nnot available',
                     ha='center', va='center', transform=ax2.transAxes,
                     fontsize=12, style='italic', color='gray')
            ax2.set_title('Material Properties', fontsize=12, fontweight='bold')

        # Panel 3: Radiation term (net emission coefficient)
        if props['nec'] is not None:
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(self.x, props['nec'], 'r-', linewidth=2)
            ax3.fill_between(self.x.flatten(), props['nec'].flatten(), alpha=0.3, color='red')
            ax3.set_xlabel('Normalized radius r/R', fontsize=11)
            ax3.set_ylabel('nec (W/m³)', fontsize=11)
            ax3.set_title('Radiation Term (Net Emission Coefficient)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.text(0.05, 0.1, f'Max nec: {props["nec"].max():.2e} W/m³',
                     transform=ax3.transAxes, fontsize=10, verticalalignment='bottom', 
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        else:
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.text(0.5, 0.5, 'Radiation term\nnot available',
                     ha='center', va='center', transform=ax3.transAxes,
                     fontsize=12, style='italic', color='gray')
            ax3.set_title('Radiation Term', fontsize=12, fontweight='bold')

        # Panel 4: Loss curve (training history)
        ax4 = fig.add_subplot(gs[1, 1])
        if self.history['losses']:
            loss_epochs = self.history['epochs']
            loss_values = self.history['losses']
            ax4.semilogy(loss_epochs, loss_values, 'purple', linewidth=2.5, marker='o', markersize=4)
            ax4.set_xlabel('Epoch', fontsize=11)
            ax4.set_ylabel('Loss (log scale)', fontsize=11)
            ax4.set_title('Training Loss Convergence', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, which='both')
            current_loss = kwargs.get('total_loss', None)
            if current_loss is not None:
                if isinstance(current_loss, torch.Tensor):
                    loss_val = current_loss.item()
                else:
                    loss_val = float(current_loss)
                ax4.text(0.95, 0.95, f'Current loss: {loss_val:.2e}',
                         transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                         horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        else:
            ax4.text(0.5, 0.5, 'No loss history yet',
                     ha='center', va='center', transform=ax4.transAxes,
                     fontsize=12, style='italic', color='gray')
            ax4.set_title('Training Loss Convergence', fontsize=12, fontweight='bold')

        fig.suptitle(f'Arc 1D PINN Model - Epoch {epoch}', fontsize=14, fontweight='bold', y=0.995)
        return fig

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
        kwargs : dict
            Additional training information:
              - 'total_loss': Total loss value (torch.Tensor or float)
              - 'loss_dict': Dictionary of individual loss terms
        
        Returns:
        --------
        dict
            Dictionary mapping plot names to matplotlib figures
        """
        network.eval()
        
        # Generate predictions on evaluation grid (in reduced units)
        with torch.no_grad():
            T_reduced = network(numpy2torch(self.x, require_grad=False)).cpu().numpy()
        
        # Compute material properties at current temperature profile
        props = self._compute_material_properties(T_reduced.flatten())
        
        # Save history for animation (only at specified frequency)
        if self.save_history and epoch % self.history_freq == 0:
            self.history['epochs'].append(epoch)
            self.history['T'].append(T_reduced.copy())
            self.history['T_center'].append(props['T_center'])
            
            # Extract total loss from training info
            total_loss = kwargs.get('total_loss', None)
            if total_loss is not None:
                self.history['losses'].append(total_loss.item())
        
        # Create figure with multiple subplots for comprehensive visualization
        fig = self._make_figure(epoch=epoch, T_reduced=T_reduced, props=props, **kwargs)

        # Optionally save frame for GIF
        if self.gif_enabled and (epoch % self.gif_freq == 0):
            frame_path = os.path.join(self.gif_tmp_dir, f'epoch_{epoch:06d}.png')
            try:
                fig.savefig(frame_path, dpi=120)
                self._gif_frames.append(frame_path)
            except Exception as e:
                print(f'Warning: failed to save GIF frame at epoch {epoch}: {e}')
        
        return {'arc_visualization': fig}

    def save_gif(self, gif_path: str = None, duration_ms: int = None, loop: int = 0):
        """
        Assemble saved frames into a GIF showing T vs T_ref and loss evolution.

        Parameters:
        - gif_path: optional output path; defaults to <gif_dir>/<gif_filename>
        - duration_ms: per-frame duration in milliseconds; defaults to gif_duration_ms
        - loop: number of loops (0 = infinite)
        """
        if not self.gif_enabled:
            print('GIF not enabled; nothing to save.')
            return
        if len(self._gif_frames) == 0:
            print('No frames collected; GIF not created.')
            return
        out_path = gif_path if gif_path is not None else os.path.join(self.gif_dir, self.gif_filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        frames_sorted = sorted(self._gif_frames)
        try:
            img2gif(frames_sorted, out_path, duration=(duration_ms or self.gif_duration_ms), loop=loop)
            print(f'GIF saved: {out_path}')
        except Exception as e:
            print(f'Failed to create GIF: {e}')
            return
        if self.gif_cleanup_tmp:
            # Best-effort cleanup of temporary frames
            try:
                for f in frames_sorted:
                    if os.path.exists(f):
                        os.remove(f)
                if os.path.isdir(self.gif_tmp_dir) and len(os.listdir(self.gif_tmp_dir)) == 0:
                    os.rmdir(self.gif_tmp_dir)
            except Exception as e:
                print(f'Warning: failed cleanup of tmp frames: {e}')

    def save_final_results(self, network: nn.Module, save_dir: str = None, epoch: int = None, **kwargs):
        """
        Save final result figures similar to TensorBoard panels and a dedicated loss plot.

        Outputs:
        - final_panels.png: the same 2x2 panel figure used in TensorBoard
        - loss_curve.png: standalone loss curve using collected history

        Parameters:
        - network: trained network to generate the final panel figure
        - save_dir: output directory; defaults to gif_dir
        - epoch: epoch number to show in the title; if None, uses last recorded or 0
        - kwargs: optional training info for annotations (e.g., total_loss)
        """
        out_dir = save_dir or self.gif_dir
        os.makedirs(out_dir, exist_ok=True)

        # Compute current prediction and properties
        network.eval()
        with torch.no_grad():
            T_reduced = network(numpy2torch(self.x, require_grad=False)).cpu().numpy()
        props = self._compute_material_properties(T_reduced.flatten())

        # Determine epoch label
        if epoch is None:
            epoch = self.history['epochs'][-1] if self.history['epochs'] else 0

        # Save panel figure
        fig = self._make_figure(epoch=epoch, T_reduced=T_reduced, props=props, **kwargs)
        panels_path = os.path.join(out_dir, 'final_panels.png')
        try:
            fig.savefig(panels_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Final panels saved: {panels_path}')
        except Exception as e:
            print(f'Failed to save final panels: {e}')

        # Save standalone loss curve
        if self.history['losses']:
            plt.figure(figsize=(7, 5))
            plt.semilogy(self.history['epochs'], self.history['losses'], 'purple', linewidth=2.0, marker='o', markersize=4)
            plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)')
            plt.title('Training Loss Curve')
            plt.grid(True, alpha=0.3, which='both')
            loss_path = os.path.join(out_dir, 'loss_curve.png')
            try:
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                print(f'Loss curve saved: {loss_path}')
            except Exception as e:
                print(f'Failed to save loss curve: {e}')
            finally:
                plt.close()
        else:
            print('No loss history recorded; loss_curve.png not created.')


class TraArc1DTempNet(nn.Module):
    """
    Neural network wrapper for solving 1D transient arc temperature equation with
    automatic boundary condition enforcement.

    This network automatically satisfies the Dirichlet boundary condition at r = R
    (arc radius) for all time steps by construction, eliminating the need for
    explicit boundary loss terms. The temperature output is modified to ensure
    T(R, t) = Tb for all t.

    The network applies a transformation to the backbone network output:
        T(r, t) = (r - R) · N(r, t) + Tb

    where N(r, t) is the backbone network output (takes both r and t as inputs),
    R is the normalized arc radius (typically 1.0), and Tb is the boundary
    temperature (normalized). At r = R, the boundary condition T(R, t) = Tb is
    satisfied for all t. Unlike the stationary case (StaArc1DNet), this network
    handles time-dependent temperature evolution.

    Parameters (Constructor)
    ------------------------
    network : nn.Module
        Backbone neural network (e.g., FNN) that maps (r, t) → N(r, t)
        Input shape: [batch_size, 2] where [:, 0]=r and [:, 1]=t
        Output shape: [batch_size, 1] representing network prediction
    R : float, optional
        Normalized arc radius (default: 1.0)
    Tb : float, optional
        Normalized boundary temperature at r = R (default: 0.03)
    """
    def __init__(self, network, R=1.0, Tb=0.03):
        super(TraArc1DTempNet, self).__init__()

        self.network = network
        self.R = R  # Reduced arc radius
        self.Tb = Tb  # Reduced boundary temperature at r=R

    def forward(self, x):
        """
        Forward pass with automatic boundary condition enforcement.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, 2] where:
            x[:, 0] : r (normalized radius, range [0, 1])
            x[:, 1] : t (normalized time, range [0, 1])

        Returns
        -------
        torch.Tensor
            Temperature T(r, t) in normalized units, shape [batch_size, 1]
            Satisfies T(R, t) = Tb for all t by construction
        """
        out = self.network(x)
        T = out*(x[:,0:1] - self.R) + self.Tb  # Enforce T(R, t) = Tb
        return T


class TraArc1DTempModel(PINN):
    """
    PINN model for solving 1D transient arc discharge energy equation without radial velocity.
    
    This class implements a Physics-Informed Neural Network specifically designed for
    simulating transient (time-dependent) arc discharge phenomena. The model solves the
    nonlinear transient energy balance equation considering time-dependent temperature
    evolution, thermal conduction, and radiation effects.
    
    Physical Constraints:
    ---------------------
    - T(R, t) = Tb (Dirichlet boundary condition at arc radius)
    - ∂T/∂r(0, t) = 0 (symmetry condition at centerline)
    - T(r, 0) = Tinit_func(r) (initial temperature distribution)
    """
    def __init__(
        self,
        R,
        I,
        Tb=300.0,
        Tinit_func=None,
        T_red=1e4,
        t_red=1e-3,
        backbone_net=FNN(layers=[2, 200, 200, 200, 200, 200, 200, 1]),
        train_data_x_size=200,
        train_data_t_size=100,
        sample_mode='uniform',
        prop:ArcPropSpline=None,
    ):
        self.R = R 
        self.I = I 
        self.T_red = T_red
        self.Tb = Tb
        if Tinit_func is None:
            raise ValueError('Tinit_func (initial condition) must be provided for transient arc model.')
        else:
            self.Tinit_func = Tinit_func
        self.t_red = t_red
        self.train_data_x_size = train_data_x_size
        self.train_data_t_size = train_data_t_size
        self.sample_mode = sample_mode
        self.prop = prop

        self.geo = Geo1DTime([0.0, 1.0], ts=0.0, te=1.0)
        network = TraArc1DTempNet(backbone_net, R=1.0, Tb=Tb/T_red)
        super().__init__(network)

        self.set_loss_func(F.smooth_l1_loss)

    
    def _define_loss_terms(self):
        """
        Define physics-informed loss terms for the transient arc model without radial velocity.
        
        This method constructs the complete loss function by defining residuals for:
        1. Energy PDE in the domain (transient temperature evolution without convection)
        2. Boundary condition at r=0 (symmetry: ∂T/∂r = 0)
        3. Initial condition at t=0 (prescribed temperature distribution)
        """
        def _pde_residual(network, x):
            """
            PDE residual for transient arc equation without radial velocity.
            
            Parameters
            ----------
            network : nn.Module
                Neural network model
            x : torch.Tensor
                Input tensor where x[:,0] is r and x[:,1] is t
            """
            T = network(x)
            kappa = self.prop.kappa(T.view(-1)*self.T_red).view(-1,1)
            Cp = self.prop.Cp(T.view(-1)*self.T_red).view(-1,1)
            rho = self.prop.rho(T.view(-1)*self.T_red).view(-1,1)
            nec = self.prop.nec(T.view(-1)*self.T_red).view(-1,1)

            joule = 0
            radiation = 4*np.pi*nec
            net_energy = joule - radiation

            T_x = df_dX(T, x)
            T_r = T_x[:,0:1]
            T_t = T_x[:,1:2]

            r = x[:,0:1]
            T_term = r*kappa*T_r
            T_xx = df_dX(T_term, x)
            T_rr = T_xx[:,0:1]

            func = T_t - (net_energy*(self.t_red/self.T_red) + T_rr/r*(self.t_red/self.R/self.R))/(rho*Cp)

            return func
        
        def _bc_residual(network, x):
            """
            Boundary condition residual at r=0 (symmetry condition).
            """
            T = network(x)
            T_x = df_dX(T, x)

            func_bc = T_x[:,0:1]
            return func_bc
        

        def _init_residual(network, x):
            """
            Initial condition residual at t=0.
            """
            T = network(x)
            func_ic = T - Ti
            return func_ic
        
        # Sample domain collocation points
        xt_domain, xt_bc = self.geo.sample_all_domain(Nx=self.train_data_x_size, 
                                                      Nt=self.train_data_t_size, 
                                                      mode=[self.sample_mode, self.sample_mode])
        xb = xt_bc[0][0]
        xi = xt_bc[1]

        if isinstance(xi, torch.Tensor):
            _xi = xi.detach().cpu().numpy()
            Ti = self.Tinit_func(_xi[:,0:1]*self.R)/self.T_red
            Ti = numpy2torch(Ti)
        else: # np.ndarray
            Ti = self.Tinit_func(xi[:,0:1]*self.R)/self.T_red
        
        # Add equation terms with weights
        self.add_equation('Domain', _pde_residual, weight=1.0, data=xt_domain)
        self.add_equation('Left Boundary', _bc_residual, weight=10.0, data=xb)
        self.add_equation('Initial Condition', _init_residual, weight=10.0, data=xi)


class TraArc1DTempVisCallback(VisualizationCallback):
    """
    Custom visualization callback for 1D transient arc PINN training (temperature only).
    
    This callback provides comprehensive monitoring of the transient arc model training:
      1. Real-time TensorBoard logging during training
      2. Training history tracking for post-training animation
      3. Temperature distribution evolution at multiple time steps
      4. Material property evolution over time
      5. Loss convergence monitoring
    """
    
    def __init__(self, model: 'TraArc1DTempModel', log_freq: int = 50, 
                 save_history: bool = True, history_freq: int = None,
                 x_eval: np.ndarray = np.linspace(0, 1, 201, dtype=REAL()).reshape(-1,1),
                 t_eval: list = [0.1, 0.5, 0.9],
                 T_csv_file: list[str] = ['','',''], # Paths to CSV files for reference temperature data at different times
                #  num_time_snapshots: int = 5,
                 gif_enabled: bool = False,
                 gif_dir: str = None,
                 gif_freq: int = None,
                 gif_duration_ms: int = 300,
                 gif_cleanup_tmp: bool = True):
        """
        Initialize the transient arc visualization callback for real-time training monitoring.
        
        This callback provides comprehensive real-time monitoring and post-training visualization
        capabilities for transient arc simulations. It tracks temperature evolution at multiple
        time points and generates publication-quality figures and animations.
        
        Parameters:
        -----------
        model : TraArc1DTempModel
            The transient arc model instance for temperature-only simulation.
            Used to access geometry, physical parameters, and material properties.
        
        log_freq : int, default=50
            Frequency (in epochs) for logging visualizations to TensorBoard.
            Controls how often multi-panel figures are generated and logged.
        
        save_history : bool, default=True
            Whether to save prediction snapshots for creating training animations.
            Set to False to reduce memory consumption if animations are not needed.
        
        history_freq : int, optional
            Frequency (in epochs) for saving history snapshots.
            If None, defaults to log_freq.
            Use larger values (e.g., 200) to reduce memory for long training runs.
        
        x_eval : np.ndarray, shape (n_r, 1)
            Radial evaluation grid for visualization (normalized radius, 0 to 1).
            Default: 201 points linearly spaced from 0 to 1.
            Use finer grids for higher-resolution visualizations (more detail).
            Use coarser grids for faster evaluation and smaller GIF files.
        
        t_eval : list or np.ndarray
            List of normalized time points (0 to 1) at which to display temperature
            snapshots in the multi-panel figures.
            Example: [0.1, 0.5, 0.9] shows temperature at 10%, 50%, 90% of total time.
            Default: [0.1, 0.5, 0.9] (3 snapshots per epoch).
        
        T_csv_file : list[str], optional
            Paths to CSV files containing reference temperature data at different times
            for comparison with PINN predictions.
            - One CSV file per time point in t_eval (must match length)
            - CSV columns: 'r(m)' (radius in meters), 'T(K)' (temperature in Kelvin)
            - Empty string '' or None skips reference comparison for that time point
            Example: ['ref_0.1ms.csv', 'ref_0.5ms.csv', 'ref_0.9ms.csv']
        
        gif_enabled : bool, default=False
            Whether to save per-epoch frames and generate a training animation GIF.
            When True, PNG frames are saved at gif_freq intervals and assembled into
            a GIF showing loss convergence and temperature evolution.
            Enables visual inspection of training progress and convergence behavior.
        
        gif_dir : str, optional (default=None)
            Output directory for GIF animation and final summary plots.
            If None, defaults to current working directory.
        
        gif_freq : int, optional
            Frequency (in epochs) to save frames for GIF assembly.
            If None, uses history_freq.
            Use larger values (e.g., 500) for smaller GIF files and faster creation.
        
        gif_duration_ms : int, default=300
            Duration per frame in milliseconds for the final GIF animation.
        
        gif_cleanup_tmp : bool, default=True
            Whether to automatically delete temporary PNG frames after GIF creation.
            Set to False to retain frames for manual inspection or re-assembly.
        """
        super().__init__(name='CS-PINN_1D_Arc_Transient_noV', log_freq=log_freq)
        
        self.model = model
        self.x_eval = x_eval
        self.t_eval = t_eval
        self.xt_list = model.geo.sample_space_time_list(x_eval, t_eval, require_grad=False)
        self.T_ref_func_list = [get_Tfunc_from_file(csv_file) for csv_file in T_csv_file]
        
        # Physical parameters
        self.T_red = model.T_red
        self.t_red = model.t_red
        self.R = model.R
        self.I = model.I
        self.Tb = model.Tb
        self.Tinit_func = model.Tinit_func
        
        # Training history tracking
        self.save_history = save_history
        self.history_freq = history_freq if history_freq is not None else log_freq
        self.history = {
            'epochs': [],           # List of epoch numbers
            'r_eval': self.x_eval,  # Radial evaluation grid (fixed)
            't_eval': self.t_eval,  # Time evaluation points (fixed)
            'T': [],                # List of T(r,t) arrays [n_epochs, n_time, n_r]
            'losses': [],           # List of total loss values
            'T_center_t': [],       # List of center temp at all time points
        }
        
        # GIF configuration/state
        self.gif_enabled = gif_enabled
        if gif_dir is None:
            self.gif_dir = os.getcwd()
        else:
            self.gif_dir = gif_dir
        self.gif_tmp_dir = os.path.join(self.gif_dir, 'tmp_frames')
        self.gif_filename = 'training_animation.gif'
        self.gif_freq = gif_freq if gif_freq is not None else self.history_freq
        self.gif_duration_ms = gif_duration_ms
        self.gif_cleanup_tmp = gif_cleanup_tmp
        self._gif_frames = []
        if self.gif_enabled:
            os.makedirs(self.gif_dir, exist_ok=True)
            os.makedirs(self.gif_tmp_dir, exist_ok=True)

    def _compute_material_properties_at_t(self, T_physical: np.ndarray, t_reduced: float) -> dict:
        """
        Compute material properties at a given time step and temperature profile.
        
        Parameters:
        -----------
        T_physical : np.ndarray
            Temperature in physical units (K), shape (n_r,)
        t_reduced : float
            Normalized time value for reference in output
        
        Returns:
        --------
        dict
            Dictionary containing material properties and statistics
        """
        if self.model.prop is None:
            return {
                'T_physical': T_physical,
                'kappa': None,
                'sigma': None,
                'nec': None,
                'max_T': T_physical.max(),
                'T_center': T_physical[0],
                't_physical': t_reduced * self.t_red
            }
        
        with torch.no_grad():
            T_tensor = numpy2torch(T_physical, require_grad=False)
            kappa = self.model.prop.kappa(T_tensor).cpu().numpy()
            sigma = self.model.prop.sigma(T_tensor).cpu().numpy()
            nec = self.model.prop.nec(T_tensor).cpu().numpy()
        
        return {
            'T_physical': T_physical,
            'kappa': kappa,
            'sigma': sigma,
            'nec': nec,
            'max_T': T_physical.max(),
            'T_center': T_physical[0],
            't_physical': t_reduced * self.t_red
        }

    def _make_figure(self, epoch: int, T_reduced_all: np.ndarray, **kwargs) -> plt.Figure:
        """
        Build multi-panel figure showing temperature evolution over time steps.
        
        Parameters:
        -----------
        epoch : int
            Current training epoch
        T_reduced_all : np.ndarray
            Temperature array of shape (n_time, n_r) in reduced units
        kwargs : dict
            Optional training info (e.g., total_loss)
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with temperature panels in grid layout and 
            loss curve panel spanning the bottom row
        """
        n_time = len(self.t_eval)
        
        # Determine layout: temperature panels in grid, loss panel spans bottom
        n_cols = min(3, n_time)  # Max 3 columns for temperature panels
        n_rows_temp = (n_time + n_cols - 1) // n_cols  # Rows needed for temperature panels
        n_rows_total = n_rows_temp + 1  # Add one row for loss panel
        
        fig = plt.figure(figsize=(5*n_cols, 4*n_rows_temp + 3))
        gs = fig.add_gridspec(n_rows_total, n_cols, hspace=0.4, wspace=0.3)
        
        # Plot temperature at each time step
        for i in range(n_time):
            ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
            
            T_phys = T_reduced_all[i] * self.T_red
            t_val = self.t_eval[i]
            t_physical = t_val * self.t_red

            props = self._compute_material_properties_at_t(T_phys, t_val)
            
            ax.plot(self.x_eval, T_phys, 'b-', linewidth=2.5, label='CS-PINN Prediction')

            # Plot reference temperature if available
            T_ref_func = self.T_ref_func_list[i]
            if T_ref_func is not None:
                T_ref = T_ref_func(self.x_eval*self.R)
                ax.plot(self.x_eval, T_ref, 'r--', linewidth=2.0, label='Reference', alpha=0.8)
                # Compute and display error metrics
                error = np.abs(T_phys.flatten() - T_ref.flatten())
                relative_error = error / (T_ref.flatten() + 1e-10) * 100
                max_error = error.max()
                mean_error = error.mean()
                max_rel_error = relative_error.max()
                rel_l2_error = calc_relative_l2_err(T_ref, T_phys)
                info_text = (f'Max Error: {max_error:.1f} K\n'
                            f'Mean Error: {mean_error:.1f} K\n'
                            f'Max Rel Error: {max_rel_error:.2f}%\n'
                            f'Rel L2 Error: {rel_l2_error:.5g}')
            else:
                info_text = (f'Epoch {epoch}\n'
                            f'Max T: {props["max_T"]:.0f} K\n'
                            f'Center T: {props["T_center"]:.0f} K')
            
            ax.set_xlabel('Normalized radius r/R', fontsize=10)
            ax.set_ylabel('Temperature (K)', fontsize=10)
            ax.set_title(f'Time t = {t_physical*1e3:.2f} ms (epoch={epoch})', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)
            
            # Add info box
            ax.text(0.05, 0.05, info_text,
                   transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add loss curve panel spanning the entire bottom row
        ax_loss = fig.add_subplot(gs[n_rows_temp, :])  # Span all columns in last row
        if self.history['losses']:
            loss_epochs = self.history['epochs']
            loss_values = self.history['losses']
            ax_loss.semilogy(loss_epochs, loss_values, 'purple', linewidth=2.5, marker='o', markersize=4)
            ax_loss.set_xlabel('Epoch', fontsize=11)
            ax_loss.set_ylabel('Loss (log scale)', fontsize=11)
            ax_loss.set_title('Training Loss Convergence', fontsize=12, fontweight='bold')
            ax_loss.grid(True, alpha=0.3, which='both')
            
            current_loss = kwargs.get('total_loss', None)
            if current_loss is not None:
                if isinstance(current_loss, torch.Tensor):
                    loss_val = current_loss.item()
                else:
                    loss_val = float(current_loss)
                ax_loss.text(0.98, 0.95, f'Current loss: {loss_val:.2e}',
                            transform=ax_loss.transAxes, fontsize=10, verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        else:
            ax_loss.text(0.5, 0.5, 'No loss history yet',
                        ha='center', va='center', transform=ax_loss.transAxes,
                        fontsize=12, style='italic', color='gray')
            ax_loss.set_title('Training Loss', fontsize=12, fontweight='bold')
        
        fig.suptitle(f'Arc 1D Transient PINN Model - Epoch {epoch}', 
                    fontsize=14, fontweight='bold', y=0.995)
        return fig

    def visualize(self, network, epoch: int, writer: SummaryWriter, **kwargs):
        """
        Generate visualization plots for the current training epoch.
        
        Parameters:
        -----------
        network : nn.Module
            The neural network being trained
        epoch : int
            Current training epoch number
        writer : SummaryWriter
            TensorBoard writer for logging figures
        kwargs : dict
            Additional training information (e.g., 'total_loss')
        
        Returns:
        --------
        dict
            Dictionary mapping plot names to matplotlib figures
        """
        network.eval()
        
        # Generate predictions on evaluation grid at all time points
        T_reduced_all = []
        with torch.no_grad():
            for t_val in self.t_eval:
                # Create (r, t) pairs for this time step
                x_r = self.x_eval.reshape(-1, 1)
                t_reps = np.repeat(t_val, len(x_r)).astype(REAL()).reshape(-1, 1)
                xt_grid = np.hstack([x_r, t_reps])
                
                # Predict
                T_reduced = network(numpy2torch(xt_grid, require_grad=False)).cpu().numpy()
                T_reduced_all.append(T_reduced)
        
        T_reduced_all = np.array(T_reduced_all)  # shape: (n_time, n_r)
        
        # Save history for animation
        if self.save_history and epoch % self.history_freq == 0:
            self.history['epochs'].append(epoch)
            self.history['T'].append(T_reduced_all.copy())
            
            # Track center temperature at each time point
            T_center_at_times = T_reduced_all[:, 0]
            self.history['T_center_t'].append(T_center_at_times)
            
            # Extract loss
            total_loss = kwargs.get('total_loss', None)
            if total_loss is not None:
                if isinstance(total_loss, torch.Tensor):
                    self.history['losses'].append(total_loss.item())
                else:
                    self.history['losses'].append(float(total_loss))
        
        # Create figure
        fig = self._make_figure(epoch=epoch, T_reduced_all=T_reduced_all, **kwargs)
        
        # Optionally save frame for GIF
        if self.gif_enabled and (epoch % self.gif_freq == 0):
            frame_path = os.path.join(self.gif_tmp_dir, f'epoch_{epoch:06d}.png')
            try:
                fig.savefig(frame_path, dpi=120)
                self._gif_frames.append(frame_path)
            except Exception as e:
                print(f'Warning: failed to save GIF frame at epoch {epoch}: {e}')
        
        return {'transient_visualization': fig}

    def save_gif(self, gif_path: str = None, duration_ms: int = None, loop: int = 0):
        """
        Assemble saved frames into a GIF showing transient temperature evolution and loss.
        
        Parameters:
        -----------
        gif_path : str, optional
            Output path for the GIF; defaults to <gif_dir>/training_animation_transient.gif
        duration_ms : int, optional
            Per-frame duration in milliseconds; defaults to gif_duration_ms
        loop : int, default=0
            Number of loops (0 = infinite)
        """
        if not self.gif_enabled:
            print('GIF not enabled; nothing to save.')
            return
        if len(self._gif_frames) == 0:
            print('No frames collected; GIF not created.')
            return
        
        out_path = gif_path if gif_path is not None else os.path.join(self.gif_dir, self.gif_filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        frames_sorted = sorted(self._gif_frames)
        
        try:
            img2gif(frames_sorted, out_path, duration=(duration_ms or self.gif_duration_ms), loop=loop)
            print(f'GIF saved: {out_path}')
        except Exception as e:
            print(f'Failed to create GIF: {e}')
            return
        
        if self.gif_cleanup_tmp:
            try:
                for f in frames_sorted:
                    if os.path.exists(f):
                        os.remove(f)
                if os.path.isdir(self.gif_tmp_dir) and len(os.listdir(self.gif_tmp_dir)) == 0:
                    os.rmdir(self.gif_tmp_dir)
            except Exception as e:
                print(f'Warning: failed cleanup of tmp frames: {e}')

    def save_final_results(self, network: nn.Module, save_dir: str = None, epoch: int = None, **kwargs):
        """
        Save final result figures showing temperature evolution and loss curve.
        
        Outputs:
        - final_panels_transient.png: multi-panel figure with temperature at different times and loss curve
        - loss_curve_transient.png: standalone loss curve plot
        - center_temp_evolution.png: center temperature evolution over time at different epochs
        
        Parameters:
        -----------
        network : nn.Module
            Trained network for final prediction
        save_dir : str, optional
            Output directory; defaults to gif_dir
        epoch : int, optional
            Epoch number for title; if None, uses last recorded or 0
        kwargs : dict
            Optional training info
        """
        out_dir = save_dir or self.gif_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # Generate final predictions
        network.eval()
        T_reduced_all = []
        with torch.no_grad():
            for t_val in self.t_eval:
                x_r = self.x_eval.reshape(-1, 1)
                t_reps = np.repeat(t_val, len(x_r)).astype(REAL()).reshape(-1, 1)
                xt_grid = np.hstack([x_r, t_reps])
                T_reduced = network(numpy2torch(xt_grid, require_grad=False)).cpu().numpy()
                T_reduced_all.append(T_reduced)
        
        T_reduced_all = np.array(T_reduced_all)
        
        # Determine epoch label
        if epoch is None:
            epoch = self.history['epochs'][-1] if self.history['epochs'] else 0
        
        # Save multi-panel figure (temperature profiles + loss curve)
        fig = self._make_figure(epoch=epoch, T_reduced_all=T_reduced_all, **kwargs)
        panels_path = os.path.join(out_dir, 'final_panels.png')
        try:
            fig.savefig(panels_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Final panels saved: {panels_path}')
        except Exception as e:
            print(f'Failed to save final panels: {e}')
        
        # Save standalone loss curve
        if self.history['losses']:
            plt.figure(figsize=(7, 5))
            plt.semilogy(self.history['epochs'], self.history['losses'], 'purple', 
                        linewidth=2.0, marker='o', markersize=4)
            plt.xlabel('Epoch', fontsize=11)
            plt.ylabel('Loss (log scale)', fontsize=11)
            plt.title('Training Loss Curve (Transient Arc Model)', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3, which='both')
            loss_path = os.path.join(out_dir, 'loss_curve.png')
            try:
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                print(f'Loss curve saved: {loss_path}')
            except Exception as e:
                print(f'Failed to save loss curve: {e}')
            finally:
                plt.close()
        else:
            print('No loss history recorded; loss_curve.png not created.')
        
        
class TraArc1DVelNet(nn.Module):
    """
    Neural network wrapper for solving 1D transient arc with radial velocity using
    automatic boundary condition enforcement.

    This network automatically satisfies the boundary conditions at r = 0 (symmetry)
    by construction. The velocity output is modified to ensure V(0, t) = 0 for all
    time steps, which is a physical requirement due to radial symmetry.

    The network applies a transformation to the backbone network output:
        V(r, t) = r · N(r, t)

    where N(r, t) is the backbone network output (takes both r and t as inputs),
    and r is the normalized radial coordinate. At r = 0, the symmetry condition
    V(0, t) = 0 is automatically satisfied. In arc discharge simulations, the
    radial velocity must be zero due to cylindrical symmetry.

    Parameters (Constructor)
    ------------------------
    network : nn.Module
        Backbone neural network (e.g., FNN) that maps (r, t) → N(r, t)
        Input shape: [batch_size, 2] where [:, 0]=r and [:, 1]=t
        Output shape: [batch_size, 1] representing velocity prediction
    """
    def __init__(self, network):
        super(TraArc1DVelNet, self).__init__()

        self.network = network

    def forward(self, x):
        """
        Forward pass with automatic symmetry condition enforcement.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, 2] where:
            x[:, 0] : r (normalized radius, range [0, 1])
            x[:, 1] : t (normalized time, range [0, 1])

        Returns
        -------
        torch.Tensor
            Radial velocity V(r, t) in normalized units, shape [batch_size, 1]
            Satisfies V(0, t) = 0 for all t by construction (symmetry)
        """
        out = self.network(x)
        V = out*x[:,0:1]  # Enforce V(0, t) = 0 by multiplication with r
        return V
    

class TraArc1DNet(nn.Module):
    """
    Neural network wrapper for coupled 1D transient arc equations with automatic
    boundary condition enforcement.

    This network simultaneously predicts both temperature T(r, t) and radial velocity
    V(r, t) while automatically satisfying multiple boundary conditions:
    1. Temperature boundary condition: T(R, t) = Tb (at arc boundary)
    2. Velocity symmetry condition: V(0, t) = 0 (at centerline)

    The network applies transformations to the backbone network outputs:
        T(r, t) = (r - R) · N₁(r, t) + Tb
        V(r, t) = r · N₂(r, t)

    where N₁(r, t) and N₂(r, t) are the two outputs of the backbone network.
    This design is for coupled energy and momentum transport in arc discharges,
    where the automatic boundary condition enforcement reduces training complexity.

    Parameters (Constructor)
    ------------------------
    network : nn.Module
        Backbone neural network (e.g., FNN) that maps (r, t) → [N₁(r, t), N₂(r, t)]
        Input shape: [batch_size, 2] where [:, 0]=r and [:, 1]=t
        Output shape: [batch_size, 2] representing [temperature, velocity] predictions
    R : float, optional
        Normalized arc radius (default: 1.0)
    Tb : float, optional
        Normalized boundary temperature at r = R (default: 0.03)
    """
    def __init__(self, network, R=1.0, Tb=0.03):
        super(TraArc1DNet, self).__init__()

        self.network = network
        self.R = R  # Reduced arc radius
        self.Tb = Tb  # Reduced boundary temperature at r=R

    def forward(self, x):
        """
        Forward pass with automatic boundary condition enforcement for both T and V.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, 2] where:
            x[:, 0] : r (normalized radius, range [0, 1])
            x[:, 1] : t (normalized time, range [0, 1])

        Returns
        -------
        T : torch.Tensor
            Temperature T(r, t) in normalized units, shape [batch_size, 1]
            Satisfies T(R, t) = Tb for all t by construction
        V : torch.Tensor
            Radial velocity V(r, t) in normalized units, shape [batch_size, 1]
            Satisfies V(0, t) = 0 for all t by construction (symmetry)
        """
        out = self.network(x)
        T = out[:,0:1]*(x[:,0:1] - self.R) + self.Tb  # Enforce T(R, t) = Tb
        V = out[:,1:2]*x[:,0:1]  # Enforce V(0, t) = 0
        return T, V


def get_TVfunc_from_file(csv_file):
    """
    Load reference temperature and velocity profiles from CSV file for comparison.

    Parameters
    ----------
    csv_file : str
        Path to CSV file containing reference temperature and velocity data.
        The CSV should have columns: 'r(m)' (radius in meters),
        'T(K)' (temperature in Kelvin), and 'V(m/s)' (velocity in m/s).

    Returns
    -------
    T_spline : function
        Cubic spline interpolation function for temperature at radius r
    V_spline : function
        Cubic spline interpolation function for velocity at radius r

    Raises
    ------
    FileNotFoundError
        If csv_file does not exist.
    """

    # Check file existence
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    r_data = df['r(m)'].values.astype(REAL())
    V_data = df['V(m/s)'].values.astype(REAL())
    T_data = df['T(K)'].values.astype(REAL())
    V_spline = intp.CubicSpline(r_data, V_data, extrapolate=True)
    T_spline = intp.CubicSpline(r_data, T_data, extrapolate=True)
    
    return T_spline, V_spline


class TraArc1DModel(PINN):
    """
    PINN model for solving coupled 1D transient arc plasma equations with temperature
    and radial velocity.

    Implements a Physics-Informed Neural Network specifically designed for simulating
    transient (time-dependent) arc discharge phenomena with full coupling between energy
    and momentum transport. Solves coupled nonlinear transient equations considering
    temperature evolution, radial velocity, thermal conduction, convection, and
    radiation effects.

    Physical Constraints
    --------------------
    - T(R, t) = Tb (Dirichlet boundary condition at arc radius)
    - ∂T/∂r(0, t) = 0 (symmetry condition for temperature at centerline)
    - V(0, t) = 0 (symmetry condition for velocity at centerline)
    - T(r, 0) = Tinit_func(r) (initial temperature distribution)
    - Mass conservation through continuity equation

    Parameters (Constructor)
    ------------------------
    R : float
        Arc radius [m]
    I : float
        Arc current [A]
    Tb : float, optional
        Boundary temperature at r = R [K] (default: 300.0)
    Tinit_func : callable
        Function that returns initial temperature profile T(r)
    T_red : float, optional
        Temperature reduction factor for normalization [K] (default: 1e4)
    t_red : float, optional
        Time reduction factor for normalization [s] (default: 1e-3)
    backbone_net : nn.Module, optional
        Backbone neural network with 2 outputs (T, V)
        (default: FNN with 7 layers)
    train_data_x_size : int, optional
        Number of spatial training collocation points (default: 200)
    train_data_t_size : int, optional
        Number of temporal training collocation points (default: 100)
    sample_mode : str, optional
        Sampling strategy: 'uniform', 'lhs', or 'random' (default: 'uniform')
    prop : ArcPropSpline, optional
        Arc material properties object (default: None)
    """
    def __init__(
        self,
        R,
        I,
        Tb=300.0,
        Tinit_func=None,
        T_red=1e4,
        t_red=1e-3,
        backbone_net=FNN(layers=[2, 300, 300, 300, 300, 300, 300, 2]),
        train_data_x_size=200,
        train_data_t_size=100,
        sample_mode='uniform',
        prop:ArcPropSpline=None,
    ):
        self.R = R 
        self.I = I 
        self.T_red = T_red
        self.Tb = Tb
        if Tinit_func is None:
            raise ValueError('Tinit_func (initial condition) must be provided for transient arc model.')
        else:
            self.Tinit_func = Tinit_func
        self.t_red = t_red
        self.train_data_x_size = train_data_x_size
        self.train_data_t_size = train_data_t_size
        self.sample_mode = sample_mode
        self.prop = prop

        self.geo = Geo1DTime([0.0, 1.0], ts=0.0, te=1.0)
        network = TraArc1DNet(backbone_net, R=1.0, Tb=Tb/T_red)
        super().__init__(network)

        self.set_loss_func(F.smooth_l1_loss)

    
    def _define_loss_terms(self):
        """
        Define physics-informed loss terms for the coupled transient arc model.
        
        This method constructs the complete loss function by defining residuals for:
        1. Energy PDE in the domain (temperature evolution with convection)
        2. Continuity PDE in the domain (mass conservation with velocity)
        3. Boundary condition at r=0 (symmetry: ∂T/∂r = 0)
        4. Initial condition at t=0 (prescribed temperature distribution)
        """
        def _pde_T_residual(network, x):
            """
            PDE residual for transient arc equation (Temperature equation).
            
            Parameters
            ----------
            network : nn.Module
                Neural network model
            x : torch.Tensor
                Input tensor where x[:,0] is r and x[:,1] is t
            """
            T, V = network(x)
            kappa = self.prop.kappa(T.view(-1)*self.T_red).view(-1,1)
            Cp = self.prop.Cp(T.view(-1)*self.T_red).view(-1,1)
            rho = self.prop.rho(T.view(-1)*self.T_red).view(-1,1)
            nec = self.prop.nec(T.view(-1)*self.T_red).view(-1,1)

            joule = 0
            radiation = 4*np.pi*nec
            net_energy = joule - radiation

            T_x = df_dX(T, x)
            T_r = T_x[:,0:1]
            T_t = T_x[:,1:2]

            r = x[:,0:1]
            T_term = r*kappa*T_r
            T_xx = df_dX(T_term, x)
            T_rr = T_xx[:,0:1]

            func = T_t + V*T_r*(self.t_red/self.R) - (net_energy*(self.t_red/self.T_red) + T_rr/r*(self.t_red/self.R/self.R))/(rho*Cp)

            return func
        
        def _pde_V_residual(network, x):
            """
            PDE residual for transient arc equation (Velocity equation).
            
            Parameters
            ----------
            network : nn.Module
                Neural network model
            x : torch.Tensor
                Input tensor where x[:,0] is r and x[:,1] is t
            """
            T, V = network(x)
            rho = self.prop.rho(T.view(-1)*self.T_red).view(-1,1)

            r = x[:,0:1]
            rho_grad = df_dX(rho, x)
            rho_t = rho_grad[:,1:2]
            V_term = r*rho*V
            V_term_grad = df_dX(V_term, x)
            V_r = V_term_grad[:,0:1]

            func = rho_t + V_r/r*(self.t_red/self.R)

            return func
        
        def _bc_T_residual(network, x):
            """
            Boundary condition residual at r=0 (symmetry condition).
            """
            T, _ = network(x)
            T_x = df_dX(T, x)

            func_bc = T_x[:,0:1]
            return func_bc
        

        def _init_T_residual(network, x):
            """
            Initial condition residual at t=0.
            """
            T, _ = network(x)
            func_ic = T - Ti
            return func_ic
        
        # Sample domain collocation points
        xt_domain, xt_bc = self.geo.sample_all_domain(Nx=self.train_data_x_size, 
                                                      Nt=self.train_data_t_size, 
                                                      mode=[self.sample_mode, self.sample_mode])
        xb = xt_bc[0][0]
        xi = xt_bc[1]

        if isinstance(xi, torch.Tensor):
            _xi = xi.detach().cpu().numpy()
            Ti = self.Tinit_func(_xi[:,0:1]*self.R)/self.T_red
            Ti = numpy2torch(Ti)
        else: # np.ndarray
            Ti = self.Tinit_func(xi[:,0:1]*self.R)/self.T_red
        
        # Add equation terms with weights
        self.add_equation('Domain_T', _pde_T_residual, weight=1.0, data=xt_domain)
        self.add_equation('Domain_V', _pde_V_residual, weight=1.0, data=xt_domain)
        self.add_equation('Left Boundary', _bc_T_residual, weight=10.0, data=xb)
        self.add_equation('Initial Condition', _init_T_residual, weight=10.0, data=xi)


class TraArc1DVisCallback(VisualizationCallback):
    """
    Custom visualization callback for 1D transient arc PINN training with temperature and radial velocity.
    
    This callback provides comprehensive monitoring of the coupled transient arc model training:
      1. Real-time TensorBoard logging during training
      2. Side-by-side comparison of temperature and velocity at multiple time steps
      3. Training history tracking for post-training animation
      4. Reference data comparison with error metrics (when CSV files provided)
      5. Material property evolution monitoring over time
      6. Loss convergence tracking with logarithmic scale visualization
    """
    
    def __init__(self, model: 'TraArc1DModel', log_freq: int = 50, 
                 save_history: bool = True, history_freq: int = None,
                 x_eval: np.ndarray = np.linspace(0, 1, 201, dtype=REAL()).reshape(-1,1),
                 t_eval: list = [0.1, 0.5, 0.9],
                 TV_csv_file: list[str] = ['','',''], # Paths to CSV files for reference temperature and velocity data at different times
                #  num_time_snapshots: int = 5,
                 gif_enabled: bool = False,
                 gif_dir: str = None,
                 gif_freq: int = None,
                 gif_duration_ms: int = 300,
                 gif_cleanup_tmp: bool = True):
        """
        Initialize the transient arc visualization callback for real-time training monitoring.
        
        This callback provides comprehensive real-time monitoring and post-training visualization
        capabilities for coupled transient arc simulations with both temperature and velocity.
        It tracks the evolution of both fields at multiple time points and generates
        publication-quality figures with side-by-side T/V comparison and animations.
        
        Parameters:
        -----------
        model : TraArc1DModel
            The coupled transient arc model instance for temperature and velocity simulation.
            Used to access geometry, physical parameters, and material properties.
        
        log_freq : int, default=50
            Frequency (in epochs) for logging visualizations to TensorBoard.
            Controls how often multi-panel figures with T/V comparison are generated and logged.
        
        save_history : bool, default=True
            Whether to save prediction snapshots for creating training animations.
            Set to False to reduce memory consumption if animations are not needed.
        
        history_freq : int, optional
            Frequency (in epochs) for saving history snapshots.
            If None, defaults to log_freq.
            Use larger values (e.g., 200) to reduce memory for long training runs.
        
        x_eval : np.ndarray, shape (n_r, 1)
            Radial evaluation grid for visualization (normalized radius, 0 to 1).
            Default: 201 points linearly spaced from 0 to 1.
            Finer grids provide higher-resolution visualizations but slower evaluation.
        
        t_eval : list or np.ndarray
            List of normalized time points (0 to 1) at which to display T and V
            snapshots in the multi-panel figures.
            Example: [0.1, 0.5, 0.9] shows distributions at 10%, 50%, 90% of total time.
            Default: [0.1, 0.5, 0.9] (3 time snapshots per epoch).
            Each time point creates one row with side-by-side T/V panels.
        
        TV_csv_file : list[str], optional
            Paths to CSV files containing reference temperature and velocity data
            at different times for comparison with PINN predictions.
            - One CSV file per time point in t_eval (must match length)
            - CSV columns: 'r(m)' (radius in meters), 'T(K)' (temperature in Kelvin),
                          'V(m/s)' (velocity in meters per second)
            - Empty string '' or None skips reference comparison for that time point
            Example: ['ref_0.1ms.csv', 'ref_0.5ms.csv', 'ref_0.9ms.csv']
        
        gif_enabled : bool, default=False
            Whether to save per-epoch frames and generate a training animation GIF.
            When True, PNG frames are saved at gif_freq intervals showing the evolution
            of T and V distributions and loss convergence over training epochs.
            Enables visual inspection of training dynamics and convergence behavior.
        
        gif_dir : str, optional (default=None)
            Output directory for GIF animation and final summary plots.
            If None, defaults to current working directory.
        
        gif_freq : int, optional
            Frequency (in epochs) to save frames for GIF assembly.
            If None, uses history_freq.
            Use larger values (e.g., 500) for smaller GIF files and faster creation.
        
        gif_duration_ms : int, default=300
            Duration per frame in milliseconds for the final GIF animation.
        
        gif_cleanup_tmp : bool, default=True
            Whether to automatically delete temporary PNG frames after GIF creation.
            Set to False to retain frames for manual inspection or re-assembly.
        """
        super().__init__(name='CS-PINN_1D_Arc_Transient', log_freq=log_freq)
        
        self.model = model
        self.x_eval = x_eval
        self.t_eval = t_eval
        self.xt_list = model.geo.sample_space_time_list(x_eval, t_eval, require_grad=False)
        
        # Load reference T and V functions from CSV files
        TV_funcs = [get_TVfunc_from_file(csv_file) if csv_file else (None, None) for csv_file in TV_csv_file]
        self.T_ref_func_list = [T_func for T_func, _ in TV_funcs]
        self.V_ref_func_list = [V_func for _, V_func in TV_funcs]
        
        # Physical parameters
        self.T_red = model.T_red
        self.t_red = model.t_red
        self.R = model.R
        self.I = model.I
        self.Tb = model.Tb
        self.Tinit_func = model.Tinit_func
        
        # Training history tracking
        self.save_history = save_history
        self.history_freq = history_freq if history_freq is not None else log_freq
        self.history = {
            'epochs': [],           # List of epoch numbers
            'r_eval': self.x_eval,  # Radial evaluation grid (fixed)
            't_eval': self.t_eval,  # Time evaluation points (fixed)
            'T': [],                # List of T(r,t) arrays [n_epochs, n_time, n_r]
            'V': [],                # List of V(r,t) arrays [n_epochs, n_time, n_r]
            'losses': [],           # List of total loss values
            'T_center_t': [],       # List of center temp at all time points
        }
        
        # GIF configuration/state
        self.gif_enabled = gif_enabled
        if gif_dir is None:
            self.gif_dir = os.getcwd()
        else:
            self.gif_dir = gif_dir
        self.gif_tmp_dir = os.path.join(self.gif_dir, 'tmp_frames')
        self.gif_filename = 'training_animation.gif'
        self.gif_freq = gif_freq if gif_freq is not None else self.history_freq
        self.gif_duration_ms = gif_duration_ms
        self.gif_cleanup_tmp = gif_cleanup_tmp
        self._gif_frames = []
        if self.gif_enabled:
            os.makedirs(self.gif_dir, exist_ok=True)
            os.makedirs(self.gif_tmp_dir, exist_ok=True)

    def _compute_material_properties_at_t(self, T_physical: np.ndarray, t_reduced: float) -> dict:
        """
        Compute material properties at a given time step and temperature profile.
        
        Parameters:
        -----------
        T_physical : np.ndarray
            Temperature in physical units (K), shape (n_r,)
        t_reduced : float
            Normalized time value for reference in output
        
        Returns:
        --------
        dict
            Dictionary containing material properties and statistics
        """
        if self.model.prop is None:
            return {
                'T_physical': T_physical,
                'kappa': None,
                'sigma': None,
                'nec': None,
                'max_T': T_physical.max(),
                'T_center': T_physical[0],
                't_physical': t_reduced * self.t_red
            }
        
        with torch.no_grad():
            T_tensor = numpy2torch(T_physical, require_grad=False)
            kappa = self.model.prop.kappa(T_tensor).cpu().numpy()
            sigma = self.model.prop.sigma(T_tensor).cpu().numpy()
            nec = self.model.prop.nec(T_tensor).cpu().numpy()
        
        return {
            'T_physical': T_physical,
            'kappa': kappa,
            'sigma': sigma,
            'nec': nec,
            'max_T': T_physical.max(),
            'T_center': T_physical[0],
            't_physical': t_reduced * self.t_red
        }

    def _make_figure(self, epoch: int, T_reduced_all: np.ndarray, V_reduced_all: np.ndarray, **kwargs) -> plt.Figure:
        """
        Build multi-panel figure showing temperature and velocity evolution over time steps.
        Each time step shows T and V side by side for easy comparison.
        
        Parameters:
        -----------
        epoch : int
            Current training epoch
        T_reduced_all : np.ndarray
            Temperature array of shape (n_time, n_r) in reduced units
        V_reduced_all : np.ndarray
            Velocity array of shape (n_time, n_r) in reduced units
        kwargs : dict
            Optional training info (e.g., total_loss)
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure with T/V panels side by side for each time step,
            plus loss curve panel spanning the bottom row
        """
        n_time = len(self.t_eval)
        
        # Layout: Each row shows one time step with T and V side by side (2 columns per time)
        # Plus 1 extra row for loss curve at the bottom
        n_cols = 2  # T and V always side by side
        n_rows = n_time + 1  # One row per time step, plus loss row
        
        fig = plt.figure(figsize=(14, 4*n_time + 3))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.35, wspace=0.25,
                             height_ratios=[1]*n_time + [0.8])
        
        # Plot temperature and velocity at each time step (side by side)
        for i in range(n_time):
            T_phys = T_reduced_all[i] * self.T_red
            V_phys = V_reduced_all[i]
            t_val = self.t_eval[i]
            t_physical = t_val * self.t_red

            props = self._compute_material_properties_at_t(T_phys, t_val)
            
            # Temperature subplot (left column)
            ax_T = fig.add_subplot(gs[i, 0])
            ax_T.plot(self.x_eval, T_phys, 'b-', linewidth=2.5, label='CS-PINN')

            # Plot reference temperature if available
            T_ref_func = self.T_ref_func_list[i]
            if T_ref_func is not None:
                T_ref = T_ref_func(self.x_eval*self.R)
                ax_T.plot(self.x_eval, T_ref, 'r--', linewidth=2.0, label='Reference', alpha=0.8)
                error_T = np.abs(T_phys.flatten() - T_ref.flatten())
                relative_error = error_T / (T_ref.flatten() + 1e-10) * 100
                max_error = error_T.max()
                mean_error = error_T.mean()
                max_rel_error = relative_error.max()
                rel_l2_error = calc_relative_l2_err(T_ref, T_phys)
                info_text_T = (f'Max Error: {max_error:.1f} K\n'
                            f'Mean Error: {mean_error:.1f} K\n'
                            f'Max Rel Error: {max_rel_error:.2f}%\n'
                            f'Rel L2 Error: {rel_l2_error:.5g}')
            else:
                info_text_T = (f'Max T: {props["max_T"]:.0f} K\n'
                              f'Center: {props["T_center"]:.0f} K')
            
            ax_T.set_xlabel('Normalized radius r/R', fontsize=11)
            ax_T.set_ylabel('Temperature (K)', fontsize=11)
            ax_T.set_title(f'Temperature @ t = {t_physical*1e3:.2f} ms', fontsize=12, fontweight='bold')
            ax_T.grid(True, alpha=0.3)
            ax_T.legend(loc='best', fontsize=10)
            
            # Add info box for temperature
            ax_T.text(0.05, 0.05, info_text_T,
                     transform=ax_T.transAxes, fontsize=9, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

            # Velocity subplot (right column, same row)
            ax_V = fig.add_subplot(gs[i, 1])
            ax_V.plot(self.x_eval, V_phys, 'g-', linewidth=2.5, label='CS-PINN')

            V_ref_func = self.V_ref_func_list[i]
            if V_ref_func is not None:
                V_ref = V_ref_func(self.x_eval*self.R)
                ax_V.plot(self.x_eval, V_ref, 'r--', linewidth=2.0, label='Reference', alpha=0.8)
                error_V = np.abs(V_phys.flatten() - V_ref.flatten())

                relative_error = error_V / (np.abs(V_ref.flatten()) + 1e-10) * 100
                max_error = error_V.max()
                mean_error = error_V.mean()
                max_rel_error = relative_error.max()
                rel_l2_error = calc_relative_l2_err(V_ref, V_phys)
                info_text_V = (f'Max Error: {max_error:.3g} m/s\n'
                            f'Mean Error: {mean_error:.3g} m/s\n'
                            f'Max Rel Error: {max_rel_error:.2f}%\n'
                            f'Rel L2 Error: {rel_l2_error:.5g}')
            else:
                info_text_V = f'Max: {V_phys.max():.3g} m/s'

            ax_V.set_xlabel('Normalized radius r/R', fontsize=11)
            ax_V.set_ylabel('Velocity (m/s)', fontsize=11)
            ax_V.set_title(f'Velocity @ t = {t_physical*1e3:.2f} ms', fontsize=12, fontweight='bold')
            ax_V.grid(True, alpha=0.3)
            ax_V.legend(loc='best', fontsize=10)
            ax_V.text(0.05, 0.05, info_text_V,
                     transform=ax_V.transAxes, fontsize=9, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        
        # Add loss curve panel spanning the entire bottom row
        ax_loss = fig.add_subplot(gs[n_time, :])  # Span all columns in last row
        if self.history['losses']:
            loss_epochs = self.history['epochs']
            loss_values = self.history['losses']
            ax_loss.semilogy(loss_epochs, loss_values, 'purple', linewidth=2.5, marker='o', markersize=4)
            ax_loss.set_xlabel('Epoch', fontsize=11)
            ax_loss.set_ylabel('Loss (log scale)', fontsize=11)
            ax_loss.set_title('Training Loss Convergence', fontsize=12, fontweight='bold')
            ax_loss.grid(True, alpha=0.3, which='both')
            
            current_loss = kwargs.get('total_loss', None)
            if current_loss is not None:
                if isinstance(current_loss, torch.Tensor):
                    loss_val = current_loss.item()
                else:
                    loss_val = float(current_loss)
                ax_loss.text(0.98, 0.95, f'Current loss: {loss_val:.2e}',
                            transform=ax_loss.transAxes, fontsize=10, verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        else:
            ax_loss.text(0.5, 0.5, 'No loss history yet',
                        ha='center', va='center', transform=ax_loss.transAxes,
                        fontsize=12, style='italic', color='gray')
            ax_loss.set_title('Training Loss', fontsize=12, fontweight='bold')
        
        fig.suptitle(f'Arc 1D Transient PINN Model (T & V) - Epoch {epoch}', 
                fontsize=14, fontweight='bold', y=0.995)
        return fig

    def visualize(self, network, epoch: int, writer: SummaryWriter, **kwargs):
        """
        Generate visualization plots for the current training epoch.
        
        Parameters:
        -----------
        network : nn.Module
            The neural network being trained
        epoch : int
            Current training epoch number
        writer : SummaryWriter
            TensorBoard writer for logging figures
        kwargs : dict
            Additional training information (e.g., 'total_loss')
        
        Returns:
        --------
        dict
            Dictionary mapping plot names to matplotlib figures
        """
        network.eval()
        
        # Generate predictions on evaluation grid at all time points
        T_reduced_all = []
        V_reduced_all = []
        with torch.no_grad():
            for t_val in self.t_eval:
                # Create (r, t) pairs for this time step
                x_r = self.x_eval.reshape(-1, 1)
                t_reps = np.repeat(t_val, len(x_r)).astype(REAL()).reshape(-1, 1)
                xt_grid = np.hstack([x_r, t_reps])
                
                # Predict
                T_reduced, V_reduced = network(numpy2torch(xt_grid, require_grad=False))
                T_reduced_all.append(T_reduced.cpu().numpy())
                V_reduced_all.append(V_reduced.cpu().numpy())
        
        T_reduced_all = np.array(T_reduced_all)  # shape: (n_time, n_r)
        V_reduced_all = np.array(V_reduced_all)  # shape: (n_time, n_r)
        
        # Save history for animation
        if self.save_history and epoch % self.history_freq == 0:
            self.history['epochs'].append(epoch)
            self.history['T'].append(T_reduced_all.copy())
            self.history['V'].append(V_reduced_all.copy())
            
            # Track center temperature at each time point
            T_center_at_times = T_reduced_all[:, 0]
            self.history['T_center_t'].append(T_center_at_times)
            
            # Extract loss
            total_loss = kwargs.get('total_loss', None)
            if total_loss is not None:
                if isinstance(total_loss, torch.Tensor):
                    self.history['losses'].append(total_loss.item())
                else:
                    self.history['losses'].append(float(total_loss))
        
        # Create figure
        fig = self._make_figure(epoch=epoch, T_reduced_all=T_reduced_all, V_reduced_all=V_reduced_all, **kwargs)
        
        # Optionally save frame for GIF
        if self.gif_enabled and (epoch % self.gif_freq == 0):
            frame_path = os.path.join(self.gif_tmp_dir, f'epoch_{epoch:06d}.png')
            try:
                fig.savefig(frame_path, dpi=120)
                self._gif_frames.append(frame_path)
            except Exception as e:
                print(f'Warning: failed to save GIF frame at epoch {epoch}: {e}')
        
        return {'transient_visualization': fig}

    def save_gif(self, gif_path: str = None, duration_ms: int = None, loop: int = 0):
        """
        Assemble saved frames into a GIF showing transient temperature evolution and loss.
        
        Parameters:
        -----------
        gif_path : str, optional
            Output path for the GIF; defaults to <gif_dir>/training_animation_transient.gif
        duration_ms : int, optional
            Per-frame duration in milliseconds; defaults to gif_duration_ms
        loop : int, default=0
            Number of loops (0 = infinite)
        """
        if not self.gif_enabled:
            print('GIF not enabled; nothing to save.')
            return
        if len(self._gif_frames) == 0:
            print('No frames collected; GIF not created.')
            return
        
        out_path = gif_path if gif_path is not None else os.path.join(self.gif_dir, self.gif_filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        frames_sorted = sorted(self._gif_frames)
        
        try:
            img2gif(frames_sorted, out_path, duration=(duration_ms or self.gif_duration_ms), loop=loop)
            print(f'GIF saved: {out_path}')
        except Exception as e:
            print(f'Failed to create GIF: {e}')
            return
        
        if self.gif_cleanup_tmp:
            try:
                for f in frames_sorted:
                    if os.path.exists(f):
                        os.remove(f)
                if os.path.isdir(self.gif_tmp_dir) and len(os.listdir(self.gif_tmp_dir)) == 0:
                    os.rmdir(self.gif_tmp_dir)
            except Exception as e:
                print(f'Warning: failed cleanup of tmp frames: {e}')

    def save_final_results(self, network: nn.Module, save_dir: str = None, epoch: int = None, **kwargs):
        """
        Save final result figures showing temperature evolution and loss curve.
        
        Outputs:
        - final_panels_transient.png: multi-panel figure with temperature at different times and loss curve
        - loss_curve_transient.png: standalone loss curve plot
        - center_temp_evolution.png: center temperature evolution over time at different epochs
        
        Parameters:
        -----------
        network : nn.Module
            Trained network for final prediction
        save_dir : str, optional
            Output directory; defaults to gif_dir
        epoch : int, optional
            Epoch number for title; if None, uses last recorded or 0
        kwargs : dict
            Optional training info
        """
        out_dir = save_dir or self.gif_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # Generate final predictions
        network.eval()
        T_reduced_all = []
        V_reduced_all = []
        with torch.no_grad():
            for t_val in self.t_eval:
                x_r = self.x_eval.reshape(-1, 1)
                t_reps = np.repeat(t_val, len(x_r)).astype(REAL()).reshape(-1, 1)
                xt_grid = np.hstack([x_r, t_reps])

                T_reduced, V_reduced = network(numpy2torch(xt_grid, require_grad=False))
                T_reduced_all.append(T_reduced.cpu().numpy())
                V_reduced_all.append(V_reduced.cpu().numpy())
        
            T_reduced_all = np.array(T_reduced_all)
            V_reduced_all = np.array(V_reduced_all)
        
        # Determine epoch label
        if epoch is None:
            epoch = self.history['epochs'][-1] if self.history['epochs'] else 0
        
        # Save multi-panel figure (temperature & velocity profiles + loss curve)
        fig = self._make_figure(epoch=epoch, T_reduced_all=T_reduced_all, V_reduced_all=V_reduced_all, **kwargs)
        panels_path = os.path.join(out_dir, 'final_panels.png')
        try:
            fig.savefig(panels_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Final panels saved: {panels_path}')
        except Exception as e:
            print(f'Failed to save final panels: {e}')
        
        # Save standalone loss curve
        if self.history['losses']:
            plt.figure(figsize=(7, 5))
            plt.semilogy(self.history['epochs'], self.history['losses'], 'purple', 
                        linewidth=2.0, marker='o', markersize=4)
            plt.xlabel('Epoch', fontsize=11)
            plt.ylabel('Loss (log scale)', fontsize=11)
            plt.title('Training Loss Curve (Transient Arc Model)', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3, which='both')
            loss_path = os.path.join(out_dir, 'loss_curve.png')
            try:
                plt.savefig(loss_path, dpi=150, bbox_inches='tight')
                print(f'Loss curve saved: {loss_path}')
            except Exception as e:
                print(f'Failed to save loss curve: {e}')
            finally:
                plt.close()
        else:
            print('No loss history recorded; loss_curve.png not created.')
  
