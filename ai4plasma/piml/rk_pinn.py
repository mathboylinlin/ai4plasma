"""RK-PINN models for plasma simulations in AI4Plasma.

This module implements a Runge-Kutta Physics-Informed Neural Network (RK-PINN)
for solving 1D corona discharge problems. It extends standard PINN methodology
with implicit Runge-Kutta time integration schemes for improved accuracy and
stability in temporal evolution of corona discharge phenomena.

RK-PINN Classes
---------------
- `Corona1DRKNet`: Neural network with built-in boundary condition enforcement.
- `Corona1DRKModel`: PINN model for corona discharge using RK time stepping.
- `Corona1DRKVisCallback`: Visualization callback with multi-panel plots.

RK-PINN References
------------------
[1] L. Zhong, B. Wu, and Y. Wang, "Low-temperature plasma simulation
    based on physics-informed neural networks: Frameworks and preliminary
    applications," Physics of Fluids, vol. 34, no. 8, p. 087116, 2022.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as intp
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ai4plasma.core.network import FNN
from ai4plasma.config import REAL
from ai4plasma.utils.common import numpy2torch
from ai4plasma.utils.common import Boltz_k, Elec, Epsilon_0
from ai4plasma.utils.math import calc_relative_l2_err
from ai4plasma.plasma.prop import CoronaPropSpline
from ai4plasma.utils.io import img2gif
from .geo import Geo1D
from .pinn import PINN, VisualizationCallback


def load_butcher_table(q):
    """
    Load Butcher tableau for implicit Runge-Kutta method.
    
    This function loads pre-computed Butcher tableau coefficients for implicit
    Runge-Kutta methods from a .npy file. The file path is constructed relative
    to the location of this module file, ensuring correct loading regardless of
    the current working directory.
    
    If the local file is not found, the function will attempt to download it from
    HuggingFace Datasets (repository: mathboylinlin/ai4plasma_butcher_table).
    
    Parameters
    ----------
    q : int
        Order of the Runge-Kutta method.
    
    Returns
    -------
    torch.Tensor
        Butcher tableau coefficients for the specified order, shape (q+1, q).
    
    Raises
    ------
    FileNotFoundError
        If the Butcher table cannot be loaded locally and download fails.
    ImportError
        If huggingface_hub is not installed when download is attempted.
    """
    # Get the directory of the current file (rk_pinn.py)
    butcher_dir = 'ButcherTable/'
    butcher_file = os.path.join(butcher_dir, 'Butcher_%d.npy' % q)
    
    # Check if file exists locally
    if not os.path.exists(butcher_file):
        print(f"Butcher table file not found locally: {butcher_file}")
        print(f"Attempting to download from HuggingFace...")
        
        # Try to download from HuggingFace
        try:
            # Create ButcherTable directory if it doesn't exist
            os.makedirs(butcher_dir, exist_ok=True)
            
            # Download file from HuggingFace (Dataset repository)
            hf_hub_download(
                repo_id="mathboylinlin/ai4plasma_butcher_table",
                repo_type="dataset",
                filename=f"Butcher_{q}.npy",
                local_dir=butcher_dir
            )
            print(f"Successfully downloaded Butcher table from HuggingFace to {butcher_file}")
            
        except ImportError:
            raise ImportError(
                "huggingface_hub package is required to download Butcher tables. "
                "Please install it using: pip install huggingface_hub"
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load Butcher table for order {q}:\n"
                f"  Local file not found: {butcher_file}\n"
                f"  HuggingFace download failed: {str(e)}\n"
                f"  Dataset repository: https://huggingface.co/datasets/mathboylinlin/ai4plasma_butcher_table\n"
                f"  Please ensure the dataset exists and check your internet connection."
            )
    
    # Load the Butcher table
    _weights = np.load(butcher_file).astype(REAL())
    weights = numpy2torch(np.reshape(_weights[0:q*(q+1)], (q+1, q)), require_grad=False)
    return weights

def get_PhiNe_func_from_file(csv_file):
    """
    Load reference potential and electron density profiles from CSV file for comparison.
    
    This function reads experimental or reference simulation data from a CSV file
    containing electric potential (Φ) and electron density (Ne) profiles as functions
    of radius. The data is interpolated using cubic splines for smooth evaluation
    at arbitrary radial positions.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV file containing reference corona discharge data.
    
    Returns
    -------
    Phi_spline : scipy.interpolate.CubicSpline
        Returns interpolated potential (V) at radius r (m).
    Ne_spline : scipy.interpolate.CubicSpline
        Returns interpolated electron density (m⁻³) at radius r (m).
    
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the specified path.
    KeyError
        If required columns are missing from the CSV file.
    """

    # Check file existence
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    r_data = df['r(cm)'].values.astype(REAL()) * 1e-2  # Convert cm to m
    Phi_data = df['U(V)'].values.astype(REAL())
    Ne_data = df['Ne(m^-3)'].values.astype(REAL())
    Phi_spline = intp.CubicSpline(r_data, Phi_data, extrapolate=True)
    Ne_spline = intp.CubicSpline(r_data, Ne_data, extrapolate=True)
    
    return Phi_spline, Ne_spline


class Corona1DRKNet(nn.Module):
    """
    Neural network wrapper for solving 1D corona discharge with automatic boundary 
    condition enforcement for coupled potential and electron density fields.
    
    This network automatically satisfies boundary conditions at the domain edges
    by construction, reducing training complexity and improving physical consistency.
    
    Parameters
    ----------
    network : nn.Module
        Backbone neural network (e.g., FNN) that maps r → [N₁, N₂].
        Input shape: [batch_size, 1] representing radial coordinate r.
        Output shape: [batch_size, 2*(q+1)] representing [Φ stages, Ne stages].
    q : int
        Order of the implicit Runge-Kutta method.
        Determines number of RK stages: q+1.
        Total network outputs: 2*(q+1) [q+1 for Φ, q+1 for Ne].
    R : float, optional
        Normalized domain radius (default: 1.0).
        For physical radius Rphys meters, normalize as r_norm = r_phys / Rphys.
    V0 : float, optional
        Normalized applied voltage at electrode r=0 (default: None).
        Physical voltage V₀phys is normalized as V_norm = V₀phys / V_red.
    """
    def __init__(self, network, q, R=1.0, V0=None):
        super(Corona1DRKNet, self).__init__()

        self.network = network
        self.q = q
        self.R = R
        self.V0 = V0

    def forward(self, x):
        """
        Forward pass through the network with boundary condition enforcement.
        
        Parameters
        ----------
        x : torch.Tensor
            Radial coordinates, shape (batch_size, 1).
        
        Returns
        -------
        Phi : torch.Tensor
            Electric potential at all RK stages, shape (batch_size, q+1).
        Ne : torch.Tensor
            Electron density at all RK stages, shape (batch_size, q+1).
        """
        out = self.network(x)
        q = self.q
        Phi = out[:, 0:q+1]*((x[:,0:1] - self.R)*x[:,0:1]) + (1 - x[:,0:1]/self.R)*self.V0
        Ne = out[:, q+1:2*(q+1)]
        return Phi, Ne


class Corona1DRKModel(PINN):
    """
    PINN model for solving 1D corona discharge equations using implicit Runge-Kutta time integration.
    
    This class implements a Physics-Informed Neural Network with implicit Runge-Kutta (RK) temporal
    discretization for corona discharge simulations. Unlike standard PINNs that typically use automatic
    differentiation for temporal derivatives, RK-PINN discretizes time explicitly using RK formulas,
    enabling better control over temporal accuracy and stability.
    
    The model solves coupled nonlinear equations for electric potential (Φ) and electron density (Ne)
    in a cylindrical corona discharge geometry, with physics-based loss functions that enforce both
    the governing PDEs and boundary conditions.
    
    Parameters
    ----------
    R : float
        Domain radius [m].
    T : float
        Gas temperature [K].
    P : float
        Gas pressure [Pa].
    V0 : float
        Applied voltage at electrode [V].
    dt : float
        Normalized time step (Δt_norm).
    Ne_init_func : callable, optional
        Initial condition function for electron density Ne(r). Required.
    Np_func : callable, optional
        Positive ion density function Np(r). Required.
    N_red : float, default=1e15
        Electron density reduction factor [m⁻³].
    t_red : float, default=5e-9
        Time reduction factor [s].
    V_red : float, default=10e3
        Voltage reduction factor [V].
    gamma : float, default=0.066
        Secondary electron emission coefficient [dimensionless].
    train_data_size : int, default=500
        Number of training collocation points.
    sample_mode : {'uniform', 'lhs', 'random'}, default='uniform'
        Sampling strategy for collocation points.
    q : int, default=50
        Order of implicit Runge-Kutta method (stages = q+1).
    backbone_net : nn.Module, default=FNN([1, 300, 300, 300, 300, 102])
        Neural network architecture.
    prop : CoronaPropSpline, optional
        Material property object (transport coefficients).
    """
    def __init__(
        self,
        R,              # Domain radius [m]
        T,              # Gas temperature [K]
        P,              # Gas pressure [Pa]
        V0,             # Applied voltage at electrode [V]
        dt,             # Normalized time step (Δt_norm)
        Ne_init_func=None,      # Initial condition function for electron density Ne(r)
        Np_func=None,           # Positive ion density function Np(r)
        N_red=1e15,             # Electron density reduction factor [m⁻³]
        t_red=5e-9,             # Time reduction factor [s]
        V_red=10e3,             # Voltage reduction factor [V]
        gamma=0.066,            # Secondary electron emission coefficient [dimensionless]
        train_data_size=500,    # Number of training collocation points
        sample_mode='uniform',  # Sampling strategy: 'uniform', 'lhs', or 'random'
        q=50,                   # Order of implicit Runge-Kutta method (stages = q+1)
        backbone_net=FNN(layers=[1, 300, 300, 300, 300, 102]),  # Neural network architecture
        prop:CoronaPropSpline=None,  # Material property object (transport coefficients)
    ):
        """
        Initialize the CORONA discharge RK-PINN model.
        
        Parameters are documented in class docstring.
        
        Raises
        ------
        ValueError
            If Ne_init_func is None.
        ValueError
            If Np_func is None.
        """
        self.R = R
        self.T = T
        self.P = P
        self.V0 = V0
        self.N_red = N_red
        self.t_red = t_red
        self.V_red = V_red
        self.E_red = V_red / R
        self.gamma = gamma
        self.Neu = P/(Boltz_k*T)                        # neutral number density
        self.dt = dt
        self.q = q
        self.sample_mode = sample_mode
        self.train_data_size = train_data_size
        self.prop = prop
        
        if Ne_init_func is None:
            raise ValueError('Ne_init_func (initial condition of electron density) must be provided for corona discharge model.')
        else:
            self.Ne_init_func = Ne_init_func
        
        if Np_func is None:
            raise ValueError('Np_func (positive ion density function) must be provided for corona discharge model.')
        else:
            self.Np_func = Np_func

        self.geo = Geo1D([0.0, 1.0])
        network = Corona1DRKNet(backbone_net, q, R=1.0, V0=V0/V_red)

        super().__init__(network)

        self.set_loss_func(F.mse_loss)

        self.RK_weights = load_butcher_table(q)

    
    def _define_loss_terms(self):
        """
        Define physics-informed loss terms for the corona discharge model.
        
        This method constructs the complete loss function for the RK-PINN model by
        combining:
        1. PDE residuals for electron continuity (Ne equation)
        2. PDE residuals for electrostatic potential (Φ equation)  
        3. Boundary conditions for electron density
        """
        def _grad_f0(G, X):
            """
            Compute gradient with respect to X for q-stage outputs.
            
            Parameters
            ----------
            G : torch.Tensor
                Output tensor with q stages.
            X : torch.Tensor
                Input tensor.
            
            Returns
            -------
            torch.Tensor
                Gradient of G with respect to X.
            """
            grad_tmp = torch.autograd.grad(G, X, grad_outputs=Gx0, retain_graph=True, create_graph=True)[0]
            grad = torch.autograd.grad(grad_tmp, Gx0, grad_outputs=torch.ones_like(grad_tmp), retain_graph=True, create_graph=True)[0]

            return grad

        def _grad_f1(G, X):
            """
            Compute gradient with respect to X for (q+1)-stage outputs.
            
            Parameters
            ----------
            G : torch.Tensor
                Output tensor with (q+1) stages.
            X : torch.Tensor
                Input tensor.
            
            Returns
            -------
            torch.Tensor
                Gradient of G with respect to X.
            """
            grad_tmp = torch.autograd.grad(G, X, grad_outputs=Gx1, retain_graph=True, create_graph=True)[0]
            grad = torch.autograd.grad(grad_tmp, Gx1, grad_outputs=torch.ones_like(grad_tmp), retain_graph=True, create_graph=True)[0]
            
            return grad
        
        def _RK_Ne_residual(network, x):
            """
            RK-PINN Residual for corona discharge model (Electron density equation).
            
            Parameters
            ----------
            network : nn.Module
                Neural network model.
            x : torch.Tensor
                Spatial coordinates.
            
            Returns
            -------
            torch.Tensor
                Residuals of the electron density equation.
            """
            Phi, Ne = network(x)  # N*(q+1)
            
            ## gradients of Phi  ###
            Phi_grad = _grad_f1(Phi, x)

            ### E-dependent coefficients ###
            E = -Phi_grad[:,:]
            EN = torch.abs(E)*self.E_red/self.Neu/1e-21  # in unit of Td
            alpha = self.prop.alpha(EN.view(-1)).view(-1, E.shape[1])
            mu_e = self.prop.mu_e(EN.view(-1)).view(-1, E.shape[1])
            D_e = self.prop.D_e(EN.view(-1)).view(-1, E.shape[1])

            ### gradients of Ne ###
            Ne_grad = _grad_f1(Ne[:, :], x)
            Ne_r = Ne_grad[:, :]

            Ne_term1 = mu_e[:,:-1]*E[:,:-1]*Ne[:,:-1]
            Ne_term1_grad = _grad_f0(Ne_term1, x)
            Ne_term1_r = Ne_term1_grad[:, :]

            Ne_term2 = D_e[:,:-1]*Ne_r[:,:-1]
            Ne_term2_grad = _grad_f0(Ne_term2, x)
            Ne_term2_r = Ne_term2_grad[:, :]

            ### equation N[u] of Ne (N*1) ###
            func_Ne = -Ne_term1_r*(self.E_red*self.t_red/self.R) - Ne_term2_r*(self.t_red/(self.R*self.R)) - alpha[:,:-1]*Ne[:,:-1]*mu_e[:,:-1]*torch.abs(E[:,:-1])*(self.t_red*self.E_red)

            ### residual
            func = Ne + self.dt*func_Ne @ self.RK_weights.T - Ne0

            return func

        def _RK_Phi_residual(network, x):
            """
            RK-PINN Residual for corona discharge model (Potential equation).
            
            Parameters
            ----------
            network : nn.Module
                Neural network model.
            x : torch.Tensor
                Spatial coordinates.
            
            Returns
            -------
            torch.Tensor
                Residuals of the potential equation.
            """
            Phi, Ne = network(x)  # N*(q+1)
            
            ## gradients of Phi  ###
            Phi_grad = _grad_f1(Phi, x)
            E_grad = _grad_f1(Phi_grad, x)
            Phi_laplace = E_grad[:, :]

            ### equation of Phi ###
            func_Phi = Phi_laplace + Phi_coeff*(Np - Ne)

            return func_Phi

        def _RK_bc_Ne_residual(network, x):
            """
            RK-PINN Boundary Residual for corona discharge model (Electron density equation).
            
            Parameters
            ----------
            network : nn.Module
                Neural network model.
            x : torch.Tensor
                Spatial coordinates.
            
            Returns
            -------
            torch.Tensor
                Boundary residuals of the electron density equation.
            """
            Phi, Ne = network(x)  # N*(q+1)
            
            ## gradients of Phi  ###
            Phi_grad = _grad_f1(Phi, x)

            ### E-dependent coefficients ###
            E = -Phi_grad[:,:]
            EN = torch.abs(E)*self.E_red/self.Neu/1e-21  # in unit of Td
            mu_p = self.prop.mu_p(EN.view(-1)).view(-1, E.shape[1])

            ### gradients of Ne ###
            Ne_grad = _grad_f1(Ne[:, :], x)
            Ne_r = Ne_grad[:, :]

            ### boundary condition of Ne ###
            func_Ne_b_left = Ne_r[0:1,:] + Np[0:1,0:1]*mu_p[0:1,:]*torch.abs(E[0:1,:])*(self.gamma*self.R*self.E_red)
            func_Ne_b_right = Ne_r[-2:-1,:]

            ### container of Ne_b ###
            func_Ne_b = torch.cat((func_Ne_b_left, func_Ne_b_right), 0)

            return func_Ne_b
        

        def _RK_all_residual(network, x):
            """
            RK-PINN Residual for corona discharge model.
            
            Combines electron density equation, potential equation,
            and electron density boundary condition.
            
            Parameters
            ----------
            network : nn.Module
                Neural network model.
            x : torch.Tensor
                Spatial coordinates.
            
            Returns
            -------
            torch.Tensor
                Combined residuals.
            """
            Phi, Ne = network(x)  # N*(q+1)
            
           ## gradients of Phi  ###
            Phi_grad = _grad_f1(Phi, x)
            E_grad = _grad_f1(Phi_grad, x)
            Phi_laplace = E_grad[:, :]

            ### equation of Phi ###
            func_Phi = Phi_laplace + Phi_coeff*(Np - Ne)

            ### E-dependent coefficients ###
            E = -Phi_grad[:,:]
            EN = torch.abs(E)*self.E_red/self.Neu/1e-21  # in unit of Td
            alpha = self.prop.alpha(EN.view(-1)).view(-1, E.shape[1])
            mu_e = self.prop.mu_e(EN.view(-1)).view(-1, E.shape[1])
            mu_p = self.prop.mu_p(EN.view(-1)).view(-1, E.shape[1])
            D_e = self.prop.D_e(EN.view(-1)).view(-1, E.shape[1])

            ### gradients of Ne ###
            Ne_grad = _grad_f1(Ne[:, :], x)
            Ne_r = Ne_grad[:, :]

            Ne_term1 = mu_e[:,:-1]*E[:,:-1]*Ne[:,:-1]
            Ne_term1_grad = _grad_f0(Ne_term1, x)
            Ne_term1_r = Ne_term1_grad[:, :]

            Ne_term2 = D_e[:,:-1]*Ne_r[:,:-1]
            Ne_term2_grad = _grad_f0(Ne_term2, x)
            Ne_term2_r = Ne_term2_grad[:, :]

            ### equation N[u] of Ne (N*1) ###
            func_Ne_equ = -Ne_term1_r*(self.E_red*self.t_red/self.R) - Ne_term2_r*(self.t_red/(self.R*self.R)) - alpha[:,:-1]*Ne[:,:-1]*mu_e[:,:-1]*torch.abs(E[:,:-1])*(self.t_red*self.E_red)

            ### residual
            func_Ne = Ne + self.dt*func_Ne_equ @ self.RK_weights.T - Ne0

            ### boundary condition of Ne ###
            func_Ne_b_left = Ne_r[0:1,:] + Np[0:1,0:1]*mu_p[0:1,:]*torch.abs(E[0:1,:])*(self.gamma*self.R*self.E_red)
            func_Ne_b_right = Ne_r[-2:-1,:]

            ### container of Ne_b ###
            func_Ne_b = torch.cat((func_Ne_b_left, func_Ne_b_right), 0)

            ## all
            weight_Ne, weight_Phi, weight_Ne_b = 1.0, 1.0,  1.0
            func = torch.cat((func_Ne*weight_Ne, func_Phi*weight_Phi, func_Ne_b*weight_Ne_b), 0)

            return func
        
        # Sample domain collocation points
        x = self.geo.sample_domain(self.train_data_size, mode=self.sample_mode, include_boundary=True)
        
        ##
        Gx0 = numpy2torch(np.ones((self.train_data_size, self.q), dtype=REAL()))
        Gx1 = numpy2torch(np.ones((self.train_data_size, self.q+1), dtype=REAL()))

        # constant
        Phi_coeff = Elec*self.N_red*self.R*self.R/Epsilon_0/self.V_red
        x_np = x.detach().cpu().numpy()
        Np = numpy2torch(self.Np_func(x_np))
        Ne0 = numpy2torch(self.Ne_init_func(x_np*self.R) / self.N_red)
        
        # Add equation terms with weights
        # to avoid too many separate terms, we combine all residuals into one term with balanced weights
        # instead of adding them separately, which may lead to better convergence and stability in training
        # self.add_equation('Electron', _RK_Ne_residual, weight=1.0, data=x)
        # self.add_equation('Potential', _RK_Phi_residual, weight=1.0, data=x)
        # self.add_equation('Electron Boundary', _RK_bc_Ne_residual, weight=1.0, data=x)
        self.add_equation('Equ_all', _RK_all_residual, weight=1.0, data=x)


class Corona1DRKVisCallback(VisualizationCallback):
    """
    Custom visualization callback for 1D corona discharge RK-PINN model training.
    
    This callback provides comprehensive real-time monitoring and post-training visualization
    capabilities for corona discharge simulations using the RK-PINN framework. It generates
    publication-quality figures showing electric potential (Φ), electron density (Ne) evolution,
    and training convergence metrics.
    
    Parameters
    ----------
    model : Corona1DRKModel
        The corona discharge model instance.
        Provides access to geometry, physical parameters, and material properties.
    log_freq : int, default=50
        Frequency (in epochs) for logging visualizations to TensorBoard.
        Example: log_freq=50 means visualize every 50 epochs.
    save_history : bool, default=True
        Whether to save prediction snapshots for creating training animations.
        Set to False to save memory if animations are not needed.
    history_freq : int, optional
        Frequency (in epochs) for saving history snapshots.
        If None, defaults to log_freq.
        Use larger values (e.g., 200) for very long training runs.
    x_eval : np.ndarray, shape (n_r, 1), optional
        Radial evaluation grid for visualization (normalized radius 0→1).
        Default: 201 points linearly spaced from 0 to 1.
        Use finer grids for higher-resolution visualizations.
        Use coarser grids for faster evaluation.
    corona_csv_file : str, optional
        Path to CSV file containing reference Φ and Ne profiles.
        CSV columns required: 'r(cm)', 'U(V)', 'Ne(m^-3)'.
        If None, reference comparison is skipped.
        Enables error analysis and validation against experimental/reference data.
    gif_enabled : bool, default=False
        Whether to save per-epoch frames and generate a training animation GIF.
        When True, PNG frames are automatically saved at gif_freq intervals
        and assembled into an animated GIF at training end.
        Useful for post-training analysis and presentations.
    gif_dir : str, optional
        Output directory for GIF animation and final plots.
        If None, defaults to current working directory.
    gif_freq : int, optional
        Frequency (in epochs) to save frames for GIF creation.
        If None, uses history_freq.
        Larger values (e.g., 500) produce shorter GIFs with fewer frames.
    gif_duration_ms : int, default=300
        Duration per frame in milliseconds for the GIF animation.
    gif_cleanup_tmp : bool, default=True
        Whether to automatically delete temporary PNG frames after GIF creation.
        Set to False to retain frames for manual re-assembly or inspection.
    """
    
    def __init__(self, model: 'Corona1DRKModel', log_freq: int = 50, 
                 save_history: bool = True, history_freq: int = None,
                 x_eval: np.ndarray = np.linspace(0, 1, 201, dtype=REAL()).reshape(-1,1),
                 corona_csv_file: str = None, 
                 gif_enabled: bool = False,
                 gif_dir: str = None,
                 gif_freq: int = None,
                 gif_duration_ms: int = 300,
                 gif_cleanup_tmp: bool = True):
        """
        Initialize the corona discharge RK-PINN visualization callback.
        
        See class docstring for detailed parameter descriptions and examples.
        
        Parameters
        ----------
        model : Corona1DRKModel
            The corona discharge model instance.
        log_freq : int, default=50
            Frequency (in epochs) for logging visualizations to TensorBoard.
        save_history : bool, default=True
            Whether to save prediction snapshots for creating training animations.
        history_freq : int, optional
            Frequency (in epochs) for saving history snapshots.
        x_eval : np.ndarray, default=np.linspace(0, 1, 201, dtype=REAL()).reshape(-1,1)
            Radial evaluation grid for visualization (normalized radius 0→1).
        corona_csv_file : str, optional
            Path to CSV file containing reference Φ and Ne profiles.
        gif_enabled : bool, default=False
            Whether to save per-epoch frames and generate a training animation GIF.
        gif_dir : str, optional
            Output directory for GIF animation and final plots.
        gif_freq : int, optional
            Frequency (in epochs) to save frames for GIF creation.
        gif_duration_ms : int, default=300
            Duration per frame in milliseconds for the GIF animation.
        gif_cleanup_tmp : bool, default=True
            Whether to automatically delete temporary PNG frames after GIF creation.
        """
        super().__init__(name='RK-PINN_1D_Corona', log_freq=log_freq)
        
        self.model = model
        self.x_eval = x_eval
        
        # Load reference Phi and Ne functions from CSV files
        self.Phi_ref_func, self.Ne_ref_func = get_PhiNe_func_from_file(corona_csv_file) if corona_csv_file else (None, None)
        
        # Physical parameters
        self.t_eval = model.dt
        self.t_red = model.t_red
        self.R = model.R
        self.N_red = model.N_red
        self.V_red = model.V_red
        self.Ne_init_func = model.Ne_init_func
        
        # Training history tracking
        self.save_history = save_history
        self.history_freq = history_freq if history_freq is not None else log_freq
        self.history = {
            'epochs': [],           # List of epoch numbers
            'r_eval': self.x_eval,  # Radial evaluation grid (fixed)
            't_eval': model.dt,  # Time evaluation points (fixed)
            'Phi': [],                # List of Phi(r,t) arrays [n_epochs, n_time, n_r]
            'Ne': [],                # List of Ne(r,t) arrays [n_epochs, n_time, n_r]
            'losses': [],           # List of total loss values
            'Ne_center_t': [],       # List of center Ne at all time points
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

    def _compute_material_properties_at_t(self, Phi_phys: np.ndarray, Ne_phys: np.ndarray, t_reduced: float) -> dict:
        """
        Compute diagnostic material properties at a given time step.
        
        This method extracts the final RK stage predictions and computes summary
        statistics for visualization and analysis. The method focuses on the
        converged solution (q+1)-th RK stage representing the solution at
        the end of the time step.
        
        Parameters
        ----------
        Phi_phys : np.ndarray
            Electric potential array in physical units (V), shape (n_r, q+1)
            with columns [stage_1, stage_2, ..., stage_q, stage_q+1].
        Ne_phys : np.ndarray
            Electron density array in physical units (m⁻³), shape (n_r, q+1)
            with columns corresponding to RK stages.
        t_reduced : float
            Normalized time value for reference in output.
        
        Returns
        -------
        dict
            Dictionary containing diagnostic quantities:
              - 'Phi_at_t': Final (Φ) profile (n_r,) from last RK stage
              - 'Ne_at_t': Final (Ne) profile (n_r,) from last RK stage
              - 'max_Phi': Maximum absolute potential magnitude |V|
              - 'Ne_center': Electron density at centerline r=0
              - 't_physical': Time in physical units (s)
        """
        return {
            'Phi_at_t': Phi_phys[:,-1].reshape(-1, 1),
            'Ne_at_t': Ne_phys[:,-1].reshape(-1, 1),
            'max_Phi': np.abs(Phi_phys[:,-1]).max(),
            'Ne_center': Ne_phys[0,-1],
            't_physical': t_reduced * self.t_red
        }

    def _make_figure(self, epoch: int, Phi_reduced: np.ndarray, Ne_reduced: np.ndarray, **kwargs) -> plt.Figure:
        """
        Build multi-panel matplotlib figure for TensorBoard logging and GIF frames.
        
        Generates a comprehensive 2x2 panel figure showing:
        - Left column: Electron density (Ne) predictions vs reference
        - Right column: Electric potential (Φ) predictions vs reference
        - Bottom row: Training loss convergence curve
        
        Parameters
        ----------
        epoch : int
            Current training epoch number (used in title).
        Phi_reduced : np.ndarray
            Electric potential in normalized units, shape (n_r, q+1).
        Ne_reduced : np.ndarray
            Electron density in normalized units, shape (n_r, q+1).
        kwargs : dict
            Optional training info:
              - 'total_loss': Current loss value for annotation
        
        Returns
        -------
        plt.Figure
            Matplotlib figure object with all panels configured.
        """
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, height_ratios=[1, 0.8])
        
        # Plot 
        Phi_phys = Phi_reduced * self.V_red
        Ne_phys = Ne_reduced * self.N_red
        t_physical = self.t_eval * self.t_red

        props = self._compute_material_properties_at_t(Phi_phys, Ne_phys, t_physical)
        Ne_pred = props['Ne_at_t']
        Phi_pred = props['Phi_at_t']
        
        # Temperature subplot (left column)
        ax_Ne = fig.add_subplot(gs[0, 0])
        ax_Ne.plot(self.x_eval, Ne_pred, 'b-', linewidth=2.5, label='RK-PINN')
        # Plot reference Ne if available
        if self.Ne_ref_func is not None:
            Ne_ref = self.Ne_ref_func(self.x_eval*self.R)
            ax_Ne.plot(self.x_eval, Ne_ref, 'r--', linewidth=2.0, label='Reference', alpha=0.8)
            error_Ne = np.abs(Ne_pred.flatten() - Ne_ref.flatten())
            relative_error = error_Ne / (Ne_ref.flatten() + 1e-10) * 100
            max_error = error_Ne.max()
            mean_error = error_Ne.mean()
            max_rel_error = relative_error.max()
            rel_l2_error = calc_relative_l2_err(Ne_ref, Ne_pred)
            info_text_Ne = (f'Max Error: {max_error:.2g} m^-3\n'
                        f'Mean Error: {mean_error:.2g} m^-3\n'
                        f'Max Rel Error: {max_rel_error:.2f}%\n'
                        f'Rel L2 Error: {rel_l2_error:.5g}')
        else:
            info_text_Ne = (f'Center: {props["Ne_center"]:.2g} m^-3\n')
        
        ax_Ne.set_xlabel('Normalized radius r/R', fontsize=11)
        ax_Ne.set_ylabel('Ne (m^-3)', fontsize=11)
        ax_Ne.set_title(f'Ne @ t = {t_physical*1e9:.2f} ns', fontsize=12, fontweight='bold')
        ax_Ne.grid(True, alpha=0.3)
        ax_Ne.legend(loc='best', fontsize=10)
        
        # Add info box for Ne
        ax_Ne.text(0.05, 0.05, info_text_Ne,
                    transform=ax_Ne.transAxes, fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        # Potential subplot (right column, same row)
        ax_Phi = fig.add_subplot(gs[0, 1])
        ax_Phi.plot(self.x_eval, Phi_pred, 'g-', linewidth=2.5, label='RK-PINN')

        if self.Phi_ref_func is not None:
            Phi_ref = self.Phi_ref_func(self.x_eval*self.R)
            ax_Phi.plot(self.x_eval, Phi_ref, 'r--', linewidth=2.0, label='Reference', alpha=0.8)
            error_Phi = np.abs(Phi_pred.flatten() - Phi_ref.flatten())
            relative_error = error_Phi / (np.abs(Phi_ref.flatten()) + 1e-10) * 100
            max_error = error_Phi.max()
            mean_error = error_Phi.mean()
            max_rel_error = relative_error.max()
            rel_l2_error = calc_relative_l2_err(Phi_ref, Phi_pred)
            info_text_Phi = (f'Max Error: {max_error:.2g} m/s\n'
                        f'Mean Error: {mean_error:.2g} m/s\n'
                        f'Max Rel Error: {max_rel_error:.2f}%\n'
                        f'Rel L2 Error: {rel_l2_error:.5g}')
        else:
            info_text_Phi = f'Abs. Max: {props["max_Phi"]:.2g} m/s'

        ax_Phi.set_xlabel('Normalized radius r/R', fontsize=11)
        ax_Phi.set_ylabel('Potential (V)', fontsize=11)
        ax_Phi.set_title(f'Potential @ t = {t_physical*1e9:.2f} ns', fontsize=12, fontweight='bold')
        ax_Phi.grid(True, alpha=0.3)
        ax_Phi.legend(loc='best', fontsize=10)
        ax_Phi.text(0.05, 0.05, info_text_Phi,
                    transform=ax_Phi.transAxes, fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        
        # Add loss curve panel spanning the entire bottom row
        ax_loss = fig.add_subplot(gs[1, :])  # Span all columns in last row
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
        
        fig.suptitle(f'1D Corona RK-PINN Model - Epoch {epoch}', 
                fontsize=14, fontweight='bold', y=0.995)
        return fig

    def visualize(self, network, epoch: int, writer: SummaryWriter, **kwargs):
        """
        Generate visualization plots for the current training epoch.
        
        This method is the main visualization callback invoked automatically by
        the PINN.train() method at specified intervals (controlled by log_freq).
        It performs three main tasks:
        1. Evaluates the network on a uniform radial grid
        2. Saves predictions to history (for animation)
        3. Creates multi-panel figure and saves for TensorBoard and GIF
        
        Parameters
        ----------
        network : nn.Module
            The neural network being trained (Corona1DRKNet).
        epoch : int
            Current training epoch number.
        writer : SummaryWriter
            TensorBoard writer for logging figures and metrics.
        kwargs : dict
            Additional training information:
              - 'total_loss': Total loss value (torch.Tensor or float)
              - 'loss_dict': Dictionary of individual loss terms
        
        Returns
        -------
        dict
            Dictionary mapping visualization names to matplotlib figures
        """
        network.eval()
        
        # Generate predictions on evaluation grid at all time points
        with torch.no_grad():
            # Create (r, t) pairs for this time step
            x_r = numpy2torch(self.x_eval.reshape(-1, 1), require_grad=False)
            
            # Predict
            Phi, Ne = network(x_r)
            Phi = Phi.cpu().numpy()
            Ne = Ne.cpu().numpy()
        
        # Save history for animation
        if self.save_history and epoch % self.history_freq == 0:
            self.history['epochs'].append(epoch)
            self.history['Phi'].append(Phi.copy())
            self.history['Ne'].append(Ne.copy())
            
            # Track center Ne at each time point
            Ne_center = Ne[:, 0]
            self.history['Ne_center_t'].append(Ne_center)
            
            # Extract loss
            total_loss = kwargs.get('total_loss', None)
            if total_loss is not None:
                if isinstance(total_loss, torch.Tensor):
                    self.history['losses'].append(total_loss.item())
                else:
                    self.history['losses'].append(float(total_loss))
        
        # Create figure
        fig = self._make_figure(epoch=epoch, Phi_reduced=Phi, Ne_reduced=Ne, **kwargs)
        
        # Optionally save frame for GIF
        if self.gif_enabled and (epoch % self.gif_freq == 0):
            frame_path = os.path.join(self.gif_tmp_dir, f'epoch_{epoch:06d}.png')
            try:
                fig.savefig(frame_path, dpi=120)
                self._gif_frames.append(frame_path)
            except Exception as e:
                print(f'Warning: failed to save GIF frame at epoch {epoch}: {e}')
        
        return {'1d_corona_visualization': fig}

    def save_gif(self, gif_path: str = None, duration_ms: int = None, loop: int = 0):
        """
        Assemble saved PNG frames into an animated GIF showing training progress.
        
        This method combines temporary PNG frames collected during training into
        a single GIF animation. The animation shows how the Ne and Φ predictions
        evolve during training, along with the loss curve convergence.
        
        Parameters
        ----------
        gif_path : str, optional
            Output path for the GIF animation file.
            If None, defaults to <gif_dir>/training_animation.gif.
            Directory is created if it doesn't exist.
        duration_ms : int, optional
            Per-frame duration in milliseconds for the GIF playback.
            If None, uses self.gif_duration_ms (default 300 ms).
        loop : int, default=0
            Number of animation loops.
            0 = infinite loop (gif continues cycling).
            n > 0 = gif cycles n times then stops.
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
        Save final result figures showing potential and density distributions.
        
        This method generates and saves publication-quality figures at the end of
        training. It demonstrates the trained model's final predictions without
        further training dynamics.
        
        Parameters
        ----------
        network : nn.Module
            Trained neural network for final prediction generation.
            Should be in evaluation mode (will be set by this method).
        save_dir : str, optional
            Output directory for final figure files.
            If None, defaults to self.gif_dir.
            Directory is created if it doesn't exist.
        epoch : int, optional
            Epoch number to display in figure title.
            If None, uses the last recorded epoch from history.
            If no history, defaults to 0.
        kwargs : dict
            Optional training information:
              - 'total_loss': Final loss value for annotation
        """
        out_dir = save_dir or self.gif_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # Generate final predictions
        network.eval()
        with torch.no_grad():
            # Create (r, t) pairs for this time step
            x_r = numpy2torch(self.x_eval.reshape(-1, 1), require_grad=False)
            
            # Predict
            Phi, Ne = network(x_r)
            Phi = Phi.cpu().numpy()
            Ne = Ne.cpu().numpy()
        
        # Determine epoch label
        if epoch is None:
            epoch = self.history['epochs'][-1] if self.history['epochs'] else 0
        
        # Save multi-panel figure
        fig = self._make_figure(epoch=epoch, Phi_reduced=Phi, Ne_reduced=Ne, **kwargs)
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
  





