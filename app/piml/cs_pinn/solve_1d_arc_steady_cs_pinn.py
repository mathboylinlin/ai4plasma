###
## Author: Linlin Zhong
## Email: linlin@seu.edu.cn
## Created: Jan. 19, 2022
## Updated: Jan. 21, 2026
## Description: Example of solving 1D steady-state arc discharge model using CS-PINN 
##              (Coefficient-Subnet Physics-Informed Neural Network) for SF6 plasma.
##              This script demonstrates how to:
##              - Set up a physics-based neural network for arc discharge simulation
##              - Use temperature-dependent material properties (conductivity, etc.)
##              - Train the model with adaptive learning rate scheduling
##              - Visualize and compare results with reference FVM data
###


import sys
sys.path.append('.')

import torch.nn as nn

from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import set_seed, Timer
from ai4plasma.config import DEVICE
from ai4plasma.core.network import FNN
from ai4plasma.plasma.prop import ArcPropSpline
from ai4plasma.piml.cs_pinn import StaArc1DModel, StaArc1DVisCallback


# ============================================================================
# Configuration and Initialization
# ============================================================================

## Fix random seed for reproducibility ##
set_seed(2023)

## Set computing device (GPU if available, otherwise CPU) ##
if check_gpu(print_required=True):
    DEVICE.set_device(0)  # Using cuda:0 for GPU acceleration
else:
    DEVICE.set_device(-1) # Using CPU
print(DEVICE)

## Start timer to measure total execution time ##
my_timer = Timer()


# ============================================================================
# Arc Discharge Physical Parameters
# ============================================================================

## Gas type and arc geometry
gas = 'SF6'          # Working gas: Sulfur hexafluoride
R = 10e-3           # Arc radius in meters (10 mm)
I = 200             # Arc current in Amperes
T_red = 1e4         # Temperature reduction factor for normalization (10,000 K)
Tb = 2000.0         # Boundary temperature in Kelvin at r = R


# ============================================================================
# Training Configuration
# ============================================================================

## Training hyperparameters
num_epochs = 100000           # Total number of training epochs
learning_rate = 1e-3         # Initial learning rate for Adam optimizer
train_data_size = 500        # Number of collocation points for PDE residual
test_data_size = 600         # Number of points for visualization and evaluation


# ============================================================================
# Material Properties Setup
# ============================================================================

## Load arc plasma properties (temperature-dependent transport coefficients)
# These files contain thermodynamic and transport properties:
# - Electrical conductivity σ(T)
# - Thermal conductivity κ(T)
# - Density ρ(T)
# - Specific heat capacity Cp(T)
# - Net emission coefficient (radiation) nec(T)
thermo_file = f'app/piml/cs_pinn/data/{gas.lower()}_p1.dat'
nec_file = f'app/piml/cs_pinn/data/{gas.lower()}_p1_nec.dat'

# Create spline interpolator for material properties
# Automatically clamps temperatures to [300, 30000] K range
prop = ArcPropSpline(thermo_file, nec_file, R)


# ============================================================================
# Neural Network Architecture
# ============================================================================

## Define feedforward neural network architecture
# Input: 1D (normalized radius r/R)
# Hidden layers: 6 layers with 50 neurons each
# Output: 1D (normalized temperature T/T_red)
# Activation: Hyperbolic tangent (Tanh) for smooth gradients
layers = [1, 50, 50, 50, 50, 50, 50, 1]
backbone_net = FNN(layers, act_fun=nn.Tanh())

## Reference data file for comparison (from Finite Volume Method simulation)
T_csv_file = f'app/piml/cs_pinn/data/sta_arc1d_{gas}.csv'

## Create CS-PINN model for 1D steady-state arc
# This model solves the energy balance equation:
# ∇·(κ∇T) = σE² - 4πε (Joule heating - Radiation loss)
arc_model = StaArc1DModel(
    R,                      # Arc radius
    I,                      # Arc current
    Tb,                     # Boundary temperature
    T_red,                  # Temperature normalization factor
    backbone_net,           # Neural network backbone
    train_data_size,        # Number of training points
    test_data_size,         # Number of test points
    sample_mode='uniform',  # Uniform spatial sampling
    prop=prop              # Material property interpolator
)


# ============================================================================
# Visualization and Monitoring
# ============================================================================

## Setup visualization callback for training monitoring
log_freq = 50  # Log to TensorBoard every 50 epochs

## Setup GIF animation for visualizing training progress
# The GIF will show temperature evolution and loss convergence
gif_results_dir = 'app/piml/cs_pinn/results/sta/'

viz_callback = StaArc1DVisCallback(
    model=arc_model,
    log_freq=log_freq,          # TensorBoard logging frequency
    save_history=True,          # Enable history tracking for creating animations
    history_freq=1000,          # Save snapshot every 1000 epochs (memory efficient)
    T_csv_file=T_csv_file,      # Reference FVM data for comparison and error metrics
    gif_enabled=True,           # Enable training animation (GIF) generation
    gif_dir=gif_results_dir,    # Output directory for GIF and final plots
    gif_freq=1000,              # Save frame every 1000 epochs for smooth GIF
    gif_duration_ms=300         # Duration per frame in milliseconds (300ms = 3.33 fps)
)

# Register callback to be executed during training
arc_model.register_visualization_callback(viz_callback)


# ============================================================================
# Training Process
# ============================================================================

## Setup optimizer with adaptive learning rate
# Create Adam optimizer with initial learning rate
arc_model.create_optimizer('Adam', lr=learning_rate)

# Create learning rate scheduler for adaptive training
# - Reduces LR by factor of 0.5 at epochs 3000 and 40000
# - Helps achieve better convergence in later stages
arc_model.create_lr_scheduler('MultiStepLR', milestones=[3000, 40000], gamma=0.5)

## Start training
arc_model.train(
    num_epochs=num_epochs,
    print_loss=True,                              # Print loss to console
    print_loss_freq=log_freq,                     # Print frequency (every 50 epochs)
    tensorboard_logdir='app/piml/cs_pinn/runs/sta/',  # TensorBoard log directory
    save_final_model=True,                        # Save final trained model
    checkpoint_dir='./app/piml/cs_pinn/models/sta/',  # Checkpoint save directory
    checkpoint_freq=5000                          # Save checkpoint every 5000 epochs
)

print('\n' + '='*70)
print('Training completed!')
print('='*70)


# ============================================================================
# Post-Processing: Animation and Final Results Export
# ============================================================================

## Generate training animation GIF
# This creates an animated GIF showing the evolution of temperature profile
# and loss curve throughout the training process
print('\nGenerating training animation GIF...')
viz_callback.save_gif()
print(f'Animation saved to: {gif_results_dir}')

## Export final results
# This saves the final 2x2 panel plot (similar to TensorBoard) and the
# final loss curve in high resolution for publications or reports
print('\nExporting final result plots...')
viz_callback.save_final_results(
    network=arc_model.network,
    save_dir=gif_results_dir,
    epoch=num_epochs
)
print(f'Final plots saved to: {gif_results_dir}/')
print('  - final_panels.png: 2x2 panel figure (T, properties, radiation, loss)')
print('  - loss_curve.png: Final training loss curve')


# ============================================================================
# Post-Processing
# ============================================================================

## Print total execution time ##
my_timer.current()