###############################################################################
## Author: Linlin Zhong
## Email: linlin@seu.edu.cn
## Created: Jan. 19, 2022
## Updated: Jan. 21, 2026
##
## Description:
##   Complete training pipeline for 1D transient arc discharge model using
##   CS-PINN (Coefficient-Subnet Physics-Informed Neural Network) for SF6 plasma.
##
###############################################################################


import sys
sys.path.append('.')

import numpy as np
import torch.nn as nn

from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import set_seed, Timer
from ai4plasma.config import DEVICE, REAL
from ai4plasma.core.network import FNN
from ai4plasma.plasma.prop import ArcPropSpline
from ai4plasma.piml.cs_pinn import get_Tfunc_from_file, TraArc1DTempModel, TraArc1DTempVisCallback


# ============================================================================
# CONFIGURATION AND INITIALIZATION
# ============================================================================
# This section sets up the computing environment and ensures reproducibility
# of results through fixed random seeding and device configuration.

## Fix random seed for reproducibility
# All random operations (NumPy, PyTorch, Python) use this seed
set_seed(2023)

## Set computing device: GPU if available, otherwise CPU
# GPU acceleration significantly speeds up neural network training (~10-100x)
if check_gpu(print_required=True):
    DEVICE.set_device(0)  # Use cuda:0 for GPU-accelerated computation
else:
    DEVICE.set_device(-1) # Fall back to CPU if no GPU available
print(DEVICE)

## Initialize timer for tracking total execution time
# Useful for benchmarking and understanding computational requirements
my_timer = Timer()


# ============================================================================
# ARC DISCHARGE PHYSICAL PARAMETERS
# ============================================================================
# These parameters define the physical properties and geometry of the arc
# discharge system being simulated. All values are in SI units.

## Gas type and arc geometry
gas = 'SF6'          # Working gas medium: Sulfur hexafluoride (common in power systems)
R = 10e-3           # Arc radius [m]: 10 mm (typical for circuit breaker arcs)
I = 200             # Arc current [A]: 200 Amperes (transient discharge condition)

## Normalization factors for neural network input/output
# These improve numerical stability and training convergence by scaling
# physically large/small quantities into order-unity ranges
T_red = 1e4         # Temperature normalization [K]: divide by 10,000 K
                    # Typical range: [300, 30000] K → normalized: [0.03, 3.0]
t_red = 1e-3        # Time normalization [s]: divide by 1 ms
                    # Typical range: [0, 10] ms → normalized: [0, 10]

## Boundary conditions
Tb = 2000.0         # Boundary temperature [K] at arc radius r = R
                    # Physical constraint: T(R, t) = Tb (automatically enforced)


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# These hyperparameters control the training process, convergence behavior,
# and computational cost.

## Training hyperparameters
num_epochs = 50000           # Total training iterations

learning_rate = 1e-4         # Initial learning rate: 0.0001 for Adam optimizer
                             # Will be reduced by scheduler at milestones

train_data_x_size = 200      # Number of collocation points for PDE residual
                             # Spatial sampling: 200 points uniformly in [0, R]

train_data_t_size = 100      # Number of time points for temporal PDE residual
                             # Temporal sampling: 100 points uniformly in [0, T]

## Evaluation/visualization grid (for output and error computation)
test_data_x = np.linspace(0, 1, 210)  # Spatial evaluation: 210 points [0, R]
test_data_t = [0.1, 0.5, 0.9]         # Time snapshots (normalized): early, middle, late


# ============================================================================
# Material Properties Setup
# ============================================================================

## Load arc plasma properties (temperature-dependent transport coefficients)
thermo_file = f'app/piml/cs_pinn/data/{gas.lower()}_p1.dat'
nec_file = f'app/piml/cs_pinn/data/{gas.lower()}_p1_nec.dat'

# Create spline interpolator for material properties
prop = ArcPropSpline(thermo_file, nec_file, R)


# ============================================================================
# Neural Network Architecture
# ============================================================================

## Define feedforward neural network architecture
layers = [2, 200, 200, 200, 200, 200, 200, 1]
backbone_net = FNN(layers, act_fun=nn.Tanh())

initial_file = f'app/piml/cs_pinn/data/sta_arc1d_{gas}.csv'
Tinit_func = get_Tfunc_from_file(initial_file)

## Create CS-PINN model for 1D steady-state arc
arc_model = TraArc1DTempModel(
    R,                      # Arc radius
    I,                      # Arc current
    Tb,                     # Boundary temperature
    Tinit_func,          # Initial temperature function
    T_red,                  # Temperature normalization factor
    t_red,                  # Time normalization factor
    backbone_net,           # Neural network backbone
    train_data_x_size,        # Number of training points
    train_data_t_size,         # Number of test points
    sample_mode='uniform',  # Uniform spatial sampling
    prop=prop              # Material property interpolator
)


# ============================================================================
# Visualization and Monitoring
# ============================================================================

## Setup visualization callback for training monitoring
log_freq = 50                 # Log to TensorBoard every 50 epochs
                              # Provides frequent monitoring without excessive overhead

## Setup GIF animation for visualizing training progress
gif_results_dir = 'app/piml/cs_pinn/results/tra_noV/'

viz_callback = TraArc1DTempVisCallback(
    model=arc_model,
    log_freq=log_freq,          # TensorBoard logging frequency: every 50 epochs
                                # Balance between monitoring granularity and I/O cost
    save_history=True,          # Enable history tracking for creating training animations
                                # Stores snapshots for GIF generation
    history_freq=1000,          # Save snapshot every 1000 epochs for memory efficiency
                                # Typical: 500-2000 depending on memory constraints
    x_eval=np.linspace(0, 1, 201, dtype=REAL()).reshape(-1,1),
                                # Spatial evaluation grid: 201 points uniformly in [0, R]
                                # Used for plotting temperature profiles
    t_eval=[0.1, 0.5, 0.9],     # Time snapshots (normalized): early (0.1), middle (0.5), late (0.9)
                                # Representative times showing transient progression
    T_csv_file=[f'app/piml/cs_pinn/data/tra_arc1d_noV_{gas}_t0.1.csv',
                f'app/piml/cs_pinn/data/tra_arc1d_noV_{gas}_t0.5.csv',
                f'app/piml/cs_pinn/data/tra_arc1d_noV_{gas}_t0.9.csv'],
                                # Reference data from numerical simulation for comparison
    gif_enabled=True,           # Enable training animation (GIF) generation
                                # Creates animated visualization of convergence
    gif_dir=gif_results_dir,    # Output directory for GIF and final publication-quality plots
    gif_freq=1000,              # Save frame every 1000 epochs (50 frames for 50K epochs)
                                # Typical: 500-2000; higher = smoother animation but more frames
    gif_duration_ms=300         # Duration per frame: 300 ms (3.33 fps)
                                # Results in ~15-25 second animation; adjust for desired speed
)

# Register callback to be executed during training
arc_model.register_visualization_callback(viz_callback)


# ============================================================================
# Training Process
# ============================================================================

## Setup optimizer with adaptive learning rate
arc_model.create_optimizer('Adam', lr=learning_rate)

## Create learning rate scheduler for multi-stage training
arc_model.create_lr_scheduler('MultiStepLR', milestones=[20000, 50000], gamma=0.5)

## Start training process
arc_model.train(
    num_epochs=num_epochs,          # Total 50,000 training iterations
    print_loss=True,                # Display loss to console
    print_loss_freq=log_freq,       # Print loss every 50 epochs (~1000 printouts total)
    tensorboard_logdir='app/piml/cs_pinn/runs/tra_noV/',   # TensorBoard event directory
                                    # Useful for plotting loss curves, learning rate changes
    save_final_model=True,          # Save trained network at training completion
    checkpoint_dir='./app/piml/cs_pinn/models/tra_noV/',   # Checkpoint/model save directory
                                    # Contains intermediate models and final trained network
    checkpoint_freq=5000                          # Save checkpoint every 5000 epochs
)

print('\n' + '='*70)
print('Training completed!')
print('='*70)


# ============================================================================
# Post-Processing: Animation and Final Results Export
# ============================================================================

## Generate training animation GIF
print('\nGenerating training animation GIF...')
viz_callback.save_gif()
print(f'Animation saved to: {gif_results_dir}')

## Export final results
print('\nExporting final result plots...')
viz_callback.save_final_results(
    network=arc_model.network,
    save_dir=gif_results_dir,
    epoch=num_epochs
)
print(f'Final plots saved to: {gif_results_dir}')
print('  - final_panels.png: Multi-panel comparison figure')
print('  - loss_curve.png: Detailed loss evolution with 4 components')


# ============================================================================
# Timing and Summary
# ============================================================================

## Print total execution time and training statistics
print('\n' + '='*70)
print('TRAINING SUMMARY')
print('='*70)
print(f'Total training time:', end=' ')
my_timer.current()

print(f'Total epochs trained: {num_epochs}')
print(f'Results and artifacts saved to: {gif_results_dir}')
print('='*70 + '\n')