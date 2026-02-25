###############################################################################
## Author: Linlin Zhong
## Email: linlin@seu.edu.cn
## Created: Jan. 19, 2022
###############################################################################


import sys
sys.path.append('.')

import numpy as np
import torch.nn as nn

from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import Boltz_k, set_seed, Timer
from ai4plasma.config import DEVICE, REAL
from ai4plasma.core.network import FNN
from ai4plasma.plasma.prop import CoronaPropSpline
from ai4plasma.piml.rk_pinn import get_PhiNe_func_from_file, Corona1DRKModel, Corona1DRKVisCallback


# ============================================================================
# CONFIGURATION AND INITIALIZATION
# ============================================================================
set_seed(2023)

## Set computing device: GPU if available, otherwise CPU
if check_gpu(print_required=True):
    DEVICE.set_device(0)  # Use cuda:0 for GPU-accelerated computation
else:
    DEVICE.set_device(-1) # Fall back to CPU if no GPU available
print(DEVICE)

## Initialize timer for tracking total execution time
my_timer = Timer()


# ============================================================================
# CORONA DISCHARGE PHYSICAL PARAMETERS
# ============================================================================

## Gas type and arc geometry
gas = 'Ar'          # Working gas medium: Argon (common in plasma physics)
R = 0.01            # radius [m]: 10 mm (typical for corona discharge experiments)
T = 600             # Temperature [K]
P = 101325          # Pressure [Pa]: 1 atm
gamma = 0.066
Neu = P/(Boltz_k*T)  # Neutral particle density [m^-3] from ideal gas law

## Normalization factors for neural network input/output
N_red = 1e15
t_red = 5e-9
V_red = 10e3

## Boundary conditions
V0 = -10e3          # Voltage at r=0 (cathode)
na, nb, nr = 1.0, 25e-2, 1e-3
Np_func = lambda x, na=na, nb=nb, nr=nr: na*(nr + np.exp(-x*x/(2*nb*nb)))
_, Ne_init_func = get_PhiNe_func_from_file(f'app/piml/rk_pinn/data/corona_1d_{gas}_t0.0.csv')


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

## Training hyperparameters
num_epochs = 100000   
learning_rate = 1e-4 
train_data_x_size = 500 

## Evaluation/visualization grid (for output and error computation)
test_data_x = np.linspace(0, 1, 210)  # Spatial evaluation: 210 points [0, R]

# ============================================================================
# Material Properties Setup
# ============================================================================

## Create material property interpolator for corona discharge model
coeff_file = {
    'alpha': f'app/piml/rk_pinn/data/{gas}_alpha.dat',
    'D_e': f'app/piml/rk_pinn/data/{gas}_D_e.dat',
    'D_p': f'app/piml/rk_pinn/data/{gas}_D_p.dat',
    'mu_e': f'app/piml/rk_pinn/data/{gas}_mu_e.dat',
    'mu_p': f'app/piml/rk_pinn/data/{gas}_mu_p.dat'
}
prop = CoronaPropSpline(coeff_file, N_neutral=Neu)


# ============================================================================
# Neural Network Architecture
# ============================================================================

## Define feedforward neural network architecture
q = 300
layers = [1, 300, 300, 300, 300, 2*(q+1)]
backbone_net = FNN(layers, act_fun=nn.Tanh())

## 
corona_model = Corona1DRKModel(
    R=R,
    T=T,
    P=P,
    V0=V0,
    dt=1.0,
    Ne_init_func=Ne_init_func,
    Np_func=Np_func,
    N_red=N_red,
    t_red=t_red,
    V_red=V_red,
    gamma=gamma,
    train_data_size=train_data_x_size,
    sample_mode='uniform',
    q=q,
    backbone_net=backbone_net,
    prop=prop
)


# ============================================================================
# Visualization and Monitoring
# ============================================================================

## Setup visualization callback for training monitoring
log_freq = 50                 # Log to TensorBoard every 50 epochs
                              # Provides frequent monitoring without excessive overhead

## Setup GIF animation for visualizing training progress
gif_results_dir = 'app/piml/rk_pinn/results/corona/'

viz_callback = Corona1DRKVisCallback(  
    model=corona_model,
    log_freq=log_freq,          # TensorBoard logging frequency: every 50 epochs
                                # Balance between monitoring granularity and I/O cost
    save_history=True,          # Enable history tracking for creating training animations
                                # Stores snapshots for GIF generation
    history_freq=1000,          # Save snapshot every 1000 epochs for memory efficiency
                                # Typical: 500-2000 depending on memory constraints
    x_eval=np.linspace(0, 1, 201, dtype=REAL()).reshape(-1,1),
                                # Spatial evaluation grid: 201 points uniformly in [0, R]
                                # Used for plotting temperature profiles
    corona_csv_file=f'app/piml/rk_pinn/data/corona_1d_{gas}_t5.0.csv',
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
corona_model.register_visualization_callback(viz_callback)


# ============================================================================
# Training Process
# ============================================================================

## Setup optimizer with adaptive learning rate
corona_model.create_optimizer('Adam', lr=learning_rate)

## Create learning rate scheduler for multi-stage training
corona_model.create_lr_scheduler('MultiStepLR', milestones=[50000,100000], gamma=0.5)

## Start training process
corona_model.train(
    num_epochs=num_epochs,          # Total 50,000 training iterations
    print_loss=True,                # Display loss to console
    print_loss_freq=log_freq,       # Print loss every 50 epochs (~1000 printouts total)
    tensorboard_logdir='app/piml/rk_pinn/runs/corona/',   # TensorBoard event directory
                                    # Useful for plotting loss curves, learning rate changes
    save_final_model=True,          # Save trained network at training completion
    checkpoint_dir='./app/piml/rk_pinn/models/corona/',   # Checkpoint/model save directory
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

## Export final results
print('\nExporting final result plots...')
viz_callback.save_final_results(
    network=corona_model.network,
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