###############################################################################
## Author: Linlin Zhong
## Email: linlin@seu.edu.cn
## Created: Jan. 19, 2022
## Updated: Feb. 3, 2026
##
## Description:
##   Resume training of 1D transient arc CS-PINN model from a saved checkpoint.
##   This script loads a previously trained model and continues training for
##   additional epochs, useful for:
##   - Fine-tuning with different hyperparameters
##   - Extended training to improve convergence
##   - Adaptive training with reduced learning rate
##   - Recovery from interrupted training runs
##
###############################################################################


import sys
sys.path.append('.')

import os
import numpy as np
import torch
import torch.nn as nn

from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import set_seed, Timer
from ai4plasma.config import DEVICE, REAL
from ai4plasma.core.network import FNN
from ai4plasma.plasma.prop import ArcPropSpline
from ai4plasma.piml.cs_pinn import get_Tfunc_from_file, TraArc1DModel, TraArc1DVisCallback


# ============================================================================
# CONFIGURATION AND INITIALIZATION
# ============================================================================

## Fix random seed for reproducibility
set_seed(2023)

## Set computing device
if check_gpu(print_required=True):
    DEVICE.set_device(0)  # Use CUDA GPU
else:
    DEVICE.set_device(-1) # Use CPU

print(DEVICE)

## Initialize timer
my_timer = Timer()


# ============================================================================
# ARC DISCHARGE PHYSICAL PARAMETERS
# ============================================================================
gas = 'SF6'
R = 10e-3
I = 200
T_red = 1e4
t_red = 1e-3
Tb = 2000.0

# ============================================================================
# CHECKPOINT AND TRAINING CONFIGURATION
# ============================================================================

## Resume training configuration
start_epoch = 150000
num_epochs = 350000 
initial_learning_rate = 1e-4       # Initial LR (if not loading from optimizer state)
resume_learning_rate = 5e-5        # Learning rate for resume phase (often lower)
                                   # Common: 0.1x to 0.5x of initial rate for fine-tuning


## Checkpoint loading configuration
checkpoint_dir = './app/piml/cs_pinn/models/tra/'
checkpoint_name = f'checkpoint_epoch_{start_epoch}.pth'
checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name)

if not os.path.exists(checkpoint_file):
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

## Training data configuration
train_data_x_size = 200
train_data_t_size = 100

## Evaluation grid
test_data_x = np.linspace(0, 1, 210)
test_data_t = [0.1, 0.5, 0.9]


# ============================================================================
# MATERIAL PROPERTIES
# ============================================================================

thermo_file = f'app/piml/cs_pinn/data/{gas.lower()}_p1.dat'
nec_file = f'app/piml/cs_pinn/data/{gas.lower()}_p1_nec.dat'

prop = ArcPropSpline(thermo_file, nec_file, R)


# ============================================================================
# LOAD CHECKPOINT AND INITIALIZE MODEL
# ============================================================================

## Reconstruct neural network backbone
layers = [2, 300, 300, 300, 300, 300, 300, 2]
backbone_net = FNN(layers, act_fun=nn.Tanh())

## Load initial temperature function
initial_file = f'app/piml/cs_pinn/data/sta_arc1d_{gas}.csv'
Tinit_func = get_Tfunc_from_file(initial_file)

## Create TraArc1DModel instance
arc_model = TraArc1DModel(
    R,
    I,
    Tb,
    Tinit_func,
    T_red,
    t_red,
    backbone_net,
    train_data_x_size,
    train_data_t_size,
    sample_mode='uniform',
    prop=prop
)

# ============================================================================
# VISUALIZATION CALLBACK SETUP
# ============================================================================

gif_results_dir = 'app/piml/cs_pinn/results/tra_resume/'

viz_callback = TraArc1DVisCallback(
    model=arc_model,
    log_freq=50,
    save_history=True,
    history_freq=1000,
    x_eval=np.linspace(0, 1, 201, dtype=REAL()).reshape(-1, 1),
    t_eval=[0.1, 0.5, 0.9],
    TV_csv_file=[f'app/piml/cs_pinn/data/tra_arc1d_{gas}_t0.1.csv',
                 f'app/piml/cs_pinn/data/tra_arc1d_{gas}_t0.5.csv',
                 f'app/piml/cs_pinn/data/tra_arc1d_{gas}_t0.9.csv'],
    gif_enabled=True,
    gif_dir=gif_results_dir,
    gif_freq=1000,
    gif_duration_ms=300
)

arc_model.register_visualization_callback(viz_callback)


# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================
arc_model.create_optimizer('Adam', lr=initial_learning_rate)
arc_model.create_lr_scheduler('MultiStepLR', 
                               milestones=[20000, 50000], 
                               gamma=0.5)

# ============================================================================
# RESUME TRAINING
# ============================================================================

arc_model.train(
    num_epochs=num_epochs,
    print_loss=True,
    print_loss_freq=50,
    tensorboard_logdir='app/piml/cs_pinn/runs/tra_resume/',
    save_final_model=True,
    checkpoint_dir='./app/piml/cs_pinn/models/tra_resume/',
    checkpoint_freq=5000,
    resume_from=checkpoint_file  # Pass loaded epoch to continue numbering
)

print('\n' + '='*70)
print('Training resumed and completed!')
print('='*70)

# ============================================================================
# POST-PROCESSING
# ============================================================================

print('\nGenerating updated training animation GIF...')
viz_callback.save_gif()

print('\nExporting updated final result plots...')
viz_callback.save_final_results(
    network=arc_model.network,
    save_dir=gif_results_dir,
    epoch=start_epoch + num_epochs
)
print(f'Updated plots saved to: {gif_results_dir}')

# ============================================================================
# SUMMARY
# ============================================================================

print('\n' + '='*70)
my_timer.current()
print('='*70 + '\n')
