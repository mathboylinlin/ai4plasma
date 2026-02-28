###############################################################################
## Author: Yifan Wang
## Email: ee_wangyf@seu.edu.cn
## Created: Feb. 13, 2026
## Updated: Feb. 13, 2026
##
## Description:
##   Demonstration script for predicting total ionization cross sections of 
##   molecules using DeepCSNet.
##
##   This example showcases how to apply DeepCSNet to approximate total electron-
##   impact ionization cross sections as a function of molecular composition and
##   incident electron energy. The model learns the complex nonlinear relationship
##   between molecular descriptors (C, H, O, N, F atom counts) and energy-dependent
##   cross sections.
##
## References:
##   [1] Y. Wang and L. Zhong, "DeepCSNet: a deep learning method for predicting
##       electron-impact doubly differential ionization cross sections,"
##       Plasma Sources Science and Technology, vol. 33, no. 10, p. 105012, 2024.
##
###############################################################################

import os
import sys
import pandas as pd
import torch
sys.path.append('.')

import numpy as np
import torch.nn as nn

from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import set_seed, numpy2torch, Timer
from ai4plasma.utils.math import calc_relative_l2_err
from ai4plasma.config import DEVICE, REAL
from ai4plasma.core.network import FNN
from ai4plasma.operator.deepcsnet import DeepCSNet, DeepCSNetModel


###############################################################################
## 1. Environment Setup
###############################################################################

## Fix random seed for reproducibility ##
set_seed(2026)

## Configure computational device (CPU/GPU) ##
if check_gpu(print_required=True):
    DEVICE.set_device(0)  # Use CUDA device 0 (first GPU)
else:
    DEVICE.set_device(-1) # Fall back to CPU if no GPU available
print(f"Using device: {DEVICE}")

## Initialize timer for performance monitoring ##
my_timer = Timer()

###############################################################################
## 2. Network Architecture Definition
###############################################################################

## Define network layer configurations ##
molecule_layers, trunk_layers = [5, 80, 80, 80], [1, 80, 80, 80]

## Select activation function ##
activation = nn.Sigmoid()

## Instantiate sub-networks ##
molecule_network = FNN(layers=molecule_layers, act_fun=activation)
trunk_network = FNN(layers=trunk_layers, act_fun=activation)

## Construct DeepCSNet in MMC (Multi-Molecule Configuration) mode ##
network = DeepCSNet(trunk_net=trunk_network, molecule_net=molecule_network)

## Wrap network in high-level model interface ##
mymodel = DeepCSNetModel(network=network)


###############################################################################
## 3. Data Loading and Preprocessing
###############################################################################

## Initialize data containers ##
Energy = np.empty((0, 1), dtype=REAL())
Molecule = np.empty((0, 5), dtype=REAL())
data = np.empty((0, 1), dtype=REAL())

## Parse cross section data from CSV files ##
data_dir = 'app/operator/deepcsnet/data/csv'
csvs = os.listdir(data_dir)  # List all CSV files in directory
csvs.sort()  # Sort alphabetically for consistent processing order

# Process each molecule's cross section data
for csv in csvs:
    # Read cross section data from CSV
    raw = pd.read_csv(os.path.join(data_dir, csv))
    
    # Extract cross section values (skip header row)
    raw_data = np.array(raw['Q_combined(A^2)'][1:], dtype=REAL())
    
    # Extract energy values (skip header row)
    rawa_energy = np.array(raw['E_BEB(eV)'][1:], dtype=REAL())
    
    # Filter data: keep only energies ≥ 30 eV
    index = np.where(rawa_energy >= 30)
    
    # Append filtered data to global arrays
    data = np.append(data, raw_data[index]).reshape(-1, 1)
    Energy = np.append(Energy, rawa_energy[index]).reshape(-1, 1)

    # Parse molecular formula from filename to extract atom counts
    # Example: "C2H4O.csv" → {C:2, H:4, O:1, N:0, F:0}
    molecule = {'C': 0, 'H': 0, 'O': 0, 'N': 0, 'F': 0}
    formula = csv[:-4]  # Remove '.csv' extension
    
    # Parse formula from right to left to handle multi-digit numbers
    atom_cnt = 0
    i = 0
    for c in formula[::-1]:
        if c.isdigit():
            # Accumulate digits for atom count (handles ≥10 atoms)
            atom_cnt += int(c) * (10 ** i)
            i += 1
        elif c.isupper():
            # Uppercase letter indicates new element
            # If no digit follows, count is 1 (e.g., H in CH4)
            molecule[c] += (atom_cnt or 1)
            i = atom_cnt = 0
    
    # Convert dictionary to array in consistent order: [C, H, O, N, F]
    molecule_list = []
    for num in molecule.items():
        molecule_list.append(num[1])
    molecule = np.array(molecule_list, dtype=REAL()).reshape(-1, 5)
    
    # Append molecular descriptor to global array
    Molecule = np.append(Molecule, molecule, axis=0)

## Data organization and preprocessing ##

# Trim energy grid to common range across all molecules
# Ensures all molecules have cross sections at same energy points
Energy = Energy[:176]

# Organize data into matrices
MoleAttri = Molecule  # Shape: [n_molecules, 5] - molecular descriptors
DiffCS = data.reshape(MoleAttri.shape[0], Energy.shape[0])  # Shape: [n_molecules, n_energies]

## Shuffle molecules for random train/test split ##
index = np.random.choice(np.arange(MoleAttri.shape[0]), MoleAttri.shape[0], replace=False)
MoleAttri = MoleAttri[index, :]  # Shuffle molecular descriptors
DiffCS = DiffCS[index, :]         # Shuffle cross sections accordingly

## Apply logarithmic transformation ##
Energy = np.log(Energy)
DiffCS = np.log(DiffCS)

## Normalize data to [0.05, 0.95] range ##
Energy_max = np.max(Energy)
Energy_min = np.min(Energy)
Energy = (Energy - Energy_min) * 0.9 / (Energy_max - Energy_min) + 0.05
Energy = Energy.reshape(-1, 1)

# Normalize cross sections (targets)
DiffCS_max = np.max(DiffCS)
DiffCS_min = np.min(DiffCS)
DiffCS = (DiffCS - DiffCS_min) * 0.9 / (DiffCS_max - DiffCS_min) + 0.05

## Prepare model inputs ##
Trunk_Input = Energy  # Shape: [176, 1] - energy grid for trunk network

## Split into training and test sets ##
Trunk_Input = numpy2torch(Trunk_Input)  # Energy grid (shared by all molecules)
Mole_Input_Train = numpy2torch(MoleAttri[:70, :])  # Training molecular descriptors
DiffCS_Train = numpy2torch(DiffCS[:70, :])          # Training cross sections

# Test data kept as NumPy for error calculation (no gradients needed)
Mole_Input_Test = numpy2torch(MoleAttri[70:, :], require_grad=False)  # Test molecular descriptors
DiffCS_Test = DiffCS[70:, :]  # Test cross sections (ground truth)

## Configure training data in model ##
mymodel.prepare_train_data(
    trunk_inputs=Trunk_Input,        # Shape: [176, 1] - energy coordinates
    molecule_inputs=Mole_Input_Train, # Shape: [70, 5] - molecular features
    targets=DiffCS_Train              # Shape: [70, 176] - target cross sections
)


###############################################################################
## 4. Model Training
###############################################################################

## Define learning rate schedule ##
def update_lr(epoch):
    """
    Learning rate multiplier as a function of epoch.
    
    Strategy:
    - Epochs 0-99,999: lr_multiplier = 1.0 (full learning rate)
    - Epochs 100,000+: lr_multiplier = 0.5 (reduced for fine-tuning)
    
    Parameters:
    -----------
    epoch : int
        Current training epoch number.
    
    Returns:
    --------
    float
        Learning rate multiplier (applied to base learning rate).
    """
    if epoch < 100000:
        lr = 1.0     # Full learning rate: 5e-4
    else:
        lr = 0.5     # Reduced learning rate: 2.5e-4
    return lr

## Configure optimizer and learning rate scheduler ##
optimizer = torch.optim.Adam(mymodel.network.parameters(), lr=5e-4)

# LambdaLR scheduler: applies update_lr multiplier at each step
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=update_lr)

## Train the model ##
print("\nStarting training...")
print("Training configuration:")
print(f"  - Number of epochs: 200,000")
print(f"  - Initial learning rate: 5e-4")
print(f"  - Training molecules: 70")
print(f"  - Energy points: 176")
print(f"  - Total training samples: 70 × 176 = 12,320")
print("")

mymodel.train(
    num_epochs=200000,
    print_loss_freq=100,      # Print loss every 100 epochs
    optimizer=optimizer,
    lr_scheduler=scheduler, 
    tensorboard_logdir='app/operator/deepcsnet/runs',
    checkpoint_dir='app/operator/deepcsnet/models',
    checkpoint_freq=10000
)

print("\nTraining completed!")


###############################################################################
## 5. Model Evaluation and Testing
###############################################################################

print("\n" + "="*80)
print("Testing Phase")
print("="*80)

## Make predictions on test set ##
DiffCS_Predict = mymodel.predict(
    trunk_input=Trunk_Input,         # Same energy grid as training
    molecule_input=Mole_Input_Test   # Test molecular descriptors [18, 5]
)
# Output shape: [18, 176] - cross sections for 18 molecules at 176 energies

# Convert PyTorch tensor to NumPy array for error calculation
DiffCS_Predict = DiffCS_Predict.cpu().detach().numpy()

## Calculate relative L2 error ##
DiffCS_Predict = (DiffCS_Predict - 0.05) * (DiffCS_max - DiffCS_min) / 0.9 + DiffCS_min  # Inverse normalization
DiffCS_Test = (DiffCS_Test - 0.05) * (DiffCS_max - DiffCS_min) / 0.9 + DiffCS_min  # Inverse normalization

DiffCS_Predict = np.exp(DiffCS_Predict)  # Inverse log transform to original scale
DiffCS_Test = np.exp(DiffCS_Test)          # Inverse log transform to original scale

l2_err = calc_relative_l2_err(DiffCS_Test, DiffCS_Predict)

print(f"\nTest Results:")
print(f"  - Number of test molecules: {DiffCS_Test.shape[0]}")
print(f"  - Energy points per molecule: {DiffCS_Test.shape[1]}")
print(f"  - Total test predictions: {DiffCS_Test.shape[0] * DiffCS_Test.shape[1]}")
print(f"  - Relative L2 error: {l2_err:.6g}")
print(f"\nNote: Error calculated in normalized log-space.")
print(f"      For physical interpretation, inverse transform to original scale.")

###############################################################################
## 6. Performance Summary
###############################################################################

print("\n" + "="*80)
print("Execution Summary")
print("="*80)

## Report total execution time ##
my_timer.current()

print("\n" + "="*80)
print("Workflow completed successfully!")
print("="*80)