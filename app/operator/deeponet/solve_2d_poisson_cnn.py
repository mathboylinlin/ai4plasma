###
## Author: Linlin Zhong
## Email: linlin@seu.edu.cn
## Created: Jan 02, 2026
## Description: Example using CNN-based DeepONet for 2D Poisson equation (Improved).
##              With enhanced DeepONet classes supporting both FNN and CNN branch networks.
##              PDE: -Δu(x,y) = f(x,y), with analytical family
##              f(x,y) = 2*v*pi*pi*sin(pi*x)*sin(pi*y)
##              u(x,y) = v*sin(pi*x)*sin(pi*y)
###

import sys
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn

from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import set_seed, numpy2torch, Timer
from ai4plasma.utils.math import calc_relative_l2_err
from ai4plasma.config import DEVICE, REAL
from ai4plasma.core.network import FNN, CNN
from ai4plasma.operator.deeponet import DeepONet, DeepONetModel

# Set random seed for reproducibility
set_seed(2023)

# Configure device (GPU or CPU)
if check_gpu(print_required=True):
    DEVICE.set_device(0)  # Use GPU 0
else:
    DEVICE.set_device(-1)  # Use CPU
print(DEVICE)

# Start timer
my_timer = Timer()

# Define spatial grid resolution for branch input (2D field)
grid_size = 16  # Resolution of input field (16x16 grid)

# Define network architectures
# Branch network: CNN to process 2D RHS field
# Input: (batch_size, 1, grid_size, grid_size) - single channel 2D field
# Output: feature vector of size 32
conv_layers = [1, 8, 16, 32]  # Channel progression: 1 -> 8 -> 16 -> 32 (deeper network)
fc_layers = [32, 32]  # Fully connected layers: flatten features to 32-dim vector

branch_net = CNN(
    conv_layers=conv_layers,
    fc_layers=fc_layers,
    input_dim=2,  # 2D spatial data
    act_fun=nn.ReLU(),
    use_BN=True,  # Use batch normalization
    use_pooling=True,  # Use pooling layers
    pooling_type='max',
    # Convolutional layer parameters
    kernel_size=3,
    stride=1,
    padding=1,
    # Pooling layer parameters (newly added for fine control)
    pooling_kernel_size=2,  # 2x2 pooling window
    pooling_stride=2,        # Non-overlapping pooling (stride = kernel_size)
    pooling_padding=0,       # No padding in pooling
    init_method='kaiming'    # Kaiming initialization for ReLU
)

# Trunk network: FNN to process spatial coordinates (x, y)
trunk_layers = [2, 32, 32, 32]  # Input: 2D coordinates, Output: 32-dim vector
trunk_net = FNN(layers=trunk_layers)

# Create DeepONet with CNN branch and FNN trunk
# The enhanced DeepONet class now automatically detects and handles CNN inputs
network = DeepONet(branch_net, trunk_net)
model = DeepONetModel(network=network)

print("Network configuration:")
print(f"  Branch network type: {branch_net.__class__.__name__}")
print(f"  Trunk network type: {trunk_net.__class__.__name__}")
print(f"  DeepONet recognizes CNN branch: {network.branch_is_cnn}")

# Generate training data
n_params = 20  # Number of different parameter values
v_values = np.linspace(1.0, 10.0, n_params, dtype=REAL())  # Parameter range

# Create RHS fields on grid for branch network input
x_grid = np.linspace(0, 1, grid_size, dtype=REAL())
y_grid = np.linspace(0, 1, grid_size, dtype=REAL())
X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='xy')

# Branch inputs: 2D RHS fields f(x,y) = 2*v*pi^2*sin(pi*x)*sin(pi*y)
# Shape: (n_params, 1, grid_size, grid_size) - 4D tensor for CNN
branch_inputs = np.zeros((n_params, 1, grid_size, grid_size), dtype=REAL())
for i, v in enumerate(v_values):
    f_field = 2 * v * np.pi**2 * np.sin(np.pi * X_grid) * np.sin(np.pi * Y_grid)
    branch_inputs[i, 0, :, :] = f_field

# Trunk inputs: evaluation points (x, y) coordinates
# Use a finer grid for evaluation
nx_eval, ny_eval = 32, 32
x_eval = np.linspace(0, 1, nx_eval, dtype=REAL())
y_eval = np.linspace(0, 1, ny_eval, dtype=REAL())
X_eval, Y_eval = np.meshgrid(x_eval, y_eval, indexing='xy')
trunk_inputs = np.stack([X_eval.ravel(), Y_eval.ravel()], axis=-1)  # Shape: (B, 2)
B = trunk_inputs.shape[0]  # Number of evaluation points

# Target outputs: analytical solution u(x,y) = v*sin(pi*x)*sin(pi*y)
# Shape: (n_params, B)
u_targets = np.zeros((n_params, B), dtype=REAL())
for i, v in enumerate(v_values):
    u_field = v * np.sin(np.pi * trunk_inputs[:, 0]) * np.sin(np.pi * trunk_inputs[:, 1])
    u_targets[i, :] = u_field

# Convert to PyTorch tensors
# NOTE: branch_train is now 4D (n_params, 1, grid_size, grid_size) - CNN format
branch_train = numpy2torch(branch_inputs)
trunk_train = numpy2torch(trunk_inputs)
u_train = numpy2torch(u_targets)

print(f"\nData shapes:")
print(f"  Branch train shape: {branch_train.shape} (CNN format: batch, channels, height, width)")
print(f"  Trunk train shape: {trunk_train.shape}")
print(f"  Targets shape: {u_train.shape}")

# Prepare training data with batching
# The enhanced prepare_train_data now automatically handles 4D CNN inputs
model.prepare_train_data(
    branch_train, 
    trunk_train, 
    u_train,
    batch_size=4,  # Process 4 parameter samples per batch
    shuffle=True   # Shuffle batches for better training
)

print(f"\nDataset information:")
print(f"  Is CNN input detected: {model.is_cnn_input}")
print(f"  Dataset size: {len(model.dataset)}")
print(f"  Batch size: {model.dataloader.batch_size}")

# Prepare test data: interpolate parameter v = 5.5
v_test = 5.5
f_test_field = 2 * v_test * np.pi**2 * np.sin(np.pi * X_grid) * np.sin(np.pi * Y_grid)
# Keep 4D format: (1, 1, grid_size, grid_size)
branch_test = numpy2torch(f_test_field.reshape(1, 1, grid_size, grid_size), require_grad=False)

trunk_test = numpy2torch(trunk_inputs, require_grad=False)

# Analytical solution for test parameter
u_test_true = v_test * np.sin(np.pi * trunk_inputs[:, 0]) * np.sin(np.pi * trunk_inputs[:, 1])
u_test_true = u_test_true.reshape(1, -1)  # Shape: (1, B)

# Train the model
print("\nTraining CNN-DeepONet model...")
print("Configuration summary:")
print(f"  - Conv layers: {conv_layers}")
print(f"  - Pooling: kernel={branch_net.pooling_kernel_size}, stride={branch_net.pooling_stride}, padding={branch_net.pooling_padding}")

# Use learning rate scheduler for better convergence
optimizer = torch.optim.Adam(model.network.parameters(), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.9)

model.train(
    num_epochs=1000, 
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    print_loss_freq=10
)

# Make predictions
u_pred = model.predict(branch_test, trunk_test)
u_pred_np = u_pred.cpu().detach().numpy()

# Calculate L2 relative error
l2_err = calc_relative_l2_err(u_test_true, u_pred_np)
print(f'L2 relative error = {l2_err:.6g}')

# Print total running time
my_timer.current()

# Optional: Visualize results
try:
    import matplotlib.pyplot as plt
    
    # Reshape predictions to 2D grid for visualization
    u_pred_grid = u_pred_np.reshape(ny_eval, nx_eval)
    u_true_grid = u_test_true.reshape(ny_eval, nx_eval)
    error_grid = np.abs(u_pred_grid - u_true_grid)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot true solution
    im1 = axes[0].contourf(X_eval, Y_eval, u_true_grid, levels=20, cmap='viridis')
    axes[0].set_title('True Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot predicted solution
    im2 = axes[1].contourf(X_eval, Y_eval, u_pred_grid, levels=20, cmap='viridis')
    axes[1].set_title('CNN-DeepONet Prediction')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot absolute error
    im3 = axes[2].contourf(X_eval, Y_eval, error_grid, levels=20, cmap='hot')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()

    fig_dir = 'app/operator/deeponet/results/cnn_deeponet_2d_poisson.png'
    plt.savefig(fig_dir, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to '{fig_dir}'")
    
except ImportError:
    print("Matplotlib not available. Skipping visualization.")
