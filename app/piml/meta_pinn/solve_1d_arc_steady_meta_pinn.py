"""
Meta-PINN example for 1D steady arc tasks with SF6-N2 mixtures.

This script builds multiple StaArc1DTask instances with different SF6:N2
mixing ratios (ratio affects arc plasma properties) and runs a short
meta-training loop, followed by a meta-test and a comparison study.
"""

import sys
sys.path.append('.')

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import set_seed, numpy2torch, Timer
from ai4plasma.config import DEVICE, REAL
from ai4plasma.piml.meta_pinn import MetaPINN, StaArc1DTask, MetaStaArc1DNet
from ai4plasma.piml.cs_pinn import get_Tfunc_from_file
from ai4plasma.core.network import FNN



# Fix random seed for reproducibility
set_seed(2023)

# Set computing device (GPU if available, otherwise CPU)
if check_gpu(print_required=True):
    DEVICE.set_device(0)  # Using cuda:0 for GPU acceleration
else:
    DEVICE.set_device(-1) # Using CPU
print(DEVICE)

# Start timer to measure total execution time
my_timer = Timer()

# Gas type and arc geometry
gas = 'SF6_N2'          # Working gas: Sulfur hexafluoride and Nitrogen mixture
R = 10e-3           # Arc radius in meters (10 mm)
I = 200             # Arc current in Amperes
T_red = 1e4         # Temperature reduction factor for normalization (10,000 K)
Tb = 2000.0         # Boundary temperature in Kelvin at r = R

layers = [1, 50, 50, 50, 50, 50, 50, 1]
backbone_net = FNN(layers, act_fun=nn.Tanh())
support_data_size = 500
query_data_size = 600

base_dir = 'app/piml/meta_pinn/data/'

def build_tasks(mix_ratios):
    """
    Build a list of StaArc1DTask instances for given SF6:N2 mix ratios.

    Parameters:
    -----------
    mix_ratios : list[float]
        N2 fractions in the mixture, e.g., 0.1 means 10% N2.

    Returns:
    --------
    list[StaArc1DTask]
        Task list with arc plasma properties mapped by ratio.
    """
    task_list = []
    for i, ratio in enumerate(mix_ratios):
        gas_str = f"{gas.lower()}_c{ratio*100:.0f}"
        thermo_file = os.path.join(base_dir, f"prop/{gas_str}_p1.dat")
        nec_file = os.path.join(base_dir, f"prop/{gas_str}_p1_nec.dat")

        task = StaArc1DTask(
            task_id=gas_str,
            R=R,
            I=I,
            Tb=Tb,
            T_red=T_red,
            backbone_net=backbone_net,
            thermo_file=thermo_file,
            nec_file=nec_file,
            support_data_size=support_data_size,
            query_data_size=query_data_size
        )
        task_list.append(task)

    return task_list


train_mix_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]  # N2 fractions for training tasks
test_mix_ratios = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]  # N2 fractions for test tasks
train_tasks = build_tasks(train_mix_ratios)
test_tasks = build_tasks(test_mix_ratios)

test_tasks_ref_Tfunc = [get_Tfunc_from_file(os.path.join(base_dir, f"sta_arc1d_{gas.lower()}_c{ratio*100:.0f}_p1.csv")) for ratio in test_mix_ratios]

# Meta-training setup
meta_pinn = MetaPINN(
    train_tasks=train_tasks,
)

# Meta-training (short run)
meta_pinn.meta_train(
    outer_epochs=10000,
    inner_epochs=5,
    outer_lr=1e-4,
    inner_lr=1e-5,
    print_freq=10,
    tensorboard_logdir='app/piml/meta_pinn/runs',
    checkpoint_dir='app/piml/meta_pinn/models',
    checkpoint_freq=1000,
)

meta_pinn.meta_test(
    test_tasks=test_tasks,
    results_dir='app/piml/meta_pinn/results'
)


# Comparison Test: Meta-learned vs From-scratch Training
print("\n" + "=" * 80)
print("Starting Comparison Test: Meta-learned vs From-scratch")
print("=" * 80)

# Select a test task for detailed comparison
comparison_task = test_tasks[0]
comparison_task_ref_Tfunc = test_tasks_ref_Tfunc[0]
print(f"\nComparison Task: {comparison_task.task_id}")
print(f"N2 ratio: {comparison_task.task_id.split('_c')[1][:2]}%")

# Training parameters for fine-tuning
fine_tune_epochs = 1000
fine_tune_lr = 1e-4  # Learning rate for Adam optimizer

# Get query data for fine-tuning and evaluation
query_data = comparison_task.get_query_data()

# 1. Meta-learned Model Training
print("\n" + "-" * 80)
print("1. Fine-tuning Meta-learned Model")
print("-" * 80)

# Clone the meta-learned network and load pretrained weights
meta_learned_backbone_net = FNN(layers, act_fun=nn.Tanh())
meta_learned_network = MetaStaArc1DNet(meta_learned_backbone_net)
meta_learned_network.to(DEVICE())

# Load meta-learned weights
checkpoint_path = f'app/piml/meta_pinn/results/{comparison_task.get_task_id()}_meta_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=DEVICE())
meta_learned_network.load_state_dict(checkpoint['network_state_dict'])
meta_learned_network.train()

# Create optimizer for meta-learned model
meta_optimizer = torch.optim.Adam(meta_learned_network.parameters(), lr=fine_tune_lr)

# Storage for training history
meta_learned_history = {
    'query_loss': [],
    'epochs': []
}

# Fine-tune the meta-learned model
for epoch in range(fine_tune_epochs):
    meta_optimizer.zero_grad()
    
    loss_qry, _ = comparison_task.compute_loss(meta_learned_network, query_data)
    
    loss_qry.backward()
    meta_optimizer.step()
    
    meta_learned_history['query_loss'].append(loss_qry.item())
    meta_learned_history['epochs'].append(epoch + 1)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{fine_tune_epochs}] Query Loss: {loss_qry.item():.6f}")

# 2. From-scratch Model Training
print("\n" + "-" * 80)
print("2. Training From-scratch Model")
print("-" * 80)

# Create a new network with random initialization
from_scratch_backbone_net = FNN(layers, act_fun=nn.Tanh())
from_scratch_network = MetaStaArc1DNet(from_scratch_backbone_net)
from_scratch_network.to(DEVICE())
from_scratch_network.train()

# Create optimizer for from-scratch model
scratch_optimizer = torch.optim.Adam(from_scratch_network.parameters(), lr=fine_tune_lr)

# Storage for training history
from_scratch_history = {
    'query_loss': [],
    'epochs': []
}

# Train from scratch
for epoch in range(fine_tune_epochs):
    scratch_optimizer.zero_grad()
    
    loss_qry, _ = comparison_task.compute_loss(from_scratch_network, query_data)
    
    loss_qry.backward()
    scratch_optimizer.step()
    
    from_scratch_history['query_loss'].append(loss_qry.item())
    from_scratch_history['epochs'].append(epoch + 1)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{fine_tune_epochs}] Query Loss: {loss_qry.item():.6f}")

# 3. Visualization
print("\n" + "-" * 80)
print("3. Generating Comparison Plots")
print("-" * 80)

results_dir = 'app/piml/meta_pinn/results'
os.makedirs(results_dir, exist_ok=True)

# Generate a uniform grid for visualization
r_values = np.linspace(0, 1, query_data_size, dtype=REAL()).reshape(-1, 1)  # Normalized radius values from 0 to 1
r_torch = numpy2torch(r_values, require_grad=False)

# Get predictions from both models
meta_learned_network.eval()
from_scratch_network.eval()
with torch.no_grad():
    pred_meta = meta_learned_network(r_torch).detach().cpu().numpy()
    pred_scratch = from_scratch_network(r_torch).detach().cpu().numpy()

pred_meta = pred_meta*(1 - r_values)*T_red + Tb
pred_scratch = pred_scratch*(1 - r_values)*T_red + Tb

# Use reference temperature profile as ground truth
reference_pred = comparison_task_ref_Tfunc(r_values*R)

# Create combined figure with 3 subplots (2 on top, 1 on bottom)
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.3, wspace=0.25)

# Top left: Temperature predictions
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(r_values * 1000, reference_pred * T_red, 'k-', linewidth=2, label='Reference', alpha=0.7)
ax0.plot(r_values * 1000, pred_meta * T_red, 'b-', linewidth=2, label='Meta-learned')
ax0.plot(r_values * 1000, pred_scratch * T_red, 'r--', linewidth=2, label='From-scratch')
ax0.set_xlabel('Radius r (mm)', fontsize=12)
ax0.set_ylabel('Temperature (K)', fontsize=12)
ax0.set_title('Temperature Profiles', fontsize=14, fontweight='bold')
ax0.legend(fontsize=11)
ax0.grid(True, alpha=0.3)

# Top right: Absolute errors
ax1 = fig.add_subplot(gs[0, 1])
error_meta = np.abs(pred_meta - reference_pred) * T_red
error_scratch = np.abs(pred_scratch - reference_pred) * T_red
ax1.plot(r_values * 1000, error_meta, 'b-', linewidth=2, label='Meta-learned')
ax1.plot(r_values * 1000, error_scratch, 'r--', linewidth=2, label='From-scratch')
ax1.set_xlabel('Radius r (mm)', fontsize=12)
ax1.set_ylabel('Absolute Error (K)', fontsize=12)
ax1.set_title('Prediction Errors', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Bottom: Loss curves (spanning both columns)
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(meta_learned_history['epochs'], meta_learned_history['query_loss'], 
        'b-', linewidth=2, label='Meta-learned', marker='o', markersize=4, markevery=10)
ax2.plot(from_scratch_history['epochs'], from_scratch_history['query_loss'], 
        'r--', linewidth=2, label='From-scratch', marker='s', markersize=4, markevery=10)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.savefig(os.path.join(results_dir, 'comparison_results.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(results_dir, 'comparison_results.png')}")
plt.close()

# 4. Summary Statistics
print("\n" + "=" * 80)
print("Comparison Summary")
print("=" * 80)
print(f"\nFinal Results after {fine_tune_epochs} epochs:")
print("-" * 80)
print(f"{'Model':<20} {'Loss':<15}")
print("-" * 80)
print(f"{'Meta-learned':<20} {meta_learned_history['query_loss'][-1]:<15.6f}")
print(f"{'From-scratch':<20} {from_scratch_history['query_loss'][-1]:<15.6f}")
print("-" * 80)

# Calculate improvement
loss_improvement = (from_scratch_history['query_loss'][-1] - meta_learned_history['query_loss'][-1]) / from_scratch_history['query_loss'][-1] * 100

print(f"\nImprovement by Meta-learning:")
print(f"  Final Loss: {loss_improvement:+.2f}%")

# Calculate convergence speed (epochs to reach target loss)
target_loss = min(meta_learned_history['query_loss'][-1], from_scratch_history['query_loss'][-1]) * 1.1
meta_converge_epoch = next((i for i, loss in enumerate(meta_learned_history['query_loss']) if loss < target_loss), fine_tune_epochs)
scratch_converge_epoch = next((i for i, loss in enumerate(from_scratch_history['query_loss']) if loss < target_loss), fine_tune_epochs)

print(f"\nConvergence Speed (to reach {target_loss:.6f}):")
print(f"  Meta-learned: {meta_converge_epoch} epochs")
print(f"  From-scratch: {scratch_converge_epoch} epochs")
print(f"  Speedup: {scratch_converge_epoch / max(meta_converge_epoch, 1):.2f}x")
print("=" * 80)


# Print total execution time
my_timer.current()
