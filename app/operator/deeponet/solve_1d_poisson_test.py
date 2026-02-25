###
## Author: Linlin Zhong
## Email: linlin@seu.edu.cn
## Created: Jan 01, 2023
## Description: Example approximating 2D Poisson equation solution by DeepONet
##              for testing DeepONetModel.prepare_train_data batch_size.
##              PDE: -Δu(x) = f(x), with analytical family
##              f(x) = v*pi*pi*sin(pi*x)
##              u(x) = v*sin(pi*x)
###
## Description: Test script for DeepONetModel.train features
###

import sys
sys.path.append('.')


import os
import numpy as np
from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import set_seed, numpy2torch, Timer
from ai4plasma.utils.math import calc_relative_l2_err
from ai4plasma.config import DEVICE, REAL
from ai4plasma.core.network import FNN
from ai4plasma.operator.deeponet import DeepONet, DeepONetModel

## Fix random seed ##
set_seed(2023)

## Set device ##
if check_gpu(print_required=True):
    DEVICE.set_device(0)  # Using cuda:0
else:
    DEVICE.set_device(-1) # Using cpu
print(DEVICE)

## start timer ##
my_timer = Timer()

# Data: 1D Poisson (same as example)
n_params = 20
n_points = 40
v = np.linspace(1.0, 10.0, n_params, dtype=REAL()).reshape((-1, 1))
x = np.linspace(-1, 1, n_points, dtype=REAL()).reshape((-1, 1))
u = v * np.sin(np.pi * x.T)
v, x, u = numpy2torch(v), numpy2torch(x), numpy2torch(u)

# Network
branch_layers, trunk_layers = [1, 10, 10, 10, 10], [1, 10, 10, 10, 10]
branch_network, trunk_network = FNN(layers=branch_layers), FNN(layers=trunk_layers)
network = DeepONet(branch_network, trunk_network)
model = DeepONetModel(network=network)

# Prepare data
model.prepare_train_data(v, x, u, batch_size=2, shuffle=True)

# Test data
vv = np.array([[5.8]], dtype=REAL())
xx = np.linspace(-1, 1, 15, dtype=REAL()).reshape((-1, 1))
uu = vv * np.sin(np.pi * xx.T)
vv, xx = numpy2torch(vv), numpy2torch(xx)

# --- 1. Default Adam optimizer, tensorboard, checkpoint, final model ---
print('Test 1: Adam optimizer, tensorboard, checkpoint, final model')
ckpt_dir = 'app/operator/deeponet/results/ckpt'
tb_dir = 'app/operator/deeponet/results/tb'
final_model_path = 'app/operator/deeponet/results/test_final1.pth'
model.train(
    num_epochs=1000,
    lr=1e-3,
    print_loss=True,
    print_loss_freq=100,
    tensorboard_logdir=tb_dir,
    checkpoint_dir=ckpt_dir,
    checkpoint_freq=500,
    save_final_model=True,
    final_model_path=final_model_path
)

# --- 2. Resume from checkpoint ---
print('Test 2: Resume from checkpoint')
model2 = DeepONetModel(network=DeepONet(FNN(branch_layers), FNN(trunk_layers)))
model2.prepare_train_data(v, x, u, batch_size=2, shuffle=True)
model2.train(
    num_epochs=2000,
    lr=1e-3,
    resume_from=os.path.join(ckpt_dir, 'checkpoint_epoch_1000.pth'),
    print_loss=True,
    print_loss_freq=100,
    checkpoint_dir=ckpt_dir,
    checkpoint_freq=500
)

# --- 3. Custom optimizer (SGD) ---
print('Test 3: Custom optimizer (SGD)')
model3 = DeepONetModel(network=DeepONet(FNN(branch_layers), FNN(trunk_layers)))
model3.prepare_train_data(v, x, u, batch_size=2, shuffle=True)
opt_cfg = {'name':'SGD', 'params':{'lr':1e-4, 'momentum':0.8}}
model3.train(
    num_epochs=1000,
    optimizer_cfg=opt_cfg,
    print_loss=True,
    print_loss_freq=100
)

# --- 4. Prediction and error ---
u_pred = model.predict(vv, xx).cpu().detach().numpy()
l2_err = calc_relative_l2_err(uu, u_pred)
print('Test 4: L2 error Adam = %.6g' % l2_err)

u_pred3 = model3.predict(vv, xx).cpu().detach().numpy()
l2_err3 = calc_relative_l2_err(uu, u_pred3)
print('Test 4: L2 error SGD = %.6g' % l2_err3)

print('All tests done.')
print('TensorBoard logdir:', tb_dir)
print('Checkpoint dir:', ckpt_dir)
print('Final model path:', final_model_path)
print('To view loss curve: tensorboard --logdir', tb_dir)


## Print running time ##
my_timer.current()
