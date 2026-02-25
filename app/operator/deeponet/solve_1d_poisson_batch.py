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
## Description: 1D Poisson equation example for testing DeepONetModel.prepare_train_data batch_size.
###

import sys
sys.path.append('.')

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

## Network (branch net & trunk net) ##
branch_layers, trunk_layers = [1, 10, 10, 10, 10], [1, 10, 10, 10, 10]
branch_network, trunk_network = FNN(layers=branch_layers), FNN(layers=trunk_layers)

## Create DeepONet ##
network = DeepONet(branch_network, trunk_network)

## Model ##
mymodel = DeepONetModel(network=network)

## Prepare dataset ##
# Branch input: v with shape (A, 1)
n_params = 20
v = np.linspace(1.0, 20.0, n_params, dtype=REAL()).reshape((-1, 1))  # branch input shape (A,1)
# Trunk input: x with shape (B, 1)
x = np.linspace(-1, 1, 40, endpoint=True, dtype=REAL()).reshape((-1, 1))
# Target output: u with shape (A, B)
u = v * np.sin(np.pi * x.T)  # Broadcasting

# Convert numpy arrays to torch tensors
v, x, u = numpy2torch(v), numpy2torch(x), numpy2torch(u)

# batch_size
batch_size = 5
mymodel.prepare_train_data(v, x, u, batch_size=batch_size, shuffle=False)

## Test data ##
vv = np.array([[5.5]], dtype=REAL())
xx = np.linspace(-1, 1, 30, endpoint=True, dtype=REAL()).reshape((-1, 1))
uu = vv * np.sin(np.pi * xx.T)

vv = numpy2torch(vv)
xx = numpy2torch(xx)

## Training ##
print(f"Training with batch_size = {batch_size}")
mymodel.train(num_epochs=10000, lr=1e-4) 

## Test ##
u_predict = mymodel.predict(vv, xx)
u_predict = u_predict.cpu().detach().numpy()

## Calculate L2 error ##
l2_err = calc_relative_l2_err(uu, u_predict)
print('L2 error = %.6g' % (l2_err))

## Print running time ##
my_timer.current()