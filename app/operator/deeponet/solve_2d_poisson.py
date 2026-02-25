###
## Author: Linlin Zhong
## Email: linlin@seu.edu.cn
## Created: Jan 01, 2023
## Description: Example approximating 2D Poisson equation solution by DeepONet.
##              PDE: -Δu(x,y) = f(x,y), with analytical family
##              f(x,y) = 2*v*pi*pi*sin(pi*x)*sin(pi*y)
##              u(x,y) = v*sin(pi*x)*sin(pi*y)
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

# reproducible
set_seed(2023)

# device
if check_gpu(print_required=True):
    DEVICE.set_device(0)
else:
    DEVICE.set_device(-1)
print(DEVICE)

# timer
my_timer = Timer()

# networks
branch_layers = [1, 32, 32, 32]
trunk_layers = [2, 32, 32, 32]  # trunk takes (x,y)
branch_net = FNN(layers=branch_layers)
trunk_net = FNN(layers=trunk_layers)
network = DeepONet(branch_net, trunk_net)
model = DeepONetModel(network=network)

# training data (parameterized RHS amplitude v)
n_params = 10
v = np.linspace(1.0, 10.0, n_params, dtype=REAL()).reshape((-1, 1))  # branch input shape (A,1)

# trunk points: uniform grid in [0,1]x[0,1]
nx, ny = 32, 32
x = np.linspace(0, 1, nx, dtype=REAL())
y = np.linspace(0, 1, ny, dtype=REAL())
X, Y = np.meshgrid(x, y, indexing='xy')
xy_points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # shape (B,2)
B = xy_points.shape[0]

# construct targets: u = v * sin(pi x) sin(pi y)
u = v @ (np.sin(np.pi * xy_points[:, 0:1]) * np.sin(np.pi * xy_points[:, 1:2])).T  # shape (A,B)

# convert to torch
v_train = numpy2torch(v)
x_train = numpy2torch(xy_points)
u_train = numpy2torch(u)

# prepare dataset
model.prepare_train_data(v_train, x_train, u_train)

# test data: pick v = 5.5
v_test = np.array([[5.5]], dtype=REAL())
xy_test = xy_points.copy()  # same grid for evaluation
# true solution on grid (1, B)
uu_true = (v_test @ (np.sin(np.pi * xy_test[:, 0:1]) * np.sin(np.pi * xy_test[:, 1:2])).T)

# convert test tensors and move to device
v_test = numpy2torch(v_test, require_grad=False)
xy_test = numpy2torch(xy_test, require_grad=False)

# train (adjust epochs/lr as needed)
model.train(num_epochs=10000, lr=1e-4)

# predict
u_pred = model.predict(v_test, xy_test)  # expected shape (1, B) as torch tensor
u_pred_np = u_pred.cpu().detach().numpy()

# L2 relative error
l2_err = calc_relative_l2_err(uu_true, u_pred_np)
print(f'L2 relative error = {l2_err:.6g}')

# print running time
my_timer.current()