import sys
sys.path.append('.')

import numpy as np
from ai4plasma.core.network import FNN
from ai4plasma.operator.deeponet import DeepONet, DeepONetModel
from ai4plasma.utils.common import set_seed, numpy2torch
from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.math import calc_relative_l2_err
from ai4plasma.config import DEVICE, REAL

# Set seed for reproducibility
set_seed(2026)

## Set device ##
if check_gpu(print_required=True):
    DEVICE.set_device(0)  # Using cuda:0
else:
    DEVICE.set_device(-1) # Using cpu
print(DEVICE)

# Define branch and trunk networks
branch_net = FNN([1, 10, 10, 10, 10])
trunk_net = FNN([1, 10, 10, 10, 10])

# Create DeepONet model
network = DeepONet(branch_net, trunk_net)
model = DeepONetModel(network=network)

# Prepare training data
# Branch input: parameter v (A samples)
v = np.array([[2, 4, 6, 8, 10]], dtype=REAL()).reshape((-1, 1))
# Trunk input: spatial coordinate x (B points)
x = np.linspace(-1, 1, 40, endpoint=True, dtype=REAL()).reshape((-1, 1))
# Target output: solution u (A × B)
u = v * np.sin(np.pi * x.T)

# Convert to tensors and prepare data
v, x, u = numpy2torch(v), numpy2torch(x), numpy2torch(u)
model.prepare_train_data(v, x, u)

# Train the model
model.train(num_epochs=10000, lr=1e-4, print_loss_freq=100)

# Test on unseen parameter
vv = np.array([[5.5]], dtype=REAL())
xx = np.linspace(-1, 1, 30, endpoint=True, dtype=REAL()).reshape((-1, 1))
uu = vv * np.sin(np.pi * xx.T)  # True solution

# Predict
vv, xx = numpy2torch(vv), numpy2torch(xx)
u_pred = model.predict(vv, xx).cpu().detach().numpy()

# Evaluate
l2_err = calc_relative_l2_err(uu, u_pred)
print(f"Relative L2 error: {l2_err:.4e}")