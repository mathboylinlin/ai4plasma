import sys
sys.path.append('.')

import torch
import torch.nn as nn
from ai4plasma.config import REAL
from ai4plasma.piml.pinn import PINN
from ai4plasma.piml.geo import Geo1D
from ai4plasma.core.network import FNN
from ai4plasma.utils.math import df_dX, calc_relative_l2_err
from ai4plasma.utils.common import set_seed, numpy2torch

# Set seed for reproducibility
set_seed(2026)

# Define custom PINN class
class SimplePINN(PINN):
    def __init__(self, network):
        self.geo = Geo1D([0.0, 1.0])
        super().__init__(network)
    
    @staticmethod
    def _pde_residual(network, x):
        """Compute residual: d²u/dx² + sin(x)"""
        u = network(x)
        u_x = df_dX(u, x)
        u_xx = df_dX(u_x, x)
        return u_xx + torch.sin(x)
    
    @staticmethod
    def _bc_residual(network, x):
        """Dirichlet BC: u(x) = 0"""
        return network(x)
    
    def _define_loss_terms(self):
        """Set up domain and boundary loss terms"""
        x_domain = self.geo.sample_domain(100, mode='uniform')
        x_bc = self.geo.sample_boundary()
        
        self.add_equation('Domain', self._pde_residual, weight=1.0, data=x_domain)
        self.add_equation('Left BC', self._bc_residual, weight=10.0, data=x_bc[0])
        self.add_equation('Right BC', self._bc_residual, weight=10.0, data=x_bc[1])

# Create and train PINN
network = FNN([1, 64, 64, 64, 1])
pinn = SimplePINN(network)
pinn.set_loss_func(nn.MSELoss())

pinn.train(num_epochs=5000, lr=1e-3, print_loss=True, print_loss_freq=500)

# Predict on evaluation points
import numpy as np
x_eval = np.linspace(0, 1, 200, dtype=REAL()).reshape(-1, 1)
u_true = np.sin(x_eval) - x_eval * np.sin(1)  # Analytical solution for comparison
x_eval_tensor = numpy2torch(x_eval)
u_pred = pinn.predict(x_eval_tensor).detach().numpy()

print(f"Relative L2 error: {calc_relative_l2_err(u_true, u_pred):.4e}")