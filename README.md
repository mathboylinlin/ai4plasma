# AI4Plasma

<p align="center">
    <picture>
        <img src="https://raw.githubusercontent.com/mathboylinlin/ai4plasma/main/docs/images/AI4Plasma_Logo.svg">
    </picture>
</p>

[![License](https://img.shields.io/github/license/mathboylinlin/ai4plasma)](https://github.com/mathboylinlin/ai4plasma/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-green)](https://ai4plasma.readthedocs.io)

**AI4Plasma** is a Python- and PyTorch-based library for physics-informed machine learning and operator learning, specifically designed for plasma physics simulation. As the world's first AI package dedicated to plasma simulation, AI4Plasma bridges the gap between cutting-edge AI techniques and plasma physics research, making it easier for plasma researchers to leverage AI tools to enhance both the efficiency and accuracy of their simulations.

Although there are several excellent predecessors such as DeepXDE, AI4Plasma is (and will always be) the AI algorithm package that understands plasma the most and the plasma algorithm package that understands AI the most. We welcome your testing and use, and also invite you to join the AI4Plasma development team.

<p align="center">
    <picture>
        <img src="https://raw.githubusercontent.com/mathboylinlin/ai4plasma/main/docs/images/AI4Plasma_Code.svg" width="800">
    </picture>
</p>

## ✨ Key Features

- **Physics-Informed Machine Learning (PIML)**
  - 🧠 **PINNs**: Classic Physics-Informed Neural Networks for solving PDEs
  - 🎯 **CS-PINNs**: Coefficient-Subnet PINNs optimized for solving plasma equations with variable coefficients
  - ⏰ **RK-PINNs**: Runge-Kutta PINNs for time-dependent problems
  - 🔄 **Meta-PINNs**: Meta-learning approach for rapid adaptation across different physics scenarios
  - 🔍 **NAS-PINNs**: Neural Architecture Search for automatic PINN architecture optimization

- **Operator Learning**
  - 🌐 **DeepONet**: Deep Operator Networks for learning solution operators
  - 🔬 **DeepCSNet**: Deep Operator Networks for predicting cross sections

- **Plasma Physics Models**
  - ⚡ **Arc Plasma Simulation**: Steady and transient arc plasma simulations (1-D in current version)
  - 📊 **Plasma Properties**: Built-in plasma property calculations and estimations

- **Comprehensive Utilities**
  - 📐 **Geometry Tools**: 1D, 2D, and 3D domain definitions with flexible boundary conditions
  - 🎨 **Visualization**: TensorBoard integration and custom plotting callbacks
  - 🚀 **Auto-differentiation**: Efficient gradient computation for complex PDEs
  - 💾 **I/O Management**: Easy model saving/loading and checkpoint management
  - 🖥️ **GPU Support**: Automatic device detection and optimization

<p align="center">
    <picture>
        <img src="https://raw.githubusercontent.com/mathboylinlin/ai4plasma/main/docs/images/CS-PINN-Sta-Arc.gif" width="800">
    </picture>
</p>
<p align="center">
    <picture>
        <img src="https://raw.githubusercontent.com/mathboylinlin/ai4plasma/main/docs/images/RK-PINN.gif" width="800">
    </picture>
</p>

## 📦 Installation

### Install from PyPI

The easiest way to install AI4Plasma is via pip:

```bash
pip install ai4plasma
```

Or upgrade to the latest version:

```bash
pip install --upgrade ai4plasma
```

### Install from Source

For the latest development version:

```bash
git clone https://github.com/ai4plasma/ai4plasma.git
cd ai4plasma
pip install -e .
```

### Install with Conda

If you prefer using Conda/Mamba:

```bash
conda create -n ai4plasma python=3.12
conda activate ai4plasma
pip install -e .
```

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0
- NumPy, SciPy, Pandas, Matplotlib
- TensorBoard, Imageio (for visualization)
- FiPy (for traditional numerical methods comparison)
- Shapely (for geometry processing)
- Huggingface-hub (for downloading dataset from Hugging Face)

## 🚀 Quick Start

### Example 1: Solving a Simple PDE with PINN

Solve the 1D ODE: **d²u/dx² = -sin(x)** with boundary conditions **u(0) = u(1) = 0**

```python
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
```

### Example 2: Learning Operators with DeepONet

Learn the solution operator for **-Δu(x) = f(x)** where **f(x) = v·π²·sin(πx)** and **u(x) = v·sin(πx)**

```python
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
```

## 📚 Module Overview

### Core Modules (`ai4plasma.core`)

- **`model.py`**: Base model class with training utilities and checkpoint management
- **`network.py`**: Neural network architectures (FNN, CNN, ResNet, etc.)

### Physics-Informed ML (`ai4plasma.piml`)

- **`geo.py`**: Geometry definitions and domain sampling
- **`pinn.py`**: Standard Physics-Informed Neural Networks
- **`cs_pinn.py`**: Coefficient-Subnet PINNs (CS-PINN) for plasma applications
- **`meta_pinn.py`**: Meta-learning PINNs (Meta-PINN) for multi-task scenarios
- **`rk_pinn.py`**: Runge-Kutta PINNs (RK-PINN) for temporal problems
- **`nas_pinn.py`**: Neural Architecture Search for PINNs (NAS-PINN)

### Operator Learning (`ai4plasma.operator`)

- **`deeponet.py`**: Deep Operator Networks implementation
- **`deepcsnet.py`**: Deep Operator Networks for cross section prediction

### Plasma Physics (`ai4plasma.plasma`)

- **`arc.py`**: Arc plasma models and solvers
- **`prop.py`**: Plasma property calculations

### Utilities (`ai4plasma.utils`)

- **`common.py`**: Common utilities (seed setting, timers, etc.)
- **`device.py`**: GPU/CPU device management
- **`io.py`**: File I/O and checkpoint utilities
- **`math.py`**: Mathematical utilities (automatic differentiation, operators)

## 📖 Documentation

Full documentation is available at [https://ai4plasma.readthedocs.io](https://ai4plasma.readthedocs.io)

- **API Reference**: Detailed documentation of all modules and classes
- **User Guide**: Step-by-step tutorials for getting started
- **Examples**: Comprehensive examples covering various use cases

## 🎯 Example Applications

The `app/` directory contains numerous ready-to-run examples:

### Physics-Informed Neural Networks

- **1D/2D Poisson Equation**: [`app/piml/pinn/solve_1d_pinn.py`](app/piml/pinn/solve_1d_pinn.py)
- **2D Rectangular Domain**: [`app/piml/pinn/solve_2d_rect_pinn.py`](app/piml/pinn/solve_2d_rect_pinn.py)
- **Polynomial Domain**: [`app/piml/pinn/solve_2d_poly_pinn.py`](app/piml/pinn/solve_2d_poly_pinn.py)

### Plasma Simulations

- **Steady Arc by FVM**: [`app/plasma/arc/solve_1d_arc_steady.py`](app/plasma/arc/solve_1d_arc_steady.py)
- **Transient Arc by FVM**: [`app/plasma/arc/solve_1d_arc_transient_explicit.py`](app/plasma/arc/solve_1d_arc_transient_explicit.py)
- **Steady Arc by CS-PINN**: [`app/piml/cs_pinn/solve_1d_arc_steady_cs_pinn.py`](app/piml/cs_pinn/solve_1d_arc_steady_cs_pinn.py)
- **Transient Arc (with radial velocity) by CS-PINN**: [`app/piml/cs_pinn/solve_1d_arc_transient_cs_pinn.py`](app/piml/cs_pinn/solve_1d_arc_transient_cs_pinn.py)
- **Transient Arc (without radial velocity) by CS-PINN**: [`app/piml/cs_pinn/solve_1d_arc_transient_noV_cs_pinn.py`](app/piml/cs_pinn/solve_1d_arc_transient_noV_cs_pinn.py)
- **Corona Discharge by RK-PINN**: [`app/piml/rk_pinn/solve_1d_corona_rk_pinn.py`](app/piml/rk_pinn/solve_1d_corona_rk_pinn.py)
- **Resume Training**: [`app/piml/cs_pinn/resume_1d_arc_transient_cs_pinn.py`](app/piml/cs_pinn/resume_1d_arc_transient_cs_pinn.py)

### Operator Learning

- **DeepONet for Poisson**: [`app/operator/deeponet/solve_1d_poisson.py`](app/operator/deeponet/solve_1d_poisson.py)
- **2D CNN-based DeepONet**: [`app/operator/deeponet/solve_2d_poisson_cnn.py`](app/operator/deeponet/solve_2d_poisson_cnn.py)
- **DeepCSNet for Cross-Sections**: [`app/operator/deepcsnet/predict_total_ionxsec.py`](app/operator/deepcsnet/predict_total_ionxsec.py)

### Meta-Learning & NAS

- **Meta-PINN**: [`app/piml/meta_pinn/solve_1d_arc_steady_meta_pinn.py`](app/piml/meta_pinn/solve_1d_arc_steady_meta_pinn.py)
- **NAS-PINN**: [`app/piml/nas_pinn/search_pinn_2d_poisson.py`](app/piml/nas_pinn/search_pinn_2d_poisson.py)

## 📊 Project Structure

```
ai4plasma/
├── ai4plasma/              # Main package
│   ├── core/              # Core model and network architectures
│   ├── piml/              # Physics-informed machine learning
│   ├── operator/          # Operator learning methods
│   ├── plasma/            # Plasma physics models
│   └── utils/             # Utility functions
├── app/                   # Example applications
│   ├── piml/              # PIML examples
│   ├── operator/          # Operator learning examples
│   └── plasma/            # Plasma simulation examples
├── docs/                  # Documentation source files
```

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding new features, improving documentation, or sharing examples, your help is appreciated.

### How to Contribute

1. **Fork the repository** on GitHub
2. **Create a new branch** for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and commit with clear messages
4. **Submit a pull request** with a detailed description

### Code Style

- Follow PEP 8 guidelines & "Numpy" style comment
- Use type hints where applicable
- Write docstrings for all public functions and classes
- Keep functions focused and modular

## 📝 Citation

If you use AI4Plasma in your research, please cite:

```bibtex
@software{ai4plasma2026,
  title={AI4Plasma: An AI Library for Plasma Physics Simulation},
  author={Zhong, Linlin and contributors},
  year={2026},
  url={https://github.com/mathboylinlin/ai4plasma},
  version={0.1.0}
}
```

For specific methods, please also cite the relevant papers:

- **PINNs**: M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics* 378, 686 (2019).
  
- **CS-PINN**: L. Zhong, B. Wu, and Y. Wang, "Low-temperature plasma simulation based on physics-informed neural networks: Frameworks and preliminary applications," *Physics of Fluids* 34, 087116 (2022)
  
- **RK-PINN**: L. Zhong, B. Wu, and Y. Wang, "Low-temperature plasma simulation based on physics-informed neural networks: Frameworks and preliminary applications," *Physics of Fluids* 34, 087116 (2022)
  
- **Meta-PINN**: L. Zhong, B. Wu, and Y. Wang, "Accelerating physics-informed neural network based 1D arc simulation by meta learning," *Journal of Physics D: Applied Physics* 56, 074006 (2023).
  
- **NAS-PINN**: Y. Wang, and L. Zhong, "NAS-PINN: Neural architecture search-guided physics-informed neural network for solving PDEs," *Journal of Computational Physics* 496, 112603 (2024).

- **DeepONet**: L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators," *Nature Machine Intelligence* 3, 218 (2021).
  
- **DeepCSNet**: Y. Wang, and L. Zhong, "DeepCSNet: a deep learning method for predicting electron-impact doubly differential ionization cross sections," *Plasma Sources Science and Technology* 33, 105012 (2024).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape AI4Plasma
- Inspired by DeepXDE and other physics-informed ML and operator learning frameworks
- Built on top of PyTorch and the broader Python scientific computing ecosystem
- Special thanks to the plasma physics and machine learning communities

## 📬 Contact

- **Author**: [Linlin Zhong](http://mathboylinlin.com)
- **Email**: linlin@seu.edu.cn
- **Homepage**: [https://github.com/mathboylinlin/ai4plasma](https://github.com/mathboylinlin/ai4plasma)
- **Documentation**: [https://ai4plasma.readthedocs.io](https://ai4plasma.readthedocs.io)

## 🌟 Star History

If you find AI4Plasma useful, please consider giving it a star ⭐ on GitHub!

---

**Made with ❤️ by the AI4Plasma Team**
