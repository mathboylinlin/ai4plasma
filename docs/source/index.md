# AI4Plasma Documentation

Welcome to the **AI4Plasma** documentation! AI4Plasma is a python- and pytorch-based library for physics-informed and operator learning, especially designed for plasma simulation. As the world's first AI package for plasma simulation, AI4Plasma has been determined from its inception to make it easier for plasma researchers to use AI tools, thereby enhancing the efficiency and even the accuracy of plasma simulation. Although there are several excellent predecessors, such as DeepXDE, AI4Plasma is (and will always be) the AI algorithm package that understands plasma the most and the plasma algorithm package that understands AI the most.

**Attention: The documentation is prepared with the help of generative AI due to the limitd time. It will be checked and improved in the next version**

```{toctree}
:maxdepth: 2
:caption: Getting Started
:numbered:

guides/installation
guides/quickstart
guides/basic_concepts
guides/configuration
```

```{toctree}
:maxdepth: 2
:caption: User Guide
:numbered:

guides/piml
guides/operator
guides/training
guides/utilities
guides/plasma_models
```

```{toctree}
:maxdepth: 2
:caption: Examples
:numbered:

examples/piml
examples/operator
examples/plasma
```

```{toctree}
:maxdepth: 2
:caption: Core Modules
:numbered:

api/core
api/piml
api/operator
api/plasma
api/utils
```

```{toctree}
:maxdepth: 2
:caption: Development
:numbered:

dev/index
```

## Overview

AI4Plasma provides a flexible framework for:

- **Operator**: A neural network architecture for learning nonlinear operators
- **PIML**: Physics-Informed Neural Networks for solving PDEs
- **Utilities**: Common tools for data handling, device management, and mathematical operations
- **Core Models**: Base classes and network architectures for neural network models

## Key Features

✨ **Modular Design**: Clean separation of concerns with dedicated modules for operators, PINNs, and utilities

⚡ **GPU Support**: Automatic GPU/CPU device management for efficient computation

🔬 **Physics-Informed**: Built-in support for physics-informed learning approaches

📊 **Flexible Data Handling**: Support for various data formats and batch processing strategies

## Quick Example

```python
import torch
import numpy as np
from ai4plasma.operator.deeponet import DeepONet, DeepONetModel
from ai4plasma.core.network import FNN

# Create branch and trunk networks
branch_net = FNN(layers=[1, 64, 64, 64])
trunk_net = FNN(layers=[1, 64, 64, 64])

# Initialize DeepONet
network = DeepONet(branch_net, trunk_net)
model = DeepONetModel(network=network)

# Prepare training data
model.prepare_train_data(branch_inputs, trunk_inputs, targets, batch_size=32)

# Train the model
model.train(num_epochs=100, lr=1e-3)

# Make predictions
predictions = model.predict(test_branch, test_trunk)
```

## Package Structure

```
ai4plasma/
├── core/          # Base models and network architectures
├── operator/      # Neural operator models (DeepONet, etc.)
├── piml/          # Physics-Informed Machine Learning (PINN, CS-PINN, RK-PINN, ...)
└── utils/         # Utility functions and common tools
```

## Installation

See [Installation Guide](guides/installation.md) for detailed setup instructions.

## Getting Help

- 📖 Read the [Getting Started Guide](guides/quickstart.md)
- 🔍 Check the [API Reference](api/operator.md)
- 📚 Browse [Examples](examples/index.md)
- 💬 Open an issue on GitHub

## Citation

If you use AI4Plasma in your research, please cite:

```bibtex
@software{ai4plasma2026,
  author = {Zhong, Linlin},
  title = {AI4Plasma: An AI Library for Plasma Physics Simulation},
  year = {2026},
  url = {https://github.com/mathboylinlin/ai4plasma}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/mathboylinlin/ai4plasma/blob/main/LICENSE) file for details.

---

**Last Updated**: January 14, 2026