# Neural Operators

This guide explains the operator-learning models under `ai4plasma.operator`. It focuses on the intuition, data organization, and training workflow for neural operators in AI4Plasma.

```{contents}
:local:
:depth: 2
```

## What is an Operator?

In many plasma problems, we want a model that maps an *entire input function* to an *output function*, not just a single input to a single output. For example:

- Input: a spatially varying source term, boundary profile, or material property field
- Output: a temperature field, potential field, or density field across a domain

Neural operators approximate this mapping directly, enabling fast evaluation of parametric PDEs without solving the PDE from scratch each time.

## DeepONet

DeepONet learns an operator $\mathcal{G}$ that maps an input function $u$ to an output function $\mathcal{G}(u)(y)$.

In AI4Plasma, `ai4plasma.operator.deeponet.DeepONet` is implemented as a **branch network** + **trunk network**:

- Branch network: encodes the *input function / parameterization* (e.g., forcing term amplitude, boundary condition parameters, or a discretized field).
- Trunk network: encodes the *query coordinates* (e.g., $x$ for 1D, or $(x,y)$ for 2D).

The outputs are combined via Einstein summation:

$$
\text{out}_{b,n} = \sum_i \text{branch}_{b,i}\,\text{trunk}_{n,i} + \text{bias}
$$

### Input / output shapes

The implementation supports two typical branch modes:

- **FNN branch**: `branch_inputs` is 2D: `(batch_size, features)`
- **CNN branch**: `branch_inputs` is 4D: `(batch_size, channels, height, width)`

`trunk_inputs` is typically `(num_points, coord_dim)`.

The output has shape `(batch_size, num_points)`.

### Data organization

DeepONet training data typically contains:

- `branch_inputs`: sampled input functions, either as vectors (FNN) or images (CNN)
- `trunk_inputs`: coordinates where the output is evaluated
- `targets`: ground-truth output values at those coordinates

In many applications, the same `trunk_inputs` grid is shared across all samples, while `branch_inputs` varies across cases.

### Training wrapper

`ai4plasma.operator.deeponet.DeepONetModel` provides a ready-to-use training loop with:

- configurable optimizer / scheduler
- TensorBoard logging
- checkpointing and resuming

See also: the training guide in `guides/training.md`.

### Practical tips

- Normalize both inputs and outputs when possible (especially for multi-physics datasets).
- Ensure the branch and trunk output dimensions match the chosen basis dimension.
- If the output field is smooth, use `tanh` activations and moderate depth; for sharp features, consider deeper networks or richer basis dimension.

## DeepCSNet

DeepCSNet (`ai4plasma.operator.deepcsnet.DeepCSNet`) is a specialized operator-like architecture for cross-section prediction.

It separates inputs into **coefficient subnets** and a **trunk subnet**:

- Molecule Net (optional): molecular descriptors (multi-molecule mode)
- Energy Net (optional): incident energy features
- Trunk Net (required): output coordinates (angles, ejected energies, etc.)

The final prediction is computed via tensor contraction (einsum), similar to DeepONet.

### Modes

- **SMC** (single-molecule configuration): Energy Net + Trunk Net
- **MMC** (multi-molecule configuration): Molecule Net (+ optional Energy Net) + Trunk Net

### Data organization

DeepCSNet datasets often use a shared coordinate grid (angles, ejected energies) across samples. Organize the inputs so that the coefficient subnets capture *case-specific* variation while the trunk subnet captures *evaluation coordinates*.

### Practical notes

- Ensure the hidden dimension of trunk outputs matches the branch output dimension (or matches the concatenated dimension in MMC).
- When building datasets, the coordinate grid (trunk input) is often shared across all cases.
- For MMC, be careful to align molecular descriptors with the corresponding output samples.

## Losses and Metrics

Operator learning in AI4Plasma is typically supervised with mean-squared error on the output field values:

$$
\mathcal{L}_{data} = \frac{1}{N}\sum_{i=1}^N |\hat{u}(y_i) - u(y_i)|^2
$$

Common evaluation metrics include:

- Relative $L_2$ error: $\|\hat{u}-u\|_2 / \|u\|_2$
- Mean absolute error (MAE)
- Task-specific physics diagnostics (e.g., integral quantities, extrema, or conservation checks)

If you want physics constraints during operator training, consider combining operator models with PINN-style residual losses for hybrid supervision.

## Minimal working example (DeepONet)

This mirrors the scripts in `app/operator/deeponet/`:

```python
import numpy as np
import torch.nn as nn

from ai4plasma.config import DEVICE, REAL
from ai4plasma.utils.device import check_gpu
from ai4plasma.utils.common import set_seed, numpy2torch
from ai4plasma.core.network import FNN
from ai4plasma.operator.deeponet import DeepONet, DeepONetModel

set_seed(2023)
DEVICE.set_device(0 if check_gpu() else -1)

branch_net = FNN([1, 32, 32, 32], act_fun=nn.Tanh())
trunk_net = FNN([1, 32, 32, 32], act_fun=nn.Tanh())

net = DeepONet(branch_net, trunk_net)
model = DeepONetModel(net)

v = np.linspace(1.0, 10.0, 10, dtype=REAL()).reshape(-1, 1)
x = np.linspace(-1, 1, 64, dtype=REAL()).reshape(-1, 1)
u = v * np.sin(np.pi * x.T)

model.prepare_train_data(numpy2torch(v), numpy2torch(x), numpy2torch(u))
model.train(num_epochs=10000, lr=1e-4)
```

## Common pitfalls

- **Mismatched dimensions**: ensure branch output dimension equals trunk output dimension (or the concatenation in multi-branch setups).
- **Overfitting on small datasets**: use early stopping, weight decay, or data augmentation.
- **Inconsistent scaling**: apply consistent normalization across train/validation/test splits.

## Where to go next

- See the training workflow in [guides/training.md](training.md)
- Explore PINN-based alternatives in [guides/piml.md](piml.md)
- Check scripts in `app/operator/` for practical end-to-end examples
