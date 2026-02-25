# Utilities

AI4Plasma includes a focused set of utilities under `ai4plasma.utils` used across examples and models. These helpers cover reproducibility, device handling, data conversion, timing, I/O, and math utilities commonly needed in PINNs and operator learning.

```{contents}
:local:
:depth: 2
```

## Reproducibility

- `set_seed(seed)`: seeds Python, NumPy, and PyTorch for deterministic runs. Also configures CuDNN determinism.

Suggested practice:

- Call `set_seed(...)` once at the start of each script.
- Log the seed, device, and precision to your experiment notes.

## Device utilities

- `check_gpu(print_required=False)`: reports GPU availability.
- `select_gpu_by_id(gpu_id)`: sets the active GPU by ID with validation.
- `torch_device(device_id)`: returns a `torch.device` object for CPU or GPU.
- `Device`: centralized device manager (used by `ai4plasma.config.DEVICE`).

Example:

```python
from ai4plasma.utils.device import check_gpu
from ai4plasma.config import DEVICE

DEVICE.set_device(0 if check_gpu() else -1)
```

## Data conversion

- `numpy2torch(x, require_grad=True)`: converts NumPy arrays to Torch tensors, moves to `DEVICE()`, and optionally enables gradients.
- `list2torch(list_of_arrays, require_grad=True)`: converts a list of NumPy arrays to a list of tensors.

Example:

```python
from ai4plasma.utils.common import numpy2torch

X = numpy2torch(np.random.randn(100, 2), require_grad=True)
```

## Timing

- `Timer`: lightweight wall-clock timer used by many scripts.
- `print_runing_time(t)`: prints time in seconds, minutes, or hours.

## I/O helpers

- `read_json(path)`: JSON configuration file loading with error handling.
- `img2gif(img_file_list, gif_file, ...)`: build training animations from saved frames.

Example:

```python
from ai4plasma.utils.io import img2gif

img2gif(['epoch_0.png', 'epoch_100.png'], 'training.gif', duration=300)
```

## Math helpers

- `df_dX(f, X)`: autograd derivatives used in PINN residuals.
- `calc_l2_err(true, pred)`: L2 error for quick diagnostics.
- `calc_relative_l2_err(true, pred)`: relative error for normalized comparisons.
- `Real`: floating-point precision manager for NumPy and PyTorch.

Example: derivatives

```python
from ai4plasma.utils.math import df_dX

u = net(x)
u_x = df_dX(u, x)
```

Example: precision control

```python
from ai4plasma.utils.math import Real

real = Real(precision=64)
real.set_torch_dtype(64)
```

## Physical constants

Common physical constants are exposed in `ai4plasma.utils.common`:

- `Boltz_k`: Boltzmann constant (J/K)
- `Elec`: elementary charge (C)
- `Epsilon_0`: vacuum permittivity (F/m)
