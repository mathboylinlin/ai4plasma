# Configuration

AI4Plasma centralizes two global settings in `ai4plasma.config`:

- `REAL`: controls the floating-point precision used when creating NumPy arrays and Torch tensors.
- `DEVICE`: controls whether tensors and models run on CPU or a specific CUDA GPU.

These settings are used throughout the package (for example, `FNN` constructs `nn.Linear(..., dtype=REAL('torch'))`, and helper utilities move tensors to `DEVICE()`).

## Floating-point precision: `REAL`

`REAL` is an instance of `ai4plasma.utils.math.Real`.

```python
from ai4plasma.config import REAL

print(REAL)              # e.g. Float32
np_dtype = REAL()        # numpy dtype (default)
torch_dtype = REAL('torch')
```

If you need a different precision, create a new `Real` (or extend the project-level config pattern):

```python
from ai4plasma.utils.math import Real

REAL64 = Real(precision=64)
```

Notes:

- Many examples use Float32 for speed.
- PINN-style training can sometimes benefit from Float64 in stiff problems, but it is slower.

## Compute device: `DEVICE`

`DEVICE` is an instance of `ai4plasma.utils.device.Device`.

```python
from ai4plasma.config import DEVICE
from ai4plasma.utils.device import check_gpu

if check_gpu(print_required=True):
    DEVICE.set_device(0)   # use cuda:0
else:
    DEVICE.set_device(-1)  # use cpu

print(DEVICE)              # Device(cuda:0) or Device(cpu)
```

When a function calls `DEVICE()`, it receives a `torch.device`.

## Common pitfalls

- **Mixed device**: If you see errors like *expected all tensors to be on the same device*, ensure inputs were created via `numpy2torch(...)` or explicitly moved to `DEVICE()`.
- **Precision mismatch**: If you create tensors manually with `torch.tensor(...)`, they may default to `float64`. Prefer `REAL('torch')` or `numpy2torch`.
