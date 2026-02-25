# Training, Logging, Checkpoints

AI4Plasma training utilities show up in multiple places:

- `ai4plasma.core.model.BaseModel`: minimal base wrapper
- `ai4plasma.operator.*Model`: operator-learning training wrappers
- `ai4plasma.piml.pinn.PINN`: multi-term physics loss and callbacks

```{contents}
:local:
:depth: 2
```

## Training components

Most training loops share the same building blocks:

- **Model**: network + physics or operator wrapper
- **Optimizer**: Adam/AdamW commonly used for PINNs and operators
- **Scheduler**: optional learning rate schedule for long runs
- **Data**: supervised datasets or collocation points
- **Callbacks**: visualization and diagnostics

## TensorBoard

Many training loops accept `tensorboard_logdir=...` and will write event files.

Run TensorBoard from the repository root:

```bash
tensorboard --logdir app
```

Common log directories:

- `app/operator/deeponet/runs/...`
- `app/piml/cs_pinn/runs/...`
- `app/piml/rk_pinn/runs/...`

### What to log

Recommended scalars:

- total loss and per-term loss
- learning rate
- validation error (if available)
- physical diagnostics (e.g., conductance, radiation power, max/min temperature)

Recommended figures:

- solution profiles and contours
- residual maps or error heatmaps
- training curves (loss, error)

## Checkpointing

Typical options (model-specific):

- `checkpoint_dir`: directory for saving periodic checkpoints
- `checkpoint_freq`: save every N epochs
- `resume_from`: resume training from a checkpoint file

A checkpoint usually contains:

- model weights
- optimizer state
- current epoch

### When to save

- Save frequently in long runs (every 100-1000 epochs)
- Save more frequently during hyperparameter exploration
- Always save the best validation checkpoint when a validation metric is available

## Reproducibility

Use `set_seed(...)` from `ai4plasma.utils.common` and record:

- seed
- device
- precision
- code version

Also consider recording:

- git commit or tag
- dataset version and preprocessing
- training configuration (hyperparameters)

## Practical tips for PINN training

- Start with a conservative learning rate (e.g. `1e-4` to `1e-3`).
- Use a scheduler (MultiStepLR or ReduceLROnPlateau) for long runs.
- Balance loss term weights (domain vs boundary) when residual magnitudes differ.
- Log intermediate physical quantities (conductivity, radiation terms) to catch non-physical regimes early.

## Example: basic training skeleton

```python
from ai4plasma.utils.common import set_seed
from ai4plasma.core.network import FNN

set_seed(2023)

net = FNN([2, 64, 64, 64, 1])
# Create model wrapper, define equation terms, then train
# model.train(num_epochs=..., lr=...)
```

## Troubleshooting

- **Loss does not decrease**: reduce learning rate, increase collocation points, or check derivative computations.
- **Boundary drift**: increase boundary term weight or sample more boundary points.
- **Exploding gradients**: use gradient clipping or smooth loss functions, and check property interpolation ranges.
- **Slow training**: reduce batch size, simplify network, or use mixed precision where appropriate.

## Checkpoint resumption workflow

1. Identify the latest checkpoint file in `checkpoint_dir`.
2. Pass `resume_from=...` to the model training method.
3. Verify logs continue from the expected epoch in TensorBoard.

## Next steps

- Operator training details: [guides/operator.md](operator.md)
- PINN workflows: [guides/piml.md](piml.md)
- Example scripts: [examples/index.md](../examples/index.md)
