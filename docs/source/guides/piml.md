# Physics-Informed Machine Learning (PIML)

AI4Plasma provides a general physics-informed learning framework built around `ai4plasma.piml.pinn`. This guide explains the core abstractions, typical workflows, and practical training guidance for physics-informed neural networks (PINNs) and related models.

```{contents}
:local:
:depth: 2
```

## What is PIML?

Physics-Informed Machine Learning integrates physical laws (typically PDEs) directly into the learning objective. Instead of learning only from data pairs, the model is trained to satisfy the governing equations, boundary conditions, and initial conditions.

For a PDE:

$$
\mathcal{F}[u](x, t) = 0, \quad (x, t) \in \Omega \times [0, T]
$$

with boundary conditions $\mathcal{B}[u] = 0$ and initial conditions $u(x, 0) = u_0(x)$, a PINN minimizes a weighted loss:

$$
\mathcal{L} = w_{pde}\mathcal{L}_{pde} + w_{bc}\mathcal{L}_{bc} + w_{ic}\mathcal{L}_{ic}
$$

where each loss term evaluates residuals at sampled points in the corresponding domain.

## Core abstractions

### `EquationTerm`

An `EquationTerm` represents one residual component of the physics loss, such as:

- PDE interior residual
- boundary condition residual (Dirichlet or Neumann)
- initial condition residual
- constraints or regularization

Each term has:

- `residual_fn(network, data) -> residual`
- `weight`
- `data` (collocation points)
- optional `DataLoader` for batching

This design makes it straightforward to build multi-physics problems where different residuals must be balanced.

### `PINN` training loop

The `PINN` class (see `ai4plasma.piml.pinn.PINN`) extends the base model concept with:

- multiple equation terms
- optional adaptive loss weighting
- checkpointing and resume
- TensorBoard logging

Typical workflow:

1. Create a network (often an `FNN`)
2. Define physics residuals and construct `EquationTerm` objects
3. Create optimizer and optional LR scheduler
4. Train with `model.train(...)`

### Visualization callbacks

`VisualizationCallback` enables custom plotting during training, with figures logged to TensorBoard. Many scripts in `app/piml/*` use specialized callbacks to monitor:

- predicted field profiles
- derived physical quantities (conductivity, radiation terms)
- loss decomposition and error metrics

## Geometry and sampling

Physics-informed models rely on sampling points in the interior and on the boundaries of the domain. AI4Plasma provides geometry helpers in `ai4plasma.piml.geo`:

- `Geo1D`, `Geo1DTime` for 1D and 1D time-dependent problems
- `GeoRect2D`, `GeoPoly2D` for 2D domains
- `GeoPoly2DTime` for 2D time-dependent problems

Sampling strategies:

- Uniform grid sampling
- Random sampling
- LHS (Latin Hypercube Sampling) for space-filling designs

Use consistent sampling densities across PDE, boundary, and initial condition points to avoid imbalance.

## Autograd derivatives

Many residual functions require first- and second-order derivatives. AI4Plasma includes helpers like `ai4plasma.utils.math.df_dX`:

```python
from ai4plasma.utils.math import df_dX

u = net(x)
u_x = df_dX(u, x)      # du/dx
u_xx = df_dX(u_x, x)   # d2u/dx2
```

For time-dependent problems, compute temporal derivatives similarly by passing time coordinates in the input.

## Loss design and weighting

Loss weighting is crucial. If one residual dominates, the model may satisfy one constraint while ignoring others. AI4Plasma supports adaptive weighting to balance competing terms.

Guidelines:

- Start with equal weights if scales are similar
- Increase boundary or initial condition weights if they drift
- Use adaptive weights for multi-physics problems with varying magnitudes

## Metrics and evaluation

Common metrics include:

- Relative $L_2$ error: $\|\hat{u}-u\|_2 / \|u\|_2$
- Mean absolute error (MAE)
- Physics-specific diagnostics (e.g., flux integrals or conservation checks)

Evaluate on a held-out grid or reference solution when possible.

## Worked example: 1D steady arc (CS-PINN)

The CS-PINN steady arc example in [app/piml/cs_pinn/solve_1d_arc_steady_cs_pinn.py](../../../app/piml/cs_pinn/solve_1d_arc_steady_cs_pinn.py) demonstrates:

- physics residual definition for the Elenbaas-Heller energy equation
- temperature-dependent properties via `ArcPropSpline`
- visualization callbacks producing TensorBoard panels and training GIFs

See [guides/properties.md](plasma_models.md) and [examples/piml.md](../examples/piml.md) for run commands and outputs.

## Practical tips

- Normalize inputs and outputs when ranges differ by orders of magnitude.
- Use smooth activations like `tanh` for continuous fields; consider deeper networks for sharp gradients.
- Validate residuals separately by inspecting each `EquationTerm` loss.
- If training is unstable, reduce learning rate or increase collocation points.

## Common pitfalls

- **Insufficient boundary sampling**: leads to boundary drift even with correct PDE residuals.
- **Scale mismatch**: results in one loss term dominating the optimization.
- **Overfitting to collocation points**: validate on a separate grid if possible.

## Next steps

- Review operator learning alternatives in [guides/operators.md](operator.md)
- Explore example workflows in [examples/piml.md](../examples/piml.md)
- Check PINN variants in the API reference at [api/index.md](../api/index.md)
