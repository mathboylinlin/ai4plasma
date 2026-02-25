# Plasma Models

AI4Plasma contains physics solvers and supporting utilities under `ai4plasma.plasma`. This guide introduces the built-in plasma models and the property-interpolation toolkit used across solvers and PINN workflows.

```{contents}
:local:
:depth: 2
```

## Arc discharge models (`ai4plasma.plasma.arc`)

The arc module implements 1D radial thermal arc models based on the Elenbaas-Heller equation. It includes:

- `StaArc1D`: stationary (steady-state) arc model
- `TraArc1DNoV`: transient arc model without radial velocity
- `TraArc1D`: transient arc model with radial velocity (convection)

These solvers use conventional discretization (via FiPy) and temperature-dependent properties.

### Governing equations (overview)

Steady-state energy balance (cylindrical coordinates):

$$
\frac{1}{r}\frac{d}{dr}\left(r\kappa\frac{dT}{dr}\right) + \sigma E^2 - S_{rad} = 0
$$

Transient energy equation:

$$
\rho C_p \frac{dT}{dt} + \rho C_p V \frac{dT}{dr} = \frac{1}{r}\frac{d}{dr}\left(r\kappa\frac{dT}{dr}\right) + \sigma E^2 - S_{rad}
$$

where $\kappa$ is thermal conductivity, $\sigma$ is electrical conductivity, and $S_{rad}$ is radiation loss computed from NEC tables.

### How this relates to CS-PINN

The CS-PINN implementation in `ai4plasma.piml.cs_pinn` solves a closely related 1D steady energy equation, but replaces the discretization with a neural network and physics residuals.

- Use `ai4plasma.plasma.arc` when you want a traditional solver baseline.
- Use `ai4plasma.piml.cs_pinn` when you want PINN-based solutions, differentiability, and easy coupling with learning.

## Plasma properties and interpolation (`ai4plasma.plasma.prop`)

Thermal-plasma and discharge models require temperature- or field-dependent coefficients, such as conductivity $\sigma(T)$, thermal conductivity $\kappa(T)$, diffusion coefficients, and radiation loss tables. AI4Plasma provides property utilities in `ai4plasma.plasma.prop`.

### Table reading

- `read_thermo_data(...)` reads temperature-dependent thermo/transport properties.
- `read_nec_data(...)` reads net emission coefficient (NEC) tables used for radiation losses.

Data files are expected to be whitespace-separated tables with headers (for example: `T(K)`, `rho(kg/m3)`, `sigma(S/m)`), matching the fields used in `read_thermo_data`.

### Interpolation helpers

- `interp_prop(...)`: 1D interpolation in temperature using `RegularGridInterpolator` (SciPy 1.14+ compatible).
- `interp_prop_log(...)`: interpolation in log-space for quantities spanning multiple orders of magnitude.
- `interp_nec(...)` / `interp_nec_log(...)`: 2D interpolation for NEC tables depending on temperature and radius.

Use log-space interpolation for coefficients that change exponentially with temperature to improve numerical stability.

### PyTorch-ready spline classes

#### `ArcPropSpline`

`ArcPropSpline` is designed for PINNs and neural operators:

- reads thermo/NEC data files
- exposes `sigma(T)`, `kappa(T)`, `nec(T)` and other properties
- clamps temperature to a physical range (commonly `[300, 30000]` K)
- returns Torch tensors on the current device and supports autograd

#### `CoronaPropSpline`

For corona discharge RK-PINN examples, `CoronaPropSpline` provides interpolation of transport coefficients such as $\alpha(E/N)$, $\mu_e(E/N)$, and $D_e(E/N)$.

### Example: creating `ArcPropSpline`

```python
from ai4plasma.plasma.prop import ArcPropSpline

thermo_file = 'app/piml/cs_pinn/data/sf6_p1.dat'
nec_file = 'app/piml/cs_pinn/data/sf6_p1_nec.dat'
R = 10e-3

prop = ArcPropSpline(thermo_file, nec_file, R)
```

### Example: property interpolation in a residual

```python
def pde_residual(model, X, prop):
	r = X[:, 0:1]
	T = model(X)
	kappa = prop.kappa(T)
	sigma = prop.sigma(T)
	# Use kappa and sigma in residual definition
	return residual
```

### Numerical stability tips

- Prefer log-interpolation for coefficients spanning many orders of magnitude.
- Clamp inputs (temperature or reduced field) to the validity range of the tables.
- In PINN residuals, watch for exploding gradients when coefficients change sharply; lowering LR or using smooth loss functions can help.

## Where this fits in the workflow

- Classical solvers (arc models) and physics-informed models (PINNs) both rely on the same property tables.
- Keep property sources consistent between baseline solvers and learning models to ensure fair comparisons.
- When building datasets for PINNs, sample temperature or field ranges within the validity of the tables.
