# Basic Concepts

AI4Plasma is the world's first AI library specifically designed for plasma physics simulation, combining state-of-the-art machine learning techniques with rigorous physics-based modeling. This guide introduces the fundamental concepts and architectural components that power AI4Plasma.

## Core Architecture

### Network Components

AI4Plasma provides flexible neural network building blocks optimized for scientific computing:

#### **FNN (Fully Connected Neural Network)**

Dense multi-layer perceptrons with:
- Customizable depth and width
- Optional batch normalization
- Multiple weight initialization strategies (Xavier, zero initialization)
- Tanh activation by default (suitable for smooth physics solutions)
- Precise floating-point control (REAL precision)

**Structure:**
```
Input → Linear → [BN?] → Activation → ... → Linear (output)
```

#### **CNN (Convolutional Neural Network)**

1D/2D/3D convolution-based architectures with:
- Flexible backbone with optional fully connected head
- Adaptive global pooling strategies
- Batch normalization and max/avg pooling options
- Automatic dimension detection
- Lazy initialization based on actual feature sizes

**Structure:**
```
Input → [Conv → BN? → Activation → Pool?] × N
     → Global Pool or Flatten
     → [FC → Activation?] × M → Output
```

#### **RelaxFNN (Neural Architecture Search)**

Relaxed architecture for automatic network design:
- Soft selection over different architectural choices
- Learnable architecture parameters optimized jointly with weights
- Efficient search for problem-specific network structures
- Discrete architecture extraction after search

### Geometry System

The geometry system provides a unified interface for domain and boundary sampling across different problem types:

#### **Base Classes**
- `Geometry`: Abstract base class defining the interface for all geometric domains
- `GeoTime`: Temporal domain $[t_s, t_e]$
- `Geo1D`: 1D spatial domain $[x_l, x_u]$
- `Geo1DTime`: Space-time domain for 1D problems
- `GeoPoly2D`: 2D polygonal domains
- `GeoRect2D`: 2D rectangular domains
- `GeoPoly2DTime`: Space-time domain for 2D problems

#### **Sampling Strategies**
- **Uniform**: Evenly spaced grid sampling
- **Random**: Uniform random distribution
- **LHS**: Latin Hypercube Sampling (for efficient space-filling designs)

**Example Usage:**
```python
from ai4plasma.piml.geo import Geo1DTime, SamplingMode

# Create a space-time domain
geo = Geo1DTime()
geo.create_domain(xl=0.0, xu=1.0, ts=0.0, te=1.0)

# Sample interior points
X_interior = geo.sample_domain(nx=100, nt=50, mode=SamplingMode.UNIFORM)

# Sample initial condition
X_ic = geo.sample_ic(nx=100, mode=SamplingMode.RANDOM)
```

## Physics-Informed Machine Learning (PIML)

### Physics-Informed Neural Networks (PINNs)

PINNs embed physical laws directly into neural network training through residual-based loss functions. Instead of requiring large labeled datasets, PINNs learn solutions by minimizing physics equation residuals.

#### **Mathematical Formulation**

For a PDE of the form:

$$\mathcal{F}[u](x, t) = 0, \quad x \in \Omega, t \in [0, T]$$

with boundary conditions $\mathcal{B}[u] = 0$ and initial conditions $u(x, 0) = u_0(x)$, a PINN minimizes:

$$\mathcal{L} = w_{pde}\mathcal{L}_{pde} + w_{bc}\mathcal{L}_{bc} + w_{ic}\mathcal{L}_{ic}$$

where:
- $\mathcal{L}_{pde} = \frac{1}{N_{pde}}\sum_{i=1}^{N_{pde}} |\mathcal{F}[u_\theta](x_i, t_i)|^2$
- $\mathcal{L}_{bc} = \frac{1}{N_{bc}}\sum_{i=1}^{N_{bc}} |\mathcal{B}[u_\theta](x_i, t_i)|^2$
- $\mathcal{L}_{ic} = \frac{1}{N_{ic}}\sum_{i=1}^{N_{ic}} |u_\theta(x_i, 0) - u_0(x_i)|^2$

#### **Key Features**

The PINN framework in AI4Plasma provides:

- **Multi-Physics Support**: Handle arbitrary coupled PDEs through `EquationTerm` abstraction
- **Automatic Differentiation**: Compute spatial/temporal derivatives via PyTorch autograd
- **Adaptive Loss Weighting**: Automatically balance competing physics constraints
- **Batch Training**: Support for large datasets via DataLoader integration
- **Visualization Callbacks**: Real-time monitoring with custom visualization functions
- **Checkpoint Management**: Save and resume training with full state recovery
- **TensorBoard Integration**: Comprehensive logging and monitoring

**Example:**
```python
from ai4plasma.piml.pinn import PINN, EquationTerm

# Define PDE residual
def pde_residual(model, X):
    x, t = X[:, 0:1], X[:, 1:2]
    u = model(X)
    u_t = model.df_dt(X, u)
    u_xx = model.df_dxx(X, u, x_idx=0)
    return u_t - 0.01 * u_xx  # Heat equation

# Create equation term
pde_term = EquationTerm(
    name="pde",
    residual_fn=pde_residual,
    data=X_pde,
    weight=1.0
)

# Build and train PINN
pinn = PINN(model=net, equation_terms=[pde_term])
pinn.train(epochs=10000, lr=1e-3)
```

### PINN Variants

AI4Plasma includes several advanced PINN variants for specialized applications:

#### **CS-PINN (Coefficient-Subnet PINN)**

Specialized for problems with complex material properties and temperature-dependent coefficients:

- **Automatic Boundary Enforcement**: Network architecture guarantees boundary conditions by construction
- **Spline Interpolation**: Handles temperature-dependent properties (thermal conductivity, heat capacity)
- **Gauss-Legendre Quadrature**: Accurate computation of integral terms
- **Application**: Arc discharge simulations with temperature-dependent plasma properties

**Network Construction:**

$$T(r) = (r - R) \cdot N_\theta(r) + T_b$$

This structure automatically satisfies $T(R) = T_b$, reducing training complexity.

#### **Meta-PINN**

Enables rapid adaptation to new physics tasks through meta-learning:

- **Task Abstraction**: Support/query split for few-shot learning
- **MAML Framework**: Model-Agnostic Meta-Learning for PINN
- **Fast Adaptation**: Quickly fine-tune to new parameters (current, geometry)
- **Application**: Multi-current arc discharge, multi-geometry problems

**Meta-Learning Workflow:**
1. **Meta-training**: Sample task batch → adapt on support set → update on query set
2. **Meta-testing**: Initialize with meta-parameters → fine-tune on new task (few steps)

#### **RK-PINN (Runge-Kutta PINN)**

Incorporates Runge-Kutta time-stepping for improved temporal accuracy:

- **High-Order Time Integration**: 4th-order Runge-Kutta scheme
- **Stage-by-Stage Training**: Learn intermediate RK stages
- **Temporal Accuracy**: Better handling of time-dependent dynamics
- **Application**: Transient plasma phenomena, corona discharge

#### **NAS-PINN (Neural Architecture Search PINN)**

Automatically discovers optimal network architectures for specific physics problems:

- **Differentiable Architecture Search**: Learn architecture parameters via gradient descent
- **Relaxed Architectures**: Soft selection over architectural choices
- **Problem-Specific Optimization**: Find best width/depth for target PDE
- **Application**: Automated PINN design without manual hyperparameter tuning

## Neural Operators

Neural operators learn mappings between infinite-dimensional function spaces, enabling fast evaluation of parametric PDEs and reducing computational cost of repeated simulations.

### DeepONet (Deep Operator Network)

DeepONet learns nonlinear operators $G: \mathcal{U} \to \mathcal{V}$ mapping input functions to output functions.

#### **Architecture**

DeepONet consists of two sub-networks:

- **Branch Network**: Processes input functions $u(x)$ (supports FNN and CNN)
- **Trunk Network**: Processes evaluation coordinates $y$

**Mathematical Formulation:**

$$G(u)(y) \approx \sum_{i=1}^p b_i(u) \cdot t_i(y) + \text{bias}$$

where:
- $b_i(u)$: $i$-th basis function from branch network
- $t_i(y)$: $i$-th basis function from trunk network
- $p$: latent dimension

#### **Key Features**

- **Automatic Architecture Detection**: FNN for 2D data, CNN for 4D image-like data
- **Flexible Data Splitting**: By branch samples or trunk evaluation points
- **Distributed Training**: Full DataLoader support for large-scale problems
- **Checkpoint Management**: Resume training with full state recovery

**Applications:**
- Parametric PDE solving (e.g., Poisson equation with varying boundary conditions)
- Fast surrogate models for expensive simulations
- Real-time physics predictions

#### **Example:**
```python
from ai4plasma.operator.deeponet import DeepONetModel

model = DeepONetModel(
    branch_sizes=[100, 128, 128, 128],  # Branch network
    trunk_sizes=[2, 128, 128, 128],      # Trunk network
    basis_dim=128,                       # Latent dimension
    bias_output=True
)

model.train_model(
    train_loader=train_loader,
    epochs=1000,
    lr=1e-3
)
```

### DeepCSNet (Deep Cross Section Network)

Specialized neural operator for predicting electron-impact cross sections in plasma physics.

#### **Architecture**

DeepCSNet employs a modular coefficient-subnet structure:

- **Molecule Net**: Processes molecular features (for multi-molecule mode)
- **Energy Net**: Processes incident electron energy
- **Trunk Net**: Processes scattering angles and kinematics

#### **Operation Modes**

- **SMC (Single-Molecule Configuration)**: Energy Net + Trunk Net
  - For single molecular species
  
- **MMC (Multi-Molecule Configuration)**: Molecule Net + Energy Net + Trunk Net
  - For multiple molecular species simultaneously

#### **Applications**

- Predicting doubly differential ionization cross sections (DDCS)
- Total ionization cross sections
- Fast cross section lookup for plasma kinetic simulations

**Physical Relevance:**
Cross sections are fundamental to plasma modeling, determining collision rates, energy transfer, and species production. DeepCSNet provides orders-of-magnitude speedup compared to first-principles calculations.

## Equation Terms and Loss Components

The `EquationTerm` class provides a flexible abstraction for physics constraints:

```python
class EquationTerm:
    """Encapsulates a single physics constraint.
    
    Attributes
    ----------
    name : str
        Identifier for the constraint (e.g., 'pde', 'bc', 'ic')
    residual_fn : callable
        Function computing the physics residual
    data : torch.Tensor
        Collocation points for evaluating residual
    weight : float
        Loss weight for balancing multiple constraints
    """
```

**Benefits:**
- Modular constraint definition
- Dynamic weight updates during training
- Easy addition/removal of physics terms
- Support for data batching via DataLoader

## Visualization and Monitoring

AI4Plasma provides comprehensive tools for monitoring training progress:

### TensorBoard Integration

Automatic logging of:
- Loss components (PDE, BC, IC losses)
- Total loss evolution
- Learning rate schedules
- Custom metrics
- Solution visualizations

### Visualization Callbacks

Abstract base class `VisualizationCallback` enables custom real-time monitoring:

```python
class MyCallback(VisualizationCallback):
    def __call__(self, model, epoch, writer):
        # Custom visualization logic
        fig = plot_solution(model)
        writer.add_figure('solution', fig, epoch)
```

**Features:**
- Automatic figure logging to TensorBoard
- Configurable callback frequency
- Multiple independent visualizations
- No modification to core training loop

## Training Utilities

### Checkpoint Management

Full state saving and resumption:
```python
# Save checkpoint
model.save_checkpoint('checkpoint.pth', epoch=1000)

# Resume training
model = PINN.load_checkpoint('checkpoint.pth', model=net)
model.train(epochs=2000, lr=1e-4)  # Continue from epoch 1000
```

### Adaptive Loss Weighting

Automatically balance competing loss terms to prevent one constraint from dominating:

```python
pinn = PINN(
    model=net,
    equation_terms=terms,
    use_adaptive_weights=True  # Enable adaptive weighting
)
```

### Learning Rate Scheduling

Support for PyTorch schedulers:
```python
pinn.train(
    epochs=10000,
    lr=1e-3,
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
)
```

## Plasma Physics Applications

AI4Plasma is specifically designed for plasma simulation challenges:

### Arc Discharge Modeling

- Steady-state and transient 1D cylindrical arc
- Temperature-dependent properties (conductivity, heat capacity)
- Ohmic heating, radiation losses, convection
- Automatic boundary condition enforcement

### Cross Section Prediction

- Electron-impact ionization cross sections
- Multi-molecule species support
- Fast lookup for kinetic simulations

### Corona Discharge

- Time-dependent ionization dynamics
- Runge-Kutta time integration
- Photoionization and recombination

### Parametric Studies

- Meta-learning for fast parameter sweeps
- Neural operators for real-time predictions
- Uncertainty quantification

## Typical Workflow

A typical AI4Plasma workflow consists of:

1. **Problem Definition**
   - Define governing PDEs
   - Specify boundary/initial conditions
   - Set up computational domain (geometry)

2. **Network Design**
   - Choose network architecture (FNN, CNN, custom)
   - Set hyperparameters (depth, width, activation)
   - Optionally use NAS-PINN for automatic design

3. **Physics Encoding**
   - Implement residual functions
   - Create equation terms with appropriate weights
   - Set up sampling points (collocation points)

4. **Training**
   - Initialize PINN/operator model
   - Configure optimizer and scheduler
   - Add visualization callbacks
   - Train with TensorBoard monitoring

5. **Validation & Deployment**
   - Compare with reference solutions or experiments
   - Compute error metrics (L2 error, relative error)
   - Export model for inference
   - Use in larger simulation pipelines

## References

1. **PINNs**: M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics*, vol. 378, pp. 686-707, 2019.

2. **DeepONet**: L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators," *Nature Machine Intelligence*, vol. 3, no. 3, pp. 218-229, 2021.
   
3. **CS-PINN**: L. Zhong, B. Wu, and Y. Wang, "Low-temperature plasma simulation based on physics-informed neural networks: Frameworks and preliminary applications," Physics of Fluids, vol. 34, no. 8, p. 087116, 2022.
   
4. **RK-PINN**: L. Zhong, B. Wu, and Y. Wang, "Low-temperature plasma simulation based on physics-informed neural networks: Frameworks and preliminary applications," Physics of Fluids, vol. 34, no. 8, p. 087116, 2022.

5. **Meta-PINN**: L. Zhong, B. Wu, and Y. Wang, "Accelerating physics-informed neural network based 1D arc simulation by meta learning," *Journal of Physics D: Applied Physics*, vol. 56, p. 074006, 2023.
   
6. **NAS-PINN**: Y. Wang and L. Zhong, "NAS-PINN: Neural architecture search-guided physics-informed neural network for solving PDEs," Journal of Computational Physics, vol. 496, p. 112603, 2024.
   
7. **DeepCSNet**: Y. Wang and L. Zhong, "DeepCSNet: a deep learning method for predicting electron-impact doubly differential ionization cross sections," *Plasma Sources Science and Technology*, vol. 33, no. 10, p. 105012, 2024.

## Next Steps

- Explore the [API Reference](../api/index.md) for detailed class and function documentation
- Check out [Examples](../examples/index.md) for practical tutorials
- Read the [Developer Guide](../dev/index.md) to understand the internals
- Try the example scripts in the `app/` directory
