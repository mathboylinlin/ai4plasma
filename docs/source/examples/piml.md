# Physics-Informed Machine Learning (PIML)

This document provides comprehensive examples demonstrating Physics-Informed Neural Networks (PINNs) and their variants for solving partial differential equations (PDEs) in plasma physics and computational science.

## Overview

**Physics-Informed Machine Learning** combines the power of neural networks with physical laws encoded as PDEs. Instead of relying solely on data, PINNs incorporate governing equations, boundary conditions, and initial conditions directly into the loss function, enabling:

- **Mesh-free solutions**: No discretization or meshing required
- **Inverse problems**: Simultaneous parameter identification and forward solving
- **Data efficiency**: Physics constraints reduce data requirements
- **Continuous solutions**: Differentiable solutions at arbitrary points
- **Multi-physics coupling**: Natural handling of coupled PDEs

### Why PINNs for Plasma Physics?

Plasma physics involves complex multi-scale phenomena with:
- Highly nonlinear PDEs with temperature-dependent properties
- Multi-physics coupling (thermal, electromagnetic, fluid dynamics)
- Wide range of spatial and temporal scales
- Limited experimental data in extreme conditions

PINNs offer a flexible framework for modeling these challenges without expensive grid-based simulations.

### Available Implementations

AI4Plasma provides several PINN variants optimized for different scenarios:

- **Standard PINN**: Basic implementation for general PDEs
- **CS-PINN**: Coefficient-Subnet PINN for arc discharge with complex material properties
- **RK-PINN**: Runge-Kutta PINN for time-dependent problems with better temporal accuracy
- **Meta-PINN**: Meta-learning framework for fast adaptation across related problems

### Common Features

All scripts in this section share:

- **Device Management**: Automatic CPU/GPU selection via `ai4plasma.config.DEVICE`
- **Reproducibility**: Fixed random seeds via `ai4plasma.utils.common.set_seed`
- **Monitoring**: TensorBoard integration for real-time training visualization
- **Checkpointing**: Automatic model saving for resumption and deployment
- **Visualization**: Comparison plots and animated GIF generation

### Output Organization

```
app/piml/{method}/
├── runs/       # TensorBoard event files for training monitoring
├── models/     # Saved model checkpoints (.pth files)
├── results/    # Output figures, GIFs, and CSV files
└── data/       # Input data (material properties, reference solutions)
```

---

## Standard PINN

### Theory

Physics-Informed Neural Networks [1] solve PDEs by minimizing a composite loss function:

$$
\mathcal{L} = \mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{IC}} \mathcal{L}_{\text{IC}}
$$

where:
- $\mathcal{L}_{\text{PDE}}$: PDE residual at interior collocation points
- $\mathcal{L}_{\text{BC}}$: Boundary condition residual
- $\mathcal{L}_{\text{IC}}$: Initial condition residual (for time-dependent problems)
- $\lambda_{\text{BC}}, \lambda_{\text{IC}}$: Weighting hyperparameters

**Key Advantage**: Uses automatic differentiation to compute derivatives required for PDE evaluation—no finite differences needed.

### Examples

#### 1. One-Dimensional Poisson Equation

**File**: `app/piml/pinn/solve_1d_pinn.py`

**Problem**: Solve the second-order ODE with Dirichlet boundary conditions:

$$
\frac{d^2 u}{dx^2} = -\sin(x), \quad x \in [0, 1]
$$

**Boundary Conditions**:
- $u(0) = 0$ (left boundary)
- $u(1) = 0$ (right boundary)

**Analytical Solution**:
$$
u(x) = \sin(x) - x\sin(1)
$$

**Network Architecture**:
```python
layers = [1, 50, 50, 50, 50, 1]  # Input: x → Output: u(x)
network = FNN(layers, act_fun=nn.Tanh())
```

**Loss Function Weights**:
- PDE weight: $w_{\text{PDE}} = 1.0$
- Boundary weight: $w_{\text{BC}} = 10.0$ (higher to enforce BCs strictly)

**Features**:
- Custom visualization callback with solution evolution
- Animated training process (saved as GIF)
- Memory-efficient history recording
- Comparison with analytical solution

**Training Configuration**:
- Epochs: 20,000
- Learning rate: 0.001 (Adam optimizer)
- Learning rate decay: 0.5× at epochs 10,000 and 15,000
- Collocation points: 100 (interior), 2 (boundaries)

**Run**:
```bash
python app/piml/pinn/solve_1d_pinn.py
```

**Expected Output**:
- Final relative error: < 0.1%
- Training time: ~1-2 minutes
- Output files:
  - `results/1d_pinn_final.png`: Final solution comparison
  - `results/1d_pinn_animation.gif`: Training evolution
  - `runs/1d_pinn/`: TensorBoard logs

---

#### 2. Two-Dimensional Poisson Equation (Rectangular Domain)

**File**: `app/piml/pinn/solve_2d_rect_pinn.py`

**Problem**: Solve 2D Poisson equation on rectangular domain:

$$
\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = -8\pi^2\sin(2\pi x)\sin(2\pi y)
$$

**Domain**: $\Omega = [0, 1] \times [0, 1]$

**Boundary Conditions**: $u = 0$ on all boundaries $\partial\Omega$

**Analytical Solution**:

$$
u(x, y) = \sin(2\pi x)\sin(2\pi y)
$$

**Network Architecture**:
```python
layers = [2, 50, 50, 50, 50, 1]  # Input: (x,y) → Output: u(x,y)
network = FNN(layers, act_fun=nn.Tanh())
```

**Geometry**: Uses `GeoRect2D` class for rectangular domain sampling

**Sampling**:
- Interior points: $50 \times 50 = 2500$ collocation points
- Boundary points: $50$ per edge × 4 edges = 200 points

**Features**:
- 2D heatmap and contour plot visualization
- Real-time monitoring of solution evolution
- Automatic domain sampling with `GeoRect2D`
- Colorbar and axis labels for publication-ready figures

**Training Configuration**:
- Epochs: 30,000
- Learning rate: 0.001 with exponential decay
- Batch processing: All collocation points processed simultaneously

**Run**:
```bash
python app/piml/pinn/solve_2d_rect_pinn.py
```

**Expected Output**:
- Final relative error: < 1%
- Training time: ~5-10 minutes
- 2D solution visualization with contours

---

#### 3. Two-Dimensional Poisson Equation (Polygonal Domain)

**File**: `app/piml/pinn/solve_2d_poly_pinn.py`

**Problem**: Same 2D Poisson equation as above, but on a polygonal domain.

**Key Difference**: Uses `GeoPoly2D` for arbitrary polygon shapes

**Geometry Definition**:
```python
# Define polygon vertices (unit square as example)
points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
geo = GeoPoly2D(points)
```

**Advantages**:
- Handle complex geometries (L-shaped, triangular, arbitrary polygons)
- Automatic boundary detection and sampling
- No need for structured grids

**Use Cases**:
- Irregular computational domains
- Complex electrode geometries in plasma devices
- Multi-region coupling problems

**Run**:
```bash
python app/piml/pinn/solve_2d_poly_pinn.py
```

---

## CS-PINN (Coefficient-Subnet PINN)

### Theory

CS-PINN extends standard PINNs for problems with **complex nonlinear coefficients** (e.g., temperature-dependent material properties). It employs:

1. **Coefficient Subnets**: Separate neural networks or interpolators for material properties
2. **Automatic Boundary Enforcement**: Network architecture guarantees boundary conditions by construction
3. **Spline Interpolation**: Cubic splines for temperature-dependent plasma properties

**Architecture for Arc Discharge**:

```
Input: r (radius)  →  Neural Network  →  Temperature T(r)
                                            ↓
                            Material Properties: κ(T), σ(T), ε(T)
                                            ↓
                            PDE Residual: ∇·(κ∇T) - σE² + 4πε
```

**Boundary Condition Enforcement**:

Instead of using loss terms, CS-PINN constructs the network output as:

$$
T(r) = (r - R) \cdot N(r) + T_b
$$

where $N(r)$ is the neural network output. This **guarantees** $T(R) = T_b$ regardless of network parameters.

### Physical Model: Arc Discharge

Arc discharge is a high-temperature plasma phenomenon occurring in electrical devices (circuit breakers, welding arcs, lightning). The **energy balance equation** governs temperature distribution:

$$
\frac{1}{r}\frac{d}{dr}\left(r \kappa(T) \frac{dT}{dr}\right) = \sigma(T) E^2 - 4\pi \varepsilon_{\text{nec}}(T)
$$

**Physical Terms**:
- $\kappa(T)$: Thermal conductivity [W/(m·K)] — heat conduction
- $\sigma(T)$: Electrical conductivity [S/m] — Joule heating
- $E$: Electric field [V/m] — determined by arc current $I$
- $\varepsilon_{\text{nec}}(T)$: Net emission coefficient [W/m³] — radiation loss

**Material Property Interpolation**:
- Data from NIST, NASA, or experimental measurements
- Cubic spline interpolation in log-log space
- Automatic clamping to physical range [300 K, 30,000 K]

### Examples

#### 1. Steady-State Arc Discharge

**File**: `app/piml/cs_pinn/solve_1d_arc_steady_cs_pinn.py`

**Physical Problem**: Solve steady-state energy balance for SF₆ arc plasma

**Gas**: SF₆ (Sulfur hexafluoride) — widely used in power systems

**Physical Parameters**:
- Arc radius: $R = 10$ mm
- Arc current: $I = 200$ A
- Boundary temperature: $T_b = 2000$ K
- Temperature normalization: $T_{\text{red}} = 10,000$ K

**Network Architecture**:
```python
layers = [1, 50, 50, 50, 50, 50, 50, 1]  # 6 hidden layers, 50 neurons each
network = FNN(layers, act_fun=nn.Tanh())
```

**Training Configuration**:
- Epochs: 100,000
- Learning rate: 0.001
- Collocation points: 500 (uniformly sampled)
- Evaluation points: 600

**Material Properties**:
- Thermodynamic properties: `sf6_p1.dat` (κ, σ, ρ, Cp vs T)
- Radiation properties: `sf6_p1_nec.dat` (net emission coefficient)

**Numerical Methods**:
- Gauss-Legendre quadrature (100 points) for arc conductance integral
- Automatic differentiation for spatial derivatives
- Adaptive learning rate scheduling (decay at milestones)

**Features**:
- Real-time visualization with reference data comparison
- Material property evolution plots (κ, σ, ε vs radius)
- Automatic checkpoint saving every 10,000 epochs
- TensorBoard logging for comprehensive monitoring

**Run**:
```bash
python app/piml/cs_pinn/solve_1d_arc_steady_cs_pinn.py
```

**Expected Output**:
- Peak temperature: ~22,000-25,000 K at arc center
- Training time: ~1-2 hours (100k epochs, GPU)
- Relative error vs FVM: < 2%
- Output files:
  - `results/sta/arc_temperature.png`: Temperature profile
  - `results/sta/arc_properties.png`: Material properties
  - `results/sta/training_history.gif`: Evolution animation
  - `models/sta/model_final.pth`: Trained model

**Physical Insights**:
- Highest temperature at arc center (r = 0)
- Sharp temperature gradient near boundary
- Joule heating dominates in core, radiation in periphery
- Electric field determined by conductance integral

---

#### 2. Transient Arc Discharge

**File**: `app/piml/cs_pinn/solve_1d_arc_transient_cs_pinn.py`

**Physical Problem**: Solve time-dependent energy balance with radial flow

**Governing Equations**:

Energy equation:

$$
\rho C_p \frac{\partial T}{\partial t} + \rho C_p v \frac{\partial T}{\partial r} = \frac{1}{r}\frac{\partial}{\partial r}\left(r\kappa\frac{\partial T}{\partial r}\right) + \sigma E^2 - 4\pi\varepsilon
$$

Continuity equation (mass conservation):

$$
\frac{\partial (\rho r)}{\partial t} + \frac{\partial (\rho v r)}{\partial r} = 0
$$

**Additional Parameters**:
- Time normalization: $t_{\text{red}} = 1$ ms
- Simulation duration: 10 ms (normalized: 10)
- Initial condition: Steady-state solution from previous example

**Network Architecture**:
```python
layers = [2, 300, 300, 300, 300, 300, 300, 2]  # Input: (r,t) → Output: (T,v)
network = FNN(layers, act_fun=nn.Tanh())
```

**Key Features**:
- Coupled temperature and velocity fields
- Automatic initial condition enforcement
- Multi-time snapshot visualization
- Higher network capacity (300 neurons/layer) for complexity

**Training Configuration**:
- Epochs: 150,000
- Learning rate: 0.0001 (lower for stability)
- Spatial points: 200
- Temporal points: 100
- Evaluation times: t = 0.1, 0.5, 0.9 (normalized)

**Run**:
```bash
python app/piml/cs_pinn/solve_1d_arc_transient_cs_pinn.py
```

**Expected Output**:
- Time-dependent temperature evolution
- Radial velocity field development
- Training time: ~3-5 hours (150k epochs, GPU)
- Multi-panel plots showing different time snapshots

---

#### 3. Transient Arc Without Velocity

**File**: `app/piml/cs_pinn/solve_1d_arc_transient_noV_cs_pinn.py`

**Simplification**: Energy equation only (no radial flow)

**Governing Equation**:

$$
\rho C_p \frac{\partial T}{\partial t} = \frac{1}{r}\frac{\partial}{\partial r}\left(r\kappa\frac{\partial T}{\partial r}\right) + \sigma E^2 - 4\pi\varepsilon
$$

**Use Case**: Early-stage arc development where flow is negligible

**Advantages**:
- Faster training (single output variable)
- Lower network capacity required
- Clearer physical interpretation

**Run**:
```bash
python app/piml/cs_pinn/solve_1d_arc_transient_noV_cs_pinn.py
```

---

#### 4. Resume Training

**File**: `app/piml/cs_pinn/resume_1d_arc_transient_cs_pinn.py`

**Purpose**: Continue training from saved checkpoint

**Features**:
- Load model state and optimizer state
- Resume from specific epoch
- Useful for long training runs or hardware interruptions

**Usage**:
```python
# Specify checkpoint file
checkpoint_file = 'app/piml/cs_pinn/models/tra/model_epoch_50000.pth'

# Load and continue training
arc_model.load_checkpoint(checkpoint_file)
arc_model.train(num_epochs=100000, resume=True)
```

**Run**:
```bash
python app/piml/cs_pinn/resume_1d_arc_transient_cs_pinn.py
```

---

## RK-PINN (Runge-Kutta PINN)

### Theory

RK-PINN improves temporal accuracy for time-dependent PDEs by using **implicit Runge-Kutta schemes**. Instead of treating time as just another input coordinate, RK-PINN discretizes time explicitly while keeping spatial derivatives continuous.

**Key Idea**: At each time step $t_n$, the solution $u^{n+1}$ satisfies:

$$
u^{n+1} = u^n + \Delta t \sum_{i=1}^{q} b_i k_i
$$

where $k_i$ are stage derivatives computed from the PDE. The network outputs **all stages simultaneously**, enabling efficient implicit time stepping.

**Advantages over Standard PINN**:
- Higher temporal accuracy (order q)
- Better stability for stiff problems
- Explicit time discretization provides clearer physical interpretation

### Example: Corona Discharge

**File**: `app/piml/rk_pinn/solve_1d_corona_rk_pinn.py`

**Physical Problem**: Simulate time-dependent corona discharge in Argon gas

**Corona Discharge**: Non-equilibrium plasma near sharp electrodes (e.g., power lines, corona treaters)

**Governing Equations**:

Electron continuity:

$$
\frac{\partial n_e}{\partial t} = \nabla \cdot (D_e \nabla n_e - \mu_e n_e \mathbf{E}) + \alpha |\mu_e n_e \mathbf{E}| - \beta n_e n_p
$$

Ion continuity:

$$
\frac{\partial n_p}{\partial t} = \nabla \cdot (D_p \nabla n_p + \mu_p n_p \mathbf{E}) + \alpha |\mu_e n_e \mathbf{E}| - \beta n_e n_p
$$

Poisson equation (electric field):

$$
\nabla^2 \Phi = -\frac{e}{\epsilon_0}(n_p - n_e)
$$

**Physical Variables**:
- $n_e$: Electron density [m⁻³]
- $n_p$: Positive ion density [m⁻³]
- $\Phi$: Electric potential [V]
- $\mathbf{E} = -\nabla \Phi$: Electric field [V/m]

**Transport Coefficients** (field-dependent):
- $D_e, D_p$: Diffusion coefficients [m²/s]
- $\mu_e, \mu_p$: Mobilities [m²/(V·s)]
- $\alpha$: Ionization coefficient [m⁻¹]

**Physical Parameters**:
- Gas: Argon (Ar)
- Radius: $R = 10$ mm
- Temperature: $T = 600$ K
- Pressure: $P = 1$ atm
- Applied voltage: $V_0 = -10$ kV
- Secondary emission coefficient: $\gamma = 0.066$

**Network Architecture**:
```python
q = 300  # Number of RK stages (determines temporal accuracy)
layers = [1, 300, 300, 300, 300, 2*(q+1)]  # Output: (Φ, n_e) at all stages
network = FNN(layers, act_fun=nn.Tanh())
```

**Normalization**:
- Density: $N_{\text{red}} = 10^{15}$ m⁻³
- Time: $t_{\text{red}} = 5$ ns
- Voltage: $V_{\text{red}} = 10$ kV

**Training Configuration**:
- Epochs: 100,000
- Learning rate: 0.0001
- Collocation points: 500
- Time step: $\Delta t = 1.0$ (normalized)

**Features**:
- Implicit RK integration for stiffness
- Multi-stage output prediction
- Field-dependent transport coefficient interpolation
- Initial condition from reference solution

**Run**:
```bash
python app/piml/rk_pinn/solve_1d_corona_rk_pinn.py
```

**Expected Output**:
- Electron avalanche development
- Space charge field evolution
- Training time: ~2-4 hours (GPU)
- Output files:
  - `results/corona/electron_density.png`
  - `results/corona/potential_distribution.png`
  - `results/corona/electric_field.png`

**Physical Insights**:
- Ionization wave propagation from cathode
- Space charge effects on electric field distortion
- Avalanche-to-streamer transition at critical density

---

## Meta-PINN

### Theory

Meta-PINN applies **meta-learning** (learning-to-learn) to PINNs for rapid adaptation to new related tasks. Instead of training from scratch for each parameter configuration, Meta-PINN pre-trains on multiple tasks and fine-tunes quickly on new ones.

**MAML Algorithm** (Model-Agnostic Meta-Learning):

1. **Meta-Training**: Learn initialization parameters $\theta^*$ that work well across tasks
2. **Meta-Testing**: Fine-tune from $\theta^*$ with few gradient steps on new task

**Benefits**:
- Fast adaptation to new parameter values (e.g., different gas mixtures)
- Reduced training time for parametric studies
- Better generalization across related physics problems

### Example: Multi-Gas Arc Discharge

**File**: `app/piml/meta_pinn/solve_1d_arc_steady_meta_pinn.py`

**Problem**: Learn to solve steady arc discharge for various SF₆-N₂ gas mixtures

**Gas Mixtures**: SF₆:N₂ with varying nitrogen fractions (0%, 10%, 20%, ..., 50%)

**Why Meta-Learning?**:
- Each mixture has different material properties
- Traditional approach: Train separate model for each mixture (~100k epochs each)
- Meta-learning approach: Pre-train once, fine-tune for new mixture (~5k epochs)

**Training Tasks**: 5 SF₆-N₂ mixtures (10%, 20%, 30%, 40%, 50% N₂)

**Test Tasks**: 6 interpolated/extrapolated mixtures (5%, 15%, 25%, 35%, 45%, 55% N₂)

**Meta-Training Configuration**:
- Outer loop (meta-epochs): 1,000
- Inner loop (task adaptation): 5 gradient steps
- Support points: 500 per task
- Query points: 600 per task
- Meta learning rate: 0.001
- Task learning rate: 0.01

**Network Architecture**:
```python
layers = [1, 50, 50, 50, 50, 50, 50, 1]
backbone_net = FNN(layers, act_fun=nn.Tanh())
```

**Features**:
- Multi-task training with task sampling
- Support/query set splitting for meta-learning
- Rapid adaptation demonstration
- Property table management for different mixtures

**Run**:
```bash
python app/piml/meta_pinn/solve_1d_arc_steady_meta_pinn.py
```

**Expected Output**:
- Meta-trained model: `models/meta/meta_model.pth`
- Fine-tuned models for test tasks
- Comparison plots: meta-learning vs. training from scratch
- Adaptation curves showing fast convergence
- Training time: ~2-3 hours (meta-training)
- Fine-tuning time: ~5 minutes per new task

**Applications**:
- Parametric design studies (varying pressure, current, gas composition)
- Uncertainty quantification with parameter variations
- Real-time simulation with adaptive property updates

---

## Best Practices

### 1. Network Architecture Selection

**Depth vs. Width**:
- **Shallow & Wide** (e.g., [1, 100, 100, 1]): Fast training, good for smooth solutions
- **Deep & Narrow** (e.g., [1, 30, 30, 30, 30, 30, 1]): Better at learning complex patterns

**Activation Functions**:
- **Tanh**: Smooth gradients, traditional choice for PINNs
- **Sin**: Excellent for periodic solutions
- **SoftPlus**: Avoids vanishing gradients, good for deep networks

**Guidelines**:
- Start with 3-5 hidden layers, 50-100 neurons per layer
- Increase depth for complex physics (multi-scale, sharp gradients)
- Use wider networks for higher-dimensional problems

### 2. Loss Function Weighting

**Balancing Terms**:

$$
\mathcal{L} = w_{\text{PDE}} \mathcal{L}_{\text{PDE}} + w_{\text{BC}} \mathcal{L}_{\text{BC}} + w_{\text{IC}} \mathcal{L}_{\text{IC}}
$$

**Typical Values**:
- $w_{\text{PDE}} = 1.0$ (reference)
- $w_{\text{BC}} = 10.0$ - $100.0$ (enforce strictly)
- $w_{\text{IC}} = 10.0$ - $100.0$ (important for transient problems)

**Adaptive Weighting**:
- Monitor individual loss components
- Increase weight if corresponding term doesn't decrease
- Use gradient-based balancing (e.g., GradNorm)

### 3. Collocation Point Sampling

**Strategies**:
- **Uniform**: Simple, works for regular domains
- **Latin Hypercube Sampling (LHS)**: Better space-filling properties
- **Adaptive**: Increase density where PDE residual is high

**Number of Points**:
- 1D: 100-1000 points
- 2D: 1000-10000 points
- 3D: 10000-100000 points

**Rule of Thumb**: More points → better accuracy, but slower training

### 4. Training Strategy

**Learning Rate**:
- Initial: 0.001 - 0.01 (Adam optimizer)
- Decay schedule: Exponential or step decay
- Fine-tuning: Reduce to 0.0001 towards convergence

**Epochs**:
- Simple problems: 10,000 - 30,000
- Complex multi-physics: 100,000 - 500,000
- Monitor validation loss to avoid overfitting

**Optimization Tips**:
- Use learning rate warmup for first few hundred epochs
- Apply gradient clipping (e.g., max_norm = 1.0) for stability
- Check for loss stagnation and adjust learning rate

### 5. Validation and Verification

**Verification Methods**:
1. **Analytical Solutions**: Compare with known solutions for test cases
2. **Reference Data**: Compare with FEM/FVM/experimental results
3. **Grid Convergence**: Evaluate on increasingly fine evaluation grids
4. **Conservation Laws**: Check mass, energy, momentum conservation

**Error Metrics**:
- **Relative L2 Error**: $\frac{\|u_{\text{pred}} - u_{\text{true}}\|_2}{\|u_{\text{true}}\|_2}$
- **Pointwise Max Error**: $\max_i |u_{\text{pred}}(x_i) - u_{\text{true}}(x_i)|$
- **PDE Residual Norm**: $\|\mathcal{F}[u_{\text{pred}}]\|_2$

**Target Accuracy**:
- Simple problems: < 0.1% relative error
- Complex multi-physics: < 5% relative error
- Engineering applications: < 10% typically acceptable

### 6. Debugging Common Issues

**Loss Not Decreasing**:
- Check loss function implementation
- Verify PDE residual computation (use finite differences to validate autodiff)
- Reduce learning rate
- Increase network capacity
- Check input/output normalization

**Loss Plateaus Early**:
- Increase boundary condition weights
- Add more collocation points
- Use adaptive sampling
- Try different activation functions
- Implement learning rate warmup

**Unphysical Solutions**:
- Verify boundary conditions are enforced correctly
- Check physical parameter ranges (e.g., temperature clipping)
- Increase boundary/initial condition weights
- Use exact boundary enforcement (like CS-PINN)

**Training Instability (NaN/Inf)**:
- Apply gradient clipping
- Reduce learning rate
- Check for division by zero in PDE terms
- Normalize inputs/outputs to [0, 1] or [-1, 1]
- Use float64 precision instead of float32

### 7. Computational Efficiency

**GPU Acceleration**:
- Essential for large networks and many collocation points
- Speedup: 10-100× compared to CPU
- Recommended for problems with >10k collocation points

**Memory Management**:
- Batch collocation points if GPU memory is limited
- Use gradient checkpointing for very deep networks
- Delete intermediate tensors explicitly

**Parallelization**:
- Train multiple parameter configurations in parallel
- Use data parallelism for large-scale sampling

---

## Performance Benchmarks

Typical performance on standard hardware (single NVIDIA GPU):

| Example | Network Size | Collocation Points | Epochs | Training Time | Final Error |
|---------|-------------|-------------------|--------|---------------|-------------|
| 1D Poisson | [1,50,50,50,1] | 100 | 20k | ~2 min | <0.1% |
| 2D Poisson | [2,50,50,50,1] | 2500 | 30k | ~10 min | <1% |
| Steady Arc | [1,50×6,1] | 500 | 100k | ~2 hours | <2% |
| Transient Arc | [2,300×6,2] | 20k | 150k | ~5 hours | <5% |
| Corona RK | [1,300×4,600] | 500 | 100k | ~4 hours | <10% |
| Meta-PINN | [1,50×6,1] | 500×5 tasks | 1k meta | ~3 hours | <5% |

*Hardware: NVIDIA RTX 3090 or A100 GPU with PyTorch*

---

## Visualization and Monitoring

### TensorBoard Integration

All examples support real-time monitoring via TensorBoard:

```bash
# Start TensorBoard server
tensorboard --logdir=app/piml

# Open browser to http://localhost:6006
```

**Logged Metrics**:
- Total loss and individual components (PDE, BC, IC)
- Learning rate schedule
- Parameter histograms
- Gradient norms
- Custom metrics (temperature extrema, conservation errors)

### Output Files

**Checkpoints** (`.pth` files):
- Contain model state, optimizer state, epoch number
- Enable training resumption
- Can be loaded for inference without retraining

**Figures** (`.png`, `.pdf`):
- Final solution comparisons
- Error distributions
- Material property plots

**Animations** (`.gif`):
- Solution evolution during training
- Multi-time snapshot sequences
- Useful for presentations and debugging

**CSV Files**:
- Numerical solution values
- Error metrics over training
- Reference data for comparison

---

## Advanced Topics

### Hard vs. Soft Boundary Conditions

**Soft BCs** (standard PINN): Include BC residual in loss
- Pros: Simple implementation
- Cons: BCs may not be satisfied exactly

**Hard BCs** (CS-PINN approach): Construct network to satisfy BCs exactly
- Pros: Guaranteed satisfaction, faster convergence
- Cons: Requires special network architecture

### Transfer Learning

**Scenario**: Use trained model as initialization for related problem

**Examples**:
- Different parameter values (current 200A → 300A)
- Similar geometries (radius 10mm → 15mm)
- Related physics (steady → transient initialization)

**Approach**:
1. Load pre-trained model weights
2. Fine-tune on new problem (typically 10-20% of original training)
3. Significant time savings

### Inverse Problems

PINNs naturally handle inverse problems: infer unknown parameters from data

**Example**: Infer electrical conductivity $\sigma(T)$ from temperature measurements

$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{PDE}}}_{\text{physics}} + \underbrace{\lambda_{\text{data}} \sum_i (T_{\text{pred}}(x_i) - T_{\text{obs}}(x_i))^2}_{\text{data fidelity}}
$$

**Applications**:
- Material property identification
- Boundary condition estimation
- Source term reconstruction

---

## Troubleshooting Guide

| Problem | Possible Causes | Solutions |
|---------|----------------|-----------|
| Loss not decreasing | Poor initialization, wrong gradients | Check autodiff, reduce LR, verify PDE |
| Slow convergence | Insufficient capacity, poor sampling | Increase network size, use LHS sampling |
| Unphysical oscillations | Under-constrained BCs | Increase BC weight, use hard BCs |
| NaN/Inf values | Numerical instability | Gradient clipping, normalize data, reduce LR |
| High memory usage | Too many collocation points | Batch processing, smaller network |
| GPU out of memory | Large network or batch | Reduce network size or collocation points |

---

## Further Reading

- **PINN Theory**: Karniadakis et al., "Physics-informed machine learning," *Nature Reviews Physics*, 2021
- **Plasma Physics**: Fridman and Kennedy, "Plasma Physics and Engineering," 2nd ed., 2011
- **Neural Networks**: Goodfellow et al., "Deep Learning," MIT Press, 2016
- **Meta-Learning**: Hospedales et al., "Meta-learning in neural networks: A survey," *IEEE TPAMI*, 2021

---

## Quick Start Guide

**For Beginners**: Start with these examples in order:

1. `pinn/solve_1d_pinn.py` — Learn basic PINN concepts
2. `pinn/solve_2d_rect_pinn.py` — Extend to 2D problems
3. `cs_pinn/solve_1d_arc_steady_cs_pinn.py` — Apply to realistic physics

**For Plasma Physicists**: Jump directly to:

- Arc discharge: `cs_pinn/solve_1d_arc_steady_cs_pinn.py`
- Corona discharge: `rk_pinn/solve_1d_corona_rk_pinn.py`
- Multi-gas studies: `meta_pinn/solve_1d_arc_steady_meta_pinn.py`

**For Method Developers**: Explore advanced features:

- Custom geometries: `pinn/solve_2d_poly_pinn.py`
- Time integration: `rk_pinn/solve_1d_corona_rk_pinn.py`
- Meta-learning: `meta_pinn/solve_1d_arc_steady_meta_pinn.py`
