# Operator Learning

This document provides comprehensive examples demonstrating the use of neural operators for learning mappings between infinite-dimensional function spaces. Neural operators are powerful tools for solving parametric PDEs, surrogate modeling, and physics-informed machine learning tasks.

## Overview

**Operator Learning** aims to learn mappings between function spaces rather than finite-dimensional vectors. Given input functions (or fields) and their corresponding output functions, neural operators learn the underlying operator $\mathcal{G}: \mathcal{U} \to \mathcal{V}$ that maps from input function space $\mathcal{U}$ to output function space $\mathcal{V}$.

### Why Operator Learning?

Traditional neural networks learn point-wise mappings. Operator learning provides several advantages:

- **Discretization Invariance**: Train on one mesh, evaluate on another
- **Generalization**: Learn families of solutions parameterized by input functions
- **Efficiency**: Solve parametric PDEs without multiple FEM/FDM simulations
- **Physical Insight**: Capture underlying operator structure

### Available Implementations

AI4Plasma provides two operator learning architectures:

- **DeepONet**: Universal approximator for nonlinear operators
- **DeepCSNet**: Specialized network for electron-impact cross section prediction

## Common Setup

All scripts can be executed from the repository root directory. Most examples share common features:

- **Device Selection**: Automatically choose CPU/GPU via `ai4plasma.config.DEVICE`
- **Reproducibility**: Fix random seeds via `ai4plasma.utils.common.set_seed`
- **Performance Metrics**: Report relative L2 error for validation
- **Timing**: Monitor training and inference time using `ai4plasma.utils.common.Timer`

### Hardware Configuration

```python
from ai4plasma.utils.device import check_gpu
from ai4plasma.config import DEVICE

if check_gpu(print_required=True):
    DEVICE.set_device(0)  # Use first GPU
else:
    DEVICE.set_device(-1)  # Fall back to CPU
```

---

## DeepONet (Deep Operator Network)

### Theory

DeepONet [1] learns operators $\mathcal{G}: u(\cdot) \to \mathcal{G}(u)(\cdot)$ by leveraging the universal approximation theorem for operators. The key insight is to represent the output function as:

$$
\mathcal{G}(u)(y) \approx \sum_{k=1}^{p} b_k(u) \cdot t_k(y) + b_0
$$

where:
- $u$ is the input function (discretized as sensors)
- $y$ is the evaluation location
- $b_k(u)$ are basis functions from the **branch network** (depend on input function)
- $t_k(y)$ are basis functions from the **trunk network** (depend on location)
- $b_0$ is a learnable bias term

### Architecture

```
Input Function u --> [Branch Net] --> b = [b₁, b₂, ..., bₚ]
                                              |
                                              | Inner Product
                                              |
Location y      --> [Trunk Net]  --> t = [t₁, t₂, ..., tₚ]
                                              ↓
                                        G(u)(y) = b·t + b₀
```

**Branch Network** options:
- **FNN**: For 1D functions or feature vectors
- **CNN**: For 2D/3D field inputs (images, spatial distributions)

**Trunk Network**:
- Typically FNN processing spatial/temporal coordinates

### Examples

#### 1. One-Dimensional Poisson Equation

**File**: `app/operator/deeponet/solve_1d_poisson.py`

**Problem**: Learn the solution operator for the 1D Poisson equation family:

$$
-\frac{d^2 u}{dx^2} = f(x) = v \pi^2 \sin(\pi x), \quad x \in [-1, 1]
$$

with analytical solution $u(x) = v \sin(\pi x)$.

**Features**:
- Branch input: scalar parameter $v$ (amplitude)
- Trunk input: spatial coordinate $x$
- Simple FNN networks for both branch and trunk
- Ideal for quick testing and verification

**Network Configuration**:
```python
branch_layers = [1, 10, 10, 10, 10]  # Input: v (1D)
trunk_layers = [1, 10, 10, 10, 10]   # Input: x (1D)
```

**Training Data**:
- 5 training parameters: $v \in \{2, 4, 6, 8, 10\}$
- 40 spatial evaluation points
- Test on unseen $v = 5.5$

**Run**:
```bash
python app/operator/deeponet/solve_1d_poisson.py
```

**Expected Output**:
- Training progress with loss values
- Final L2 relative error: typically < 1e-3
- Total runtime: ~10-30 seconds (depending on hardware)

---

#### 2. One-Dimensional Poisson with Batch Training

**File**: `app/operator/deeponet/solve_1d_poisson_batch.py`

**Purpose**: Demonstrate batch-wise training with DataLoader for larger datasets.

**Key Differences**:
- Uses `batch_size=5` with PyTorch DataLoader
- Training data: 20 parameters uniformly spaced in $[1, 20]$
- Demonstrates scalability to larger datasets

**Features**:
- Efficient memory management for large datasets
- Shuffling and batching capabilities
- Same PDE as `solve_1d_poisson.py` but with more training data

**Run**:
```bash
python app/operator/deeponet/solve_1d_poisson_batch.py
```

---

#### 3. Two-Dimensional Poisson Equation

**File**: `app/operator/deeponet/solve_2d_poisson.py`

**Problem**: Learn the solution operator for 2D Poisson equation:

$$
-\Delta u(x,y) = f(x,y) = 2v\pi^2 \sin(\pi x)\sin(\pi y), \quad (x,y) \in [0,1]^2
$$

with analytical solution $u(x,y) = v \sin(\pi x)\sin(\pi y)$.

**Features**:
- Branch input: scalar parameter $v$
- Trunk input: 2D coordinates $(x, y)$
- Tests generalization to higher-dimensional spaces
- Evaluation on Cartesian grid ($32 \times 32$ points)

**Network Configuration**:
```python
branch_layers = [1, 32, 32, 32]   # Input: v (1D parameter)
trunk_layers = [2, 32, 32, 32]    # Input: (x,y) coordinates
```

**Training Data**:
- 10 training parameters uniformly spaced in $[1, 10]$
- $32 \times 32 = 1024$ spatial points per parameter
- Test on $v = 5.5$

**Run**:
```bash
python app/operator/deeponet/solve_2d_poisson.py
```

**Expected Output**:
- L2 relative error: typically < 1e-2
- Demonstrates operator learning in 2D spatial domains

---

#### 4. Two-Dimensional Poisson with CNN Branch

**File**: `app/operator/deeponet/solve_2d_poisson_cnn.py`

**Purpose**: Use CNN-based branch network for processing 2D field inputs (images).

**Key Innovation**: Instead of a scalar parameter, the branch network processes entire 2D fields:

$$
f(x,y) = 2v\pi^2 \sin(\pi x)\sin(\pi y)
$$

as a $16 \times 16$ grid (image).

**Architecture**:
```python
# CNN Branch Network
conv_layers = [1, 8, 16, 32]      # Channels: 1 → 8 → 16 → 32
fc_layers = [32, 32]               # Flattened features
# Input: (batch, 1, 16, 16)
# Output: (batch, 32)

# FNN Trunk Network
trunk_layers = [2, 32, 32, 32]    # Input: (x,y) coordinates
```

**Features**:
- Automatic CNN branch detection via `network.branch_is_cnn`
- Batch normalization and max pooling
- Kaiming initialization for ReLU activation
- Supports higher-resolution field inputs

**Training Data**:
- 20 RHS fields on $16 \times 16$ grid
- Evaluation on finer $32 \times 32$ grid
- Demonstrates resolution independence

**Benefits of CNN Branch**:
- Handles complex spatial input patterns
- Translation equivariance for physical fields
- Efficient for high-dimensional input functions

**Run**:
```bash
python app/operator/deeponet/solve_2d_poisson_cnn.py
```

**Expected Output**:
- Confirmation of CNN branch detection
- Lower error for complex input patterns
- Runtime: ~1-2 minutes

---

#### 5. Quick Test Driver

**File**: `app/operator/deeponet/solve_1d_poisson_test.py`

**Purpose**: Minimal example for rapid testing and debugging.

**Use Cases**:
- Quick verification of installation
- Testing code modifications
- Minimal computational requirements

---

### DeepONet Usage Guidelines

**When to use DeepONet**:
- Learning solution operators for parametric PDEs
- Surrogate modeling with varying input functions
- Multi-query scenarios (many evaluations with different inputs)

**Branch Network Selection**:
- **FNN**: Scalar/vector parameters, 1D functions
- **CNN**: 2D/3D fields, images, spatial distributions

**Training Tips**:
1. Start with small networks and fewer epochs for prototyping
2. Use batch training for datasets with >100 samples
3. Monitor L2 error on held-out test data
4. Increase network depth/width if underfitting
5. Add regularization if overfitting

**Performance Considerations**:
- Training time scales with: number of trunk points × batch size
- Inference is fast: single forward pass per query
- GPU acceleration recommended for CNN branches

---

## DeepCSNet (Deep Coefficient-Subnet Network)

### Theory

DeepCSNet [2] is a specialized operator network for electron-impact cross section prediction in plasma physics. It employs a modular "coefficient-subnet" architecture that processes different input feature types separately.

### Architecture

DeepCSNet consists of up to three optional sub-networks:

```
Molecular Features  --> [Molecule Net] --> m = [m₁, ..., mₚ]
                                               |
Energy Features     --> [Energy Net]   --> e = [e₁, ..., eₚ]
                                               |
Angles/Coordinates  --> [Trunk Net]    --> t = [t₁, ..., tₚ]
                                               ↓
                                    σ = Combine(m, e, t) + bias
```

**Operation Modes**:

1. **SMC (Single-Molecule Configuration)**:
   - Energy Net + Trunk Net
   - For single molecular species

2. **MMC (Multi-Molecule Configuration)**:
   - Molecule Net + Trunk Net (+ optional Energy Net)
   - For multiple molecular species

### Example: Total Ionization Cross Section Prediction

**File**: `app/operator/deepcsnet/predict_total_ionxsec.py`

**Physical Problem**: Predict total electron-impact ionization cross sections $Q(\text{molecule}, E)$ as a function of molecular composition and incident electron energy.

**Application**: Crucial for plasma modeling, mass spectrometry, and radiation chemistry simulations.

**Data**:
- 88 organic molecules (C, H, O, N, F compounds)
- Cross section measurements at various energies
- Energy range: $E \geq 30$ eV (filtered for reliability)

**Molecular Descriptors** (5 features):
- Number of Carbon atoms (C)
- Number of Hydrogen atoms (H)
- Number of Oxygen atoms (O)
- Number of Nitrogen atoms (N)
- Number of Fluorine atoms (F)

**Network Configuration**:
```python
molecule_layers = [5, 80, 80, 80]   # Molecule Net: C,H,O,N,F → features
trunk_layers = [1, 80, 80, 80]      # Trunk Net: Energy → features
```

**Data Processing Pipeline**:
1. Load CSV files (one per molecule)
2. Parse molecular formulas → extract atom counts
3. Filter energy range ($E \geq 30$ eV)
4. Logarithmic transformation: $\log_{10}(Q)$
5. Normalize to $[0.05, 0.95]$ to prevent saturation
6. Split: 70 molecules (training) + 18 molecules (testing)

**Training Configuration**:
- Optimizer: Adam with learning rate $5 \times 10^{-4}$
- Learning rate schedule: constant for 100k epochs, then $\times 0.5$
- Total epochs: 200,000
- Loss function: Mean Squared Error (MSE)

**Features**:
- TensorBoard logging for real-time monitoring
- Checkpoint saving every 50k epochs
- Resume training capability
- Comprehensive error analysis

**Run**:
```bash
python app/operator/deepcsnet/predict_total_ionxsec.py
```

**Expected Output**:
- Training progress logged to TensorBoard
- Checkpoints saved in `app/operator/deepcsnet/models/`
- Results saved in `app/operator/deepcsnet/results/`
- Final relative L2 error on test set
- Runtime: ~30-60 minutes for 200k epochs (GPU)

**Physical Insights**:
- Learns complex electron-molecule scattering physics
- Captures energy-dependent ionization thresholds
- Generalizes to unseen molecular compositions
- Typical test accuracy: relative error < 10%

---

## Best Practices

### 1. Data Preparation
- **Normalization**: Scale inputs/outputs to $[0, 1]$ or $[-1, 1]$
- **Logarithmic Transform**: Use for quantities spanning orders of magnitude
- **Train/Test Split**: Hold out diverse test cases (e.g., different molecules, parameter ranges)

### 2. Network Design
- **Start Simple**: Begin with shallow networks (3-4 layers)
- **Scale Up Gradually**: Increase depth/width if needed
- **Match Dimensions**: Ensure branch and trunk output same dimension $p$

### 3. Training Strategy
- **Learning Rate**: Start with $10^{-4}$ to $10^{-3}$
- **Learning Rate Decay**: Apply exponential or step decay
- **Early Stopping**: Monitor validation loss
- **Checkpointing**: Save models periodically for long training runs

### 4. Validation
- **Visual Inspection**: Plot predictions vs. ground truth
- **Error Metrics**: Compute L2, L∞, pointwise errors
- **Extrapolation Tests**: Test on parameter values outside training range

### 5. Debugging
- **Overfitting**: Add dropout, reduce network size, or increase data
- **Underfitting**: Increase network capacity or training epochs
- **Numerical Issues**: Check for NaN/Inf, adjust learning rate or normalization

---

## Performance Benchmarks

Typical performance on standard hardware:

| Example | Training Time | GPU Memory | Test L2 Error |
|---------|--------------|------------|---------------|
| 1D Poisson (FNN) | ~15 sec | <500 MB | <10⁻³ |
| 2D Poisson (FNN) | ~30 sec | <1 GB | <10⁻² |
| 2D Poisson (CNN) | ~2 min | ~2 GB | <10⁻² |
| Total Ionization XS | ~45 min | ~3 GB | ~10% |

*Hardware: Single NVIDIA GPU (e.g., RTX 3090, A100)*

---

## Troubleshooting

### Training Not Converging
- Reduce learning rate by 10×
- Check data normalization
- Verify network architecture matches data dimensions

### GPU Out of Memory
- Reduce batch size
- Use smaller networks
- Process data in smaller chunks

### High Test Error
- Increase training data diversity
- Try deeper/wider networks
- Check for data leakage or poor train/test split

---

## References

[1] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators," *Nature Machine Intelligence*, vol. 3, no. 3, pp. 218-229, 2021.

[2] Y. Wang and L. Zhong, "DeepCSNet: a deep learning method for predicting electron-impact doubly differential ionization cross sections," *Plasma Sources Science and Technology*, vol. 33, no. 10, p. 105012, 2024.

---

## Further Reading

- **Operator Learning**: Lu et al., "DeepXDE: A deep learning library for solving differential equations" (2021)
- **Physics-Informed ML**: Karniadakis et al., "Physics-informed machine learning" (2021)
- **Plasma Physics Applications**: Wang et al., "Machine learning methods for plasma physics" (2024)
