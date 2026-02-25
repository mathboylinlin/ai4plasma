###
## Author: Linlin Zhong
## Email: linlin@seu.edu.cn
## Date: June 1, 2017
## Updated: Jan 4, 2026
## Description: Test script for 1D transient arc plasma model with radial velocity (explicit method)
##              Demonstrates the use of TraArc1D class with explicit time integration
##              to simulate time-dependent arc behavior including convective heat transport.
##              This method is RECOMMENDED over implicit methods for better stability.
###

import sys
sys.path.append('.')

import os
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.io import savemat
import matplotlib.pyplot as plt
from ai4plasma.plasma.prop import read_thermo_data, read_nec_data, interp_nec_log
from ai4plasma.plasma.arc import TraArc1D

# ============================================================================
# Configuration Section
# ============================================================================

# Gas selection
# Options: 'SF6', 'Air', 'CO2', etc. (depending on available data files)
gas = 'SF6'

# Arc parameters
I = 0          # Arc current in Amperes [A] (set to 0 for pure decay simulation)
R = 10e-3      # Arc radius in meters [m] (10 mm)

# Time integration parameters
dt = 1e-9              # Time step size in seconds [s] (1 nanosecond - small for stability)
step_num = 100000      # Number of time steps (total time = dt * step_num = 100 μs)
save_freq = 100        # Save solution every N time steps (reduce memory usage)

# Numerical parameters
mesh_num = 500              # Number of spatial mesh cells
Tb = 2000                   # Boundary temperature at r=R [K]

# Simulation mode
enable_joule = False        # Include Joule heating? (False = pure decay mode)

# Output control
is_print = True             # Print time step progress?
flag = gas                  # Output identifier

# ============================================================================
# Data Loading Section
# ============================================================================

print("=" * 70)
print(f"1D Transient Arc Model WITH Radial Velocity (Explicit Method) - {gas} Plasma")
print("=" * 70)

# Construct data file paths
thermo_file = f'./app/plasma/arc/data/{gas.lower()}_p1.dat'
nec_file = f'./app/plasma/arc/data/{gas.lower()}_p1_nec.dat'

# Check if data files exist
if not os.path.exists(thermo_file):
    raise FileNotFoundError(f"Thermodynamic data file not found: {thermo_file}")
if not os.path.exists(nec_file):
    raise FileNotFoundError(f"NEC data file not found: {nec_file}")

print(f"\nLoading plasma property data for {gas}...")

# Read thermodynamic and transport properties
# For transient simulations with velocity, we need all properties including density and Cp
temp_list, rho_list, h_list, Cp_list, sigma_list, kappa_list = read_thermo_data(thermo_file)
print(f"  - Thermodynamic data loaded: {len(temp_list)} temperature points")
print(f"  - Temperature range: {temp_list[0]:.0f} - {temp_list[-1]:.0f} K")

# Read Net Emission Coefficient (NEC) data for radiation modeling
nec_temp_list, nec_R_list, nec_array = read_nec_data(nec_file)
print(f"  - NEC data loaded: {len(nec_temp_list)} temperatures × {len(nec_R_list)} radii")

# ============================================================================
# Property Interpolation Section
# ============================================================================

print(f"\nInterpolating NEC for arc radius R = {R*1e3:.1f} mm...")

# Interpolate NEC values at the specified arc radius
# Note: using temp_list (not nec_temp_list) to match other property arrays
nec_list = interp_nec_log(nec_temp_list, nec_R_list, nec_array, R, temp_list.copy())
print(f"  - NEC interpolated for {len(nec_list)} temperature points")

# Assemble property tuple for arc model
# Format: (temperature, density, Cp, electrical conductivity, thermal conductivity, NEC)
prop = (temp_list, rho_list, Cp_list, sigma_list, kappa_list, nec_list)

# ============================================================================
# Initial Condition Section
# ============================================================================

print(f"\nLoading initial temperature profile...")

# Initial condition: Load from steady-state solution or define analytically
initial_file = f'./app/plasma/arc/results/sta_arc1d_{gas}.csv'

if os.path.exists(initial_file):
    # Load from previously computed steady-state solution
    print(f"  - Reading from file: {initial_file}")
    dat = pd.read_csv(initial_file, sep=',')
    x_list = dat['R(m)'].values
    T_list = dat['T(K)'].values
    
    # Create interpolation function for initial temperature distribution
    # Use cubic spline for smooth initial profile
    Tfunc_init = interpolate.interp1d(x_list, T_list, kind='cubic', 
                                     fill_value='extrapolate')
    
    print(f"  - Initial profile loaded: T_max = {T_list.max():.2f} K")
else:
    # Define analytical initial profile if steady-state file not available
    print(f"  - Warning: Steady-state file not found. Using analytical profile.")
    T_center = 15000  # Center temperature [K] (higher for velocity case)
    T_boundary = Tb   # Boundary temperature [K]
    # Parabolic profile often more realistic for hot arcs
    Tfunc_init = lambda x: (1 - (x/R)**2) * (T_center - T_boundary) + T_boundary
    print(f"  - Using parabolic profile: T_center = {T_center} K")

# ============================================================================
# Arc Model Setup Section
# ============================================================================

print(f"\nInitializing transient arc model with radial velocity...")
print(f"  - Arc current: {I} A {'(Decay mode - no Joule heating)' if I == 0 else ''}")
print(f"  - Arc radius: {R*1e3:.1f} mm")
print(f"  - Mesh cells: {mesh_num}")
print(f"  - Boundary temperature: {Tb} K")
print(f"  - Solution method: EXPLICIT (recommended for stability)")

# Create transient arc model instance with velocity
myarc = TraArc1D(I, R, prop)

# ============================================================================
# Time Integration Section
# ============================================================================

print(f"\nTime integration parameters:")
print(f"  - Time step: {dt:.2e} s")
print(f"  - Number of steps: {step_num}")
print(f"  - Total simulation time: {dt*step_num:.2e} s ({dt*step_num*1e6:.2f} μs)")
print(f"  - Save frequency: every {save_freq} step(s)")
print(f"  - Saved snapshots: {step_num//save_freq + 1}")
print(f"  - Joule heating: {'Enabled' if enable_joule else 'Disabled'}")

# Calculate approximate CFL numbers for information
dx = R / mesh_num
print(f"\nMesh and stability information:")
print(f"  - Cell size: {dx*1e6:.2f} μm")
print(f"  - Estimated diffusion CFL: dt*kappa/(rho*Cp*dx²) ~ {dt*1.0/(1.0*1000*dx**2):.3f}")
print(f"    (should be < 0.5 for stability)")
print("-" * 70)

# Solve the transient arc equation using explicit method
# This method is more stable and faster than the implicit method
t, g, x, T, V = myarc.solve_explicit(
    mesh_num, 
    Tfunc_init, 
    Tb,
    dt, 
    step_num,
    enable_joule=enable_joule,
    save_freq=save_freq, 
    is_print=is_print, 
    flag=flag
)

print("-" * 70)

# ============================================================================
# Results Analysis Section
# ============================================================================

print(f"\nTime integration completed successfully!")
print(f"\nSolution statistics:")
print(f"  - Number of saved time steps: {len(t)}")
print(f"  - Time range: {t[0]:.2e} - {t[-1]:.2e} s ({t[-1]*1e6:.2f} μs)")
print(f"  - Initial center temperature: {T[0, 0]:.2f} K")
print(f"  - Final center temperature: {T[-1, 0]:.2f} K")
print(f"  - Temperature decay: {T[0, 0] - T[-1, 0]:.2f} K ({(1-T[-1,0]/T[0,0])*100:.1f}%)")

# Analyze velocity field
print(f"\nVelocity field statistics:")
print(f"  - Initial maximum velocity: {np.abs(V[0, :]).max():.4f} m/s")
print(f"  - Final maximum velocity: {np.abs(V[-1, :]).max():.4f} m/s")
print(f"  - Peak velocity magnitude: {np.abs(V).max():.4f} m/s")

# Analyze conductance evolution
print(f"\nArc conductance evolution:")
print(f"  - Initial conductance: {g[0]:.4f} S")
print(f"  - Final conductance: {g[-1]:.4f} S")
print(f"  - Conductance ratio: {g[-1]/g[0]:.4f}")

# Estimate decay time constant (if applicable)
if not enable_joule and len(g) > 10:
    # Fit exponential decay: g(t) ≈ g0 * exp(-t/tau)
    # Using log-linear fit: ln(g) ≈ ln(g0) - t/tau
    ln_g = np.log(g[g > g[0]*0.01])  # Avoid very small values
    t_valid = t[:len(ln_g)]
    if len(ln_g) > 5:
        coeffs = np.polyfit(t_valid, ln_g, 1)
        tau = -1.0 / coeffs[0] if coeffs[0] < 0 else np.inf
        print(f"  - Estimated decay time constant: {tau:.2e} s ({tau*1e6:.2f} μs)")

# ============================================================================
# Data Export Section
# ============================================================================

# Create results directory if it doesn't exist
results_dir = './app/plasma/arc/results'
os.makedirs(results_dir, exist_ok=True)

# Save results to MATLAB format (.mat file)
output_mat = f'{results_dir}/tra_arc1d_explicit_{gas}.mat'
savemat(output_mat, {
    't': t,      # Time array [s]
    'g': g,      # Conductance array [S]
    'x': x,      # Radial position array [m]
    'T': T,      # Temperature array [K] - shape: (num_times, num_positions)
    'V': V       # Velocity array [m/s] - shape: (num_times, num_positions)
})
print(f"\nResults saved to MATLAB format: {output_mat}")

# Also save summary to CSV for easy inspection
output_csv = f'{results_dir}/tra_arc1d_explicit_{gas}_summary.csv'
summary_data = {
    't(s)': t,
    't(us)': t * 1e6,
    'g(S)': g,
    'T_center(K)': T[:, 0],         # Temperature at axis
    'T_max(K)': T.max(axis=1),      # Maximum temperature at each time
    'V_max(m/s)': np.abs(V).max(axis=1),  # Maximum velocity magnitude
}
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(output_csv, mode='w', index=False, float_format='%.6e')
print(f"Summary saved to CSV: {output_csv}")

# ============================================================================
# Visualization Section
# ============================================================================

print(f"\nGenerating visualizations...")

# Create comprehensive figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# Subplot 1: Temperature evolution at center (axis)
ax1 = fig.add_subplot(2, 4, 1)
ax1.plot(t * 1e6, T[:, 0], 'b-', linewidth=2)
ax1.set_xlabel('Time (μs)', fontsize=11)
ax1.set_ylabel('Center Temperature (K)', fontsize=11)
ax1.set_title('Temperature at Arc Axis', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Subplot 2: Arc conductance evolution
ax2 = fig.add_subplot(2, 4, 2)
ax2.plot(t * 1e6, g, 'r-', linewidth=2)
ax2.set_xlabel('Time (μs)', fontsize=11)
ax2.set_ylabel('Arc Conductance (S)', fontsize=11)
ax2.set_title('Arc Conductance Evolution', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Subplot 3: Maximum temperature evolution
ax3 = fig.add_subplot(2, 4, 3)
T_max_vs_time = T.max(axis=1)
ax3.plot(t * 1e6, T_max_vs_time, 'g-', linewidth=2)
ax3.set_xlabel('Time (μs)', fontsize=11)
ax3.set_ylabel('Maximum Temperature (K)', fontsize=11)
ax3.set_title('Maximum Temperature vs Time', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Subplot 4: Maximum velocity evolution
ax4 = fig.add_subplot(2, 4, 4)
V_max_vs_time = np.abs(V).max(axis=1)
ax4.plot(t * 1e6, V_max_vs_time, 'm-', linewidth=2)
ax4.set_xlabel('Time (μs)', fontsize=11)
ax4.set_ylabel('Maximum Velocity (m/s)', fontsize=11)
ax4.set_title('Maximum Radial Velocity vs Time', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Subplot 5: Radial temperature profiles at selected times
ax5 = fig.add_subplot(2, 4, 5)
num_profiles = min(5, len(t))  # Show up to 5 profiles
indices = np.linspace(0, len(t)-1, num_profiles, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, num_profiles))

for idx, color in zip(indices, colors):
    ax5.plot(x * 1e3, T[idx, :], '-', color=color, linewidth=2,
            label=f't = {t[idx]*1e6:.2f} μs')
ax5.set_xlabel('Radial Position (mm)', fontsize=11)
ax5.set_ylabel('Temperature (K)', fontsize=11)
ax5.set_title('Radial Temperature Profiles', fontsize=12, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Subplot 6: Radial velocity profiles at selected times
ax6 = fig.add_subplot(2, 4, 6)
for idx, color in zip(indices, colors):
    ax6.plot(x * 1e3, V[idx, :], '-', color=color, linewidth=2,
            label=f't = {t[idx]*1e6:.2f} μs')
ax6.set_xlabel('Radial Position (mm)', fontsize=11)
ax6.set_ylabel('Velocity (m/s)', fontsize=11)
ax6.set_title('Radial Velocity Profiles', fontsize=12, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Subplot 7: 2D temperature evolution (contour plot)
ax7 = fig.add_subplot(2, 4, 7)
X, T_mesh = np.meshgrid(x * 1e3, t * 1e6)
contour_T = ax7.contourf(X, T_mesh, T, levels=20, cmap='hot')
ax7.set_xlabel('Radial Position (mm)', fontsize=11)
ax7.set_ylabel('Time (μs)', fontsize=11)
ax7.set_title('Temperature Evolution (2D)', fontsize=12, fontweight='bold')
cbar_T = plt.colorbar(contour_T, ax=ax7)
cbar_T.set_label('Temperature (K)', fontsize=10)

# Subplot 8: 2D velocity evolution (contour plot)
ax8 = fig.add_subplot(2, 4, 8)
contour_V = ax8.contourf(X, T_mesh, V, levels=20, cmap='coolwarm')
ax8.set_xlabel('Radial Position (mm)', fontsize=11)
ax8.set_ylabel('Time (μs)', fontsize=11)
ax8.set_title('Velocity Evolution (2D)', fontsize=12, fontweight='bold')
cbar_V = plt.colorbar(contour_V, ax=ax8)
cbar_V.set_label('Velocity (m/s)', fontsize=10)

# Add overall title
mode_str = 'Decay Mode' if not enable_joule else f'I = {I} A'
fig.suptitle(f'{gas} Transient Arc WITH Velocity (Explicit) - {mode_str}, R = {R*1e3:.1f} mm', 
             fontsize=14, fontweight='bold', y=0.98)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_png = f'{results_dir}/tra_arc1d_explicit_{gas}.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_png}")

# ============================================================================
# Summary Section
# ============================================================================

print("\n" + "=" * 70)
print("Transient simulation with velocity (explicit method) completed successfully!")
print("=" * 70)
print(f"\nOutput files:")
print(f"  - MATLAB data: {output_mat}")
print(f"  - CSV summary: {output_csv}")
print(f"  - PNG plot: {output_png}")
print(f"\nSimulation details:")
print(f"  - Total time steps: {step_num}")
print(f"  - Saved snapshots: {len(t)}")
print(f"  - Simulation time: {t[-1]:.2e} s ({t[-1]*1e6:.2f} μs)")
print(f"  - Method: Explicit time integration (RECOMMENDED)")
print(f"\nKey features:")
print(f"  - Radial velocity field computed")
print(f"  - Convective heat transport included")
print(f"  - Stable explicit integration")
print(f"\nFor more information, see:")
print(f"  - Arc model: ai4plasma/plasma/arc.py (TraArc1D class)")
print(f"  - Property utilities: ai4plasma/plasma/prop.py")
print("=" * 70)



