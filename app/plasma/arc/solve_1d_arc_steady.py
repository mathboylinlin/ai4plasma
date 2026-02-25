###
## Author: Linlin Zhong
## Email: linlin@seu.edu.cn
## Date: June 1, 2017
## Updated: Jan 3, 2026
## Description: Test script for 1D stationary arc plasma model
##              Demonstrates the use of StaArc1D class to solve the 
##              Elenbaas-Heller equation for thermal arc plasmas.
###

import sys
sys.path.append('.')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ai4plasma.plasma.prop import read_thermo_data, read_nec_data, interp_nec_log
from ai4plasma.plasma.arc import StaArc1D

# ============================================================================
# Configuration Section
# ============================================================================

# Gas selection
# Options: 'SF6', 'Air', 'CO2', etc. (depending on available data files)
gas = 'SF6'

# Arc parameters
I = 200        # Arc current in Amperes [A]
R = 10e-3      # Arc radius in meters [m] (10 mm)

# Numerical parameters
mesh_num = 500         # Number of mesh cells for spatial discretization
T_min = 300           # Minimum temperature (boundary/ambient) [K]
T_max = 12000         # Maximum temperature (initial center estimate) [K]
Tb = 2000             # Boundary temperature at r=R [K]
relax = 0.1           # Relaxation factor for iteration stability (0 < relax <= 1)
converge_tol = 1e-6   # Convergence tolerance [K]
max_ite = 6000        # Maximum number of iterations
is_print = True       # Print iteration progress

# Output flag for identification
flag = gas

# ============================================================================
# Data Loading Section
# ============================================================================

print("=" * 70)
print(f"1D Stationary Arc Model - {gas} Plasma")
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
# Returns: temperature, density, enthalpy, Cp, electrical conductivity, thermal conductivity
temp_list, rho_list, h_list, Cp_list, sigma_list, kappa_list = read_thermo_data(thermo_file)
print(f"  - Thermodynamic data loaded: {len(temp_list)} temperature points")
print(f"  - Temperature range: {temp_list[0]:.0f} - {temp_list[-1]:.0f} K")

# Read Net Emission Coefficient (NEC) data for radiation modeling
# NEC represents radiation power per unit volume as function of T and R
nec_temp_list, nec_R_list, nec_array = read_nec_data(nec_file)
print(f"  - NEC data loaded: {len(nec_temp_list)} temperatures × {len(nec_R_list)} radii")
print(f"  - Radius range: {nec_R_list[0]*1e3:.3f} - {nec_R_list[-1]*1e3:.3f} mm")

# ============================================================================
# Property Interpolation Section
# ============================================================================

print(f"\nInterpolating NEC for arc radius R = {R*1e3:.1f} mm...")

# Interpolate NEC values at the specified arc radius for all temperatures
# Using logarithmic interpolation for better accuracy across wide ranges
nec_list = interp_nec_log(nec_temp_list, nec_R_list, nec_array, R, temp_list.copy())
print(f"  - NEC interpolated for {len(nec_list)} temperature points")

# Assemble property tuple for arc model
# Format: (temperature, electrical conductivity, thermal conductivity, NEC)
prop = (temp_list, sigma_list, kappa_list, nec_list)

# ============================================================================
# Arc Model Setup Section
# ============================================================================

print(f"\nInitializing arc model...")
print(f"  - Arc current: {I} A")
print(f"  - Arc radius: {R*1e3:.1f} mm")
print(f"  - Mesh cells: {mesh_num}")
print(f"  - Boundary temperature: {Tb} K")

# Create arc model instance
myarc = StaArc1D(I, R, prop)

# Define initial temperature distribution function
# Uses linear profile: T(r) = T_max at center, T_min at boundary
# This provides a reasonable starting guess for the iterative solver
Tfunc_init = lambda x: (1 - x/R) * (T_max - T_min) + T_min

# ============================================================================
# Solution Section
# ============================================================================

print(f"\nSolving 1D stationary arc equation...")
print(f"  - Relaxation factor: {relax}")
print(f"  - Convergence tolerance: {converge_tol:.1e} K")
print(f"  - Maximum iterations: {max_ite}")
print("-" * 70)

# Solve the arc equation
# Returns: x (cell centers), T (temperatures at cells), 
#          xface (face centers), Tface (temperatures at faces)
x, T, xface, Tface = myarc.solve_onestep(
    mesh_num, 
    Tfunc_init, 
    Tb, 
    relax, 
    converge_tol, 
    max_ite, 
    is_print, 
    flag
)

print("-" * 70)

# ============================================================================
# Results Analysis Section
# ============================================================================

print(f"\nSolution obtained successfully!")
print(f"\nTemperature profile statistics:")
print(f"  - Maximum temperature: {T.max():.2f} K (at axis)")
print(f"  - Minimum temperature: {T.min():.2f} K")
print(f"  - Boundary temperature: {Tface[-1]:.2f} K")

# Calculate arc conductance for reference
arc_conductance = myarc.calc_arc_cond(temp_list, sigma_list, xface, Tface, R)
print(f"\nElectrical properties:")
print(f"  - Arc conductance: {arc_conductance:.4f} S")

# ============================================================================
# Data Export Section
# ============================================================================

# Create results directory if it doesn't exist
results_dir = './app/plasma/arc/results'
os.makedirs(results_dir, exist_ok=True)

# Save numerical results to CSV file
output_csv = f'{results_dir}/sta_arc1d_{gas}.csv'
data = {
    'R(m)': xface,           # Radial position in meters
    'T(K)': Tface,           # Temperature in Kelvin
}
df = pd.DataFrame(data)
df.to_csv(output_csv, mode='w', index=False, float_format='%.5e')
print(f"\nResults saved to: {output_csv}")

# ============================================================================
# Visualization Section
# ============================================================================

print(f"\nGenerating visualization...")

# Create figure with multiple subplots for comprehensive analysis
fig = plt.figure(figsize=(12, 10))

# Subplot 1: Temperature profile
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x * 1e3, T, 'b-', linewidth=2, label='Cell centers')
ax1.plot(xface * 1e3, Tface, 'r--', linewidth=1, alpha=0.7, label='Face centers')
ax1.set_xlabel('Radial Position (mm)', fontsize=11)
ax1.set_ylabel('Temperature (K)', fontsize=11)
ax1.set_title(f'{gas} Arc Temperature Profile', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Subplot 2: Electrical conductivity profile
ax2 = fig.add_subplot(2, 2, 2)
sigma_at_cells = np.interp(T, temp_list, sigma_list)
ax2.semilogy(x * 1e3, sigma_at_cells, 'g-', linewidth=2)
ax2.set_xlabel('Radial Position (mm)', fontsize=11)
ax2.set_ylabel('Electrical Conductivity (S/m)', fontsize=11)
ax2.set_title('Electrical Conductivity Profile', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')

# Subplot 3: Thermal conductivity profile
ax3 = fig.add_subplot(2, 2, 3)
kappa_at_cells = np.interp(T, temp_list, kappa_list)
ax3.plot(x * 1e3, kappa_at_cells, 'm-', linewidth=2)
ax3.set_xlabel('Radial Position (mm)', fontsize=11)
ax3.set_ylabel('Thermal Conductivity (W/(m·K))', fontsize=11)
ax3.set_title('Thermal Conductivity Profile', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Subplot 4: Radiation profile
ax4 = fig.add_subplot(2, 2, 4)
nec_at_cells = np.interp(T, temp_list, nec_list)
radiation_power = 4 * np.pi * nec_at_cells  # Total radiation power density
ax4.semilogy(x * 1e3, radiation_power, 'r-', linewidth=2)
ax4.set_xlabel('Radial Position (mm)', fontsize=11)
ax4.set_ylabel('Radiation Power Density (W/m³)', fontsize=11)
ax4.set_title('Radiation Loss Profile', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, which='both')

# Add overall title with simulation parameters
fig.suptitle(f'{gas} Arc Plasma - I = {I} A, R = {R*1e3:.1f} mm', 
             fontsize=14, fontweight='bold', y=0.98)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_png = f'{results_dir}/sta_arc1d_{gas}.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_png}")

# ============================================================================
# Summary Section
# ============================================================================

print("\n" + "=" * 70)
print("Simulation completed successfully!")
print("=" * 70)
print(f"\nOutput files:")
print(f"  - CSV data: {output_csv}")
print(f"  - PNG plot: {output_png}")
print(f"\nFor more information, see:")
print(f"  - Arc model: ai4plasma/plasma/arc.py")
print(f"  - Property utilities: ai4plasma/plasma/prop.py")
print("=" * 70)


