"""Utility module for plasma property interpolation and processing.

This module provides efficient methods for handling temperature-dependent thermodynamic,
transport, and radiation properties of thermal plasmas. It combines NumPy-based
interpolation functions with PyTorch-compatible spline interpolators for use in
neural network training and inference.

Plasma Propreties Classes
-------------------------
- `BaseModel` abstract base class components:
- `ArcPropSpline`: PyTorch-based cubic spline class for arc plasma properties
- `CoronaPropSpline`: PyTorch-based spline class for corona discharge
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline
import torch

from ai4plasma.utils.common import numpy2torch
from ai4plasma.config import REAL


def read_thermo_data(infile):
    """
    Read thermodynamic and transport properties data from a file.

    This function loads plasma property data including density, enthalpy,
    specific heat capacity, electrical conductivity, and thermal conductivity
    as functions of temperature.

    Parameters
    ----------
    infile : str
        Path to the input data file. Expected format: whitespace-separated
        columns with headers: T(K), rho(kg/m3), h(J/kg), Cp(J/K/kg),
        sigma(S/m), kappa(W/m/K).

    Returns
    -------
    tuple: (temp_list, rho_list, h_list, Cp_list, sigma_list, kappa_list)
        - temp_list : ndarray
            Temperature values in Kelvin
        - rho_list : ndarray
            Mass density in kg/m³
        - h_list : ndarray
            Specific enthalpy in J/kg
        - Cp_list : ndarray
            Specific heat capacity at constant pressure in J/(K·kg)
        - sigma_list : ndarray
            Electrical conductivity in S/m
        - kappa_list : ndarray
            Thermal conductivity in W/(m·K)

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    KeyError
        If required columns are missing from the data file.
    """
    dat = pd.read_csv(infile, sep=r'\s+')
    temp_list = dat['T(K)'].values
    rho_list = dat['rho(kg/m3)'].values
    h_list = dat['h(J/kg)'].values
    Cp_list = dat['Cp(J/K/kg)'].values
    sigma_list = dat['sigma(S/m)'].values
    kappa_list = dat['kappa(W/m/K)'].values

    return temp_list, rho_list, h_list, Cp_list, sigma_list, kappa_list


def read_nec_data(infile):
    """
    Read Net Emission Coefficient (NEC) data from a file.

    NEC represents the radiation emission power per unit volume of plasma
    as a function of both temperature and arc radius. This data is essential
    for modeling radiation losses in arc plasma simulations.

    Parameters
    ----------
    infile : str
        Path to the NEC data file. Expected format: first column is T(K),
        subsequent columns are NEC values at different arc radii (column
        headers represent radius values).

    Returns
    -------
    tuple: (nec_temp_list, nec_R_list, nec_array)
        - nec_temp_list : ndarray
            Temperature values in Kelvin
        - nec_R_list : ndarray
            Arc radius values in meters
        - nec_array : ndarray (2D)
            NEC values with shape (n_temp, n_radius) in W/m³. Zero or negative
            values are replaced with 1e-50 to avoid numerical issues in
            logarithmic calculations.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the data format is invalid.
    """
    dat = pd.read_csv(infile, sep=r'\s+')
    nec_temp_list = dat['T(K)'].values

    # Extract radius values from column headers (skip first column which is temperature)
    cols = dat.axes[1]
    R = []
    for i in range(1, cols.size):
        R.append(float(cols[i]))
    nec_R_list = np.array(R)

    # Extract NEC data array and replace non-positive values
    nec_array = dat.values[:, 1:]
    nec_array[nec_array <= 0] = 1e-50  # Avoid zero values for logarithmic interpolation

    return nec_temp_list, nec_R_list, nec_array


def interp_prop(temp_list, prop_list, T):
    """
    Interpolate plasma properties at given temperature values using cubic splines.

    This function performs 1D cubic spline interpolation of plasma properties
    (e.g., conductivity, thermal conductivity) as a function of temperature.
    Input temperatures are automatically clamped to the valid data range to
    prevent extrapolation errors.

    Parameters
    ----------
    temp_list : ndarray
        Array of temperature values (in Kelvin) from the property table.
        Must be monotonically increasing.
    prop_list : ndarray
        Array of property values corresponding to temp_list.
        Can represent any plasma property (conductivity, density, etc.).
    T : ndarray
        Array of query temperature values (in Kelvin) where interpolation
        is desired. Values outside [temp_list[0], temp_list[-1]] are
        automatically clamped to boundary values. Note: T is modified in-place.

    Returns
    -------
    ndarray
        Interpolated property values at the query temperatures T.
    """
    # Clamp temperature values to valid range to avoid extrapolation issues
    T = np.clip(T, temp_list[0], temp_list[-1]) 

    # Create cubic spline interpolator (replacement for deprecated interp1d)
    func = interpolate.RegularGridInterpolator(
        (temp_list,), 
        prop_list, 
        method='cubic', 
        bounds_error=False, 
        fill_value=None
    )
    
    # Reshape input for RegularGridInterpolator (requires 2D input)
    # and flatten output to match expected 1D output shape
    prop = func(T).flatten()

    return prop


def interp_prop_log(temp_list, prop_list, T):
    """
    Interpolate plasma properties using logarithmic transformation for better
    accuracy with exponentially varying quantities.

    This function is particularly useful for properties that vary over many
    orders of magnitude (e.g., electrical conductivity, radiation coefficients).
    Interpolation is performed in log-space to maintain relative accuracy across
    the entire range, then transformed back to linear space.

    Parameters
    ----------
    temp_list : ndarray
        Array of temperature values (in Kelvin) from the property table.
        Must be monotonically increasing.
    prop_list : ndarray
        Array of property values corresponding to temp_list. All values
        must be positive (required for logarithmic transformation).
    T : ndarray
        Array of query temperature values (in Kelvin) where interpolation
        is desired. Values outside [temp_list[0], temp_list[-1]] are
        automatically clamped. Note: T is modified in-place.

    Returns
    -------
    ndarray
        Interpolated property values at the query temperatures T,
        transformed back to linear space.
    """
    # Clamp temperature values to valid range
    T = np.clip(T, temp_list[0], temp_list[-1])

    # Create interpolator in log-space for better accuracy
    func = interpolate.RegularGridInterpolator(
        (temp_list,), 
        np.log(prop_list),  # Interpolate in log-space
        method='cubic', 
        bounds_error=False, 
        fill_value=None
    )
    
    # Interpolate and transform back to linear space
    prop = np.exp(func(T).flatten())

    return prop


def interp_nec(temp_list, R_list, nec_array, R, T):
    """
    Interpolate Net Emission Coefficient (NEC) at given arc radius and temperatures.

    This function performs 2D cubic spline interpolation of radiation emission
    coefficients that depend on both temperature and arc radius. This is crucial
    for accurate modeling of radiation losses in arc plasma simulations.

    Parameters
    ----------
    temp_list : ndarray
        Array of temperature values (in Kelvin) from the NEC table.
        Must be monotonically increasing.
    R_list : ndarray
        Array of arc radius values (in meters) from the NEC table.
        Must be monotonically increasing.
    nec_array : ndarray (2D)
        Array of NEC values with shape (len(temp_list), len(R_list)).
        Units: W/m³ (power per unit volume).
    R : float
        Query arc radius value (in meters). Automatically clamped to
        [R_list[0], R_list[-1]] if outside valid range.
    T : ndarray
        Array of query temperature values (in Kelvin). Values outside
        [temp_list[0], temp_list[-1]] are clamped. Note: T is modified in-place.

    Returns
    -------
    ndarray
        Interpolated NEC values at the given radius R and temperatures T.
        Shape matches T.shape.
    """
    # Clamp temperature values to valid range
    T = np.clip(T, temp_list[0], temp_list[-1]) 

    # Clamp radius value to valid range
    R = R_list[0] if R < R_list[0] else R
    R = R_list[-1] if R > R_list[-1] else R

    # Create 2D interpolator (replacement for deprecated interp2d)
    # Note: RegularGridInterpolator uses grid coordinate order (temp, radius)
    func = interpolate.RegularGridInterpolator(
        (temp_list, R_list), 
        nec_array, 
        method='cubic', 
        bounds_error=False, 
        fill_value=None
    )
    
    # Create query points array with shape (n, 2) where each row is (T, R)
    # Use np.full_like to broadcast the scalar R to match T's shape
    points = np.column_stack([T, np.full_like(T, R)])
    nec = func(points)

    return nec


def interp_nec_log(temp_list, R_list, nec_array, R, T):
    """
    Interpolate Net Emission Coefficient (NEC) using logarithmic transformation
    for improved accuracy across multiple orders of magnitude.

    This is the recommended method for NEC interpolation since radiation
    coefficients typically vary exponentially with temperature and can span
    many orders of magnitude. Logarithmic interpolation preserves relative
    accuracy better than linear interpolation.

    Parameters
    ----------
    temp_list : ndarray
        Array of temperature values (in Kelvin) from the NEC table.
        Must be monotonically increasing.
    R_list : ndarray
        Array of arc radius values (in meters) from the NEC table.
        Must be monotonically increasing.
    nec_array : ndarray (2D)
        Array of NEC values with shape (len(temp_list), len(R_list)).
        Units: W/m³. All values must be positive for log transformation.
    R : float
        Query arc radius value (in meters). Automatically clamped to
        [R_list[0], R_list[-1]] if outside valid range.
    T : ndarray
        Array of query temperature values (in Kelvin). Values outside
        [temp_list[0], temp_list[-1]] are clamped. Note: T is modified in-place.

    Returns
    -------
    ndarray
        Interpolated NEC values at the given radius R and temperatures T,
        transformed back to linear space. Shape matches T.shape.
    """
    # Clamp temperature values to valid range
    T = np.clip(T, temp_list[0], temp_list[-1]) 
    
    # Clamp radius value to valid range
    R = R_list[0] if R < R_list[0] else R
    R = R_list[-1] if R > R_list[-1] else R

    # Create 2D interpolator in log-space
    func = interpolate.RegularGridInterpolator(
        (temp_list, R_list), 
        np.log(nec_array),  # Interpolate in log-space
        method='cubic', 
        bounds_error=False, 
        fill_value=None
    )
    
    # Create query points array with shape (n, 2) where each row is (T, R)
    points = np.column_stack([T, np.full_like(T, R)])
    
    # Interpolate and transform back to linear space
    nec = np.exp(func(points))

    return nec


def interp_x(x0, f0, x, kind='linear'):
    """
    General-purpose 1D interpolation function with automatic boundary clamping.

    This utility function provides flexible interpolation for arbitrary 1D
    functions using either linear or cubic spline methods. Query points
    outside the data range are automatically clamped to boundary values
    to prevent extrapolation errors.

    Parameters
    ----------
    x0 : ndarray
        Array of x-coordinates of the data points. Must be monotonically
        increasing. Can represent any physical quantity (position, time, etc.).
    f0 : ndarray
        Array of y-coordinates (function values) corresponding to x0.
        Shape must match x0.shape.
    x : ndarray
        Array of query x-coordinates where interpolation is desired.
        Values outside [x0[0], x0[-1]] are clamped to boundary values.
    kind : str, optional
        Interpolation method. Options:
        - 'linear': Linear interpolation (default, faster)
        - 'cubic': Cubic spline interpolation (smoother, more accurate)

    Returns
    -------
    ndarray
        Interpolated function values at query points x. Shape matches x.shape.
    """
    # Select interpolation method
    method = 'linear' if kind == 'linear' else 'cubic'
    
    # Create 1D interpolator (replacement for deprecated interp1d)
    func = interpolate.RegularGridInterpolator(
        (x0,), 
        f0, 
        method=method, 
        bounds_error=False, 
        fill_value=None
    )
    
    # Clamp query points to valid data range to prevent extrapolation
    # This replaces the fill_value='extrapolate' behavior from interp1d
    x_clipped = np.clip(x, x0[0], x0[-1])
    
    # Perform interpolation and flatten result to 1D
    f = func(x_clipped).flatten()

    return f


class ArcPropSpline:
    """
    Spline interpolator for arc plasma properties with temperature range enforcement.

    This class provides cubic spline interpolation for thermodynamic properties,
    transport coefficients, and net emission coefficient (NEC) of arc plasma.
    All input temperatures are automatically clamped to the valid range of 300-30,000 K
    to ensure physical validity and prevent extrapolation errors.

    Properties Available
    --------------------
    - sigma(T): Electrical conductivity (S/m)
    - kappa(T): Thermal conductivity (W/(m·K))
    - rho(T): Mass density (kg/m³)
    - Cp(T): Specific heat capacity at constant pressure (J/(K·kg))
    - nec(T): Net emission coefficient (W/m³)

    Temperature Range
    -----------------
    All methods automatically clamp input temperatures to [300, 30000] K using torch.clamp.
    This ensures that property calculations remain within the physically valid and
    data-supported temperature range.

    Parameters
    ----------
    thermo_file : str
        Path to thermodynamic and transport property data file
    nec_file : str
        Path to net emission coefficient (NEC) data file
    R : float
        Arc radius in meters (for NEC interpolation)
    T_min : float, optional
        Minimum temperature limit in Kelvin (default: 300 K)
    T_max : float, optional
        Maximum temperature limit in Kelvin (default: 30000 K)
    """
    def __init__(self, thermo_file, nec_file, R, T_min=300.0, T_max=30000.0):
        temp, rho, _, Cp, sigma, kappa = read_thermo_data(thermo_file)
        nec_temp, nec_R, nec_array = read_nec_data(nec_file)
        nec_log = np.log(interp_nec_log(nec_temp, nec_R, nec_array, R, temp))

        # Store temperature limits for clamping
        self.T_min = T_min
        self.T_max = T_max

        self.x_sigma, self.coefs_sigma = self._cubic_spline(temp, sigma)
        self.x_kappa, self.coefs_kappa = self._cubic_spline(temp, kappa)
        self.x_rho, self.coefs_rho = self._cubic_spline(temp, rho)
        self.x_Cp, self.coefs_Cp = self._cubic_spline(temp, Cp)
        self.x_nec, self.coefs_nec = self._cubic_spline(nec_temp, nec_log)

    def _cubic_spline(self, temp, prop):
        """
        Create cubic spline interpolator and convert to PyTorch tensors.

        Parameters
        ----------
        temp : ndarray
            Temperature values (nodes)
        prop : ndarray
            Property values at temperature nodes

        Returns
        -------
        tuple
            (x, coefs) - PyTorch tensors for spline nodes and coefficients
        """
        spl = CubicSpline(temp, prop)
        x = numpy2torch(spl.x.astype(REAL()))
        coefs = numpy2torch(spl.c.astype(REAL()))

        return x, coefs

    def _torch_spline(self, nodes, coefs, x):
        """
        Evaluate cubic spline using PyTorch operations with automatic temperature clamping.

        This method performs cubic spline interpolation using PyTorch tensors,
        enabling GPU acceleration and automatic differentiation. Input temperatures
        are automatically clamped to [T_min, T_max] to ensure validity.

        Parameters
        ----------
        nodes : torch.Tensor
            Spline node positions (temperature values from data)
        coefs : torch.Tensor
            Spline coefficients (4 x n_segments array)
        x : torch.Tensor
            Query points (temperatures) - will be clamped to valid range

        Returns
        -------
        torch.Tensor
            Interpolated property values at (clamped) query points
        """
        # Clamp temperature to valid range [T_min, T_max]
        x_clamped = torch.clamp(x, min=self.T_min, max=self.T_max)
        
        # Find spline segment for each point
        locs = (x_clamped[:,None] >= nodes[None,:]).long().sum(1) - 1
        
        # Compute local coordinate within segment
        xx = x_clamped - nodes[locs]
        
        # Evaluate cubic polynomial
        y = coefs[0,locs]*xx**3 + coefs[1,locs]*xx**2 + coefs[2,locs]*xx + coefs[3,locs]

        return y

    def sigma(self, T):
        """
        Compute electrical conductivity at given temperatures.

        Parameters
        ----------
        T : torch.Tensor
            Temperature values in Kelvin (automatically clamped to [300, 30000] K)

        Returns
        -------
        torch.Tensor
            Electrical conductivity in S/m
        """
        return self._torch_spline(self.x_sigma, self.coefs_sigma, T)

    def kappa(self, T):
        """
        Compute thermal conductivity at given temperatures.

        Parameters
        ----------
        T : torch.Tensor
            Temperature values in Kelvin (automatically clamped to [300, 30000] K)

        Returns
        -------
        torch.Tensor
            Thermal conductivity in W/(m·K)
        """
        return self._torch_spline(self.x_kappa, self.coefs_kappa, T)

    def rho(self, T):
        """
        Compute mass density at given temperatures.

        Parameters
        ----------
        T : torch.Tensor
            Temperature values in Kelvin (automatically clamped to [300, 30000] K)

        Returns
        -------
        torch.Tensor
            Mass density in kg/m³
        """
        return self._torch_spline(self.x_rho, self.coefs_rho, T)

    def Cp(self, T):
        """
        Compute specific heat capacity at constant pressure at given temperatures.

        Parameters
        ----------
        T : torch.Tensor
            Temperature values in Kelvin (automatically clamped to [300, 30000] K)

        Returns
        -------
        torch.Tensor
            Specific heat capacity in J/(K·kg)
        """
        return self._torch_spline(self.x_Cp, self.coefs_Cp, T)

    def nec(self, T):
        """
        Compute net emission coefficient (NEC) at given temperatures.

        Parameters
        ----------
        T : torch.Tensor
            Temperature values in Kelvin (automatically clamped to [300, 30000] K)

        Returns
        -------
        torch.Tensor
            Net emission coefficient in W/m³
        """
        return torch.exp(self._torch_spline(self.x_nec, self.coefs_nec, T))


class CoronaPropSpline:
    """
    Spline interpolator for corona discharge transport coefficients with E/N range enforcement.

    This class provides cubic spline interpolation for transport coefficients in corona
    discharge modeling, including ionization coefficient, electron/ion mobilities, and
    diffusion coefficients. It supports arbitrary gas species (not limited to argon).
    All input E/N values are automatically clamped to the valid data range to ensure
    physical validity and prevent extrapolation errors.

    Properties Available
    --------------------
    - alpha(E/N): Ionization coefficient (1/m or m²/V depending on normalization)
    - mu_e(E/N): Electron mobility (m²/(V·s))
    - mu_p(E/N): Positive ion mobility (m²/(V·s))
    - D_e(E/N): Electron diffusion coefficient (m²/s)
    - D_p(E/N): Positive ion diffusion coefficient (m²/s)

    E/N Units
    ----------
    E/N is the reduced electric field in Townsend units (Td), where:
    1 Td = 10⁻²¹ V·m²

    Typical range: 10-1000 Td depending on gas species and pressure

    Normalization Options
    ----------------------
    The class supports flexible normalization of transport coefficients:
    - alpha: Can be stored in log-space for numerical stability
    - mu_e, D_e: Can be normalized by neutral gas density (N_neutral)
    - mu_p, D_p: Direct interpolation or custom normalization

    Parameters
    ----------
    base_path : str
        Directory path containing transport coefficient data files.
        Expected files: alpha.dat, mu_e.dat, mu_p.dat, D_e.dat, D_p.dat
    N_neutral : float, optional
        Neutral gas number density [1/m³] for normalizing transport coefficients.
        If None, no normalization is applied (default: None).
    EN_min : float, optional
        Minimum E/N limit in Td (default: None, uses data range).
    EN_max : float, optional
        Maximum E/N limit in Td (default: None, uses data range).
    alpha_log_scale : bool, optional
        Whether to store alpha in log-space for better numerical stability (default: True).
    alpha_multiply_N : bool, optional
        Whether to multiply alpha by N_neutral when evaluating (default: True).
        Set to False if alpha data is already in absolute units.
    normalize_mobility : bool, optional
        Whether to normalize mobilities and diffusion coefficients by N_neutral (default: True).
        Set to False if data is already in absolute units.

    File Format
    -----------
    Each data file should contain two columns:
    - Column 1: E/N values in Td (monotonically increasing)
    - Column 2: Transport coefficient values

    Example file (alpha.dat)::

        # E/N(Td)  alpha(1/m)
        10.0       1.5e5
        50.0       8.2e6
        100.0      2.1e7
        ...
    """
    def __init__(self, file_path_dict, N_neutral=None, EN_min=None, EN_max=None,
                 alpha_log_scale=True, alpha_multiply_N=True, normalize_mobility=True):
        """
        Initialize corona discharge transport coefficient interpolators.

        This constructor loads transport coefficient data from files and constructs
        cubic spline interpolators for efficient evaluation during training.
        """
        # Store normalization parameters
        self.N_neutral = N_neutral
        self.alpha_multiply_N = alpha_multiply_N
        self.normalize_mobility = normalize_mobility
        self.alpha_log_scale = alpha_log_scale
        
        # Load and process alpha (ionization coefficient)
        EN_alpha, alpha = self._read_prop_data(file_path_dict['alpha'])
        if alpha_log_scale:
            alpha = np.log(alpha)  # Store in log-space for numerical stability
        self.x_alpha, self.coefs_alpha = self._cubic_spline(EN_alpha, alpha)
        
        # Load and process electron mobility
        EN_mu_e, mu_e = self._read_prop_data(file_path_dict['mu_e'])
        if normalize_mobility and N_neutral is not None:
            mu_e /= N_neutral  # Normalize by neutral density
        self.x_mu_e, self.coefs_mu_e = self._cubic_spline(EN_mu_e, mu_e)
        
        # Load and process electron diffusion coefficient
        EN_D_e, D_e = self._read_prop_data(file_path_dict['D_e'])
        if normalize_mobility and N_neutral is not None:
            D_e /= N_neutral  # Normalize by neutral density
        self.x_D_e, self.coefs_D_e = self._cubic_spline(EN_D_e, D_e)
        
        # Load and process positive ion mobility
        EN_mu_p, mu_p = self._read_prop_data(file_path_dict['mu_p'])
        if normalize_mobility and N_neutral is not None:
            mu_p /= N_neutral  # Normalize by neutral density
        self.x_mu_p, self.coefs_mu_p = self._cubic_spline(EN_mu_p, mu_p)
        
        # Load and process positive ion diffusion coefficient
        EN_D_p, D_p = self._read_prop_data(file_path_dict['D_p'])
        if normalize_mobility and N_neutral is not None:
            D_p /= N_neutral  # Normalize by neutral density
        self.x_D_p, self.coefs_D_p = self._cubic_spline(EN_D_p, D_p)
        
        # Determine E/N range limits from data if not specified
        all_EN_ranges = [EN_alpha, EN_mu_e, EN_D_e, EN_mu_p, EN_D_p]
        self.EN_min = EN_min if EN_min is not None else max(en[0] for en in all_EN_ranges)
        self.EN_max = EN_max if EN_max is not None else min(en[-1] for en in all_EN_ranges)
    
    def _read_prop_data(self, infile):
        """
        Read transport coefficient data from file.

        Parameters
        ----------
        infile : str
            Path to data file with two columns: E/N and property value

        Returns
        -------
        tuple
            (EN, prop) - E/N values and property values as numpy arrays

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        """
        dat = pd.read_csv(infile, sep=r'\s+')
        EN = dat.values[:, 0].astype(REAL())
        prop = dat.values[:, 1].astype(REAL())
        return EN, prop
    
    def _cubic_spline(self, EN, prop):
        """
        Create cubic spline interpolator and convert to PyTorch tensors.

        Parameters
        ----------
        EN : ndarray
            E/N values (nodes)
        prop : ndarray
            Property values at E/N nodes

        Returns
        -------
        tuple
            (x, coefs) - PyTorch tensors for spline nodes and coefficients
        """
        spl = CubicSpline(EN, prop)
        x = numpy2torch(spl.x.astype(REAL()))
        coefs = numpy2torch(spl.c.astype(REAL()))
        return x, coefs
    
    def _torch_spline(self, nodes, coefs, x):
        """
        Evaluate cubic spline using PyTorch operations with automatic E/N clamping.

        This method performs cubic spline interpolation using PyTorch tensors,
        enabling GPU acceleration and automatic differentiation. Input E/N values
        are automatically clamped to [EN_min, EN_max] to ensure validity.

        Parameters
        ----------
        nodes : torch.Tensor
            Spline node positions (E/N values from data)
        coefs : torch.Tensor
            Spline coefficients (4 x n_segments array)
        x : torch.Tensor
            Query points (E/N values) - will be clamped to valid range

        Returns
        -------
        torch.Tensor
            Interpolated property values at (clamped) query points
        """
        # Clamp E/N to valid range [EN_min, EN_max]
        x_clamped = torch.clamp(x, min=self.EN_min, max=self.EN_max)
        
        # Find spline segment for each point
        locs = (x_clamped[:, None] >= nodes[None, :]).long().sum(1) - 1
        
        # Compute local coordinate within segment
        xx = x_clamped - nodes[locs]
        
        # Evaluate cubic polynomial
        y = coefs[0, locs] * xx**3 + coefs[1, locs] * xx**2 + coefs[2, locs] * xx + coefs[3, locs]
        
        return y
    
    def alpha(self, EN):
        """
        Compute ionization coefficient at given E/N values.

        Parameters
        ----------
        EN : torch.Tensor
            Reduced electric field E/N in Td (automatically clamped to valid range)

        Returns
        -------
        torch.Tensor
            Ionization coefficient in 1/m (or m²/V depending on normalization)

        Notes
        -----
        - If alpha_log_scale=True, exp() is applied to convert from log-space
        - If alpha_multiply_N=True and N_neutral is provided, result is multiplied by N_neutral
        """
        alpha_val = self._torch_spline(self.x_alpha, self.coefs_alpha, EN)
        
        # Convert from log-space if necessary
        if self.alpha_log_scale:
            alpha_val = torch.exp(alpha_val)
        
        # Multiply by neutral density if required
        if self.alpha_multiply_N and self.N_neutral is not None:
            alpha_val = alpha_val * self.N_neutral
        
        return alpha_val
    
    def mu_e(self, EN):
        """
        Compute electron mobility at given E/N values.

        Parameters
        ----------
        EN : torch.Tensor
            Reduced electric field E/N in Td (automatically clamped to valid range)

        Returns
        -------
        torch.Tensor
            Electron mobility in m²/(V·s)
        """
        return self._torch_spline(self.x_mu_e, self.coefs_mu_e, EN)
    
    def D_e(self, EN):
        """
        Compute electron diffusion coefficient at given E/N values.

        Parameters
        ----------
        EN : torch.Tensor
            Reduced electric field E/N in Td (automatically clamped to valid range)

        Returns
        -------
        torch.Tensor
            Electron diffusion coefficient in m²/s
        """
        return self._torch_spline(self.x_D_e, self.coefs_D_e, EN)
    
    def mu_p(self, EN):
        """
        Compute positive ion mobility at given E/N values.

        Parameters
        ----------
        EN : torch.Tensor
            Reduced electric field E/N in Td (automatically clamped to valid range)

        Returns
        -------
        torch.Tensor
            Positive ion mobility in m²/(V·s)
        """
        return self._torch_spline(self.x_mu_p, self.coefs_mu_p, EN)
    
    def D_p(self, EN):
        """
        Compute positive ion diffusion coefficient at given E/N values.

        Parameters
        ----------
        EN : torch.Tensor
            Reduced electric field E/N in Td (automatically clamped to valid range)

        Returns
        -------
        torch.Tensor
            Positive ion diffusion coefficient in m²/s
        """
        return self._torch_spline(self.x_D_p, self.coefs_D_p, EN)