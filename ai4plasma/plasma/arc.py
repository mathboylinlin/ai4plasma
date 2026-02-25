"""Arc plasma models.

This module provides a comprehensive framework for simulating thermal plasma arcs
in cylindrical geometry using the Elenbaas-Heller energy balance equation. It implements
steady-state and transient arc models with optional radial velocity effects for
high-pressure systems.

Arc Model Classes
-----------------
- `StaArc1D`: One-dimensional stationary arc model.
- `TraArc1DNoV`: Transient arc model without radial velocity.
- `TraArc1D`: Transient arc model with radial velocity (full convection).

Arc Model References
--------------------
[1] L. Zhong, Y. Cressault, and P. Teulet, "Evaluation of Arc Quenching Ability
    for a Gas by Combining 1-D Hydrokinetic Modeling and Boltzmann Equation
    Analysis," IEEE Trans. Plasma Sci., vol. 47, no. 4, pp. 1835-1840, 2019.

[2] L. Zhong, Q. Gu, and S. Zheng, "An improved method for fast evaluating arc
    quenching performance of a gas based on 1D arc decaying model," Phys. Plasmas,
    vol. 26, no. 10, p. 103507, 2019.
"""

import numpy as np
import fipy
from scipy import integrate, interpolate

from .prop import interp_prop, interp_prop_log


class StaArc1D:
    """One-dimensional stationary arc plasma model (Elenbaas-Heller equation).

    This class implements a cylindrically symmetric (1D radial) model for
    steady-state thermal plasma arcs. It solves the energy balance equation
    including Joule heating, thermal conduction, and radiation losses.

    Attributes
    ----------
    I : float
        Arc current in Amperes
    R : float
        Arc radius in meters
    temp_list : ndarray
        Temperature values for property tables (K)
    sigma_list : ndarray
        Electrical conductivity values (S/m)
    kappa_list : ndarray
        Thermal conductivity values (W/(m·K))
    nec_list : ndarray
        Net emission coefficient values (W/m³)
    mesh : fipy.Grid1D
        Finite volume mesh for spatial discretization
    T : fipy.CellVariable
        Temperature field variable
    solver : fipy.Solver
        Linear solver for the equation system
    """
    
    def __init__(self, I, R, prop):
        """Initialize the 1D stationary arc model.

        Parameters
        ----------
        I : float
            Arc current in Amperes. Typical range: 10 - 10000 A
        R : float
            Arc radius in meters. Typical range: 1e-4 - 1e-2 m
        prop : tuple
            Tuple of (temp_list, sigma_list, kappa_list, nec_list) containing
            thermodynamic, transport, and radiation properties as functions
            of temperature. All arrays should have the same length.
        """
        self.I = I
        self.R = R
        self.temp_list, self.sigma_list, self.kappa_list, self.nec_list = prop
    
    def generate_mesh(self, mesh_num):
        """Generate uniform 1D mesh in radial direction.

        Creates a uniform finite volume mesh from r=0 (axis) to r=R
        (arc boundary). Cell centers are used for temperature values,
        and face centers are used for evaluating gradients and fluxes.

        Parameters
        ----------
        mesh_num : int
            Number of mesh cells. Higher values give better spatial resolution
            but increase computational cost. Typical range: 100-1000.
        """
        self.mesh = fipy.Grid1D(nx=mesh_num, dx=self.R/mesh_num)
        self.x = self.mesh.cellCenters[0]
        self.xface = self.mesh.faceCenters[0]
    
    def init_temp(self, Tfunc_init):
        """Initialize temperature field with a given distribution function.

        Parameters
        ----------
        Tfunc_init : callable
            Function that takes radial position r (array) and returns
            initial temperature values. Example:
            lambda r: T_max * (1 - r/R) + T_min
        """
        self.T = fipy.CellVariable(
            name="Temperature", 
            mesh=self.mesh, 
            value=Tfunc_init(self.x), 
            hasOld=True  # Enable storage of previous iteration values
        )

    def set_boundary_temp(self, Tb):
        """Set boundary conditions for temperature field.

        Applies:
        - Dirichlet BC at outer boundary (r=R): T = Tb
        - Neumann BC at axis (r=0): dT/dr = 0 (symmetry condition)

        Parameters
        ----------
        Tb : float
            Boundary temperature at r=R in Kelvin. Should be close to
            ambient temperature or wall temperature. Typical: 300-3000 K.
        """
        # Fixed temperature at outer boundary (right face)
        self.T.constrain(Tb, self.mesh.facesRight)
        # Zero gradient at axis (left face) - symmetry condition
        self.T.faceGrad.constrain(0, self.mesh.facesLeft)

    def set_solver(self, ite_num=200, tol=1e-6):
        """Configure the linear solver for the equation system.

        Parameters
        ----------
        ite_num : int, optional
            Maximum number of linear solver iterations per nonlinear step.
            Default: 200. Increase if solver fails to converge.
        tol : float, optional
            Convergence tolerance for linear solver. Default: 1e-6.
            Smaller values give more accurate linear solves but take longer.
        """
        self.solver = fipy.LinearPCGSolver(iterations=ite_num, tolerance=tol)

    
    def calc_arc_cond(self, temp_list, sigma_list, x, T, R):
        """Calculate total arc electrical conductance from temperature distribution.

        The arc conductance G is computed by integrating the conductivity
        distribution over the arc cross-section:

            G = 2π ∫[0 to R] r * sigma(r) dr

        This is used to determine the electric field: E = I / G

        Parameters
        ----------
        temp_list : ndarray
            Temperature values for conductivity table (K)
        sigma_list : ndarray
            Electrical conductivity values (S/m)
        x : ndarray
            Radial positions where temperature is known (m)
        T : ndarray
            Temperature distribution at positions x (K)
        R : float
            Integration upper limit (arc radius) in meters

        Returns
        -------
        float
            Total arc conductance in Siemens (S). Typical range: 0.01-100 S
        """
        # Interpolate conductivity at face positions
        sigma = interp_prop(temp_list, sigma_list, T)
        
        # Create 1D interpolator for conductivity as function of radius
        # (replacement for deprecated interp1d)
        func_interp = interpolate.RegularGridInterpolator(
            (x,), 
            sigma, 
            method='cubic', 
            bounds_error=False, 
            fill_value=None
        )
        
        # Define integrand: r * sigma(r)
        # Note: RegularGridInterpolator expects 2D input, so we reshape
        f = lambda r: r * func_interp(np.array([[r]])).item()
        
        # Integrate from 0 to R using adaptive quadrature
        arc_cond, _ = integrate.quad(f, 0, R)
        
        # Multiply by 2π for cylindrical geometry
        arc_cond *= (2.0 * np.pi)

        return arc_cond

    def solve(self, relax, tol, max_ite=6000, is_print=True, flag=''):
        """Iteratively solve the 1D stationary arc energy balance equation.

        This method implements a relaxation iteration scheme to solve the
        nonlinear energy balance equation. The equation couples Joule heating,
        thermal conduction, and radiation through temperature-dependent
        properties.

        Parameters
        ----------
        relax : float
            Relaxation factor for under-relaxation (0 < relax <= 1).
            Smaller values improve stability but slow convergence.
            Typical values: 0.05-0.3 for difficult cases, 0.5-1.0 for easier ones.
        tol : float
            Convergence tolerance for RMS temperature change between iterations.
            Typical values: 1e-6 to 1e-4 K.
        max_ite : int, optional
            Maximum number of iterations. Default: 6000.
            Increase for difficult convergence cases.
        is_print : bool, optional
            If True, print iteration progress. Default: True.
        flag : str, optional
            Identifier string for output messages. Default: ''.

        Returns
        -------
        tuple
            (x, T, xface, Tface) where:
            - x : ndarray, cell center positions (m)
            - T : ndarray, temperature at cell centers (K)
            - xface : ndarray, face center positions (m)
            - Tface : ndarray, temperature at face centers (K)
        """
        
        ite_num = 0
        while True:
            # Update iteration counter
            ite_num += 1

            # Calculate arc conductance from current temperature distribution
            # This determines the electric field: E = I / arc_cond
            arc_cond = self.calc_arc_cond(
                self.temp_list, 
                self.sigma_list, 
                self.xface.value, 
                self.T.faceValue, 
                self.R
            )

            # Compute source terms
            # 1. Joule heating: sigma * E² = sigma * (I/G)²
            joule_energy = interp_prop(
                self.temp_list, 
                self.sigma_list, 
                self.T.value
            ) * (self.I / arc_cond) ** 2
            src_joule = fipy.CellVariable(mesh=self.mesh, value=joule_energy)
            
            # 2. Radiation loss: 4π * NEC (accounts for full solid angle)
            rad_energy = 4 * np.pi * interp_prop_log(
                self.temp_list, 
                self.nec_list, 
                self.T.value
            )
            src_rad = fipy.CellVariable(mesh=self.mesh, value=rad_energy)
            
            # Combined source term with cylindrical geometry factor r
            srcTerm = self.x * (src_joule - src_rad) + fipy.ImplicitSourceTerm(coeff=0)

            # Diffusion term: r * kappa evaluated at cell faces
            diff = self.xface * interp_prop(
                self.temp_list, 
                self.kappa_list, 
                self.T.faceValue.value
            )
            diff = fipy.FaceVariable(mesh=self.mesh, value=diff)
            diffTerm = fipy.DiffusionTerm(coeff=diff)

            # Assemble and solve equation: diffTerm + srcTerm = 0
            eqX = 0 == (diffTerm + srcTerm)
            eqX.solve(var=self.T, solver=self.solver)
            
            # Check if maximum iterations reached
            if ite_num >= max_ite:
                print('%s: Warning: Maximum iteration (%d) reached without full convergence!' 
                      % (flag, max_ite))
                break
            
            # Calculate convergence metric: RMS temperature change
            res = np.sqrt(np.sum((self.T.value - self.T.old.value) ** 2) / self.T.value.size)
            
            if is_print:
                print('%s: Iteration = %d, Residual = %g K' % (flag, ite_num, res))
            
            # Check convergence
            if res < tol:
                if is_print:
                    print('%s: Converged at iteration %d! (Residual = %g K)' 
                          % (flag, ite_num, res))
                break
            
            # Apply under-relaxation and update temperature field
            # T_new = T_old + relax * (T_solved - T_old)
            self.T.setValue(self.T.old.value + relax * (self.T.value - self.T.old.value))
            self.T.updateOld()

        return self.x.value, self.T.value, self.xface.value, self.T.faceValue.value

    def solve_onestep(self, mesh_num, Tfunc_init, Tb, relax, tol, max_ite=6000,
                     is_print=True, flag=''):
        """Complete solution procedure: mesh generation, initialization, and solving.

        This convenience method combines all steps needed to solve the arc model:
        1. Generate computational mesh
        2. Initialize temperature field
        3. Set boundary conditions
        4. Configure solver
        5. Iteratively solve the equation

        Parameters
        ----------
        mesh_num : int
            Number of mesh cells. Typical: 100-1000
        Tfunc_init : callable
            Initial temperature distribution function: T = f(r)
            Example: lambda r: (1 - r/R) * (T_max - T_min) + T_min
        Tb : float
            Boundary temperature at r=R (K). Typical: 300-3000 K
        relax : float
            Relaxation factor (0 < relax <= 1). Typical: 0.1-0.3
        tol : float
            Convergence tolerance (K). Typical: 1e-6 to 1e-4
        max_ite : int, optional
            Maximum iterations. Default: 6000
        is_print : bool, optional
            Print iteration progress. Default: True
        flag : str, optional
            Identifier for output messages. Default: ''

        Returns
        -------
        tuple
            (x, T, xface, Tface) - Position and temperature arrays
        """
        # Generate computational mesh
        self.generate_mesh(mesh_num)

        # Initialize temperature field
        self.init_temp(Tfunc_init)
        
        # Set boundary conditions
        self.set_boundary_temp(Tb)

        # Configure linear solver
        self.set_solver()

        # Solve the arc equation iteratively
        self.solve(relax, tol, max_ite, is_print, flag)

        return self.x.value, self.T.value, self.xface.value, self.T.faceValue.value
    

class TraArc1DNoV(StaArc1D):
    """One-dimensional transient arc plasma model without radial velocity (no convection).

    This class extends StaArc1D to solve time-dependent arc plasma problems where
    the arc evolves over time but without significant radial gas flow (velocity = 0).
    This is applicable during the early stages of arc decay or in cases where
    convective heat transport is negligible compared to conduction.

    Attributes
    ----------
    I : float
        Arc current in Amperes (can be zero for decay studies)
    R : float
        Arc radius in meters
    temp_list : ndarray
        Temperature values for property tables (K)
    rho_list : ndarray
        Mass density values (kg/m³)
    Cp_list : ndarray
        Specific heat capacity values (J/(kg·K))
    sigma_list : ndarray
        Electrical conductivity values (S/m)
    kappa_list : ndarray
        Thermal conductivity values (W/(m·K))
    nec_list : ndarray
        Net emission coefficient values (W/m³)
    mesh : fipy.Grid1D
        Finite volume mesh for spatial discretization
    T : fipy.CellVariable
        Temperature field variable with time history
    """

    def __init__(self, I, R, prop):
        """Initialize the 1D transient arc model without radial velocity.

        Parameters
        ----------
        I : float
            Arc current in Amperes. Set to 0 for pure decay simulations
            (no Joule heating). Typical range: 0 - 10000 A
        R : float
            Arc radius in meters. Should remain constant during simulation.
            Typical range: 1e-4 - 1e-2 m
        prop : tuple
            Tuple of (temp_list, rho_list, Cp_list, sigma_list, kappa_list, nec_list)
            containing thermodynamic, transport, and radiation properties as
            functions of temperature. All arrays should have the same length.
        """
        self.I = I
        self.R = R
        self.temp_list, self.rho_list, self.Cp_list, self.sigma_list, \
            self.kappa_list, self.nec_list = prop

    def set_solver(self):
        """Configure the solver for transient equations.

        For transient problems with sweep iterations, the solver is configured
        differently than in steady-state problems. This method is a placeholder
        and can be extended if specific solver settings are needed.
        """
        pass

    def solve(self, dt, step_num, sweep_max_num=10, sweep_res_tol=1e-6,
              is_print_sweep=False, enable_joule=False, save_freq=10,
              is_print=True, flag=''):
        """Solve the transient 1D arc equation over specified time steps.

        This method performs time integration of the energy balance equation
        using implicit time discretization. At each time step, inner sweep
        iterations are performed to handle the nonlinear coupling of
        temperature-dependent properties.

        Parameters
        ----------
        dt : float
            Time step size in seconds. Should be small enough for numerical
            stability. Typical: 1e-9 to 1e-6 s depending on the time scales
            of interest.
        step_num : int
            Total number of time steps to simulate.
            Total simulation time = dt * step_num
        sweep_max_num : int, optional
            Maximum number of sweep iterations per time step for handling
            nonlinearity. Default: 10. Increase if convergence issues occur.
        sweep_res_tol : float, optional
            Convergence tolerance for sweep iterations (RMS temperature change).
            Default: 1e-6 K. Smaller values give more accurate solutions.
        is_print_sweep : bool, optional
            If True, print detailed information for each sweep iteration.
            Default: False. Useful for debugging convergence issues.
        enable_joule : bool, optional
            If True, include Joule heating in the energy balance. If False,
            model pure cooling/decay without energy input. Default: False.
            Set to True for current-carrying arcs, False for decay studies.
        save_freq : int, optional
            Frequency of saving solution snapshots (every N time steps).
            Default: 10. Higher values save memory but lose time resolution.
        is_print : bool, optional
            If True, print progress at each saved time step. Default: True.
        flag : str, optional
            Identifier string for output messages. Default: ''.

        Returns
        -------
        tuple: (t_list, g_list, x_list, T_array) where:
            t_list : ndarray, time points where solution is saved (s)
            g_list : ndarray, arc conductance at saved times (S)
            x_list : ndarray, radial positions (face centers) (m)
            T_array : ndarray (2D), temperature profiles at saved times (K), Shape: (num_saved_steps, num_cells)
        """
        # Initialize storage arrays for time history
        # Number of saved snapshots based on save frequency
        N = step_num // save_freq + 1
        t_list = np.zeros(N)          # Time points (s)
        g_list = np.zeros(N)          # Arc conductance (S)
        T_array = np.zeros((N, len(self.T.faceValue.value)))  # Temperature profiles (K)
        
        # Calculate initial arc conductance
        arc_cond = self.calc_arc_cond(
            self.temp_list, self.sigma_list, 
            self.xface.value, self.T.faceValue, self.R
        )
        
        # Save initial state (t=0)
        t, save_i = 0, 0
        t_list[save_i] = t
        g_list[save_i] = arc_cond
        T_array[save_i, :] = self.T.faceValue.value

        # Time stepping loop
        for i in range(step_num):
            # Update current time
            t += dt

            # Store previous time step values for convergence check
            self.T.updateOld()

            # Sweep iterations to handle nonlinearity at current time step
            for j in range(sweep_max_num):
                # Store temperature at start of sweep for residual calculation
                T_last_value = self.T.faceValue.value.copy()

                # Interpolate temperature-dependent properties at current temperature
                # Mass density for thermal inertia
                rho = interp_prop(self.temp_list, self.rho_list, self.T.value)
                
                # Specific heat capacity
                Cp = interp_prop(self.temp_list, self.Cp_list, self.T.value)
                
                # Transient term coefficient: r * rho * Cp (cylindrical geometry)
                r_rho_Cp = fipy.CellVariable(mesh=self.mesh, value=self.x * rho * Cp)

                # Compute Joule heating if enabled
                if enable_joule:
                    # Recalculate arc conductance with current temperature distribution
                    arc_cond = self.calc_arc_cond(
                        self.temp_list, self.sigma_list, 
                        self.xface.value, self.T.faceValue, self.R
                    )
                    # Joule heating: sigma * E² = sigma * (I/G)²
                    sigma_interp = interp_prop(self.temp_list, self.sigma_list, self.T.value)
                    joule_energy = sigma_interp * (self.I / arc_cond) ** 2
                else:
                    # No Joule heating (decay/cooling phase)
                    joule_energy = 0

                # Create source term variables
                src_joule = fipy.CellVariable(mesh=self.mesh, value=joule_energy)
                
                # Radiation loss: 4π * NEC (full solid angle)
                # Use log interpolation for better accuracy across wide ranges
                rad_energy = 4 * np.pi * interp_prop_log(
                    self.temp_list, self.nec_list, self.T.value
                )
                src_rad = fipy.CellVariable(mesh=self.mesh, value=rad_energy)
                
                # Net source term with cylindrical geometry factor
                src_term = self.x * (src_joule - src_rad)

                # Diffusion term: div(kappa * grad(T)) in cylindrical coords
                # Coefficient: r * kappa evaluated at cell faces
                diff = self.xface * interp_prop(
                    self.temp_list, self.kappa_list, self.T.faceValue.value
                )
                diff = fipy.FaceVariable(mesh=self.mesh, value=diff)
                diff_term = fipy.DiffusionTerm(coeff=diff, var=self.T)

                # Transient term: rho * Cp * dT/dt
                tran_term = fipy.TransientTerm(coeff=r_rho_Cp, var=self.T)

                # Assemble equation: rho*Cp*dT/dt = div(kappa*grad(T)) + S_joule - S_rad
                eqX = tran_term == (diff_term + src_term)

                # Solve using implicit time integration with sweep
                # This updates self.T toward the solution at the new time level
                eqX.sweep(var=self.T, dt=dt)

                # Calculate sweep convergence metric: RMS temperature change
                res = np.sqrt(
                    np.sum((self.T.faceValue.value - T_last_value) ** 2) 
                    / self.T.faceValue.value.size
                )
                
                # Print detailed sweep information if requested
                if is_print_sweep:
                    print('%s: Time step = %d, Sweep = %d, Residual = %g K' 
                          % (flag, i + 1, j + 1, res))
                
                # Check sweep convergence
                if res < sweep_res_tol:
                    break
                
                # Warn if maximum sweeps reached without convergence
                if (j + 1) == sweep_max_num:
                    print('%s: Warning - Maximum sweep number (%d) reached at time step %d!' 
                          % (flag, sweep_max_num, i + 1))

            # Save solution at specified frequency
            if (i + 1) % save_freq == 0:
                save_i += 1
                
                # Recalculate arc conductance for saved state
                arc_cond = self.calc_arc_cond(
                    self.temp_list, self.sigma_list, 
                    self.xface.value, self.T.faceValue, self.R
                )
                
                # Store time, conductance, and temperature profile
                t_list[save_i] = t
                g_list[save_i] = arc_cond
                T_array[save_i, :] = self.T.faceValue.value

                # Print progress information
                if is_print:
                    print('%s: t = %.6e s, save_index = %d, T_max = %.2f K - Saved!' 
                          % (flag, t, save_i, self.T.value.max()))
            
        return t_list, g_list, self.xface.value, T_array

    def solve_onestep(self, mesh_num, Tfunc_init, Tb, dt, step_num,
                     sweep_max_num=10, sweep_res_tol=1e-6, is_print_sweep=False,
                     enable_joule=False, save_freq=10, is_print=True, flag=''):
        """Complete solution procedure for transient arc: setup and time integration.

        This convenience method combines all steps needed to solve the transient
        arc problem:
        1. Generate computational mesh
        2. Initialize temperature field
        3. Set boundary conditions
        4. Perform time integration

        Parameters
        ----------
        mesh_num : int
            Number of mesh cells. Typical: 100-1000
        Tfunc_init : callable
            Initial temperature distribution function: T = f(r)
            Should return temperature in Kelvin for given radial position.
            Example: lambda r: T_center * (1 - r/R) + T_boundary
        Tb : float
            Boundary temperature at r=R (K). Typically ambient or wall
            temperature. Typical: 300-3000 K
        dt : float
            Time step size (s). Choose based on required time resolution
            and numerical stability. Typical: 1e-9 to 1e-6 s
        step_num : int
            Total number of time steps. Total time = dt * step_num
        sweep_max_num : int, optional
            Maximum sweep iterations per time step. Default: 10
        sweep_res_tol : float, optional
            Sweep convergence tolerance (K). Default: 1e-6
        is_print_sweep : bool, optional
            Print sweep details. Default: False
        enable_joule : bool, optional
            Include Joule heating. Default: False (decay only)
        save_freq : int, optional
            Save solution every N time steps. Default: 10
        is_print : bool, optional
            Print time step progress. Default: True
        flag : str, optional
            Identifier for output messages. Default: ''

        Returns
        -------
        tuple: (t_list, g_list, x_list, T_array) - Time history of solution
        """
        # Generate uniform finite volume mesh
        self.generate_mesh(mesh_num)

        # Initialize temperature field with given distribution
        self.init_temp(Tfunc_init)
        
        # Set boundary conditions (Dirichlet at R, Neumann at axis)
        self.set_boundary_temp(Tb)

        # Perform time integration and return results
        t_list, g_list, x_list, T_array = self.solve(
            dt, step_num, 
            sweep_max_num, sweep_res_tol, is_print_sweep,
            enable_joule, save_freq, is_print, flag
        )

        return t_list, g_list, x_list, T_array


class TraArc1D(StaArc1D):
    """One-dimensional transient arc plasma model with radial velocity (full convection model).

    This class extends StaArc1D to solve time-dependent arc plasma problems including
    radial gas flow. The model accounts for both thermal diffusion and convective heat
    transport, providing a more complete description of arc dynamics compared to TraArc1DNoV.
    This is essential for modeling arc decay when pressure gradients drive significant
    radial flow, such as during current interruption in high-pressure circuit breakers.

    Attributes
    ----------
    I : float
        Arc current in Amperes (can be zero for decay studies)
    R : float
        Arc radius in meters
    temp_list : ndarray
        Temperature values for property tables (K)
    rho_list : ndarray
        Mass density values (kg/m³)
    Cp_list : ndarray
        Specific heat capacity values (J/(kg·K))
    sigma_list : ndarray
        Electrical conductivity values (S/m)
    kappa_list : ndarray
        Thermal conductivity values (W/(m·K))
    nec_list : ndarray
        Net emission coefficient values (W/m³)
    mesh : fipy.Grid1D
        Finite volume mesh for spatial discretization
    T : fipy.CellVariable
        Temperature field variable with time history
    V : fipy.CellVariable
        Radial velocity field variable with time history
    """

    def __init__(self, I, R, prop):
        """Initialize the 1D transient arc model with radial velocity.

        Parameters
        ----------
        I : float
            Arc current in Amperes. Set to 0 for pure decay simulations
            (no Joule heating). Typical range: 0 - 10000 A
        R : float
            Arc radius in meters. Should remain constant during simulation.
            Typical range: 1e-4 - 1e-2 m
        prop : tuple
            Tuple of (temp_list, rho_list, Cp_list, sigma_list, kappa_list, nec_list)
            containing thermodynamic, transport, and radiation properties as
            functions of temperature. All arrays should have the same length.
        """
        self.I = I
        self.R = R
        self.temp_list, self.rho_list, self.Cp_list, self.sigma_list, \
            self.kappa_list, self.nec_list = prop
    
    def init_field(self, Tfunc_init):
        """Initialize both temperature and velocity field variables.

        Creates FiPy cell variables for temperature and velocity with initial
        conditions. Both fields are configured to store old values for time
        integration schemes.

        Parameters
        ----------
        Tfunc_init : callable
            Initial temperature distribution function: T = f(r)
            Should return temperature in Kelvin for given radial position.
            Example: lambda r: T_center * (1 - (r/R)**2) + T_boundary
        """
        self.T = fipy.CellVariable(
            name="Temperature", 
            mesh=self.mesh, 
            value=Tfunc_init(self.x), 
            hasOld=True
        )
        self.V = fipy.CellVariable(
            name="Velocity", 
            mesh=self.mesh, 
            value=0, 
            hasOld=True
        )

    def set_boundary_field(self, Tb):
        """Set boundary conditions for temperature and velocity fields.

        Temperature boundary conditions:
        - At r=R (right boundary): Dirichlet BC, T = Tb (fixed wall temperature)
        - At r=0 (axis): Neumann BC, dT/dr = 0 (symmetry condition)

        Velocity boundary conditions:
        - At r=0 (axis): V = 0 (symmetry, no flow through axis)
        - At r=R (right boundary): Free (determined by temperature gradient)

        Parameters
        ----------
        Tb : float
            Boundary temperature at r=R in Kelvin. Typically represents
            wall temperature or ambient gas temperature.
            Typical range: 300 - 3000 K
        """
        # Temperature: Fixed at boundary, symmetric at axis
        self.T.constrain(Tb, self.mesh.facesRight)
        self.T.faceGrad.constrain(0, self.mesh.facesLeft)
        
        # Velocity: Zero at axis (symmetry)
        self.V.constrain(0, self.mesh.facesLeft)

    def set_solver(self):
        """Configure the solver for coupled transient equations.

        For the coupled temperature-velocity problem, solver configuration
        is handled within the solve() method. This placeholder is maintained
        for interface consistency with parent classes.
        """
        pass

    def solve(self, dt, step_num,
              sweep_T_max_num=100, sweep_T_res_tol=1e-6, is_print_sweep=False,
              enable_joule=False,
              save_freq=10, is_print=True, flag=''):
        """Solve the coupled transient temperature-velocity equations using implicit method.

        This method performs time integration of the coupled energy and momentum
        equations using a semi-implicit approach with decoupled iterations. At each
        time step, velocity is first computed from the current temperature field,
        then temperature is updated with sweep iterations accounting for convection.

        **IMPORTANT**: This implicit method may have convergence issues for some
        conditions. The explicit method (solve_explicit) is generally more stable
        and is recommended for most applications.

        Parameters
        ----------
        dt : float
            Time step size in seconds. Should be small enough for numerical
            stability of the convection term. Typical: 1e-10 to 1e-7 s
        step_num : int
            Total number of time steps to simulate.
            Total simulation time = dt * step_num
        sweep_T_max_num : int, optional
            Maximum number of sweep iterations for temperature equation per
            time step. Default: 100. May need larger values for strong coupling.
        sweep_T_res_tol : float, optional
            Convergence tolerance for temperature sweep iterations (RMS change).
            Default: 1e-6 K. Controls accuracy vs computational cost.
        is_print_sweep : bool, optional
            If True, print detailed information for each sweep iteration.
            Default: False. Useful for debugging convergence issues.
        enable_joule : bool, optional
            If True, include Joule heating in the energy balance. If False,
            model pure cooling/decay without energy input. Default: False.
        save_freq : int, optional
            Frequency of saving solution snapshots (every N time steps).
            Default: 10. Balance between time resolution and memory usage.
        is_print : bool, optional
            If True, print progress at each saved time step. Default: True.
        flag : str, optional
            Identifier string for output messages. Default: ''.

        Returns
        -------
        tuple: (t_list, g_list, x_list, T_array, V_array) where:
            t_list : ndarray, time points where solution is saved (s)
            g_list : ndarray, arc conductance at saved times (S)
            x_list : ndarray, radial positions (face centers) (m)
            T_array : ndarray (2D), temperature profiles at saved times (K), Shape: (num_saved_steps, num_cells)
            V_array : ndarray (2D), velocity profiles at saved times (m/s), Shape: (num_saved_steps, num_cells)

        Warnings
        --------
        This method has known convergence challenges and is retained mainly for
        compatibility. For production use, prefer solve_explicit() which is more
        robust and often faster.
        """
        # Initialize storage arrays for time history
        # Number of saved snapshots based on save frequency
        N = step_num // save_freq + 1
        t_list = np.zeros(N)          # Time points (s)
        g_list = np.zeros(N)          # Arc conductance (S)
        T_array = np.zeros((N, len(self.T.faceValue.value)))  # Temperature (K)
        V_array = np.zeros_like(T_array)  # Velocity (m/s)
        
        # Calculate initial arc conductance
        arc_cond = self.calc_arc_cond(
            self.temp_list, self.sigma_list, 
            self.xface.value, self.T.faceValue, self.R
        )
        
        # Save initial state (t=0)
        t, save_i = 0, 0
        t_list[save_i] = t
        g_list[save_i] = arc_cond
        T_array[save_i, :] = self.T.faceValue.value
        V_array[save_i, :] = self.V.faceValue.value

        # Create cubic spline interpolators for all properties
        # These provide smooth property variations and derivatives
        rho_func = interpolate.CubicSpline(self.temp_list, self.rho_list)
        Cp_func = interpolate.CubicSpline(self.temp_list, self.Cp_list)
        kappa_func = interpolate.CubicSpline(self.temp_list, self.kappa_list)
        sigma_func = interpolate.CubicSpline(self.temp_list, self.sigma_list)
        nec_func = interpolate.CubicSpline(self.temp_list, self.nec_list)

        # Time stepping loop
        for i in range(step_num):
            # Update current time
            t += dt

            # Store previous time step values
            self.T.updateOld()
            # Note: V.updateOld() not used as velocity is computed explicitly

            ######################### VELOCITY COMPUTATION #########################
            # Compute radial velocity from current temperature distribution
            # Velocity satisfies: dV/dr = f(T, dT/dr, d²T/dr²)
            
            # Get temperature at face centers (including boundary handling)
            T_face = self.T.faceValue.value
            T_face[0] = T_face[1]  # Apply Neumann BC at r=0 for stability
            
            # Compute Joule heating if enabled
            if enable_joule:
                arc_cond = self.calc_arc_cond(
                    self.temp_list, self.sigma_list, 
                    self.xface, T_face, self.R
                )
                joule_energy = sigma_func(T_face) * (self.I / arc_cond) ** 2
            else:
                joule_energy = 0
            
            # Radiation loss: 4π * NEC
            rad_energy = 4 * np.pi * nec_func(T_face)
            
            # Evaluate properties at face temperatures
            rho = rho_func(T_face)
            Cp = Cp_func(T_face)
            kappa = kappa_func(T_face)
            
            # Create temperature interpolator for derivative calculation
            T_func = interpolate.CubicSpline(self.xface, T_face)
            
            # Calculate temperature derivatives needed for velocity equation
            drho_dT = rho_func.derivative()(T_face)      # Density temperature sensitivity
            dkappa_dT = kappa_func.derivative()(T_face)  # Conductivity temperature sensitivity
            dT_dr = T_func.derivative()(self.xface)      # First spatial derivative
            d2T_dr2 = T_func.derivative(nu=2)(self.xface)  # Second spatial derivative

            # Compute thermal diffusion contribution to velocity
            # dV = kappa*dT/dr + r*dkappa/dT*(dT/dr)² + r*kappa*d²T/dr²
            dV = kappa * dT_dr + self.xface * dkappa_dT * dT_dr**2 + \
                 self.xface * kappa * d2T_dr2
            
            # Right-hand side of velocity equation
            # fV = -1/(rho²*Cp) * drho/dT * [r*(J - R) + d(r*kappa*dT/dr)/dr]
            fV = -drho_dT / (rho * rho * Cp) * \
                 (self.xface * (joule_energy - rad_energy) + dV)

            # Integrate velocity from axis (V=0 at r=0) to boundary
            # V(r) = ∫₀ʳ fV(r') dr' / r
            V = np.zeros_like(self.V.faceValue.value)
            V[1:] = integrate.cumulative_trapezoid(fV, self.xface) / self.xface[1:]
            self.V.faceValue.setValue(V)

            ######################### TEMPERATURE SWEEP ITERATIONS #########################
            # Solve temperature equation with decoupled iterations
            for j in range(sweep_T_max_num):
                # Store temperature at start of sweep for convergence check
                T_last_value = self.T.faceValue.value.copy()

                ######################### TEMPERATURE EQUATION #########################
                # Compute energy source terms at cell centers
                if enable_joule:
                    # Joule heating: sigma * E²
                    arc_cond = self.calc_arc_cond(
                        self.temp_list, self.sigma_list, 
                        self.xface.value, self.T.faceValue, self.R
                    )
                    joule_energy = interp_prop(
                        self.temp_list, self.sigma_list, self.T.value
                    ) * (self.I / arc_cond) ** 2
                else:
                    joule_energy = 0
                
                # Radiation loss: 4π * NEC (use log interpolation)
                rad_energy = 4 * np.pi * interp_prop_log(
                    self.temp_list, self.nec_list, self.T.value
                )

                # Transient term coefficient: r * rho * Cp
                rho = interp_prop(self.temp_list, self.rho_list, self.T.value)
                rho = fipy.CellVariable(mesh=self.mesh, value=rho)
                Cp = interp_prop(self.temp_list, self.Cp_list, self.T.value)
                Cp = fipy.CellVariable(mesh=self.mesh, value=Cp)
                r_rho_Cp = self.x * rho * Cp
                T_tra_term = fipy.TransientTerm(coeff=r_rho_Cp, var=self.T)

                # Convection term: rho * Cp * V * dT/dr (in cylindrical coords)
                # Coefficient: r * rho * Cp * V evaluated at faces
                rho_face = interp_prop(self.temp_list, self.rho_list, self.T.faceValue.value)
                Cp_face = interp_prop(self.temp_list, self.Cp_list, self.T.faceValue.value)
                temp_coef = self.xface * rho_face * Cp_face * self.V.faceValue.value
                temp_coef = np.expand_dims(temp_coef.value, axis=0)
                temp_coef = fipy.FaceVariable(mesh=self.mesh, value=temp_coef)
                T_conv_term = fipy.ConvectionTerm(coeff=temp_coef, var=self.T)

                # Diffusion term: div(kappa * grad(T)) in cylindrical coords
                # Coefficient: r * kappa evaluated at faces
                kappa_face = interp_prop(self.temp_list, self.kappa_list, self.T.faceValue.value)
                temp_diff = self.xface * kappa_face
                T_diff_term = fipy.DiffusionTerm(coeff=temp_diff, var=self.T)

                # Source term: r * (Joule heating - Radiation)
                T_src_term = self.x * fipy.CellVariable(
                    mesh=self.mesh, value=(joule_energy - rad_energy)
                )

                # Assemble and solve temperature equation
                # rho*Cp*dT/dt + rho*Cp*V*dT/dr = div(kappa*grad(T)) + S_joule - S_rad
                equ_T = (T_tra_term + T_conv_term) == (T_diff_term + T_src_term)
                
                # Solve with preconditioned conjugate gradient
                equ_T.sweep(
                    var=self.T, dt=dt, 
                    solver=fipy.LinearPCGSolver(tolerance=1e-10, iterations=1000)
                )

                # Calculate sweep convergence metric
                res_T = np.sqrt(
                    np.sum((self.T.faceValue.value - T_last_value) ** 2) / 
                    self.T.faceValue.value.size
                )
                
                # Print detailed sweep information if requested
                if is_print_sweep:
                    print('%s: Time step = %d, Sweep_T = %d, Residual = %g K' 
                          % (flag, i + 1, j + 1, res_T))
                
                # Check sweep convergence
                if res_T < sweep_T_res_tol:
                    break
                
                # Warn if maximum sweeps reached
                if (j + 1) == sweep_T_max_num:
                    print('%s: Warning - Maximum sweep number (%d) reached for T at step %d!' 
                          % (flag, sweep_T_max_num, i + 1))
            
            # Save solution at specified frequency
            if (i + 1) % save_freq == 0:
                save_i += 1
                
                # Recalculate arc conductance for saved state
                arc_cond = self.calc_arc_cond(
                    self.temp_list, self.sigma_list, 
                    self.xface.value, self.T.faceValue, self.R
                )
                
                # Store time, conductance, and field profiles
                t_list[save_i] = t
                g_list[save_i] = arc_cond
                T_array[save_i, :] = self.T.faceValue.value
                V_array[save_i, :] = self.V.faceValue.value

                # Print progress
                if is_print:
                    print('%s: t = %.6e s, save_index = %d - Saved!' 
                          % (flag, t, save_i))
        
        return t_list, g_list, self.xface.value, T_array, V_array

    def solve_onestep(self,
                      mesh_num, Tfunc_init, Tb,
                      dt, step_num,
                      sweep_T_max_num=100, sweep_T_res_tol=1e-6, is_print_sweep=False,
                      enable_joule=False, save_freq=10, is_print=True, flag=''):
        """Complete solution procedure with velocity: setup and time integration.

        This convenience method combines all steps needed to solve the coupled
        transient arc problem including radial velocity effects:
        1. Generate computational mesh
        2. Initialize temperature and velocity fields
        3. Set boundary conditions
        4. Perform coupled time integration

        Parameters
        ----------
        mesh_num : int
            Number of mesh cells. Typical: 200-1000. More cells needed
            for capturing steep velocity gradients.
        Tfunc_init : callable
            Initial temperature distribution function: T = f(r)
            Should return temperature in Kelvin for given radial position.
            Example: lambda r: T0 * np.exp(-(r/r0)**2) + Tb
        Tb : float
            Boundary temperature at r=R (K). Typically ambient or wall
            temperature. Typical: 300-3000 K
        dt : float
            Time step size (s). Must be small for convection stability.
            Typical: 1e-10 to 1e-7 s (smaller than TraArc1DNoV due to convection)
        step_num : int
            Total number of time steps. Total time = dt * step_num
        sweep_T_max_num : int, optional
            Maximum sweep iterations for temperature per time step. Default: 100
        sweep_T_res_tol : float, optional
            Temperature sweep convergence tolerance (K). Default: 1e-6
        is_print_sweep : bool, optional
            Print detailed sweep information. Default: False
        enable_joule : bool, optional
            Include Joule heating. Default: False (decay only)
        save_freq : int, optional
            Save solution every N time steps. Default: 10
        is_print : bool, optional
            Print time step progress. Default: True
        flag : str, optional
            Identifier for output messages. Default: ''

        Returns
        -------
        tuple
            (t_list, g_list, x_list, T_array, V_array) - Complete time history
            including both temperature and velocity fields
        """
        # Generate uniform finite volume mesh
        self.generate_mesh(mesh_num)

        # Initialize temperature and velocity fields
        self.init_field(Tfunc_init)
        
        # Set boundary conditions for both fields
        self.set_boundary_field(Tb)

        # Set solver configuration (if needed)
        self.set_solver()

        # Perform coupled time integration
        t_list, g_list, x_list, T_array, V_array = self.solve(
            dt, step_num, 
            sweep_T_max_num, sweep_T_res_tol, is_print_sweep,
            enable_joule, save_freq, is_print, flag
        )

        return t_list, g_list, x_list, T_array, V_array

    def solve_explicit(self, mesh_num, Tfunc_init, Tb, dt, step_num,
                      enable_joule=False,
                      save_freq=10, is_print=True, flag=''):
        """Explicitly solve the coupled temperature-velocity equations (RECOMMENDED METHOD).

        This method implements a fully explicit time integration scheme that is
        generally more stable, faster, and easier to use than the implicit method.
        The velocity and temperature fields are updated sequentially at each time
        step using direct finite difference formulas without iterative sweeps.

        **RECOMMENDED**: This is the preferred method for solving the coupled
        arc equations with velocity. It avoids convergence issues of the implicit
        solver while maintaining good accuracy.

        Parameters
        ----------
        mesh_num : int
            Number of mesh points. Typical: 200-1000. Finer mesh improves
            accuracy but reduces stable time step.
        Tfunc_init : callable
            Initial temperature distribution function: T = f(r)
            Should return temperature in Kelvin for given radial position.
            Example: lambda r: T_center * (1 - (r/R)**2) + T_boundary
        Tb : float
            Boundary temperature at r=R (K). Fixed during simulation.
            Typical: 300-3000 K (wall or ambient temperature)
        dt : float
            Time step size (s). Must satisfy CFL condition for stability.
            Typical: 1e-10 to 1e-7 s. Adjust based on max(V/dx, kappa/(rho*Cp*dx²))
        step_num : int
            Total number of time steps. Total time = dt * step_num
        enable_joule : bool, optional
            If True, include Joule heating (current-carrying arc).
            If False, pure cooling/decay without energy input.
            Default: False (decay studies)
        save_freq : int, optional
            Save solution every N time steps. Default: 10
            Balance between time resolution and memory/disk usage
        is_print : bool, optional
            Print progress information at saved steps. Default: True
        flag : str, optional
            Identifier string for output messages. Default: ''

        Returns
        -------
        tuple: (t_list, g_list, x, T_array, V_array) where:
            t_list : ndarray, time points where solution is saved (s)
            g_list : ndarray, arc conductance at saved times (S)
            x : ndarray, radial grid points (m)
            T_array : ndarray (2D), temperature profiles at saved times (K), Shape: (num_saved_steps, mesh_num)
            V_array : ndarray (2D), velocity profiles at saved times (m/s), Shape: (num_saved_steps, mesh_num)
        """
        ### Initialize mesh ###
        # Create uniform radial grid from axis (r=0) to boundary (r=R)
        x = np.linspace(0, self.R, num=mesh_num, endpoint=True)
    
        ### Initialize fields ###
        # Temperature from initial condition function
        T = Tfunc_init(x)
        # Velocity starts at zero
        V = np.zeros_like(T)

        ### Initialize storage arrays ###
        N = step_num // save_freq + 1
        t_list = np.zeros(N)          # Time points (s)
        g_list = np.zeros(N)          # Arc conductance (S)
        T_array = np.zeros((N, len(T)))  # Temperature profiles (K)
        V_array = np.zeros_like(T_array)  # Velocity profiles (m/s)
        
        # Calculate and save initial state (t=0)
        arc_cond = self.calc_arc_cond(self.temp_list, self.sigma_list, x, T, self.R)
        t, save_i = 0, 0
        t_list[save_i] = t
        g_list[save_i] = arc_cond
        T_array[save_i, :] = T
        V_array[save_i, :] = V

        ### Create property interpolators ###
        # Cubic splines provide smooth interpolation and easy derivative calculation
        rho_func = interpolate.CubicSpline(self.temp_list, self.rho_list)
        Cp_func = interpolate.CubicSpline(self.temp_list, self.Cp_list)
        kappa_func = interpolate.CubicSpline(self.temp_list, self.kappa_list)
        sigma_func = interpolate.CubicSpline(self.temp_list, self.sigma_list)
        nec_func = interpolate.CubicSpline(self.temp_list, self.nec_list)

        ### Time stepping loop ###
        for i in range(step_num):
            # Update current time
            t += dt

            # Apply Neumann boundary condition at axis (symmetry)
            # dT/dr = 0 at r=0, implemented as T[0] = T[1]
            T[0] = T[1]

            ######################### ENERGY SOURCE TERMS #########################
            # Compute Joule heating if enabled
            if enable_joule:
                # Recalculate arc conductance with current temperature
                arc_cond = self.calc_arc_cond(
                    self.temp_list, self.sigma_list, x, T, self.R
                )
                # Joule heating: sigma * E² = sigma * (I/G)²
                joule_energy = sigma_func(T) * (self.I / arc_cond) ** 2
            else:
                joule_energy = 0
            
            # Radiation loss: 4π * NEC (integrated over solid angle)
            rad_energy = 4 * np.pi * nec_func(T)
            
            ######################### PROPERTY EVALUATION #########################
            # Evaluate all properties at current temperature
            rho = rho_func(T)          # Mass density (kg/m³)
            Cp = Cp_func(T)            # Specific heat (J/(kg·K))
            kappa = kappa_func(T)      # Thermal conductivity (W/(m·K))
            
            # Create temperature spline for derivative calculation
            T_func = interpolate.CubicSpline(x, T)
            
            ######################### VELOCITY COMPUTATION #########################
            # Calculate temperature derivatives for velocity equation
            drho_dT = rho_func.derivative()(T)      # ∂ρ/∂T
            dkappa_dT = kappa_func.derivative()(T)  # ∂κ/∂T
            dT_dr = T_func.derivative()(x)          # dT/dr (first derivative)
            d2T_dr2 = T_func.derivative(nu=2)(x)    # d²T/dr² (second derivative)

            # Compute thermal diffusion contribution to velocity equation
            # This represents d/dr(r * kappa * dT/dr) expanded:
            # = kappa * dT/dr + r * dkappa/dT * (dT/dr)² + r * kappa * d²T/dr²
            dV = kappa * dT_dr + x * dkappa_dT * dT_dr**2 + x * kappa * d2T_dr2
            
            # Right-hand side of velocity ODE:
            # dV/dr = -1/(rho²*Cp) * drho/dT * [r*(J - R) + d(r*kappa*dT/dr)/dr]
            fV = -drho_dT / (rho * rho * Cp) * (x * (joule_energy - rad_energy) + dV)

            # Integrate velocity from axis to boundary
            # V(r) = ∫₀ʳ fV(r') dr' / r
            # At r=0, V=0 (symmetry condition)
            V[1:] = integrate.cumulative_trapezoid(fV, x) / x[1:]
            V[0] = 0  # Ensure exact zero at axis

            ######################### TEMPERATURE UPDATE #########################
            # Compute spatial derivatives for temperature equation
            # Thermal diffusion term in cylindrical coordinates:
            # 1/r * d/dr(r * kappa * dT/dr) = dkappa/dT * (dT/dr)² + kappa * d²T/dr²
            dT_diffusion = dkappa_dT * dT_dr**2 + kappa * d2T_dr2
            
            # Add 1/r factor for off-axis points
            dT_diffusion[x > 0] = dT_diffusion[x > 0] + kappa[x > 0] * dT_dr[x > 0] / x[x > 0]

            # Total temperature rate of change:
            # dT/dt = -V * dT/dr + [(J - R) + diffusion_term] / (rho * Cp)
            # where:
            # - First term: advection by velocity
            # - Second term: energy balance (sources + diffusion)
            delta_T = -V * dT_dr + ((joule_energy - rad_energy) + dT_diffusion) / (rho * Cp)
            
            # Explicit Euler time integration
            T += delta_T * dt
            
            # Apply Dirichlet boundary condition at r=R
            T[-1] = Tb

            ######################### SAVE SOLUTION #########################
            if (i + 1) % save_freq == 0:
                save_i += 1
                
                # Recalculate arc conductance for saved state
                arc_cond = self.calc_arc_cond(
                    self.temp_list, self.sigma_list, x, T, self.R
                )
                
                # Store time, conductance, and field profiles
                t_list[save_i] = t
                g_list[save_i] = arc_cond
                T_array[save_i, :] = T
                V_array[save_i, :] = V

                # Print progress information
                if is_print:
                    print('%s: t = %.6e s, save_index = %d, T_max = %.2f K, V_max = %.2e m/s - Saved!' 
                          % (flag, t, save_i, T.max(), np.abs(V).max()))
        
        return t_list, g_list, x, T_array, V_array


    

