import numpy as np


from numcosmo_II import cosmology
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

#EDS OM_M=1;OM_DE=0
#FLAT ACDM OM_M=0.3;OM_DE=0.7

#Flat ACDM H_N/N=-3OM_M/2

#EdS, AH=1/2

class SphericalCollapse:
    """
    Spherical-collapse models and auxiliary calculations.

    Parameters
    ----------
    cosmology : object
        Cosmology instance providing Ha(a) and a_dep_O_matter(a).
    """

    def __init__(self, cosmology):
        self.cosmology = cosmology

    def compute_theta(self, lpt, a, H):
        """
        Compute theta = div(v) / (a * H).

        The LPT class computes x_dot = dx/dt.
        Therefore, the peculiar velocity is v = a * x_dot.
        """

        vx = a * lpt.x_dot_x
        vy = a * lpt.x_dot_y
        vz = a * lpt.x_dot_z

        # Fourier transforms
        vx_k = np.fft.fftn(vx)
        vy_k = np.fft.fftn(vy)
        vz_k = np.fft.fftn(vz)

        # Fourier-space divergence
        div_v_k = 1j * (
            lpt.kx * vx_k
            + lpt.ky * vy_k
            + lpt.kz * vz_k
        )

        # Return to real space
        div_v = np.fft.ifftn(div_v_k).real

        return div_v / (a * H)

    @staticmethod
    def compute_g(delta, eps=1e-10):
        """Compute g = 1/delta where delta is nonzero."""

        g = np.full_like(delta, np.nan, dtype=float)
        valid = np.abs(delta) > eps
        g[valid] = 1.0 / delta[valid]

        return g

    def background_terms(self, N):
        """
        Compute background quantities at N = ln(a).

        Returns
        -------
        a : float
            Scale factor.
        OM_M : float
            Matter density parameter.
        A_H : float
            2 + dln(H)/dN.
        """

        a = np.exp(N)
        OM_M = self.cosmology.a_dep_O_matter(a)

        eps = 1e-5
        dlnH_dN = (
            np.log(self.cosmology.Ha(a * np.exp(eps)))
            - np.log(self.cosmology.Ha(a * np.exp(-eps)))
        ) / (2.0 * eps)

        A_H = 2.0 + dlnH_dN

        return a, OM_M, A_H

    def Model_A(self, N, y):
        """
        Model A: second-order density equation written
        as a first-order system.

        y[0] = delta
        y[1] = delta_N
        N = ln(a)
        """

        delta, delta_N = y
        a, OM_M, A_H = self.background_terms(N)

        delta_N_new = delta_N

        delta_NN = (
            -A_H * delta_N
            + (4.0 / 3.0) * delta_N**2 / (1.0 + delta)
            + 1.5 * OM_M * delta * (1.0 + delta)
        )

        return [delta_N_new, delta_NN]

    def Model_B(self, N, y):
        """
        Model B: continuity and velocity-divergence equations.

        y[0] = delta
        y[1] = theta
        N = ln(a)
        """

        delta, theta = y
        a, OM_M, A_H = self.background_terms(N)

        delta_N = -(1.0 + delta) * theta

        theta_N = (
            -A_H * theta
            - theta**2 / 3.0
            - 1.5 * OM_M * delta
        )

        return [delta_N, theta_N]

    def Model_C(self, N, y):
        """
        Model C: first-order system.

        y[0] = g
        y[1] = theta
        N = ln(a)
        """

        g, theta = y
        a, OM_M, A_H = self.background_terms(N)

        g_N = g * (1.0 + g) * theta

        theta_N = (
            -A_H * theta
            - theta**2 / 3.0
            - 1.5 * OM_M * (1.0 / (g))
        )

        return [g_N, theta_N]
    
    def integrate(self, ai, af, delta_i, f_i, n_points=1000):
      """Integrate Models A, B, and C from growing-mode initial conditions."""

      from scipy.integrate import solve_ivp

      if ai <= 0 or af <= ai:
        raise ValueError("Require 0 < ai < af.")

      if delta_i <= 0:
        raise ValueError("delta_i must be positive.")

      N_ini = np.log(ai)
      N_fin = np.log(af)
      N_eval = np.linspace(N_ini, N_fin, n_points)

      theta_i = -f_i * delta_i / (1.0 + delta_i)
      g_i = 1.0 / delta_i

      initial_conditions = {
        "A": [delta_i, f_i * delta_i],
        "B": [delta_i, theta_i],
        "C": [g_i, theta_i],}

      models = {"A": self.Model_A,
                "B": self.Model_B, 
                "C": self.Model_C,}

      solutions = {}
      for name, model in models.items():
        sol = solve_ivp(
            model,
            (N_ini, N_fin),
            initial_conditions[name],
            t_eval=N_eval,
            rtol=1e-8,
            atol=1e-10,
        )

        if not sol.success:
            raise RuntimeError(
                f"Model {name} integration failed: {sol.message}"
            )

        solutions[name] = sol

      return {
        "N": N_eval,
        "a": np.exp(N_eval),
        "theta_i": theta_i,
        "g_i": g_i,
        "f_i": f_i,
        "Model_A": solutions["A"],
        "Model_B": solutions["B"],
        "Model_C": solutions["C"],
    }

class CollapseThreshold:
    """Find the initial overdensity and linear collapse threshold."""

    def __init__(
        self,
        collapse,
        model="A",
        ai=0.01,
        delta_stop=1e4,
        f_i=1.0,
        n_points=1000,
        delta_i_min=1e-8,
        delta_i_max=1e-2,
        tol=1e-10,
        max_iter=100,
    ):
        self.collapse = collapse
        self.model = model.upper()
        self.ai = ai
        self.delta_stop = delta_stop
        self.f_i = f_i
        self.n_points = n_points
        self.delta_i_min = delta_i_min
        self.delta_i_max = delta_i_max
        self.tol = tol
        self.max_iter = max_iter

        if self.model not in ("A", "B", "C"):
            raise ValueError("model must be 'A', 'B', or 'C'.")

        if ai <= 0 or delta_stop <= 0 or n_points <= 0:
            raise ValueError("ai, delta_stop, and n_points must be positive.")

        self.rhs = getattr(collapse, f"Model_{self.model}")

    def initial_conditions(self, delta_i):
        """Return the initial conditions for the selected model."""
        theta_i = -self.f_i * delta_i / (1.0 + delta_i)

        if self.model == "A":
            return [delta_i, self.f_i * delta_i]
        elif self.model == "B":
            return [delta_i, theta_i]
        else:
            return [1.0 / delta_i, theta_i]

    def collapse_event(self, N, y):
        """Define the collapse event for the selected model."""
        if self.model in ("A", "B"):
            return y[0] - self.delta_stop

        return y[0] - 1.0 / self.delta_stop
        
    def linear_growth_equation(self, N, y):
        """Linear growth equation in N = ln(a)."""
        
        delta_L, delta_L_N = y
        a, OM_M, A_H = self.collapse.background_terms(N)
        
        delta_L_NN = (
            -A_H * delta_L_N
            + 1.5 * OM_M * delta_L
        )
    
        return [delta_L_N, delta_L_NN]

        
    def collapses_by_target(self, delta_i, N_ini, N_target):
        """Check whether collapse occurs by the target epoch."""
    
        if self.model in ("A", "B"):
            def collapse_event(N, y):
                return y[0] - self.delta_stop
        
            collapse_event.direction = 1
        
        else:  # Model C: g = 1 / delta
            epsilon = 1.0 / self.delta_stop
        
            def collapse_event(N, y):
                return y[0] - epsilon
        
            collapse_event.direction = -1
        
        collapse_event.terminal = True
        
        sol = solve_ivp(
            self.rhs,
            (N_ini, N_target),
            self.initial_conditions(delta_i),
            events=collapse_event,
            rtol=1e-8,
            atol=1e-10,
            max_step=(N_target - N_ini) / self.n_points,)
        
        if not sol.success and len(sol.t_events[0]) == 0:
            raise RuntimeError(sol.message)
    
        return len(sol.t_events[0]) > 0


    def find(self, z_c):
        """Find delta_i and delta_c for a target collapse redshift."""
        ac = 1.0 / (1.0 + z_c)

        if ac <= self.ai:
            raise ValueError("The target scale factor must exceed ai.")

        N_ini = np.log(self.ai)
        N_target = np.log(ac)

        lo = self.delta_i_min
        hi = self.delta_i_max

        # Find a lower bound that does not collapse by the target.
        for _ in range(60):
            if not self.collapses_by_target(lo, N_ini, N_target):
                break
            lo *= 0.5
        else:
            raise RuntimeError("Could not find a lower non-collapsing bound.")

        # Find an upper bound that collapses by the target.
        for _ in range(60):
            if self.collapses_by_target(hi, N_ini, N_target):
                break
            hi *= 2.0
        else:
            raise RuntimeError("Could not find an upper collapsing bound.")

        # Bisection.
        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)

            if self.collapses_by_target(mid, N_ini, N_target):
                hi = mid
            else:
                lo = mid

            if hi - lo <= self.tol * max(abs(mid), 1e-15):
                break

        delta_i = 0.5 * (lo + hi)

        # Integrate the linear perturbation to the target epoch.
        linear = solve_ivp(
            self.linear_growth_equation,
            (N_ini, N_target),
            [delta_i, self.f_i * delta_i],
            t_eval=[N_target],
            rtol=1e-10,
            atol=1e-12,)

        if not linear.success:
            raise RuntimeError(linear.message)

        delta_c = linear.y[0, -1]

        return delta_i, delta_c