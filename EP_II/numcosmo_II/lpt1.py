import numpy as np
from numcosmo_II import cosmology
from numcosmo_II import gaussian_field

class LPT:

    def __init__(
        self,
        L=100,
        Ng=128,
        aref=1.0,
        h=0.67,
        Omega_m=0.315,
        Omega_lambda=0.685,
        Omega_r=0.0,
        w0_fld=-1.0,
        wa_fld=0.0,
        a_ini=0.01,
        ai=1.0):

        self.L = L
        self.Ng = Ng
        self.aref = aref

        self.GF = gaussian_field.GaussianField(
            L=L,
            Ng=Ng,
            aref=aref
        )

        self.cosmo = cosmology.FLRW(
            h=h,
            Omega_m=Omega_m,
            Omega_lambda=Omega_lambda,
            Omega_r=Omega_r,
            w0_fld=w0_fld,
            wa_fld=wa_fld)

        self.a_ini = a_ini
        self.ai = ai


    def create_gaussian_field(self, seed=42):
      self.delta_x, self.delta_ref_k = self.GF.Gaussian_field(seed=seed)
      self.kx, self.ky, self.kz, self.k = self.GF.normal_modes(self.GF.L, self.GF.V, self.GF.Ng)
        
      return self.delta_x, self.delta_ref_k


    def calculate_potential(self):

        self.phi_k = np.zeros_like(self.delta_ref_k)

        mask = self.k != 0

        self.phi_k[mask] = -self.delta_ref_k[mask] / self.k[mask]**2

        return self.phi_k
        
    def growth(self, n_points=1000, rtol=1e-8, atol=1e-10):
      a_grid = np.linspace(self.a_ini, self.ai, n_points)
      sol = self.cosmo.solve_growth(a=a_grid, y0=[self.a_ini, 1.0], rtol=rtol, atol=atol)

      D_at_ai, f_at_ai = self.cosmo.get_growth_observables(sol, a_eval=np.array([self.ai]))
      D_at_ref, _      = self.cosmo.get_growth_observables(sol, a_eval=np.array([self.aref]))

      self.D1_ai = D_at_ai[0] / D_at_ref[0]
      self.f1_i  = f_at_ai[0]

      return self.D1_ai, self.f1_i


    def displacement_fourier(self):

        D_1=self.D1_ai

        self.Psi_x_k = np.zeros_like(self.delta_ref_k)
        self.Psi_y_k = np.zeros_like(self.delta_ref_k)
        self.Psi_z_k = np.zeros_like(self.delta_ref_k)

        mask = self.k != 0

        self.Psi_x_k[mask] = (
            1j * D_1 * self.kx[mask]
            / self.k[mask]**2
            * self.delta_ref_k[mask]
        )

        self.Psi_y_k[mask] = (
            1j * D_1 * self.ky[mask]
            / self.k[mask]**2
            * self.delta_ref_k[mask]
        )

        self.Psi_z_k[mask] = (
            1j * D_1 * self.kz[mask]
            / self.k[mask]**2
            * self.delta_ref_k[mask]
        )

        return self.Psi_x_k, self.Psi_y_k, self.Psi_z_k


    def displacement_real_space(self):

        factor = self.GF.Ng**3 / self.GF.V

        self.Psi_x = np.fft.ifftn(factor * self.Psi_x_k).real

        self.Psi_y = np.fft.ifftn(factor * self.Psi_y_k).real

        self.Psi_z = np.fft.ifftn(factor * self.Psi_z_k).real

        return self.Psi_x, self.Psi_y, self.Psi_z


    def particle_grid(self):

        q1d = np.arange(self.GF.Ng) * (self.GF.L / self.GF.Ng)

        qx, qy, qz = np.meshgrid(
            q1d,
            q1d,
            q1d,
            indexing="ij"
        )

        self.q = np.column_stack([
            qx.ravel(),
            qy.ravel(),
            qz.ravel()
        ])

        return self.q


    def displacement_at_particles(self):

        self.Psi_q = np.column_stack([
            self.Psi_x.ravel(),
            self.Psi_y.ravel(),
            self.Psi_z.ravel()
        ])

        return self.Psi_q


    def particle_positions(self):

        self.x_p = (self.q + self.Psi_q) % self.GF.L

        return self.x_p


    def particle_velocities(self):

        H_i = self.cosmo.Ha(a=self.ai)
        self.v_p = self.ai * H_i * self.f1_i * self.Psi_q

        return self.v_p

    def run(self, seed=42):
      self.create_gaussian_field(seed=seed)  
      self.calculate_potential()
      self.growth()
      self.displacement_fourier()
      self.displacement_real_space()
      self.particle_grid()
      self.displacement_at_particles()
      self.particle_positions()
      self.particle_velocities()
      return self.x_p, self.v_p
        
def particles_to_mesh_ngp(lpt):
    """
    Deposit particles onto a periodic mesh using Nearest Grid Point (NGP).

    Parameters
    ----------
    lpt : LPT
        LPT simulation object containing particle positions x_p,
        box size L, and grid resolution Ng.

    Returns
    -------
    delta_mesh : ndarray
        Three-dimensional density contrast:
        delta = rho / mean(rho) - 1.
    """
    Ng = lpt.Ng
    L = lpt.L
    dx = L / Ng

    # Convert particle positions to grid coordinates
    grid_pos = (lpt.x_p % L) / dx

    # Assign each particle to the nearest grid point
    indices = np.rint(grid_pos).astype(np.int64) % Ng

    # Count particles in each grid cell
    mesh = np.zeros((Ng, Ng, Ng), dtype=np.float64)

    np.add.at(
        mesh,
        (indices[:, 0], indices[:, 1], indices[:, 2]),
        1.0
    )

    # Convert particle counts into density contrast
    mean_density = mesh.mean()
    delta_mesh = mesh / mean_density - 1.0

    return delta_mesh

def particles_to_mesh_cic(lpt):
    """
    Deposit particles onto a periodic mesh using Cloud-in-Cell (CIC).

    Each particle is distributed among its eight neighboring grid cells
    using weights determined by its fractional position.

    Parameters
    ----------
    lpt : LPT
        LPT simulation object containing particle positions x_p,
        box size L, and grid resolution Ng.

    Returns
    -------
    delta_mesh : ndarray
        Three-dimensional density contrast:
        delta = rho / mean(rho) - 1.
    """
    Ng = lpt.Ng
    L = lpt.L
    dx = L / Ng

    # Convert particle positions to grid coordinates
    grid_pos = (lpt.x_p % L) / dx

    # Index of the lower grid point and fractional coordinates
    i0 = np.floor(grid_pos).astype(np.int64)
    frac = grid_pos - i0

    mesh = np.zeros((Ng, Ng, Ng), dtype=np.float64)

    # Distribute each particle among its eight neighboring cells
    for ox in (0, 1):
        for oy in (0, 1):
            for oz in (0, 1):
                ix = (i0[:, 0] + ox) % Ng
                iy = (i0[:, 1] + oy) % Ng
                iz = (i0[:, 2] + oz) % Ng

                wx = frac[:, 0] if ox else 1.0 - frac[:, 0]
                wy = frac[:, 1] if oy else 1.0 - frac[:, 1]
                wz = frac[:, 2] if oz else 1.0 - frac[:, 2]

                weights = wx * wy * wz

                np.add.at(mesh, (ix, iy, iz), weights)

    # Convert particle counts to density contrast
    delta_mesh = mesh / mesh.mean() - 1.0

    return delta_mesh

def mass_assignment_window_3d(Ng, L, mass_assignment='cic'):

    dx = L / Ng
    k1d = 2 * np.pi * np.fft.fftfreq(Ng, d=dx)        # angular
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')

    # Ordem do B-spline: NGP=1, CIC=2, TSC=3 → o sinc fica elevado a "p" por eixo
    p = {'ngp': 1, 'cic': 2, 'tsc': 3}[mass_assignment]

    # sinc(x) = sin(πx)/(πx); aqui x = k·dx/(2π)
    Wx = np.sinc(kx * dx / (2 * np.pi)) ** p
    Wy = np.sinc(ky * dx / (2 * np.pi)) ** p
    Wz = np.sinc(kz * dx / (2 * np.pi)) ** p

    # |W(k)|² = (Wx · Wy · Wz)²
    return (Wx * Wy * Wz) ** 2

def power_spectrum(delta_x, lpt):

    Ng = lpt.Ng
    V  = lpt.GF.V
    delta_fft = np.fft.fftn(delta_x)
    delta_k   = (V / Ng**3) * delta_fft
    P         = np.abs(delta_k)**2 / V
    return lpt.k, P

def power_spectrum_deconvolved(delta_mesh, lpt, mass_assignment='cic',k_max_deconv=None, eps=1e-3):
   
    Ng, L, V = lpt.Ng, lpt.L, lpt.GF.V
    dx = L / Ng

    delta_fft = np.fft.fftn(delta_mesh)
    delta_k = (V / Ng**3) * delta_fft

    W_sq = mass_assignment_window_3d(Ng, L, mass_assignment)

    # Máscara: não deconvoluir em |W|² < eps (k ~ Nyq)
    if k_max_deconv is None:
        k_max_deconv = np.pi / dx / 2   # k_Nyq / 2

    k_mag = lpt.k
    W_sq_safe = np.where(
        (W_sq > eps) & (k_mag <= k_max_deconv),
        W_sq,
        1.0
    )


    P_deconv = np.abs(delta_k) ** 2 / V / W_sq_safe
    # Onde |W|² ≤ eps → NaN (descartar)
    P_deconv[(W_sq <= eps) | (k_mag > k_max_deconv)] = np.nan

    return k_mag, P_deconv


def bin_power_spectrum(k, power, nbins=20, P_theory=None):

    k_flat = k.ravel()
    power_flat = power.ravel()

    mask = k_flat > 0

    k_flat = k_flat[mask]
    power_flat = power_flat[mask]

    if P_theory is not None:
        theory_flat = np.broadcast_to(
            P_theory, k.shape
        ).ravel()[mask]

    if k_flat.size == 0:
        raise ValueError("There are no non-null modes for bin")

    if nbins < 1:
        raise ValueError("nbins must be positive.")

    bins = np.linspace(
        k_flat.min(),
        k_flat.max(),
        nbins + 1,
    )

    k_values = []
    P_values = []
    N_modes = []
    P_theory_values = []

    for i in range(nbins):

        if i == nbins - 1:
            mask_shell = (
                (k_flat >= bins[i])
                & (k_flat <= bins[i + 1])
            )
        else:
            mask_shell = (
                (k_flat >= bins[i])
                & (k_flat < bins[i + 1])
            )

        if not np.any(mask_shell):
            continue

        k_values.append(np.mean(k_flat[mask_shell]))
        P_values.append(np.mean(power_flat[mask_shell]))
        N_modes.append(np.sum(mask_shell))

        if P_theory is not None:
            P_theory_values.append(
                np.mean(theory_flat[mask_shell])
            )

    result = (
        np.array(k_values),
        np.array(P_values),
        np.array(N_modes),
    )

    if P_theory is not None:
        return result + (np.array(P_theory_values),)

    return result

def bin_in_log(P_flat, k_flat, bins):
    """Average P(k) in log-bins. Returns mean P and number of modes."""
    
    P_out = np.full(len(bins) - 1, np.nan)
    N_out = np.zeros(len(bins) - 1, dtype=int)
    for b in range(len(bins) - 1):
        m = (k_flat >= bins[b]) & (k_flat < bins[b + 1]) & np.isfinite(P_flat)
        if np.any(m):
            P_out[b] = np.mean(P_flat[m])
            N_out[b] = m.sum()
    return P_out, N_out