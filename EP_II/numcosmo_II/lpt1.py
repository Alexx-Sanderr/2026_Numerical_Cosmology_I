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


    def create_gaussian_field(self):

        self.delta_x, self.delta_ref_k = self.GF.Gaussian_field()

        self.kx, self.ky, self.kz, self.k = self.GF.normal_modes(
            self.GF.L,
            self.GF.V,
            self.GF.Ng
        )

        return self.delta_x, self.delta_ref_k


    def calculate_potential(self):

        self.phi_k = np.zeros_like(self.delta_ref_k)

        mask = self.k != 0

        self.phi_k[mask] = -self.delta_ref_k[mask] / self.k[mask]**2

        return self.phi_k


    def growth(self, n_points=1000, rtol=1e-8, atol=1e-10):

        a_grid = np.linspace(self.a_ini, self.ai, n_points)

        sol = self.cosmo.solve_growth(
            a=a_grid,
            y0=[self.a_ini, 1.0],
            rtol=rtol,
            atol=atol
        )

        D1_ai, f1_i = self.cosmo.get_growth_observables(sol, a_eval=np.array([self.ai]))

        self.D1_ai = D1_ai[0]
        self.f1_i = f1_i[0]

        return self.D1_ai, self.f1_i


    def displacement_fourier(self):

        self.Psi_x_k = np.zeros_like(self.delta_ref_k)
        self.Psi_y_k = np.zeros_like(self.delta_ref_k)
        self.Psi_z_k = np.zeros_like(self.delta_ref_k)

        mask = self.k != 0

        self.Psi_x_k[mask] = (
            1j * self.D1_ai * self.kx[mask]
            / self.k[mask]**2
            * self.delta_ref_k[mask]
        )

        self.Psi_y_k[mask] = (
            1j * self.D1_ai * self.ky[mask]
            / self.k[mask]**2
            * self.delta_ref_k[mask]
        )

        self.Psi_z_k[mask] = (
            1j * self.D1_ai * self.kz[mask]
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

        self.v_p = H_i * self.f1_i * self.Psi_q

        return self.v_p

    


    def run(self):

        self.create_gaussian_field()

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

    Ng = lpt.Ng
    dx = lpt.L / Ng

    mesh = np.zeros((Ng, Ng, Ng), dtype=float)

    indices = np.floor(lpt.x_p / dx).astype(int)
    indices %= Ng

    for p in range(len(lpt.x_p)):
        i, j, k = indices[p]
        mesh[i, j, k] += 1.0

    mean = np.mean(mesh)

    if mean == 0:
        raise ValueError("The mesh contains no particles.")

    delta_mesh = mesh / mean - 1.0

    return delta_mesh


def power_spectrum(delta_x, lpt):
    Ng = lpt.Ng
    V = lpt.GF.V

    delta_fft = np.fft.fftn(delta_x)

    delta_k = (V / Ng**3) * delta_fft

    k = lpt.k
    power = np.abs(delta_k)**2 / V

    return k, power


def bin_power_spectrum(k, power, nbins=20):

    k_flat = k.ravel()
    power_flat = power.ravel()

    mask = k_flat > 0

    k_flat = k_flat[mask]
    power_flat = power_flat[mask]

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

    return (
        np.array(k_values),
        np.array(P_values),
        np.array(N_modes),
    )