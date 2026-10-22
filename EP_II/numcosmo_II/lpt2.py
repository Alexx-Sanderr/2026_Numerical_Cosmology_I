import numpy as np
from numcosmo_II import cosmology
from numcosmo_II import gaussian_field
from numcosmo_II import lpt1
from numcosmo_II import spt_field as spt

class LPT2:
    """
    Second-order Lagrangian Perturbation Theory (2LPT).

    Parameters
    ----------
    parent : object
        Main simulation object containing:
        - GF: GaussianField instance
        - cosmo: cosmology instance
    """

    def __init__(self, L, Ng, aref, h, Omega_m, Omega_lambda,a_ini, ai, GF):
        # Store simulation parameters
        self.L = L
        self.Ng = Ng
        self.aref = aref
        self.h = h
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.a_ini = a_ini
        self.ai = ai

        # Store the Gaussian field
        self.GF = GF
        self.V = GF.V

        # Fourier modes
        self.kx, self.ky, self.kz, self.k = GF.normal_modes(
            self.L, self.V, self.Ng
        )

        self.k2 = self.k**2
        self.mask = self.k2 > 0

        # Second-order quantities
        self.source2_x = None
        self.source2_k = None
        self.phi2_k = None
        self.phi2_x = None

        # Second-order displacements
        self.Psi2_x_k = None
        self.Psi2_y_k = None
        self.Psi2_z_k = None

        self.Psi2_x = None
        self.Psi2_y = None
        self.Psi2_z = None

    def Second_Order_Potential(
        self,
        delta_ref_k=None,
        phi1_k=None,
    ):
        """
        Calculate the second-order source and potential.

        Parameters
        ----------
        delta_ref_k : ndarray, optional
            Reference density field in Fourier space. This argument is
            retained for compatibility but is not used directly here.
        phi1_k : ndarray, optional
            First-order potential in Fourier space. If omitted, it is
            calculated using lpt1.calculate_potential(parent).

        Returns
        -------
        source2_x, source2_k, phi2_k, phi2_x : ndarray
            Second-order source and potential in real and Fourier space.
        """

        # First-order potential in Fourier space
        if phi1_k is None:
            phi1_k = lpt1.calculate_potential(self.parent)

        # First-order Hessian components in real space
        factor = self.Ng**3 / self.V

        phi_xx = np.fft.ifftn(factor * (-self.kx**2 * phi1_k)).real
        phi_yy = np.fft.ifftn(factor * (-self.ky**2 * phi1_k)).real
        phi_zz = np.fft.ifftn(factor * (-self.kz**2 * phi1_k)).real

        phi_xy = np.fft.ifftn(factor * (-self.kx * self.ky * phi1_k)).real
        phi_xz = np.fft.ifftn(factor * (-self.kx * self.kz * phi1_k)).real
        phi_yz = np.fft.ifftn(factor * (-self.ky * self.kz * phi1_k)).real

        # Second-order source
        self.source2_x = (
            phi_xx * phi_yy - phi_xy**2
            + phi_xx * phi_zz - phi_xz**2
            + phi_yy * phi_zz - phi_yz**2)

        # Transform source to Fourier space
        self.source2_k = (self.V / self.Ng**3) * np.fft.fftn(self.source2_x)

        # Solve the second-order Poisson equation:
        self.phi2_k = np.zeros_like(self.source2_k, dtype=complex)
        self.phi2_k[self.mask] = (-self.source2_k[self.mask]/ self.k2[self.mask])
        self.phi2_k[~self.mask] = 0.0

        # Transform potential back to real space
        self.phi2_x = np.fft.ifftn(factor * self.phi2_k).real

        return (
            self.source2_x,
            self.source2_k,
            self.phi2_k,
            self.phi2_x,
        )

    def Growth_Second_ordem(self, y, a, D1):
        """
        Second-order growth ODE, expressed in cosmic time.

        y = [D2, dD2/dt]

        The independent variable must therefore be cosmic time,
        not scale factor a.
        """

        H = self.cosmo.Ha(a=a)
        Omega_m = self.cosmo.a_dep_O_matter(a)
        D2, D2_t = y
        D2_tt = (-2.0 * H * D2_t + 1.5 * H**2 * Omega_m * (D2 - D1**2))

        return [D2_t, D2_tt]

    def displacement_2lpt_fourier(self, D2):
        """
        Calculate the second-order displacement field.

        D2 must follow the same sign convention as phi2_k.
        The second-order potential must be calculated first.
        """

        if self.phi2_k is None:
            raise RuntimeError(
                "Calculate phi2_k first using "
                "Second_Order_Potential().")

        # Fourier-space displacements
        self.Psi2_x_k = (1j * self.kx * D2 * self.phi2_k)
        self.Psi2_y_k = (1j * self.ky * D2 * self.phi2_k)
        self.Psi2_z_k = (1j * self.kz * D2 * self.phi2_k)

        # Inverse Fourier transforms
        factor = self.Ng**3 / self.V
        self.Psi2_x = np.fft.ifftn(factor * self.Psi2_x_k).real
        self.Psi2_y = np.fft.ifftn(factor * self.Psi2_y_k).real
        self.Psi2_z = np.fft.ifftn(factor * self.Psi2_z_k).real

        return (
            self.Psi2_x,
            self.Psi2_y,
            self.Psi2_z,
        )

    @staticmethod
    def direct_convolution_source(
        phi1_k, kx, ky, kz, V
    ):
        """
        Direct discrete convolution for the 2LPT source.

        Intended only for small meshes, because its computational
        cost scales as Ng**6.

        Assumes Fourier coefficients use the convention:
            f_k = (V / Ng**3) * FFT(f_x)
        """

        Ng = phi1_k.shape[0]

        Hxx = -kx**2 * phi1_k
        Hyy = -ky**2 * phi1_k
        Hzz = -kz**2 * phi1_k

        Hxy = -kx * ky * phi1_k
        Hxz = -kx * kz * phi1_k
        Hyz = -ky * kz * phi1_k

        source_k = np.zeros_like(
            phi1_k, dtype=complex
        )

        for i in range(Ng):
            for j in range(Ng):
                for l in range(Ng):

                    total = 0.0j

                    for p in range(Ng):
                        for q in range(Ng):
                            for r in range(Ng):

                                # Periodic index: k - p
                                u = (i - p) % Ng
                                v = (j - q) % Ng
                                w = (l - r) % Ng

                                total += (
                                    Hxx[p, q, r]
                                    * Hyy[u, v, w]
                                    - Hxy[p, q, r]
                                    * Hxy[u, v, w]

                                    + Hxx[p, q, r]
                                    * Hzz[u, v, w]
                                    - Hxz[p, q, r]
                                    * Hxz[u, v, w]

                                    + Hyy[p, q, r]
                                    * Hzz[u, v, w]
                                    - Hyz[p, q, r]
                                    * Hyz[u, v, w]
                                )

                    source_k[i, j, l] = total / V

        return source_k

    def particle_positions_2lpt(self, q, Psi1_q, D2):
        """
        Construct particle positions using first- and second-order
        Lagrangian displacements.
    
        Parameters
        ----------
        q : ndarray
            Lagrangian particle positions, shape (N_particles, 3).
        Psi1_q : ndarray
            First-order displacement evaluated at particle positions.
        D2 : float
            Second-order growth factor, consistent with phi2_k.
    
        Returns
        -------
        x_p_2lpt : ndarray
            Periodic 2LPT particle positions.
        """
    
        if self.phi2_k is None:
            raise RuntimeError(
                "Calculate the second-order potential first."
            )
    
        Psi2_x, Psi2_y, Psi2_z = (
            self.displacement_2lpt_fourier(D2)
        )
    
        Psi2_grid = np.stack(
            [Psi2_x, Psi2_y, Psi2_z],
            axis=-1
        )
    
        # Interpolate the second-order displacement to particle positions.
        from scipy.ndimage import map_coordinates
    
        dx = self.L / self.Ng
        grid_pos = (q % self.L) / dx
    
        Psi2_q = np.empty_like(q, dtype=float)
    
        for axis in range(3):
            Psi2_q[:, axis] = map_coordinates(
                Psi2_grid[..., axis],
                grid_pos.T,
                order=1,
                mode="grid-wrap"
            )
    
        # Combine first- and second-order displacements.
        x_p_2lpt = (
            q + Psi1_q + Psi2_q
        ) % self.L
    
        return x_p_2lpt

    def measure_statistics(self, delta_x, delta_k, num_bins, label, R_array):
        stats = spt.SPTStatistics(field_generator=self.GF)
    
        k_b, B_meas, B_tree = stats.spt_equilateral_bispectrum(
            delta_k,
            num_bins=num_bins
        )
    
        _, S3_vals = stats.spt_S3_profile(delta_x, R_array)
    
        return {
            "label": label,
            "variance": np.var(delta_x),
            "k_b": k_b,
            "B_meas": B_meas,
            "B_tree": B_tree,
            "S3": S3_vals
        }