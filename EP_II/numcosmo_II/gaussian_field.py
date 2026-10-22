import numpy as np
import camb

class GaussianField:

    def __init__(self, L=100, Ng=128, aref=1.0):

        self.L = L          # Box side length
        self.Ng = Ng        # Number of grid cells per dimension
        self.V = L**3       # Box volume
        self.aref = aref    # Reference scale factor


    def P_linear(self, k, a=1.0):

    # Parametros cosmologicos
      H0 = 67.0
      h = H0 / 100.0

      Omega_m = 0.315
      Omega_b = 0.049


      a_ref = 1.0
      z_ref = 1.0 / a_ref - 1.0

      pars = camb.CAMBparams()

      pars.set_cosmology(
        H0=H0,
        ombh2=Omega_b * h**2,
        omch2=(Omega_m - Omega_b) * h**2,
        omk=0.0)

      pars.InitPower.set_params(
        As=2.1e-9,
        ns=0.965)

      pars.set_matter_power(
        redshifts=[z_ref],
        kmax=10.0,
        nonlinear=False)

      pars.NonLinear = camb.model.NonLinear_none

      results = camb.get_results(pars)

      PK = results.get_matter_power_interpolator(
        nonlinear=False,
        hubble_units=True,
        k_hunit=True)

      k_array = np.asarray(k, dtype=float)
      P_ref = np.zeros_like(k_array)

      mask = k_array > 0.0
      P_ref[mask] = PK.P(z_ref, k_array[mask])

      if not np.isclose(a, a_ref):
        z = 1.0 / a - 1.0

        pars.set_matter_power(
            redshifts=[z, z_ref],
            kmax=10.0,
            nonlinear=False)
        results_a = camb.get_results(pars)

        PK_a = results_a.get_matter_power_interpolator(
            nonlinear=False,
            hubble_units=True,
            k_hunit=True)

        P_a = np.zeros_like(k_array)
        P_a[mask] = PK_a.P(z, k_array[mask])

        P_ref_a = np.zeros_like(k_array)
        P_ref_a[mask] = PK_a.P(z_ref, k_array[mask])

        ratio = np.ones_like(k_array)
        valid = mask & (P_ref_a > 0.0)
        ratio[valid] = P_a[valid] / P_ref_a[valid]

        P_ref = P_ref * ratio

      return P_ref.item() if P_ref.ndim == 0 else P_ref


    def normal_modes(self, L, V, Ng):
        dx = L/Ng
        k_id = 2 * np.pi * np.fft.fftfreq(Ng, d=dx)
        kx, ky, kz = np.meshgrid(k_id, k_id, k_id, indexing="ij")
        k = np.sqrt(kx**2 + ky**2 + kz**2)

        return kx, ky, kz, k


    def Gaussian_field(self, seed=42):

        L = self.L
        Ng = self.Ng
        V = self.V
        aref = self.aref

        kx, ky, kz, k = self.normal_modes(L, V, Ng)
        P = self.P_linear(k, aref)

        # Create Fourier-space Gaussian field
        delta_k = np.zeros(
            (Ng, Ng, Ng),
            dtype=np.complex128
        )

        rng = np.random.default_rng(seed=seed)
        
        for i in range(Ng):
            for j in range(Ng):
                for l in range(Ng):

                    # Negative-index cell
                    # (-i, -j, -l) modulo Ng-

                    im = (-i) % Ng
                    jm = (-j) % Ng
                    lm = (-l) % Ng

                    current = (i, j, l)
                    negative = (im, jm, lm)


                    # Only generate one member of each pair
                    if current > negative:
                        continue

                    # Zero mode
                    if i == 0 and j == 0 and l == 0:

                        delta_k[i, j, l] = 0.0

                        continue


                    # Power spectrum at this mode
                    Pk = P[i, j, l]

                    if current == negative:
                        g = rng.normal()
                        delta_k[i, j, l] = (np.sqrt(V * Pk) * g)

                    else:
                        g1 = rng.normal()
                        g2 = rng.normal()
                        delta = (
                            np.sqrt(V * Pk / 2.0)
                            * (g1 + 1j * g2)
                        )

                        delta_k[i, j, l] = delta
                        delta_k[im, jm, lm] = np.conj(delta)   

        delta_fft = (Ng**3 / V) * delta_k
        delta_x = np.fft.ifftn(delta_fft).real

        return delta_x,delta_k