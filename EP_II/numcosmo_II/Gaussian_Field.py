import numpy as np

class GaussianField:

    def __init__(self, L=100, Ng=128, aref=1.0):

        self.L = L          # Box side length
        self.Ng = Ng        # Number of grid cells per dimension
        self.V = L**3       # Box volume
        self.aref = aref    # Reference scale factor


    def P_linear(self, k, a=1.0):

        # Example spectrum only
        P0 = 1000.0
        k0 = 0.1

        P = P0 * (k / k0) * np.exp(-(k / 0.8)**2)

        # P(k=0) is not used because delta_k=0 there
        P[k == 0] = 0.0

        return P


    def normal_modes(self, L, V, Ng):
        dx = L/Ng
        k_id = 2 * np.pi * np.fft.fftfreq(Ng, d=dx)
        kx, ky, kz = np.meshgrid(k_id, k_id, k_id, indexing="ij")
        k = np.sqrt(kx**2 + ky**2 + kz**2)

        return k


    def Gaussian_field(self):

        L = self.L
        Ng = self.Ng
        V = self.V
        aref = self.aref

        P = self.P_linear(
            self.normal_modes(L, V, Ng),
            aref
        )

        # Create Fourier-space Gaussian field
        delta_k = np.zeros(
            (Ng, Ng, Ng),
            dtype=np.complex128
        )

        rng = np.random.default_rng(seed=12345)
        
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