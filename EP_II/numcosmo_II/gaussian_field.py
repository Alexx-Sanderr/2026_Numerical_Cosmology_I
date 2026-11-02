import numpy as np

class GaussianField:

    def __init__(self, L = 100, Ng = 128, aref = 1.0):

        self.L         = float( L )            # Box side length
        self.Ng        = int( Ng )             # Number of grid cells per dimension
        self.V         = float( self.L ** 3 )  # Box volume
        self.aref      = float( aref )         # Reference scale factor
        self._pk_cache = {}

    def P_linear( self, k, a = 1.0 ):
        k_array = np.asarray(k, dtype=float)
        P_ref = np.zeros_like(k_array)
        mask = k_array > 0.0
                
        if not np.any(mask):
            return P_ref.item() if P_ref.ndim == 0 else P_ref
        
        cache_key = round( float( a ), 8 )
        if cache_key not in self._pk_cache:
            import camb
        
            H0, h   = 67.37, 0.6737
            Omega_m = 0.315
            Omega_b = 0.0493

            pars = camb.CAMBparams()
            pars.set_cosmology( 
                H0 = H0, ombh2 = Omega_b * h ** 2, omch2 = ( Omega_m - Omega_b ) * h ** 2, omk = 0.0 )
            pars.InitPower.set_params( As = 2.1e-9, ns = 0.965 )

            z = 1.0 / float( a ) - 1.0
            pars.set_matter_power( redshifts = [ z ], kmax = 10.0, nonlinear = False )
            pars.NonLinear = camb.model.NonLinear_none
            
            results = camb.get_results( pars )
            self._pk_cache[ cache_key ] = results.get_matter_power_interpolator(
                nonlinear = False, hubble_units = True, k_hunit = True )

        PK = self._pk_cache[cache_key]
        z = 1.0 / float(a) - 1.0
        P_ref[mask] = PK.P(z, k_array[mask])
        
        return P_ref.item() if P_ref.ndim == 0 else P_ref


    def normal_modes(self, L = None, V = None, Ng = None):
        
        L  = self.L if L is None else float( L )
        Ng_val = int( self.Ng if Ng is None else Ng  )
        
        dx   = L / Ng_val
        k_1d = 2.0 * np.pi * np.fft.fftfreq( Ng_val, d = dx )
        
        kx, ky, kz = np.meshgrid( k_1d, k_1d, k_1d, indexing = "ij" )
        k          = np.sqrt( kx ** 2 + ky ** 2 + kz ** 2 )
        
        return kx, ky, kz, k


    def Gaussian_field( self, seed = 42 ):

        Ng, V      = self.Ng, self.V
        _, _, _, k = self.normal_modes()
        power      = np.asarray( self.P_linear( k, self.aref ), dtype = float )

        rng   = np.random.default_rng( seed = int( seed ) )
        white = ( rng.normal( size = ( Ng, Ng, Ng ) ) + 1j * rng.normal( size = ( Ng, Ng, Ng ) ) ) / np.sqrt( 2.0 )
        
        neg             = ( -np.arange( Ng ) ) % Ng
        white_neg       = white[ np.ix_( neg, neg, neg ) ]
        white_hermitian = (white + np.conjugate( white_neg ) ) / np.sqrt( 2.0 )

        delta_k            = np.sqrt( V * power ) * white_hermitian
        delta_k [0, 0, 0 ] = 0.0
        delta_x            = np.fft.ifftn( ( Ng ** 3 / V ) * delta_k ).real
        
        return delta_x, delta_k
