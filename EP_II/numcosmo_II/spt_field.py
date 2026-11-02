import numpy as np

class SPTStatistics:
    """
        Calculates 2nd-order SPT Fields, power spectrum, bispectrum, and smoothed reduced skewness
    """

    def __init__(self, field_generator):
        
        self.gf = field_generator
        self.L  = float( field_generator.L)
        self.Ng = int( field_generator.Ng )
        self.V  = float( field_generator.V )

        dx   = self.L / self.Ng
        k_1d = 2 * np.pi * np.fft.fftfreq( self.Ng, d = dx )
        
        self.kx, self.ky, self.kz  = np.meshgrid( k_1d, k_1d, k_1d, indexing = "ij" )
        self.k_mag                 = np.sqrt( self.kx ** 2 + self.ky ** 2 + self.kz ** 2 )
    
    @property
    def k2(self):
        return self.k_mag ** 2

    @property
    def k_nyquist(self):
        return np.pi * self.Ng / self.L

    @property
    def k_dealias(self):
        return ( 2.0 / 3.0 ) * self.k_nyquist

    @property
    def dealias_mask(self):
        return self.k_mag <= self.k_dealias

        
    @staticmethod
    def _top_hat_window(kR):
        """
            Top-Hat filter with expansion for kR -> 0
        """

        kR     = np.asarray( kR, dtype = float )
        window = np.ones_like( kR )
        
        small = np.abs( kR ) < 1.0e-3
        
        x                = kR[ ~small ]
        window[ ~small ] = 3.0 * ( np.sin( x ) - x * np.cos( x ) ) / x ** 3
        x2               = kR[ small] ** 2
        window[ small ]  = 1.0 - x2 / 10.0 + x2 ** 2 / 280.0
        
        return window

    def _triple_sum_from_iffts(self, f_ifft, g_ifft, h_ifft):
        return float( np.real_if_close( ( self.Ng ** 6 ) * np.sum( f_ifft * g_ifft * h_ifft ) ).real )


    def _tree_F2_weighted_triangle_sum(self, first_leg_weight, third_leg_weight):
        """
            Exact discrete sum of  F2(k1, k2) over closed triangles
        """
        
        A = np.asarray( first_leg_weight, dtype = float )
        H = np.asarray( third_leg_weight, dtype = float )
        
        H_ifft, A_ifft = np.fft.ifftn( H ), np.fft.ifftn( A )

        total   = ( 5.0 / 7.0 ) * self._triple_sum_from_iffts( A_ifft, A_ifft, H_ifft )
        nonzero = self.k2 > 0.0
        k_comps = ( self.kx, self.ky, self.kz )

        for ki in k_comps:
            D_ifft        = np.fft.ifftn( A * ki )
            E             = np.zeros_like( A )
            E[ nonzero ]  = A[ nonzero ] * ki[ nonzero ] / self.k2[ nonzero ]
            total        += self._triple_sum_from_iffts( D_ifft, np.fft.ifftn( E ), H_ifft )

        for ki in k_comps:
            for kj in k_comps:
                C             = np.zeros_like( A )
                C[ nonzero ]  = A[ nonzero ] * ki[ nonzero ] * kj[ nonzero ] / self.k2[ nonzero ]
                total        += ( 2.0 / 7.0 ) * self._triple_sum_from_iffts( np.fft.ifftn( C ), np.fft.ifftn( C ), H_ifft )

        return total

    def compute_delta_2(self, delta_1_k, dealias = True):
        """
            Calculates 2nd order SPT density delta_2(x) and delta_2(k) in real space using FFTs
        """

        Ng = self.Ng
        V  = self.V
        
        delta_input_k = np.where( self.dealias_mask, delta_1_k, 0.0 ) if dealias else delta_1_k

        safe_k2 = np.where(self.k2 == 0.0, 1.0, self.k2)
        norm = Ng**3 / V

        delta_1_x = np.fft.ifftn( norm * delta_input_k ).real
        psi       = [ np.fft.ifftn( norm * (-1j * comp / safe_k2 ) * delta_input_k ).real for comp in ( self.kx, self.ky, self.kz ) ]
        grad      = [ np.fft.ifftn( norm * ( 1j * comp ) * delta_input_k ).real for comp in ( self.kx, self.ky, self.kz ) ]
        advection = sum( p * g for p, g in zip( psi, grad ) )

        s_ij_sq = np.zeros_like( delta_1_x )
        
        components = [
            ( self.kx, self.kx, 1/3 ), ( self.ky, self.ky, 1/3 ), ( self.kz, self.kz, 1/3 ),
            ( self.kx, self.ky, 0.0 ), ( self.kx, self.kz, 0.0 ), ( self.ky, self.kz, 0.0 )
        ]
        
        for ki, kj, diag in components:
            kernel   = np.where( self.k2 == 0, 0.0, ( ki * kj / safe_k2 ) - diag )
            s_field  = np.fft.ifftn( norm * kernel * delta_input_k ).real
            weight   = 1.0 if diag != 0.0 else 2.0
            s_ij_sq += weight * ( s_field ** 2 )

        delta_2_x_raw  = ( 17.0 / 21.0 ) * ( delta_1_x ** 2 ) + advection + ( 2.0 / 7.0 ) * s_ij_sq
        delta_2_x_raw -= np.mean( delta_2_x_raw )
        
        delta_2_k = ( V / Ng**3 ) * np.fft.fftn( delta_2_x_raw )
        
        if dealias:
            delta_2_k = np.where( self.dealias_mask, delta_2_k, 0.0 )
        delta_2_k[ 0, 0, 0 ] = 0.0
        delta_2_x            = np.fft.ifftn( norm * delta_2_k ).real
        
        return delta_2_x, delta_2_k
        

    def spt_power_spectrum(self, delta_k, num_bins = 20, k_max_cutoff = None):
        """
            Measures 1D power spectrum P(k) in spherical k-shells
        """

        k_flat  = self.k_mag.flatten()
        pk_flat = ( ( np.abs( delta_k ) **2 ) / self.V ).flatten()
        

        k_min = self.k_mag[ self.k_mag > 0 ].min() 
        k_max = k_max_cutoff if k_max_cutoff is not None else self.k_nyquist / 2.0
        bins  = np.linspace( k_min, k_max, num_bins + 1 )

        

        k_centers = 0.5 * ( bins[ :-1 ] + bins[ 1: ] )
        p_binned  = np.zeros( num_bins )
        n_modes   = np.zeros( num_bins, dtype = int )

        for i in range( num_bins ):
            mask         = ( k_flat >= bins[ i ] ) & ( k_flat < bins[ i+1 ] )
            n_modes[ i ] = np.sum( mask )
            if n_modes[ i ] > 0:
                p_binned[ i ] = np.mean( pk_flat[ mask ] )

        valid = n_modes > 0

        return k_centers[ valid ], p_binned[ valid ], n_modes[ valid ]
        

    def spt_equilateral_bispectrum(self, delta_k, num_bins = 15):
        """
            Measures equilateral bispectrum B(k, k, k) using shell-filtered real-space fields
        """

        delta_k = np.asarray( delta_k, dtype = np.complex128 )
        
        expected_shape = ( self.Ng, ) * 3
        if delta_k.shape != expected_shape:
            raise ValueError( f"delta_k must have shape {expected_shape}." )
        if not isinstance(num_bins, (int, np.integer)) or num_bins < 1:
            raise ValueError( "num_bins must be a positive integer." )

        k_max_safe    = float( self.k_nyquist / 2.0 )
        k_fundamental = float( np.min( self.k_mag[ self.k_mag > 0.0 ] ) )
        edges         = np.linspace( k_fundamental, k_max_safe, int( num_bins ) + 1 )
        centers       = 0.5 * ( edges[ :-1 ] + edges[ 1: ] )

        power_grid = np.asarray(
            self.gf.P_linear( self.k_mag, getattr( self.gf, "aref", 1.0 ) ), dtype = float )
        if power_grid.shape != expected_shape:
            raise ValueError( "P_linear(k) must return an array matching the Fourier grid." )

        norm = self.Ng ** 3 / self.V
        n6   = self.Ng ** 6
        b_measured, b_tree, valid_k = [], [], []
        
        for index, k_val in enumerate(centers):
            mask = ( self.k_mag >= edges[ index ] ) & ( self.k_mag < edges[ index + 1 ] )
            
            if np.count_nonzero( mask ) < 12:
                continue
            
            indicator = mask.astype( float )
            indicator_x = np.fft.ifftn( indicator ).real
            n_tri_float = float( n6 * np.sum( indicator_x ** 3 ) )
            n_tri = int( np.rint( n_tri_float ) )
            if n_tri <= 0:
                continue

            delta_shell_k = np.where( mask, delta_k, 0.0 )
            delta_shell_x = np.fft.ifftn( norm * delta_shell_k ).real
            b_value       = ( self.V ** 2 / ( self.Ng ** 3 * n_tri ) ) * np.sum( delta_shell_x ** 3 )

            A               = np.where( mask, power_grid, 0.0 )
            f2_weighted_sum = self._tree_F2_weighted_triangle_sum( A, indicator )
            b_tree_value    = 6.0 * f2_weighted_sum / n_tri

            b_measured.append( float( b_value ) )
            b_tree.append( float( b_tree_value ) )
            valid_k.append( float( k_val ) )
            
        return np.asarray( valid_k ), np.asarray( b_measured ), np.asarray( b_tree )


    def spt_S3_profile(self, delta_x, R_array):
        """
            Computes reduced skewness S3(R) for a list of Top-Hat smoothing radii R
        """

        S3_list = []
        delta_k = np.fft.fftn( delta_x )

        for R in R_array:
            W_R      = self._top_hat_window( self.k_mag * R )
            delta_R_x = np.fft.ifftn( delta_k * W_R ).real
            var       = np.mean( delta_R_x ** 2 )
            skew      = np.mean( delta_R_x ** 3 )
            s3        = skew / ( var ** 2 ) if var > 0 else 0.0
            
            S3_list.append( s3 )
        
        return np.array( R_array ), np.array( S3_list )


    def run_ensemble_analysis(self, num_realizations = 10, num_bins = 10, R_array = None, seed_start = 12345):
        """
            Auxiliary function to obtain N realizations of a Gaussian field, the mean and uncertainty of the bispectrum, and S3(R)
        """
        
        if int( num_realizations ) < 2:
            raise ValueError( "num_realizations must be at least 2 to estimate an ensemble standard error." )
        if int( num_bins ) < 1:
            raise ValueError( "num_bins must be at least 1." )
        if R_array is None:
            R_array = np.array( [ 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0 ] )
        R_array = np.asarray( R_array, dtype = float )
        if R_array.ndim != 1 or np.any( R_array <= 0.0 ):
            raise ValueError( "R_array must be a one-dimensional array of positive radii." )



        gaussian_seed_start      = int( seed_start ) + 1_000_000
        b_meas_tot, b_meas_gauss = [], []
        s3_tot, s3_gauss         = [], []

        for seed in range( num_realizations ):

            delta_1_x, delta_1_k = self.gf.Gaussian_field( seed = seed_start + seed )
            delta_2_x, delta_2_k = self.compute_delta_2( delta_1_k )
            delta_tot_k          = delta_1_k + delta_2_k
            delta_tot_x          = delta_1_x + delta_2_x

            k_b, b_tot, b_tree = self.spt_equilateral_bispectrum( delta_tot_k, num_bins = num_bins )
            _, s3_tot_vals     = self.spt_S3_profile( delta_tot_x, R_array )

            
            null_x, null_k = self.gf.Gaussian_field( seed = gaussian_seed_start + seed )
            _, b_g, _      = self.spt_equilateral_bispectrum( null_k, num_bins = num_bins )
            _, s3_g_vals   = self.spt_S3_profile( null_x, R_array )

            b_meas_tot.append( b_tot )
            b_meas_gauss.append( b_g )
            s3_tot.append( s3_tot_vals )
            s3_gauss.append( s3_g_vals )

        n = float( num_realizations )

        return {
            "k_b": k_b,
            "B_tree": b_tree,
            "B_tot_mean": np.mean( b_meas_tot, axis = 0 ),
            "B_tot_std": np.std( b_meas_tot, axis = 0, ddof = 1 ) / np.sqrt( n ),
            "B_gauss_mean": np.mean( b_meas_gauss, axis = 0 ),
            "B_gauss_std": np.std( b_meas_gauss, axis = 0, ddof = 1) / np.sqrt( n ),
            "R_array": R_array,
            "S3_tot_mean": np.mean( s3_tot, axis = 0 ),
            "S3_tot_std": np.std( s3_tot, axis = 0, ddof = 1 ) / np.sqrt( n ),
            "S3_gauss_mean": np.mean( s3_gauss, axis = 0 ),
            "S3_gauss_std": np.std( s3_gauss, axis = 0, ddof = 1 ) / np.sqrt( n ),
        }