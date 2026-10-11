import numpy as np

class SPTStatistics:
    """
        Calculates 2nd-order SPT Fields, power spectrum, bispectrum, and smoothed reduced skewness
    """

    def __init__(self, field_generator):
        
        self.gf = field_generator
        self.L  = field_generator.L
        self.Ng = field_generator.Ng
        self.V  = field_generator.V

        dx   = self.L / self.Ng
        k_1d = 2 * np.pi * np.fft.fftfreq( self.Ng, d = dx )
        
        self.kx, self.ky, self.kz = np.meshgrid( k_1d, k_1d, k_1d, indexing = "ij" )
        self.k_mag                = np.sqrt( self.kx ** 2 + self.ky ** 2 + self.kz ** 2 )


    def compute_delta_2(self, delta_1_k):
        """
            Calculates 2nd order SPT density delta_2(x) and delta_2(k) in real space using FFTs
        """

        Ng = self.Ng
        V  = self.V
        k2 = np.where( self.k_mag == 0, 1.0, self.k_mag ** 2 )

        delta_1_x = np.fft.ifftn( ( Ng ** 3 / V ) * delta_1_k).real

        psi_x = np.fft.ifftn( ( Ng ** 3 / V ) * (-1j * self.kx / k2 ) * delta_1_k ).real
        psi_y = np.fft.ifftn( ( Ng ** 3 / V ) * (-1j * self.ky / k2 ) * delta_1_k ).real
        psi_z = np.fft.ifftn( ( Ng ** 3 / V ) * (-1j * self.kz / k2 ) * delta_1_k ).real

        g_x = np.fft.ifftn( ( Ng ** 3 / V) * (1j * self.kx) * delta_1_k ).real
        g_y = np.fft.ifftn( ( Ng ** 3 / V) * (1j * self.ky) * delta_1_k ).real
        g_z = np.fft.ifftn( ( Ng ** 3 / V) * (1j * self.kz) * delta_1_k ).real

        advection = psi_x * g_x + psi_y * g_y + psi_z * g_z

        s_ij_sq = np.zeros_like( delta_1_x )
        cmpnts  = [ ( self.kx, self.kx, 1/3 ), ( self.ky, self.ky, 1/3 ), ( self.kz, self.kz, 1/3 ), 
                    ( self.kx, self.ky, 0.0 ), ( self.kx, self.kz, 0.0 ), ( self.ky, self.kz, 0.0 )
                  ]

        for ki, kj, diag in cmpnts:
            kernel   = np.where( self.k_mag == 0, 0.0, ( ki * kj / k2 ) - diag )
            s_field  = np.fft.ifftn( ( Ng ** 3 / V ) * kernel * delta_1_k ).real
            weight   = 1.0 if diag != 0.0 else 2.0
            s_ij_sq += weight * ( s_field ** 2 ) 

        delta_2_x = ( 17.0 / 21.0 ) * ( delta_1_x ** 2 ) + advection + ( 2.0 / 7.0 ) * s_ij_sq

        delta_2_k = ( V / Ng ** 3 ) * np.fft.fftn( delta_2_x )
        delta_2_k[ self.k_mag == 0 ] = 0.0

        return delta_2_x, delta_2_k
        

    def spt_power_spectrum( self, delta_k, num_bins = 20 ):
        """
            Measures 1D power spectrum P(k) in spherical k-shells
        """

        k_flat  = self.k_mag.flatten()
        #pk_3d   = ( self.V / ( self.Ng ** 3 ) ** 2 ) * ( np.abs( delta_k ) ** 2 )
        pk_3d   = ( np.abs( delta_k ) ** 2 ) / self.V
        pk_flat = pk_3d.flatten()

        k_min = self.k_mag[ self.k_mag > 0 ].min() 
        k_max = self.k_mag.max() / 2.0
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
        

    def spt_equilateral_bispectrum(self, delta_k, num_bins = 12):
        """
            Measures equilateral bispectrum B(k, k, k) using shell-filtered real-space fields
        """

        k_centers, pk_linear, _ = self.spt_power_spectrum( delta_k, num_bins = num_bins )
        dk                      = k_centers[ 1 ] - k_centers[ 0 ] if len( k_centers ) > 1 else 0.1

        b_measured = []
        b_tree     = []
        valid_k    = []

        for k_val, p_val in zip( k_centers, pk_linear ):
            mask = ( self.k_mag >= k_val - dk / 2.0 ) & ( self.k_mag < k_val + dk / 2.0 )
            
            if np.sum( mask ) < 8:
                continue

            delta_shell_k = np.where( mask, delta_k, 0.0 )
            delta_shell_x = np.fft.ifftn( ( self.Ng ** 3 / self.V ) * delta_shell_k ).real 

            I_shell_k = np.where( mask, 1.0, 0.0 )
            I_shell_x = np.fft.ifftn( I_shell_k )

            N_tri = np.sum( I_shell_x ** 3 ).real * ( self.Ng ** 6 )

            if N_tri > 0:
                b_val = ( self.V ** 2 / ( self.Ng ** 3 * N_tri ) ) * np.sum( delta_shell_x ** 3 )
                
                b_measured.append( b_val )
                b_tree.append( ( 12.0 / 7.0 ) * ( p_val ** 2 ) )
                valid_k.append( k_val)
            
        return np.array( valid_k ), np.array( b_measured ), np.array( b_tree )


    def spt_S3_profile(self, delta_x, R_array):
        """
            Computes reduced skewness S3(R) for a list of Top-Hat smoothing radii R
        """

        S3_list = []
        delta_k = np.fft.fftn( delta_x )

        for R in R_array:
            kR = self.k_mag * R
            
            with np.errstate( divide = 'ignore', invalid = 'ignore' ):
                W_R = 3.0 * ( np.sin( kR) - kR * np.cos( kR ) ) / ( kR ** 3 )
                W_R[ self.k_mag == 0.0 ] = 1.0

            delta_R_x = np.fft.ifftn( delta_k * W_R ).real
    
            var  = np.mean( delta_R_x ** 2 )
            skew = np.mean( delta_R_x ** 3 )
    
            s3 = skew / ( var ** 2 ) if var > 0 else 0.0
            S3_list.append( s3 )
        
        return np.array( R_array ), np.array( S3_list )


    def run_ensemble_analysis(self, num_realizations = 10, num_bins = 10, R_array = None):
        """
            Auxiliary function to obtain N realizations of a Gaussian field, the mean and uncertainty of the bispectrum, and S3(R)
        """
        
        if R_array is None:
            R_array = np.array( [ 10.0, 15.0, 20.0 ] )

        b_meas_list = []
        s3_list     = []

        for seed in range( num_realizations ):
            
            rng = np.random.default_rng( seed = 1000 + seed )
           
            self.gf.rng          = rng
            delta_1_x, delta_1_k = self.gf.Gaussian_field()

            delta_2_x, delta_2_k = self.compute_delta_2( delta_1_k )
            delta_tot_k          = delta_1_k + delta_2_k
            delta_tot_x          = delta_1_x + delta_2_x

            k_b, b_meas, b_tree = self.spt_equilateral_bispectrum( delta_tot_k, num_bins = num_bins )
            _, s3_vals          = self.spt_S3_profile( delta_tot_x, R_array )

            b_meas_list.append( b_meas )
            s3_list.append( s3_vals )

        b_mean = np.mean( b_meas_list, axis = 0 )
        b_std  = np.std( b_meas_list, axis=0) / np.sqrt( num_realizations )

        s3_mean = np.mean( s3_list, axis = 0 )
        s3_std  = np.std( s3_list, axis = 0 ) / np.sqrt( num_realizations )

        return {
            "k_b": k_b,
            "B_mean": b_mean,
            "B_std": b_std,
            "B_tree": b_tree,
            "R_array": R_array,
            "S3_mean": s3_mean,
            "S3_std": s3_std,
        }