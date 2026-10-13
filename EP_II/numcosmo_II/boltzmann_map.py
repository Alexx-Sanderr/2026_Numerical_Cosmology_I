import numpy as np

from scipy.integrate import solve_ivp 

class BoltzmannMapValidator: 
    """
        Validates primordial-to-late-time power spectrum mappings from CLASS / CAMB
    """

    def __init__(self, fiducial_params = None):
        if fiducial_params is None:
            self.params = { 
                'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200, 'A_s': 2.1e-9, 'n_s': 0.9649, 'k_pivot': 0.05, 'z': 0.0
                          }
        else:
            self.params = fiducial_params


    def primordial_power_spectrum(self, k):
        """
            Calculates primordial scalar curvature power spectrum P_R(k) in Mpc^3
        """

        A_s     = self.params[ 'A_s' ]
        n_s     = self.params[ 'n_s' ]
        k_pivot = self.params[ 'k_pivot' ]

        P_R_dimentionless = A_s * ( ( k / k_pivot ) ** ( n_s - 1.0 ) )
        P_R_3D            = ( 2.0 * np.pi ** 2 / ( k ** 3 ) ) * P_R_dimentionless

        return P_R_3D

    def rec_matter_power_spectrum(self, k, T_m):
        """
            Reconstructed P_m(k)
        """

        P_R = self.primordial_power_spectrum( k )

        return P_R * ( T_m ** 2 )

    @staticmethod 
    def solve_superhorizon_potencial(y_init = 1e-4, y_final = 1e3):
        """
            Solves d(phi)/dln(y) 
        """
        
        y_init = float( y_init )
        y_final = float( y_final )

        def dphi_dlny( lny, phi ):
            lny_clamped = min( lny, 20.0 )
            y           = np.exp( lny_clamped )
            rhs         = - ( 4.0 + 3.0 * y ) / ( 2.0 * ( 1.0 + y ) ) - ( 6.0 + 5.0 * y ) / ( 2.0 * ( 1.0 + y ) ) * phi[ 0 ]
            return [ rhs ]

        phi_i    = -2.0 / 3.0 + y_init / 24.0
        lny_span = ( np.log( y_init ), np.log( y_final ) )

        sol = solve_ivp(
            dphi_dlny, lny_span, [ phi_i ], method = 'Radau', rtol = 1e-10, atol = 1e-12
                        )

        y_vals   = np.exp( sol.t )
        phi_vals = sol.y[ 0 ]

        return y_vals, phi_vals

    @staticmethod
    def free_streaming_hierarchy(t, psi, k, l_max, closure_type = 'zero'):
        """
            RHS for the source-free massless Boltzmann hierarchy
        """
        
        k     = float( k )
        l_max = int( l_max )
        dpsi  = np.zeros( l_max + 1 )

        dpsi[ 0 ] = -k * psi[ 1 ]

        for l in range( 1, l_max ):
            dpsi[ l ] = ( k / ( 2 * l + 1 ) ) * ( l * psi[ l - 1 ] - ( l + 1 ) * psi[ l + 1 ] )

        if closure_type == 'zero':
            psi_next = 0.0
        elif closure_type == 'outgoing':
            psi_next = ( ( 2 * l_max + 1 ) / max( k * t, 1e-5 ) ) * psi[ l_max ] - psi[ l_max - 1 ]
        else:
            psi_next = 0.0

        dpsi[ l_max ] = ( k / ( 2 * l_max + 1 ) ) * ( l_max * psi[ l_max - 1 ] - ( l_max + 1 ) * psi_next )
        
        return dpsi

    @classmethod
    def solve_free_streaming_hierarchy( cls, k = 1.0, l_max = 4, tau_max = 20.0, num_points = 400, closure_type = 'zero'):
        """
            Integrates the free-streaming Boltzmann hierarchy from tau=1e-3 to tau_max.
        """
        
        k        = float( k )
        l_max    = int( l_max )
        tau_span = ( 1e-3, float( tau_max ) )
        tau_eval = np.linspace( 1e-3, float( tau_max ), int( num_points ) )

        psi_init      = np.zeros( l_max + 1 )
        psi_init[ 0 ] = 1.0 

        sol = solve_ivp(
            cls.free_streaming_hierarchy,
            tau_span,
            psi_init,
            t_eval = tau_eval,
            args   = ( k, l_max, closure_type ),
            method = 'RK45',
            rtol   = 1e-8,
            atol   = 1e-10,
        )

        return sol.t, sol.y