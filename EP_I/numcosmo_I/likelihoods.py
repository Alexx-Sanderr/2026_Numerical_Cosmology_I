import numpy as np
import scipy.linalg as la
import yaml

from numcosmo_I.cosmology import FLRW
from pathlib import Path

def parser_yaml_priors(parameters_yaml: dict) -> tuple[list[str], dict[str, tuple[float, float]]]:
    """
        Names and limits extraction for defined priors on YALM
    """
    param_names = []
    bounds      = {}

    for full_keys, config in parameters_yaml.items():
        
        param_name         = full_keys.split( "." )[ -1 ]
        param_names.append( param_name )
        prior_info         = config[ "prior" ]
        bounds[param_name] = ( 
            float( prior_info[ "min" ] ), 
            float( prior_info[ "max" ] ) 
        )
    return param_names, bounds

class LikelihoodSupernovae:
    """
        Engine for nested_sampling likelihood calculations for Supernovae 
    """

    def __init__(self, z_obs: np.ndarray, mu_obs: np.ndarray, cov_matrix: np.ndarray = None, sigma_mu: np.ndarray =  None, param_names: list[str] = None, defaults: dict[str, float] = None, marginalize_M: bool = False):
        
        self.z             = np.asarray( z_obs,    dtype = float )
        self.mu_obs        = np.asarray( mu_obs,   dtype = float )  
        self.param_names   = param_names or ["Omega_m", "Omega_lambda", "M_abs"]
        self.marginalize_M = marginalize_M

        if cov_matrix is not None:
            self.cov           = np.asarray( cov_matrix, dtype = float )
            self.L             = la.cholesky( self.cov, lower=True )
            self.use_cov       = True
            self.marginalize_M = marginalize_M
            self.one_vec       = np.ones_like( self.z )
            self.y_one         = la.solve_triangular( 
                self.L, self.one_vec, 
                lower = True 
            )
            self.C_sum         = np.dot( self.y_one, self.y_one )
        elif sigma_mu is not None:
            self.sigma2  = np.asarray( sigma_mu, dtype = float ) ** 2
            self.use_cov = False
            self.C_sum   = np.sum( 1.0 / self.sigma2 )
        else:
            raise ValueError( "'cov_matrix' or 'sigma_mu' is needed" )

        self.defaults = {
            "h": 0.7,
            "Omega_m": 0.3,
            "Omega_lambda": 0.7,
            "Omega_r": 0.0,
            "w0_fld": - 1.0,
            "wa_fld": 0.0,
            "M_abs": 0.0,
        }
        
        if defaults:
            self.defaults.update( defaults )
        
    def log_likelihood(self, theta: list[float] | np.ndarray) -> float:
        """
            Calculates the log_likelihood for a SNIa model
        """

        p = self.defaults.copy()
        for name, val in zip( self.param_names, theta ):
            p[ name ] = val

        model = FLRW( 
            h            = p[ "h" ],
            Omega_m      = p[ "Omega_m" ],
            Omega_lambda = p[ "Omega_lambda" ],
            Omega_r      = p[ "Omega_r" ],
            w0_fld       = p[ "w0_fld" ],
            wa_fld       = p[ "wa_fld" ],
        )

        mu_shape = model.distance_modulus(self.z)

        if np.any( np.isnan( mu_shape ) ):
            return -np.inf

        delta_mu_0 = self.mu_obs - mu_shape

        if self.marginalize_M:
            if self.use_cov:
                y = la.solve_triangular( self.L, delta_mu_0, lower = True )
                A = np.dot( y, y )
                B = np.dot( y, self.y_one )
            else:
                w = 1.0 / self.sigma2
                A = np.sum( w * ( delta_mu_0 ** 2 ) )
                B = np.sum( w * delta_mu_0 )

            chi2 = A - ( B ** 2) / self.C_sum
        else:
            delta_mu = delta_mu_0 - p[ "M_abs" ]    
            if self.use_cov:
                y    = la.solve_triangular( self.L, delta_mu, lower = True )
                chi2 = np.dot( y, y )
            else:
                chi2 = np.sum( ( delta_mu ** 2 ) / self.sigma2 )

        return -0.5 * chi2

        
    @staticmethod
    def prior_transform( u: np.ndarray, bounds: dict[str, tuple[float, float]], param_names: list[str]) -> np.ndarray:
        """
            Maps the unitary hypercube u in [0, 1]^D to limits defined in the param_names variable
        """

        theta = np.zeros_like( u )

        for i, name in enumerate( param_names ):
            low, high   = bounds[ name ]
            theta[ i ]  = low + u[ i ] * ( high - low )

        return theta
        