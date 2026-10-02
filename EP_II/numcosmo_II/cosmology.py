import numpy as np

from scipy.integrate import quad, solve_ivp

class FLRW:
    """
        FLRW generic model with equation of state
    """
    
    def __init__ (self, h: float, Omega_m: float, Omega_lambda: float, Omega_r: float, w0_fld: float, wa_fld:float):
        
        self.c  = 299792.458     # km/s
        self.H0 = float(h) * 100 # km/s/Mpc
        self.wa = float(wa_fld)
        self.w0 = float(w0_fld)
        
        self.O_matter    = float(Omega_m)
        self.O_de        = float(Omega_lambda)
        self.O_radiation = float(Omega_r) 
        self.O_curvature = 1.0 - (self.O_matter + self.O_de + self.O_radiation)
    
    
    def _resolve_a(self, a = None, N = None):
        """
            Verifies if a or N was used as input and returns a.
        """
        
        if a is None and N is not None:
            
            return np.exp( N )
        
        if a is not None:
            return np.asarray( a )
        
        raise ValueError( "You must provide exactly one argument: either 'a_t' or 'N'." )
    
    
    def E(self, a: float = None, N: float = None) -> float:
        """ 
            E(a) = H(a)/H0 function
        """
         
        a_val = self._resolve_a( a, N )
        a_inv = 1.0 / np.asarray( a_val )
         
        f_de = ( a_inv ** ( 3.0 * ( 1.0 + self.w0 + self.wa ) ) ) * np.exp( - 3.0 * self.wa * ( 1.0 - a_val ) )
        
        E2 = ( 
             self.O_curvature * ( a_inv ** 2 ) 
             + self.O_matter * ( a_inv ** 3 ) 
             + self.O_radiation * ( a_inv ** 4 ) 
             + self.O_de * f_de
         )
        
        return np.sqrt( np.maximum( E2, 0.0 ) )
    
    
    def Ha(self, a: float = None, N: float = None):
        """
            Physical expansion rate H(a) in km/s/Mpc 
        """
        
        return self.H0 * self.E( a = a, N = N)
             
    
    def a_dep_O_matter(self, a: float = None, N: float = None):
        """
            Matter Density a-dependent function
        """
        
        a_val = self._resolve_a( a, N )
        E_val = self.E( a = a_val )
    
        return ( self.O_matter ) / ( ( a_val ** 3 ) * ( E_val ** 2 ) )
    
    def expansion_decay(self, a: float = None, N: float = None):
        """
            Expansion rate decay: A_H(a) = 2 + dlnH/dN
        """
    
        a_val = self._resolve_a( a, N )
        a_inv = 1.0 / a_val
    
        f_de     = ( a_inv ** ( 3.0 * ( 1.0 + self.w0 + self.wa ) ) ) * np.exp( - 3.0 * self.wa * ( 1.0 - a_val ) )
        df_de_da = f_de * ( 3.0 * self.wa - ( 3.0 * ( 1.0 + self.w0 + self.wa ) * a_inv ) )
    
        dE2_da = (
            - 2.0 * self.O_curvature * ( a_inv ** 3 )
            - 3.0 * self.O_matter * ( a_inv ** 4 )
            - 4.0 * self.O_radiation * ( a_inv ** 5 )
            + self.O_de * df_de_da
        )
    
        E2_val  = self.E( a = a_val) ** 2
        dlnH_dN = ( a_val / ( 2.0 * E2_val ) ) * dE2_da

        return 2.0 + dlnH_dN
    
    
    def growth_ode_N(self, N: float, y: list) -> list:
        """
            N = ln(a) ODE formulation: y = [G, G_N]
        """
        G, G_N = y
       
        return [ G_N, - self.expansion_decay( N = N) * G_N + 1.5 * self.a_dep_O_matter( N = N) * G]
    
    
    def growth_ode_a(self, a: float, y: list) -> list:
        """
            a ODE formulation: y = [D, dD/da]
        """
        
        D, dD_da = y
        dlnH_da  = ( self.expansion_decay( a = a ) + 1 ) / a  
        
        return [ dD_da, - dlnH_da * dD_da + 1.5 * ( self.a_dep_O_matter( a = a ) / a ** 2 ) * D ]
    
        
    def solve_growth(self, a = None, N = None, y0 = None, **kwargs):
        """
            Solves growth ODE system for N or a.
        """

        kwargs.setdefault( 'dense_output', True )
        
        if N is not None:
            
            return solve_ivp( self.growth_ode_N, t_span = ( N[ 0 ], N[ -1 ] ), y0 = y0, **kwargs )
        
        elif a is not None:
          
            return solve_ivp( self.growth_ode_a, t_span = ( a[ 0 ], a[ -1 ] ), y0 = y0, **kwargs )
        else:
            raise ValueError( "You must provide exactly one argument: either 'a_t' or 'N' interval." )

    def growth_quadrature(self, a: float) -> float:
        """
            Calculates G(a) from quadrature integration 
        """
        
        integrand       = lambda a_tilde: 1.0 / ( ( a_tilde ** 3 ) * ( self.E( a = a_tilde ) ** 3 ) )
        integral_val, _ = quad( integrand, 0.0, a )
        
        return 2.5 * self.O_matter * self.E( a = a ) * integral_val
            
    
    def get_growth_observables(self, sol, a_eval=None, N_eval = None):
        """
            Extracts normalized D(a) and f(a) from ODE solver
        """

        if not hasattr( sol, 'sol' ) or sol.sol is None:
            raise AttributeError( "The solution object does not contain '.sol'. Pass dense_output=True to solve_ivp." )
        
        if sol.t[ 0 ] <= 0 and sol.t[ -1 ] <= 0:
            if N_eval is None:
                N_eval = sol.t
            
            G_eval, GN_eval = sol.sol( N_eval )
            G_1             = sol.sol( 0.0 )[ 0 ] 
            
            D_a = G_eval / G_1
            f_a = GN_eval / G_eval
            
            return D_a, f_a

        else:
            if a_eval is None:
                a_eval = sol.t
            
            D_eval, dD_da_eval = sol.sol( a_eval )
            D_1                = sol.sol( 1.0 )[ 0 ] 
            
            D_a = D_eval / D_1
            f_a = a_eval * ( dD_da_eval / D_eval )
            
            return D_a, f_a