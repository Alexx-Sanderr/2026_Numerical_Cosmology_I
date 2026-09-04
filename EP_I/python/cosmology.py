#########################################
# Cosmology for SNa calculation V - 1.0 #
#            Alex Sander 2026           #
#########################################

#___________Packages________#
import scipy as sp
import numpy as np

from scipy.integrate import cumulative_trapezoid

#___________Functions_______#

def constants(h):
    H0 = h * 100                         # Hubble constant km/s/Mpc
    
    return H0

def dDc_dz (H0, c, Omega_m, Omega_lambda, Omega_r, z, w0_fld, wa_fld):
    """
        Calculates c / H(z), which is the integrand d(D_c)/dz in Mpc.
    """

    Omega_k = 1 - Omega_lambda - Omega_m - Omega_r
     
    curvature = Omega_k * ( 1 + z ) ** 2
    matter = Omega_m * ( 1 + z ) ** 3
    radiation = Omega_r * (1 + z ) ** 4
    dark_energy = Omega_lambda * ( 1+ z ) ** ( 3 * ( 1 + w0_fld + wa_fld ) ) * np.exp( - ( 3 * wa_fld * z ) / ( 1 + z ) )

    E2 = curvature + matter + radiation + dark_energy

    if np.any( E2 <= 0 ):
        return None, Omega_k

    dDc = c / ( H0 * np.sqrt( E2 ) )
    
    return dDc, Omega_k


def distances(dDc_dz_array, z, Omega_k, c, H0):
    """
        Calculates comoving radial distance D_C(z), angular diameter distance D_A(z)
        and luminosity distance D_L(z) in Mpc.
    """

    Dc_z = cumulative_trapezoid ( dDc_dz_array, z , initial = 0 ) 
    Dc_z_adm = ( H0  / c ) * Dc_z
    
    if Omega_k > 0.0:
        
        sqrt_Omega_k = np.sqrt( Omega_k )
        
        DM_z = ( c / ( H0 * sqrt_Omega_k ) ) * np.sinh( sqrt_Omega_k * Dc_z_adm  )
        
    elif Omega_k < 0.0:
        
        abs_sqrt_Omega_k = np.sqrt( np.abs( Omega_k ) )
            
        DM_z = ( c / ( H0 * abs_sqrt_Omega_k ) ) * np.sin( abs_sqrt_Omega_k * Dc_z_adm  )
        
    else:
        
        DM_z = Dc_z
        
    
    D_A = DM_z / ( 1 + z ) # Angular diameter distance
    
    
    D_L = DM_z * ( 1 + z ) # Luminosity distance 
    
    
    return {"comoving radial distance": Dc_z,
            "angular diameter distance": D_A,
            "luminosity distance": D_L,
            }