#################################################
# Cosmology for likelihood calculation V - 2.0  #
#                Modularized Version            #
#                Alex Sander - 2026             #
#################################################

#___________Packages________#
import numpy as np

from scipy.integrate import quad

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

    def E(self, z: float) -> float:
        """ 
            E(z) = H(z)/H0 function
        """

        if isinstance( z, ( list, np.ndarray ) ):
            return np.array( [ self.E( zi ) for zi in z ] )
        
        a_inv = 1.0 + z
        E2    = self.O_curvature * a_inv ** 2 + self.O_matter * a_inv ** 3 + self.O_radiation * a_inv ** 4 + self.O_de * np.exp( - (3 * ( self.wa * z ) / a_inv ) ) * a_inv ** ( 3 * ( 1 + self.wa + self.w0 ) )  

        if E2 <= 0:
            return np.nan
        
        return np.sqrt(E2)

    def comoving_distance(self, z: float) -> float:
        """
            Comoving lenght chi(z) in Mpc 
        """

        if isinstance(z, (list, np.ndarray)):
            return np.array( [ self.comoving_distance( zi ) for zi in z ] )           
        else: 
            integrand = lambda zp: 1.0 / self.E( zp )
            val, _    = quad( integrand, 0, z )
    
            return ( self.c / self.H0 ) * val

    def transverse_comoving_distance(self, z:float) -> float:
        """
            Transverse comoving distance D_M(z) in Mpc
        """

        chi = self.comoving_distance( z )
        DH  = self.c / self.H0

        if abs(self.O_curvature) < 1e-6:
            return chi
        elif self.O_curvature > 0:
            k = np.sqrt( self.O_curvature )
            return DH * ( 1.0 / k ) * np.sinh( k * chi / DH )
        else:
            k = np.sqrt( - self.O_curvature )
            return DH * ( 1.0 / k ) * np.sin( k * chi / DH )

    def angular_diameter_distance(self, z: float) -> float:
        """
            Angular diameter distance D_A(z) in Mpc
        """
        
        return self.transverse_comoving_distance( z ) / ( 1.0 + z )

    def luminosity_distance(self, z: float) -> float:
        """
            Luminosity distance D_L(z) in Mpc
        """
        
        return ( 1.0 + z ) ** 2 * self.angular_diameter_distance( z )

    def distance_modulus(self, z: float) -> float:
        """
            Distance modulus mu(z)
        """
        
        dL = self.luminosity_distance( z )
        return 5.0 * np.log10( dL ) + 25.0


    def conformal_time(self, z: float) -> float:
        """
            Conformal time eta(z) in Mpc
        """
        
        zm_standard = 1e8

        if isinstance( z, ( list, np.ndarray ) ):
            return np.array( [ self.conformal_time( zi ) for zi in z ] )
        elif zm_standard < np.max( z ):
            return self.comoving_distance( z )    
        else:
            integrand = lambda zp: 1.0 / self.E( zp )
            val, _    = quad( integrand, z, zm_standard )
            return ( self.c / self.H0 ) * val

    def volume_element(self, z: float) -> float: 
        """
            Differential comoving volume element d^2V / (dz dOmega)
        """

        return ( self.c / ( self.H0 *  self.E( z ) ) ) * self.transverse_comoving_distance( z ) ** 2

    def Hz(self, z: float) -> float:
        """
            Physical expansion rate H(z) in km/s/Mpc 
        """
        return self.H0 * self.E( z ) 
        