import numpy as np
from numcosmo_II.boltzmann_map import BoltzmannMapValidator

def test_primordial_power_spectrum_shape():
    
    validator = BoltzmannMapValidator()
    k         = np.logspace( -3, 0, 50 ) 
    P_R       = validator.primordial_power_spectrum( k )

    assert P_R.shape == ( 50, )
    assert np.all( P_R > 0 )


def test_superhorizon_potential_ration():
    
    validator        = BoltzmannMapValidator()
    y_vals, phi_vals = validator.solve_superhorizon_potencial( y_init = 1e-4, y_final = 1e3 )
    

    phi_r = - 2.0 / 3.0
    phi_m = phi_vals[ -1 ]
    ration = phi_m / phi_r


    np.testing.assert_allclose( phi_vals[ 0 ], - 2 / 3, rtol = 1e-3 )
    np.testing.assert_allclose( phi_m, - 3 / 5, rtol = 1e-3 )
    np.testing.assert_allclose( ration, 0.9, rtol = 1e-3 )