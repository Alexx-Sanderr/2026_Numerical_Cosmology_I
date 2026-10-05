import numpy as np
import pytest

from numcosmo_II.cosmology import FLRW

@pytest.fixture
def eds_model():
    """
        Einstein-de Sitter model
    """
    return FLRW(
        h=0.7, Omega_m = 1.0, Omega_lambda = 0.0, Omega_r = 0.0, w0_fld = - 1.0, wa_fld = 0.0 
    )


@pytest.fixture
def lcdm_model():
    """
        Lambda-CDM plain model 
    """
    return FLRW(
        h = 0.7, Omega_m = 0.3, Omega_lambda = 0.7, Omega_r = 0.0, w0_fld = -1.0, wa_fld = 0.0
    )

def test_background_consistency(lcdm_model):
    """
        Consistency check for early dominance and E(a) = 1
    """
    assert lcdm_model.E( 
        a = 1 
    ) == pytest.approx( 1.0, abs = 1e-6 )

    assert lcdm_model.a_dep_O_matter(
        a = 1e-4
    ) == pytest.approx( 1.0, abs = 1e-3 )
    

def test_eds_growth_limits(eds_model):
    """
        Consistency for EdS recover of D(a) = a and f(a) = 1 
    """

    a_i    = 1e-4
    N_span = [ np.log( a_i ), 0.0 ]

    sol = eds_model.solve_growth( 
        N = N_span, y0 = [ a_i, a_i ], rtol = 1e-8, atol = 1e-10 
    )

    N_eval = np.linspace( N_span[ 0 ], N_span[ 1 ], 50 ) 
    a_eval = np.exp( N_eval )

    D_a, f_a = eds_model.get_growth_observables( sol, N_eval = N_eval )

    np.testing.assert_allclose( D_a, a_eval, rtol = 1e-4 )
    np.testing.assert_allclose( f_a, np.ones_like( f_a ), rtol = 1e-4 )

def test_ode_vs_quadrature(lcdm_model):
    """
        Comparison between EDO and Quadrature solutions
    """

    a_i    = 1e-4
    N_span = [ np.log( a_i ), 0.0 ]
    
    sol = lcdm_model.solve_growth(
        N = N_span, y0 = [ a_i, a_i ], rtol = 1e-9, atol = 1e-11
    )

    test_scales = [ 0.3, 0.5, 0.8, 1.0 ]

    for a_val in test_scales:
        G_ode  = sol.sol( np.log( a_val ) )[ 0 ]
        G_quad = lcdm_model.growth_quadrature( a_val )

        np.testing.assert_allclose( G_ode, G_quad, rtol = 1e-4 )


def test_formulation_equivalence(lcdm_model):
    """
        Consistency check between EDO solvers in a and N 
    """

    a_i    = 1e-4
    N_span = [ np.log( a_i ), 0.0 ]
    a_span = [ a_i, 1.0 ]

    sol_N = lcdm_model.solve_growth(
        N = N_span, y0 = [ a_i, a_i ], rtol = 1e-8, atol = 1e-10 
    )   
    sol_a = lcdm_model.solve_growth(
        a = a_span, y0 = [ a_i, 1.0 ], rtol = 1e-8, atol = 1e-10 
    )
    a_eval = np.linspace( 0.1, 1.0, 20 )
    N_eval = np.log( a_eval )

    D_N, f_N = lcdm_model.get_growth_observables( sol_N, N_eval = N_eval )
    D_a, f_a = lcdm_model.get_growth_observables( sol_a, a_eval = a_eval )

    np.testing.assert_allclose( D_N, D_a, rtol = 1e-4 )
    np.testing.assert_allclose( f_N, f_a, rtol = 1e-4 )
    

def test_initial_conditions_sensitivity(lcdm_model):
    """
        Ensure stability with different values of a_i
    """

    sol_1 = lcdm_model.solve_growth(
        N = [ np.log( 1e-3 ), 0.0 ], y0 = [ 1e-3, 1e-3 ], rtol = 1e-8, atol = 1e-10
    )
    sol_2 = lcdm_model.solve_growth(
        N = [ np.log( 1e-5 ), 0.0 ], y0 = [ 1e-5, 1e-5 ], rtol = 1e-8, atol = 1e-10
    )

    _, f1_today = lcdm_model.get_growth_observables( sol_1, N_eval = [ 0.0 ] )
    _, f2_today = lcdm_model.get_growth_observables( sol_2, N_eval = [ 0.0 ] )

    np.testing.assert_allclose( f1_today, f2_today, rtol = 1e-4 )
    