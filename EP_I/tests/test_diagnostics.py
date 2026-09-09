import corner
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
import numpy as np

from matplotlib.figure import Figure

from numcosmo_I.diagnostics import (
    compute_stats,
    compute_nested_sampling_diagnostics,
    compute_reduced_chi2,
    compute_split_rhat,
    compute_ess,
    dunkley_power_spectrum,
    compute_mcse,
    plot_corner,
    plot_hubble_diagram,
    plot_traces,
)


@pytest.fixture
def dummy_chain():
    """
        Synthetic 2D chain for tests 
    """
    
    np.random.seed( 42 )
    return np.random.randn( 1000, 2 ) + np.array( [ 0.0, 5.0 ] )


@pytest.fixture
def dummy_multichain():
    """
        Multiples shyntetic 3D chains (M = 4, N = 500, D = 2)
    """
    np.random.seed( 42 )
    return np.random.randn( 4, 500, 2 )


# ____Statistisc____ #

def test_compute_stats(dummy_chain):
    
    weights     = np.ones( len( dummy_chain ) )
    param_names = [ "param_0", "param_1" ]

    stats = compute_stats( dummy_chain, weights, param_names )

    assert "param_0" in stats and "param_1" in stats
    for p in param_names:
        assert "mean" in stats[ p ]
        assert "std" in stats[ p ]
        assert "median" in stats[ p ]
        assert "q16" in stats[ p ]
        assert "q84" in stats[ p ]
        assert "q025" in stats[ p ]
        assert "q975" in stats[ p ]
        assert "err_minus_1sigma" in stats[ p ]
        assert "err_plus_1sigma" in stats[ p ]

    pytest.approx(stats[ "param_0" ][ "mean" ], abs = 0.1 ) == 0.0
    pytest.approx(stats[ "param_1" ][ "mean" ], abs = 0.1 ) == 5.0


def test_compute_nested_sampling_diagnostics():
    
    log_likes = np.array( [ -250.0, -240.0, -230.0 ] )
    weights   = np.array( [ 0.2, 0.3, 0.5 ] )
    logZ      = -235.0
    nlive     = 100

    diag = compute_nested_sampling_diagnostics( log_likes, weights, logZ, nlive )

    assert "shannon_information" in diag
    assert "logZ_err" in diag
    assert diag[ "logZ_err" ] >= 0.0


def test_compute_reduced_chi2():
    res = compute_reduced_chi2( best_fit_log_like = -250.0, n_data = 503, n_params = 3 )

    assert res[ "chi2" ]     == 500.0
    assert res[ "dof" ]      == 500
    assert res[ "chi2_red" ] == 1.0

    with pytest.raises( ValueError, match = "Number of data points" ):
        compute_reduced_chi2( best_fit_log_like = -10.0, n_data = 2, n_params = 3)


# ____Convergence____ #

def test_compute_split_rhat(dummy_multichain):
    rhat = compute_split_rhat( dummy_multichain )

    assert rhat.shape == (2, )
    
    np.testing.assert_allclose( rhat, 1.0, atol = 0.05 )


def test_compute_ess(dummy_chain):
    
    ess = compute_ess( dummy_chain )

    assert ess.shape == ( 2, )
    assert np.all( ess > 0.0 )


def test_dunkley_power_spectrum(dummy_chain):
    
    res = dunkley_power_spectrum( dummy_chain )

    assert "param_0" in res and "param_1" in res
    for p in [ "param_0", "param_1" ]:
        assert "P_0" in res[ p ]
        assert "j_star" in res[ p ]
        assert "alpha" in res[ p ]
        assert "r" in res[ p ]
        assert isinstance( res[ p ][ "passed" ], bool )


def test_compute_mcse(dummy_chain):
    
    ess  = compute_ess( dummy_chain )
    mcse = compute_mcse( dummy_chain, ess )

    assert mcse.shape == ( 2, )
    assert np.all( mcse > 0.0 )


# ____Plot____ #

def test_plot_corner(dummy_chain, tmp_path):
    
    weights   = np.ones( len( dummy_chain ) )
    save_file = tmp_path / "test_corner.png"

    fig = plot_corner( dummy_chain, weights = weights, param_names = [ "p0" , "p1" ], save_path = str( save_file ) )

    assert isinstance( fig, Figure )
    assert os.path.exists( save_file )
    plt.close( fig )


def test_plot_hubble_diagram( tmp_path ):
    
    z         = np.array( [ 0.1, 0.5, 1.0 ] )
    mu        = np.array( [ 38.0, 42.0, 44.0 ] )
    sig       = np.array( [ 0.1, 0.1, 0.1 ] )
    model     = np.array( [ 38.1, 41.9, 44.1 ] )
    save_file = tmp_path / "test_hubble.png"

    fig = plot_hubble_diagram( z, mu, sig, model, save_path = str( save_file ) )

    assert isinstance( fig, Figure )
    assert os.path.exists( save_file )
    plt.close( fig )


def test_plot_traces(dummy_chain, tmp_path):
    save_file = tmp_path / "test_traces.png"

    fig = plot_traces( dummy_chain, param_names = [ "p0", "p1" ], save_path =str( save_file ) ) 

    assert isinstance( fig, Figure )
    assert os.path.exists( save_file )
    plt.close( fig )