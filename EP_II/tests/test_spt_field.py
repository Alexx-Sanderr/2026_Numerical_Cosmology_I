import numpy as np
import pytest

from numcosmo_II.gaussian_field import GaussianField
from numcosmo_II.spt_field import SPTStatistics


@pytest.fixture
def sample_spt():
    
    gf                   = GaussianField( L = 100, Ng = 16, aref = 1.0 )
    delta_1_x, delta_1_k = gf.Gaussian_field()
    spt                  = SPTStatistics( gf )
    
    return spt, delta_1_x, delta_1_k


def test_delta_2_properties(sample_spt):
    """
        Verify format, finitude of real field e, and zero-mode preservation
    """
    
    spt, _, delta_1_k    = sample_spt
    delta_2_x, delta_2_k = spt.compute_delta_2( delta_1_k )

    assert delta_2_x.shape == ( spt.Ng, spt.Ng, spt.Ng )
    assert delta_2_k.shape == ( spt.Ng, spt.Ng, spt.Ng )
    
    assert np.isrealobj( delta_2_x )
    assert np.all( np.isfinite( delta_2_x ) )
    assert delta_2_k[ 0, 0, 0 ] == pytest.approx( 0.0, abs = 1e-12 )


def test_spt_power_spectrum(sample_spt):
    """
        Verifies 1D P(k) measurement
    """
    
    spt, _, delta_1_k = sample_spt
    k_centers, p_binned, n_modes = spt.spt_power_spectrum( delta_1_k, num_bins = 10 )

    assert len( k_centers ) == len( p_binned ) == len( n_modes )
    assert np.all( p_binned >= 0.0 )
    assert np.all( n_modes > 0 )


def test_spt_equilateral_bispectrum(sample_spt):
    """
        Verifies equilateral B(k,k,k) measurement
    """
    
    spt, _, delta_1_k    = sample_spt
    delta_2_x, delta_2_k = spt.compute_delta_2( delta_1_k )
    delta_tot_k          = delta_1_k + delta_2_k

    valid_k, b_meas, b_tree = spt.spt_equilateral_bispectrum( delta_tot_k, num_bins = 6 )

    assert len( valid_k ) == len( b_meas ) == len( b_tree )
    assert np.all( np.isfinite( b_meas ) )
    assert np.all( b_tree >= 0.0 )


def test_spt_S3_profile(sample_spt):
    """
        Verifies the asymmetric reduced profile S3(R) calculation 
    """
    
    spt, delta_1_x, delta_1_k = sample_spt
    delta_2_x, _              = spt.compute_delta_2( delta_1_k )
    delta_tot_x               = delta_1_x + delta_2_x

    R_array       = np.array( [ 10.0, 15.0, 20.0 ] )
    R_out, S3_out = spt.spt_S3_profile( delta_tot_x, R_array )

    assert len( R_out ) == len( S3_out )
    assert np.all( np.isfinite( S3_out ) )