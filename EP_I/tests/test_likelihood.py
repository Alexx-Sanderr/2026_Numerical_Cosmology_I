import pytest
import numpy as np
from numcosmo_I.likelihoods import LikelihoodSupernovae


def test_likelihood_initialization(mock_sn_data):
    """
        Verify correct initial file reading.
    """

    z, mu, sigma = mock_sn_data
    sn_like      = LikelihoodSupernovae( z_obs = z, mu_obs = mu, sigma_mu = sigma)
    
    assert len( sn_like.z )      == 5
    assert len( sn_like.mu_obs ) == 5

def test_prior_transform(mock_sn_data, mock_bounds):
    """
        Test hypercube mapping with real bounds 
    """
    param_names = list( mock_bounds.keys() )
    u           = np.array( [ 0.0, 1.0 ] )

    p_vals            = LikelihoodSupernovae.prior_transform( u, mock_bounds, param_names )
    assert p_vals[0] == pytest.approx( 0.0 )
    assert p_vals[1] == pytest.approx( 1.0 )


def test_log_likelihood_computation(mock_sn_data, mock_bounds):
    """
        Verify log-likelihood calculation
    """
    z, mu, sigma = mock_sn_data
    param_names  = list( mock_bounds.keys() )

    sn_like = LikelihoodSupernovae(
        z_obs = z, mu_obs = mu, sigma_mu = sigma, param_names = param_names
    )

    theta = np.array( [ 0.3, 0.7 ] )
    log_L = sn_like.log_likelihood( theta )

    assert isinstance( log_L, float )
    assert not np.isnan( log_L )
    assert not np.isinf( log_L )