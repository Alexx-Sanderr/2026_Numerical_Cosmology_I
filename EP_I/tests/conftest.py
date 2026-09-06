import numpy as np
import pytest

@pytest.fixture
def mock_sn_data(tmp_path):
    """
        Generates mock Supernovae data (z, mu, sigma_mu).
    """
    z_obs    = np.array( [ 0.01, 0.1, 0.5, 0.8, 1.0 ] )
    mu_obs   = np.array( [ 33.0, 38.2, 42.5, 43.8, 44.5 ] )
    sigma_mu = np.full_like( z_obs, 0.15 )
    
    return z_obs, mu_obs, sigma_mu
    

@pytest.fixture
def mock_bounds():
    """
        Return test param limits 
    """
    return {
        "Omega_m": ( 0, 1.0 ),
        "Omega_lambda": ( 0.0, 1.0 ),
    }


@pytest.fixture
def toy_gaussian():
    """
        Gaussian 2D function to isolate sampler validation.
    """
    
    ndim = 2
    target_mean = np.array([0.5, 0.5])

    def log_like(theta):
        return -0.5 * np.sum((theta - target_mean) ** 2 / 0.01)

    def prior_transform(u):
        return u  

    return log_like, prior_transform, ndim, target_mean