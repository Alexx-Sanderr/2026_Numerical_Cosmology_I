import os
import numpy as np
import pytest
from numcosmo_I.likelihoods import LikelihoodSupernovae
from numcosmo_I.samplers import NestedSampler


def test_sampler_toy_gaussian(toy_gaussian):
    """
        Ensures sample convergence in an analytical Gaussian 
    """
    log_like, prior_transform, ndim, _ = toy_gaussian

    sampler = NestedSampler(
        log_likelihood_fn  = log_like,
        prior_transform_fn = prior_transform,
        ndim               = ndim,
        nlive              = 50,
        dlogz              = 0.1,
        max_steps          = 500,
        seed               = 42,
    )

    results = sampler.run( resume_flag = False )

    assert "samples" in results
    assert "logZ"    in results
    assert "weights" in results
    assert results[ "samples" ].shape[ 1 ] == ndim


def test_integration_sampler_and_likelihood(mock_sn_data, mock_bounds):
    """
        Integration between LikelihoodSupernovae and NestedSampler classes
    """
    z, mu, sigma = mock_sn_data
    param_names  = list( mock_bounds.keys() )
    ndim         = len( param_names )

    sn_like = LikelihoodSupernovae(
        z_obs = z, mu_obs = mu, sigma_mu = sigma, param_names = param_names
    )
    def prior_transform_fn(u):
        return sn_like.prior_transform( u, mock_bounds, param_names )

    def log_likelihood_fn(theta):
        return sn_like.log_likelihood(theta)

    sampler = NestedSampler(
        log_likelihood_fn  = log_likelihood_fn,
        prior_transform_fn = prior_transform_fn,
        ndim               = ndim,
        nlive              = 20,
        dlogz              =0.5,
        max_steps          = 100,
        seed               = 42,
    )

    results = sampler.run( resume_flag = False )

    assert results[ "n_iterations" ]                                    > 0
    assert results[ "samples" ].shape[ 1 ]                             == 2
    assert pytest.approx( np.sum( results[ "weights" ] ), abs = 1e-3 ) == 1.0


def test_checkpoint_and_resume(tmp_path, toy_gaussian):
    """
        Checkpoint file functionality validation
     """
    log_like, prior_transform, ndim, _ = toy_gaussian
    checkpoint_file                    = str( tmp_path / "test_checkpoint.resume" )

    sampler = NestedSampler(
        log_likelihood_fn  = log_like,
        prior_transform_fn = prior_transform,
        ndim               = ndim,
        nlive              = 30,
        max_steps          = 50,
        seed               = 42,
    )

    _ = sampler.run(
        resume_flag = True, checkpoint_file = checkpoint_file, dump_length = 10
    )

    if os.path.exists( checkpoint_file ):
        results_resumed = sampler.run(
            resume_flag = True, checkpoint_file = checkpoint_file
        )
        assert results_resumed[ "n_iterations" ] >= 50
        assert not os.path.exists( checkpoint_file )