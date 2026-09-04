import cosmology
import numpy as np
import scipy as sp



def cholesky_decomp(covmatrix):
    
    decomposed_cov = sp.linalg.cho_factor(covmatrix)

    return decomposed_cov

def supernovae_likelihood(
    dec_covmatrix, obs_z, obs_mu, 
    H0, c, Omega_m, Omega_lambda, Omega_r, w0_fld, wa_fld, M):

    z_grid = np.linspace(0, np.max(obs_z), 1000)

    dDc, Omega_k = cosmology.dDc_dz(
        H0=H0,
        c=c,
        Omega_m=Omega_m,
        Omega_lambda=Omega_lambda,
        Omega_r=Omega_r,
        w0_fld=w0_fld,
        wa_fld=wa_fld,
        z=z_grid
    )

    if dDc is None:
        return -np.inf

    distances_dict = cosmology.distances(dDc, z_grid, Omega_k, c, H0) 
    DL_grid = distances_dict["luminosity distance"]
    
    DL_obs = np.interp(obs_z, z_grid, DL_grid)

    mu_th = 5.0 * np.log10(np.maximum(DL_obs, 1e-10)) + 25.0 + M

    delta_mu = obs_mu - mu_th
    cho_sol = sp.linalg.cho_solve(dec_covmatrix, delta_mu)
    chi2 = np.dot(delta_mu, cho_sol)

    log_likelihood = -0.5 * chi2
    return log_likelihood