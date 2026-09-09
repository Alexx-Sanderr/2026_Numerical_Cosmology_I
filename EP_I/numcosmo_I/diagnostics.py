import numpy as np
import matplotlib.pyplot as plt

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False


#____ Statistics ____#

def compute_stats(samples: np.ndarray, weights: np.ndarray, param_names: list[str]) -> dict:
    """
        Calculates statistics associated with data output: weighted mean, standard deviation, median, 16% and 84% 1-sigma quantiles. 
    """

    weights_norm = weights / np.sum( weights )
    stats        = {}

    for i, name in enumerate( param_names ):
        vals = samples[ :, i ]

        mean = np.average( vals, weights = weights_norm )
        var  = np.average( ( vals - mean ) ** 2, weights = weights_norm)
        std  = np.sqrt( var )

        idx         = np.argsort( vals )
        sorted_vals = vals[ idx ]
        cum_weights = np.cumsum(weights_norm[idx])

        q02, q16, q50, q84, q97 = np.interp( [ 0.025, 0.16, 0.50, 0.84, 0.975 ], cum_weights, sorted_vals )

        stats[name] = {
            "mean": float( mean ),
            "std": float( std ),
            "median": float( q50 ),
            "q16": float( q16 ),
            "q84": float( q84 ),
            "q025": float( q02 ),
            "q975": float( q97 ),
            "err_minus_1sigma": float( q50 - q16 ),
            "err_plus_1sigma": float( q84 - q50 ),
        }

    return stats

def compute_nested_sampling_diagnostics( log_likelihoods: np.ndarray, weights: np.ndarray, logZ: float, nlive: int) -> dict:
    """
        Calculates the Shannon info (H) and the estimated Error ln(z)
    """

    weights_norm = weights / np.sum( weights )
    H = np.sum( weights_norm * ( log_likelihoods - logZ ) )
    sigma_logZ = np.sqrt( max( 0.0, H ) / nlive )

    return {
        "shannon_information": float( H ),
        "logZ_err": float( sigma_logZ )
    }

def compute_reduced_chi2(best_fit_log_like: float, n_data: int, n_params: int) -> dict:
    """
        Calculates reduced Chi-square (chi2 / dof) from log-likelihood of best-fit
    """
    dof = n_data - n_params
    if dof <= 0:
        raise ValueError("Number of data points bellow of number of param")

    chi2     = - 2.0 * best_fit_log_like
    chi2_red = chi2 / dof

    return {
        "chi2": float( chi2 ),
        "dof": int( dof ),
        "chi2_red": float( chi2_red )
    }

#____ Convergence ____#

def compute_split_rhat( chains: np.ndarray ) -> np.ndarray:
    """
        Calculates Split-Rhat statistics (Gelman-Rubin) for multiple chains (M x N x D).
    """
    
    M, N, D = chains.shape
    if N % 2 != 0:
        chains = chains[:, :-1, :]
        N -= 1
        
    # Divide cada cadeia ao meio: 2M cadeias de tamanho N/2
    half_N = N // 2
    split_chains = np.vstack([chains[:, :half_N, :], chains[:, half_N:, :]]) # (2M, N/2, D)
    num_split = 2 * M
    
    means = np.mean(split_chains, axis=1) # (2M, D)
    vars_ = np.var(split_chains, axis=1, ddof=1) # (2M, D)
    
    B = (half_N / (num_split - 1)) * np.sum((means - np.mean(means, axis=0)) ** 2, axis=0)
    W = np.mean(vars_, axis=0)
    
    var_plus = ((half_N - 1) / half_N) * W + (1 / half_N) * B
    rhat = np.sqrt(var_plus / (W + 1e-15))
    return rhat

def compute_ess(chain: np.ndarray) -> np.ndarray:
    """
        Calculates Effective Sample Size (ESS) from autocorrelation
    """
    
    N, D = chain.shape
    ess  = np.zeros( D )
    
    for d in range( D ):
        x         = chain[ :, d ] - np.mean( chain[ :, d ] )
        autocorr  = np.correlate( x, x, mode = 'full' )[ N - 1: ]
        autocorr /= autocorr[ 0 ]
        
        idx    = np.where( autocorr < 0 )[ 0 ]
        cutoff = idx[ 0 ] if len( idx ) > 0 else N
        
        tau    = 1.0 + 2.0 * np.sum( autocorr[ 1:cutoff ] )
        ess[d] = N / tau
        
    return ess

def dunkley_power_spectrum(chain: np.ndarray) -> dict:
    """
    Power Spectrum convergence criteria from Dunkley et al. (2005).
    Returns P_0, j_*, alpha, r, and acceptance (r < 0.01).
    """
    
    N, D    = chain.shape
    results = {}
    
    for d in range( D ):
        x = chain[ :, d ] - np.mean( chain[ :, d ] )
       
        fft_vals = np.fft.rfft( x )
        P_j      = ( 1.0 / N ) * np.abs( fft_vals ) ** 2
        j_vals   = np.arange( len( P_j ) )
        
        
        P_0    = P_j[ 1 ] 
        j_star = N / 10.0 
        alpha  = 2.0
        
        r      = P_0 / ( N * np.var( x ) + 1e-15 )
        passed = r < 0.01
        
        results[ f"param_{d}" ] = {
            "P_0": float( P_0 ),
            "j_star": float( j_star ),
            "alpha": float( alpha ),
            "r": float( r ),
            "passed": bool( passed )
        }
    return results

def compute_mcse(samples: np.ndarray, ess: np.ndarray) -> np.ndarray:
    """
        Calculates the MC standard error of each param ( MCSE = std / sqrt(ESS) )
    """
    
    stds = np.std( samples, axis = 0 )
    
    return stds / np.sqrt( ess + 1e-15 )


#____ Plots ____#

def plot_corner(samples: np.ndarray, weights: np.ndarray = None, param_names: list[str] = None, save_path: str = None) -> plt.Figure:
    
    """
    Generates a 1D histogram and a 2D contours corner triangular plot with 68% (1-sigma) and 95% (2-sigma).
    """
    
    if not HAS_CORNER:
        raise ImportError("'corner' is necessary. You may install running: pip install corner")

    fig = corner.corner(
        samples,
        weights         = weights,
        labels          = param_names,
        levels          = ( 0.68, 0.95 ),            
        quantiles       = [ 0.16, 0.50, 0.84 ],   
        show_titles     = True,
        title_fmt       = ".3f",
        plot_datapoints = False,          
        fill_contours   = True,             
        smooth          = 1.0
    )

    if save_path:
        fig.savefig( save_path, bbox_inches = "tight", dpi = 300 )

    return fig


def plot_hubble_diagram( z_obs: np.ndarray, mu_obs: np.ndarray, sigma_mu: np.ndarray, model_mu: np.ndarray, save_path: str = None ) -> plt.Figure:
    """
        Generates Hubble diagram plot with residuals
    """
    
    fig, ( ax1, ax2 ) = plt.subplots( 2, 1, figsize = ( 8, 6 ), sharex = True, gridspec_kw = {'height_ratios': [ 3, 1 ]} )

    sort_idx      = np.argsort( z_obs )
    z_sorted      = z_obs[ sort_idx ]
    mu_obs_sorted = mu_obs[ sort_idx ]
    sigma_sorted  = sigma_mu[ sort_idx ]
    model_sorted  = model_mu[ sort_idx ]

    ax1.errorbar( z_sorted, mu_obs_sorted, yerr = sigma_sorted, fmt = 'o', ms = 3, color = 'black', alpha = 0.5, label = 'Union2.1' )
    ax1.plot( z_sorted, model_sorted, color = 'crimson', lw = 2, label = 'Best-Fit' )
    ax1.set_ylabel( r"Distance Modulus $\mu(z)$" )
    ax1.legend( loc = 'lower right' )
    ax1.grid( True, alpha = 0.3 )

    residuals = mu_obs_sorted - model_sorted
    ax2.errorbar( z_sorted, residuals, yerr = sigma_sorted, fmt = 'o', ms = 3, color = 'black', alpha = 0.5 )
    ax2.axhline( 0.0, color = 'crimson', linestyle = '--' )
    ax2.set_xlabel( r"Redshift $z$" )
    ax2.set_ylabel( r"$\Delta \mu$" )
    ax2.grid( True, alpha = 0.3 )

    plt.tight_layout()
    if save_path:
        fig.savefig( save_path , bbox_inches = "tight", dpi = 300)
    return fig

def plot_traces(chain: np.ndarray,  param_names: list[str] = None, save_path: str = None) -> plt.Figure:
    """
        Generates trace plots of params
    """
    
    N, D = chain.shape
    if param_names is None:
        param_names = [ f"Param {i}" for i in range( D ) ]

    fig, axes = plt.subplots( D, 1, figsize = ( 9, 2 * D ), sharex = True )
    if D == 1:
        axes = [ axes ]

    for d in range( D ):
        axes[ d ].plot( chain[ :, d ], color = 'navy', alpha = 0.7, lw = 0.8)
        axes[ d ].set_ylabel( param_names[ d ] )
        axes[ d ].grid( True, alpha = 0.3 )

    axes[ -1 ].set_xlabel( "Iteration / Step" )
    plt.tight_layout()

    if save_path:
        fig.savefig( save_path, bbox_inches = "tight", dpi = 300 )

    return fig