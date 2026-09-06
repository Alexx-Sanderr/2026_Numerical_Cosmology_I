import numpy as np
import os
import pickle


from typing import Callable

class NestedSampler:
    """
        Generic nested sampler
    """

    def __init__( self, log_likelihood_fn: Callable[[np.ndarray], float], prior_transform_fn: Callable[[np.ndarray], np.ndarray], ndim: int, nlive: int = 500, dlogz: float = 1e-4, max_steps: int = 50000, target_acceptance: float = 0.3, num_itx: int = 100, seed: int | None = None):

        self.log_like        = log_likelihood_fn
        self.prior_transform = prior_transform_fn
        self.ndim            = ndim
        self.nlive           = nlive
        self.dlogz           = dlogz
        self.max_steps       = max_steps
        self.target_acc      = target_acceptance
        self.num_itx         = num_itx

        if seed is not None:
            np.random.seed( seed )

    def _init_live_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Initial hypercube [0, 1]^D population
        """

        u_points    = np.random.uniform( 0.0, 1.0, size = ( self.nlive, self.ndim ) )
        live_points = np.zeros( ( self.nlive, self.ndim ) )
        log_L       = np.zeros( self.nlive )

        for i in range(self.nlive):
            live_points[ i ] = self.prior_transform( u_points[ i ] )
            log_L[ i ]       = self.log_like( live_points[ i ] )

        return u_points, live_points, log_L

    def _mcmc_reposition(
        self, u_start: np.ndarray, L_worst: float, step_size: float
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
            Random walking inside volume L > L_worst
        """

        u_curr     = u_start.copy()
        theta_curr = self.prior_transform( u_curr )
        L_curr     = self.log_like( theta_curr )
        n_accepted = 0

        for _ in range( self.num_itx ):
            u_cand = u_curr + np.random.normal( 0.0, step_size, size = self.ndim )

            if np.any( u_cand < 0.0 ) or np.any( u_cand > 1.0 ):
                continue

            theta_cand  = self.prior_transform( u_cand )
            L_cand      = self.log_like( theta_cand )

            if L_cand > L_worst:
                u_curr      = u_cand
                theta_curr  = theta_cand
                L_curr      = L_cand
                n_accepted += 1

        acc_rate = n_accepted / self.num_itx
        
        return u_curr, theta_curr, L_curr, acc_rate


    def run(self, resume_flag: bool = True, checkpoint_file: str = "checkpoint.resume", dump_length: int = 500) -> dict:
        """
            Nested Sampler engine with checkpoint support
        """

        if resume_flag and os.path.exists( checkpoint_file ):
            print( f"-> Resuming execution from checkpoint file: {checkpoint_file}"   )
            with open( checkpoint_file, "rb" ) as f:
                chk = pickle.load( f )

            step        = chk[ "step" ]
            logZ        = chk[ "logZ" ]
            X           = chk[ "X" ]
            dead_points = chk[ "dead_points" ]
            u_points    = chk[ "u_points" ]
            live_points = chk[ "live_points" ]
            log_L       = chk[ "log_L" ]
            step_size   = chk[ "step_size" ]
            delta_logZ  = chk[ "delta_logZ" ]
            np.random.set_state( chk[ "rng_state" ] )

        else:
            print( "->  Starting new Nested Sampling run ..." )

            u_points, live_points, log_L = self._init_live_points()
            X           = 1.0
            logZ        = -np.inf
            dead_points = []
            step        = 0
            delta_logZ  = np.inf
            step_size   = 0.1

        print( "\n" + "=" * 80 )
        print( f"{'Step':>7} | {'logZ':>10} | {'dlogZ':>10} | {'L_worst':>10} | {'Step Size':>10} | {'Acc Rate':>11}" )
        print( "=" * 80 )

        while delta_logZ > self.dlogz and step < self.max_steps:
            step += 1

            L_worst_idx = np.argmin( log_L )
            L_worst     = log_L[ L_worst_idx ]
            theta_worst = live_points[ L_worst_idx ].copy()

            X_prev = X
            X      = np.exp( - step / self.nlive )
            w_k    = X_prev - X
            log_w  = np.log( w_k )

            logZ = np.logaddexp( logZ, L_worst + log_w )
            dead_points.append( ( theta_worst, L_worst, log_w ) )

            logZ_remain = np.max( log_L ) + np.log( X )
            logZ_tot    = np.logaddexp( logZ, logZ_remain )
            delta_logZ  = logZ_tot - logZ

            valid_indices = [
                i for i in range( self.nlive ) if i != L_worst_idx
            ]
            rand_idx      = np.random.choice( valid_indices )
            u_start       = u_points[ rand_idx ]

            u_new, theta_new, L_new, acc_rate = self._mcmc_reposition(
                u_start = u_start, L_worst = L_worst, step_size = step_size
            )

            if acc_rate   > self.target_acc:
                step_size *= 1.05
            elif acc_rate < self.target_acc:
                step_size *= 0.95

            u_points[ L_worst_idx ]    = u_new
            live_points[ L_worst_idx ] = theta_new
            log_L[ L_worst_idx ]       = L_new

            if step % 100 == 0 or step == 1:
                print(
                    f"{step:7d} | {logZ:10.3f} | {delta_logZ:10.5f} | {L_worst:10.3f} | {step_size:10.4f} | {acc_rate:11.2%}"
                )
                
            if resume_flag and ( step % dump_length == 0 ):
                chk = {
                    "step": step,
                    "X": X,
                    "logZ": logZ,
                    "dead_points": dead_points,
                    "u_points": u_points,
                    "live_points": live_points,
                    "log_L": log_L,
                    "step_size": step_size,
                    "delta_logZ": delta_logZ,
                    "rng_state": np.random.get_state(),
                }
                with open( checkpoint_file, "wb" ) as f:
                    pickle.dump( chk, f )
                    
        w_final     = X / self.nlive
        log_w_final = np.log( w_final )

        for i in range( self.nlive ):
            logZ = np.logaddexp( logZ, log_L[ i ] + log_w_final )
            dead_points.append( ( live_points[ i ].copy(), log_L[ i ], log_w_final ) )

        if resume_flag and os.path.exists( checkpoint_file ):
            os.remove( checkpoint_file )

        samples         = np.array( [ pt[ 0 ] for pt in dead_points ] )
        log_likelihoods = np.array( [ pt[ 1 ] for pt in dead_points ] )
        log_weights     = np.array( [ pt[ 2 ] for pt in dead_points ] )

        log_p   = log_likelihoods + log_weights - logZ
        weights = np.exp( log_p )

        return {
            "samples": samples,
            "log_likelihoods": log_likelihoods,
            "weights": weights,
            "logZ": logZ,
            "n_iterations": step,
        }
            