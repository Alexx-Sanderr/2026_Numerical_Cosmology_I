import os
import pickle
import numpy as np
import likelihood


def extract_bounds(config_parameters):
    param_names = []
    bounds = []
    for key, val in config_parameters.items():
        if isinstance(val, dict) and "prior" in val:
            param_names.append(key)
            bounds.append([val["prior"]["min"], val["prior"]["max"]])
    return param_names, np.array(bounds)


def prior_transform(u_vector, bounds):
    mins = bounds[:, 0]
    maxs = bounds[:, 1]
    return mins + u_vector * (maxs - mins)


def init_live_points(
    nlive, bounds, dec_cov, z, mu_obs, H0, c, Omega_r, w0_fld, wa_fld, seed=None
):
    print(f" -> Generating {nlive} initial live points...")
    if seed is not None:
        np.random.seed(seed)

    ndim = len(bounds)
    u_points = np.random.uniform(0.0, 1.0, size=(nlive, ndim))
    live_points = np.zeros((nlive, ndim))

    for i in range(nlive):
        live_points[i] = prior_transform(u_points[i], bounds)

    log_L = np.zeros(nlive)
    for i in range(nlive):
        om = live_points[i, 0]
        ol = live_points[i, 1]
        m_abs = live_points[i, 2]

        log_L[i] = likelihood.supernovae_likelihood(
            dec_covmatrix=dec_cov,
            obs_z=z,
            obs_mu=mu_obs,
            H0=H0,
            c=c,
            Omega_m=om,
            Omega_lambda=ol,
            Omega_r=Omega_r,
            w0_fld=w0_fld,
            wa_fld=wa_fld,
            M=m_abs,
        )

    print(" -> Initial live points generated successfully.")
    return u_points, live_points, log_L


def mcmc_reposition(
    u_start,
    L_worst,
    bounds,
    dec_cov,
    z,
    mu_obs,
    H0,
    c,
    Omega_r,
    w0_fld,
    wa_fld,
    num_itx=20,
    step_size=0.1,
):
    ndim = len(bounds)
    u_curr = u_start.copy()
    theta_curr = prior_transform(u_curr, bounds)

    L_curr = likelihood.supernovae_likelihood(
        dec_covmatrix=dec_cov,
        obs_z=z,
        obs_mu=mu_obs,
        H0=H0,
        c=c,
        Omega_m=theta_curr[0],
        Omega_lambda=theta_curr[1],
        Omega_r=Omega_r,
        w0_fld=w0_fld,
        wa_fld=wa_fld,
        M=theta_curr[2],
    )

    n_accepted = 0

    for _ in range(num_itx):
        u_cand = u_curr + np.random.normal(0.0, step_size, size=ndim)

        if np.any(u_cand < 0.0) or np.any(u_cand > 1.0):
            continue

        theta_cand = prior_transform(u_cand, bounds)

        L_cand = likelihood.supernovae_likelihood(
            dec_covmatrix=dec_cov,
            obs_z=z,
            obs_mu=mu_obs,
            H0=H0,
            c=c,
            Omega_m=theta_cand[0],
            Omega_lambda=theta_cand[1],
            Omega_r=Omega_r,
            w0_fld=w0_fld,
            wa_fld=wa_fld,
            M=theta_cand[2],
        )

        if L_cand > L_worst:
            u_curr = u_cand
            theta_curr = theta_cand
            L_curr = L_cand
            n_accepted += 1

    acc_rate = n_accepted / num_itx
    return u_curr, theta_curr, L_curr, acc_rate


def run_nested_sampling(
    nlive,
    bounds,
    dec_cov,
    z,
    mu_obs,
    H0,
    c,
    Omega_r,
    w0_fld,
    wa_fld,
    dlogz=0.0001,
    max_steps=50000,
    target_acceptance=0.3,
    num_itx=20,
    seed=None,
    resume_flag=True,
    checkpoint_file="checkpoint.resume",
    dump_length=500,
):
    if resume_flag and os.path.exists(checkpoint_file):
        print(f" -> Resuming execution from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, "rb") as f:
            chk = pickle.load(f)

        step = chk["step"]
        X = chk["X"]
        logZ = chk["logZ"]
        dead_points = chk["dead_points"]
        u_points = chk["u_points"]
        live_points = chk["live_points"]
        log_L = chk["log_L"]
        step_size = chk["step_size"]
        np.random.set_state(chk["rng_state"])
        delta_logZ = chk["delta_logZ"]
        print(
            f" -> Loaded state at step {step} (logZ = {logZ:.3f}, dlogZ = {delta_logZ:.5f})"
        )
    else:
        print(" -> Starting new Nested Sampling run...")
        u_points, live_points, log_L = init_live_points(
            nlive, bounds, dec_cov, z, mu_obs, H0, c, Omega_r, w0_fld, wa_fld, seed
        )
        X = 1.0
        logZ = -np.inf
        dead_points = []
        step = 0
        delta_logZ = np.inf
        step_size = 0.1

    print("\n" + "=" * 80)
    print(
        f"{'Step':>7} | {'logZ':>10} | {'dlogZ':>10} | {'L_worst':>10} | {'Step Size':>10} | {'Acc Rate':>11}"
    )
    print("=" * 80)

    while delta_logZ > dlogz and step < max_steps:
        step += 1

        L_worst_idx = np.argmin(log_L)
        L_worst = log_L[L_worst_idx]
        theta_worst = live_points[L_worst_idx].copy()

        X_prev = X
        X = np.exp(-step / nlive)
        w_k = X_prev - X
        log_w = np.log(w_k)

        logZ = np.logaddexp(logZ, L_worst + log_w)
        dead_points.append((theta_worst, L_worst, log_w))

        logZ_remain = np.max(log_L) + np.log(X)
        logZ_tot = np.logaddexp(logZ, logZ_remain)
        delta_logZ = logZ_tot - logZ

        valid_indices = [i for i in range(nlive) if i != L_worst_idx]
        rand_idx = np.random.choice(valid_indices)
        u_start = u_points[rand_idx]

        u_new, theta_new, L_new, acc_rate = mcmc_reposition(
            u_start=u_start,
            L_worst=L_worst,
            bounds=bounds,
            dec_cov=dec_cov,
            z=z,
            mu_obs=mu_obs,
            H0=H0,
            c=c,
            Omega_r=Omega_r,
            w0_fld=w0_fld,
            wa_fld=wa_fld,
            num_itx=num_itx,
            step_size=step_size,
        )

        if acc_rate > target_acceptance:
            step_size *= 1.05
        elif acc_rate < target_acceptance:
            step_size *= 0.95

        u_points[L_worst_idx] = u_new
        live_points[L_worst_idx] = theta_new
        log_L[L_worst_idx] = L_new

        if step % 100 == 0 or step == 1:
            print(
                f"{step:7d} | {logZ:10.3f} | {delta_logZ:10.5f} | {L_worst:10.3f} | {step_size:10.4f} | {acc_rate:11.2%}"
            )

        if resume_flag and (step % dump_length == 0):
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
            with open(checkpoint_file, "wb") as f:
                pickle.dump(chk, f)
            print(f" [Checkpoint saved at step {step}]")

    print("=" * 80)
    if delta_logZ <= dlogz:
        print(
            f" -> Convergence reached at step {step}! (dlogZ = {delta_logZ:.6f} <= {dlogz})"
        )
    else:
        print(
            f" -> Maximum number of steps reached ({max_steps}). Final dlogZ = {delta_logZ:.6f}"
        )

    w_final = X / nlive
    log_w_final = np.log(w_final)

    for i in range(nlive):
        logZ = np.logaddexp(logZ, log_L[i] + log_w_final)
        dead_points.append((live_points[i].copy(), log_L[i], log_w_final))

    if resume_flag and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(" -> Final checkpoint removed successfully.")

    samples = np.array([pt[0] for pt in dead_points])
    log_likelihoods = np.array([pt[1] for pt in dead_points])
    log_weights = np.array([pt[2] for pt in dead_points])

    log_p = log_likelihoods + log_weights - logZ
    weights = np.exp(log_p)

    print(f" -> Final log-evidence log(Z): {logZ:.4f}")
    print(f" -> Total samples generated: {len(samples)}")

    return {
        "samples": samples,
        "log_likelihoods": log_likelihoods,
        "weights": weights,
        "logZ": logZ,
        "n_iterations": step,
    }