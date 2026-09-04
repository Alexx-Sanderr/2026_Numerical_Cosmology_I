import glob
import os
from datetime import datetime
import numpy as np
import cosmology


def resolve_output_paths(config):
    out_cfg = config.get("output", {})
    exp_cfg = config.get("experiment_config", {})

    root = out_cfg.get("root", ".")
    base_name = out_cfg.get("base_name", "output").rstrip("_")
    resume_flag = exp_cfg.get("resume_flag", True)

    os.makedirs(root, exist_ok=True)

    pattern = os.path.join(root, f"{base_name}_*_checkpoint.resume")
    existing_checkpoints = sorted(glob.glob(pattern)) if resume_flag else []

    if existing_checkpoints:
        checkpoint_file = existing_checkpoints[-1]
        filename = os.path.basename(checkpoint_file)
        timestamp = filename[len(base_name) + 1 : -len("_checkpoint.resume")]
        print(f" -> Found existing checkpoint matching base_name '{base_name}': {filename}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(
            root, f"{base_name}_{timestamp}_checkpoint.resume"
        )

    bestfit_file = os.path.join(
        root, f"{base_name}_{timestamp}_dz_bestfit.dat"
    )
    chain_file = os.path.join(
        root, f"{base_name}_{timestamp}_supernovae.chain"
    )

    return checkpoint_file, bestfit_file, chain_file


def write_chain(chain_filepath, results, param_names, write_headers=True):
    samples = results["samples"]
    weights = results["weights"]
    log_L = results["log_likelihoods"]

    chain_data = np.column_stack((weights, -log_L, samples))

    clean_names = [p.split(".")[-1] for p in param_names]
    header_chain = (
        "Weight\t-LogLike\t" + "\t".join(clean_names)
        if write_headers
        else ""
    )

    np.savetxt(
        chain_filepath,
        chain_data,
        fmt="%.6e",
        delimiter="\t",
        header=header_chain,
    )
    print(f" -> Saved Chain File: {chain_filepath}")


def write_best_fit(
    bestfit_filepath,
    best_params,
    config_cosmology,
    z_max,
    H0,
    c,
    write_headers=True,
):
    om_best, ol_best, m_abs_best = best_params
    Omega_r_val = config_cosmology.get("Omega_r", 0.0)
    w0_val = config_cosmology["w0_fld"]
    wa_val = config_cosmology["wa_fld"]

    z_grid = np.linspace(0.0, z_max, 300)

    dDc_arr, Omega_k = cosmology.dDc_dz(
        H0=H0,
        c=c,
        Omega_m=om_best,
        Omega_lambda=ol_best,
        Omega_r=Omega_r_val,
        z=z_grid,
        w0_fld=w0_val,
        wa_fld=wa_val,
    )

    if dDc_arr is not None:
        dist_dict = cosmology.distances(
            dDc_dz_array=dDc_arr, z=z_grid, Omega_k=Omega_k, c=c, H0=H0
        )

        Dc_best = dist_dict["comoving radial distance"]
        DA_best = dist_dict["angular diameter distance"]
        DL_best = dist_dict["luminosity distance"]

        with np.errstate(divide="ignore"):
            mu_theoretical = 5.0 * np.log10(DL_best) + 25.0 + m_abs_best
            mu_theoretical[0] = -np.inf

        bestfit_data = np.column_stack(
            (z_grid, Dc_best, DA_best, DL_best, mu_theoretical)
        )

        header_bestfit = (
            "z\tDc_Mpc\tDA_Mpc\tDL_Mpc\tmu_theoretical"
            if write_headers
            else ""
        )

        np.savetxt(
            bestfit_filepath,
            bestfit_data,
            fmt="%.6e",
            delimiter="\t",
            header=header_bestfit,
        )
        print(f" -> Saved Best-Fit D(z) File: {bestfit_filepath}")
    else:
        print(
            " -> Warning: Best-fit parameters yielded non-physical cosmology (E^2 <= 0)."
        )