import argparse
import os
import time
import numpy as np
import yaml

import cosmology
import data_extraction
import likelihood
import nested_sample as ns
import write_output


def main(args):

    initial_time = time.perf_counter()

    print(100 * "=" + "")
    print("Loading Initial Setup")
    print(100 * "=" + "\n")

    with open(args.yaml, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    print(f"{yaml.dump(config, sort_keys=False, allow_unicode=True)}")

    config_cosmology = config["cosmology"]
    config_experiment = config["experiment_config"]
    config_parameters = config["parameters"]
    config_output = config["output"]

    print(100 * "=" + "")
    print("Starting data Extraction")
    print(100 * "=" + "")

    z, mu_obs, mu_err, covariance_data = data_extraction.extraction(
        config_experiment
    )

    print(100 * "=" + "")
    print("Extraction ran Successfully")
    print(100 * "=" + "")

    c = config_cosmology["c"]
    H0 = cosmology.constants(config_cosmology["h"])

    print(f" Using H0 = {H0} km/s/Mpc")

    dec_cov = likelihood.cholesky_decomp(covariance_data)

    nlive = config_experiment["nalive"]
    seed = config_experiment["seed"]
    dlogz = config_experiment["ratio_conv"]
    max_steps = config_experiment["max_steps"]
    target_acceptance = config_experiment.get("acceptance", 0.3)
    resume_flag = config_experiment.get("resume_flag", True)
    dump_length = config_experiment.get("dump_length", 500)
    outfile_flag = config_experiment.get("outfile_flag", True) and config_output.get("write_output", True)

    checkpoint_file, bestfit_file, chain_file = write_output.resolve_output_paths(config)

    param_names, bounds = ns.extract_bounds(config_parameters)

    print(100 * "=" + "")
    print("Starting Nested Sampling Loop")
    print(100 * "=" + "")

    results = ns.run_nested_sampling(
        nlive=nlive,
        bounds=bounds,
        dec_cov=dec_cov,
        z=z,
        mu_obs=mu_obs,
        H0=H0,
        c=c,
        Omega_r=config_cosmology.get("Omega_r", 0.0),
        w0_fld=config_cosmology["w0_fld"],
        wa_fld=config_cosmology["wa_fld"],
        dlogz=dlogz,
        max_steps=max_steps,
        target_acceptance=target_acceptance,
        seed=seed,
        resume_flag=resume_flag,
        checkpoint_file=checkpoint_file,
        dump_length=dump_length,
    )

    print(100 * "=" + "")
    print("Sampling Finished Successfully")
    print(100 * "=" + "")
    print(f"Total Iterations: {results['n_iterations']}")
    print(f"Log-Evidence (ln Z): {results['logZ']:.4f}")

    samples = results["samples"]
    weights = results["weights"]
    log_L = results["log_likelihoods"]

    best_idx = np.argmax(log_L)
    best_params = samples[best_idx]

    means = np.average(samples, weights=weights, axis=0)
    stds = np.sqrt(
        np.average((samples - means) ** 2, weights=weights, axis=0)
    )

    print("\n" + 100 * "=" + "")
    print("Parameter Estimation Results (Posterior Summary)")
    print(100 * "=" + "")
    for name, mean, std, best in zip(param_names, means, stds, best_params):
        print(f" {name:25s} | Mean: {mean:8.4f} ± {std:6.4f} | MAP: {best:8.4f}")
    print(100 * "=" + "\n")

    if outfile_flag:
        write_headers = config_output.get("headers", True)

        print(100 * "=" + "")
        print(f"Saving Output Files in: {config_output.get('root', 'output')}")
        print(100 * "=" + "")

        write_output.write_chain(
            chain_filepath=chain_file,
            results=results,
            param_names=param_names,
            write_headers=write_headers,
        )

        write_output.write_best_fit(
            bestfit_filepath=bestfit_file,
            best_params=best_params,
            config_cosmology=config_cosmology,
            z_max=np.max(z),
            H0=H0,
            c=c,
            write_headers=write_headers,
        )

        print(100 * "=" + "\n")
    else:
        print(" -> Outfile flag set to False. Skipping file output creation.\n")

    final_time = time.perf_counter()
    delta_t = final_time - initial_time

    print(100 * "=" + "")
    print(f"Execution time: {delta_t:.6f} s.")
    print(100 * "=" + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--yaml", type=str, required=True, help="Initial Setup (.yaml)"
    )

    arg = parser.parse_args()

    main(arg)