# numcosmo_I: Cosmological Inference & Bayesian Diagnostics V - 1.0

A Python framework for cosmological parameter inference and MCMC/Nested Sampling diagnostics, developed for the Numerical Cosmology I graduate course. The library implements background FLRW dynamics, likelihood evaluations for Supernovae Ia datasets (e.g., Union2.1) using Cholesky decomposition, and convergence diagnostics for posterior sampling.

---

## Features

* Cosmology Core (numcosmo_I.cosmology): Statelessly computes FLRW background quantities (E(z), H(z), comoving distance chi(z), angular diameter distance D_A(z), luminosity distance D_L(z), distance modulus mu(z), lookback time, and comoving volume element).
* Likelihood Evaluation (numcosmo_I.likelihoods): Fast chi^2 likelihood calculations using Cholesky decomposition (scipy.linalg.cholesky) for stable matrix inversions on full covariance matrices.
* Sampling Engines (numcosmo_I.samplers): Modular Bayesian sampling support for both Metropolis-Hastings MCMC and Nested Sampling pipelines.
* Convergence Diagnostics (numcosmo_I.diagnostics):
  * Statistical Quantiles: Weighted mean, standard deviation, median, 16%/84% (1-sigma), and 2.5%/97.5% (2-sigma) confidence intervals.
  * Convergence Metrics: Split-Rhat (Gelman-Rubin statistic), Effective Sample Size (ESS), Monte Carlo Standard Error (MCSE), and the Dunkley et al. (2005) power spectrum convergence criterion (P_0, j_*, alpha, r).
  * Goodness-of-Fit: Reduced Chi-Square (chi^2_red) and Shannon information (H).
* Visualization Tools: Corner plots (1-sigma / 2-sigma contours), Hubble diagrams with residual subplots, and parameter trace plots.

---

## Repository Structure
```text
EP_I/
├── config/
│   ├── executable.py              # Generic example exec (python3 executable.py -y file.yaml)
│   └── Supernovae_setup.yaml      # Configuration for dataset paths and priors
│
├── data/
│   ├── SCPUnion2.1_covmat_sys.txt # Union2.1 covariance matrix with systematics
│   └── SCPUnion2.1_mu_vs_z.txt    # Union2.1 Distance and redshift data
│
├── examples_and_analysis/
│   ├── Output/                    # Output directory with test output files
│   └── Analysis.ipynb             # Jupyter notebook with package usage examples
│
├── numcosmo_I/                    # Core Python package
│   ├── __init__.py
│   ├── cosmology.py               # FLRW cosmology calculations
│   ├── likelihoods.py             # SNe Ia likelihood & Cholesky solver
│   ├── samplers.py                # Sampling algorithms
│   └── diagnostics.py             # Convergence metrics & plotters
│
├── tests/                         # Automated unit & integration tests
│   ├── conftest.py
│   ├── test_cosmology.py
│   ├── test_diagnostics.py
│   ├── test_likelihood.py
│   └── test_samplers.py
|
├── pyproject.toml                 # Package installation & metadata
│
├── README.md
│
└── requirements.txt               # Required packages 
``

---

## Installation

1. Clone the repository and enter the project folder:
   cd EP_I/

2. Activate your virtual environment and install the package:
   pip install .

---

## Running Automated Tests

The test suite validates cosmological distance limits ( E(0) = 1, flat limit D_M = chi, D_L = ( 1 + z )^2 D_A ), Cholesky likelihood evaluations, sampler consistency, and statistical diagnostic outputs.

Run all 22 unit and integration tests with pytest:

pytest tests/

---

## Usage Example

```python
import numpy as np
from numcosmo_I.cosmology import FLRW

cosmo = FLRW(
            h            =  0.70, 
            Omega_m      = 0.3, 
            Omega_lambda = 0.7
)

d_L = cosmo.luminosity_distance( 0.5 )
print( f"Luminosity Distance at z = 0.5: {d_L:.3f} Mpc" )

mu_val = cosmo.distance_modulus( 0.5 )
print( f"Distance Modulus at z = 0.5: {mu_val:.3f} mag" )
```

* Complete analysis found at

  examples_and_analysis/Analysis.ipynb
  
