# numcosmo_II: Structure Formation & Non-Linear Cosmology Dynamics V - 1.0

A Python framework for linear and non-linear cosmological structure formation, Standard Perturbation Theory (SPT), Lagrangian Perturbation Theory (1LPT / Zel'dovich approximation), spherical collapse dynamics, superhorizon potential evolution, and Boltzmann solver integrations (CLASS/CAMB), developed for the Numerical Cosmology I graduate course.

---

## Features

- **Linear Growth Dynamics (`numcosmo_II.cosmology`)**:
  * Solves linear perturbation ODEs for the growth factor $D(a)$ and logarithmic growth rate $f(a) \equiv \frac{d\ln D}{d\ln a}$.
  * Supports flat $\Lambda\text{CDM}$, Einstein-de Sitter (EdS), and open/curved FLRW background models with custom high-precision ODE integrators.

- **Gaussian Fields & Standard Perturbation Theory (`numcosmo_II.gaussian_field`, `numcosmo_II.spt_field`)**:
  * Generates 3D Gaussian random density fields and computes Standard Perturbation Theory (SPT) higher-order corrections.
  * Calculates 1D power spectra $P(k)$, bispectrum $B(k_1, k_2, k_3)$, and smoothed skewness $S_3(R)$ over multiple realizations.

- **Lagrangian Perturbation Theory (`numcosmo_II.lpt1`)**:
  * Computes 1LPT / Zel'dovich displacement fields on 3D grids using Gaussian random field realizations.
  * Particle-Mesh grid assignment algorithms (Nearest Grid Point - NGP / Cloud-In-Cell - CIC) for cosmic web density field reconstruction.
  * Calculates matter power spectra $P(k)$ with shot-noise subtraction.

- **Spherical Collapse Dynamics (`numcosmo_II.spherical_collapse`)**:
  * Integrates non-linear spherical collapse field equations across three equivalent formulations:
    - **Model A**: $(\delta, \delta_N)$
    - **Model B**: $(\delta, \theta)$
    - **Model C**: $(g, \theta)$, where $g \equiv 1/\delta$ prevents numerical overflow near turn-around and singularity.
  * Confirms analytical and numerical equivalence between Models A, B, and C with relative errors $\lesssim 10^{-7}$.
  * Implements a bisection shooting method (`CollapseThreshold`) to map the linear critical overdensity threshold $\delta_c(z)$ for $\Lambda\text{CDM}$ ($\delta_c(0) \approx 1.665$) and EdS ($\delta_c \approx 1.686$).

- **Boltzmann Solvers Integration (`numcosmo_II.boltzmann_map`)**:
  * Wrappers for CLASS and CAMB Boltzmann solvers to extract linear matter power spectra $P(k, z)$.
  * Automated precision and sampling convergence benchmarks.

---

## Repository Structure

```text
EP_II/
├── analysis/
│   ├── output/                            # Generated figure outputs and session diagnostics
│   ├── analysis.ipynb                     # Main notebook executing Ex 1, Ex 2, Ex 3, and Ex 5
│   ├── analysis_spherical_collapse.ipynb  # Dedicated analysis notebook for Ex 4 (Spherical Collapse)
│   └── analysis.pdf                       # PDF compiling the complete analysis of results
│
├── config/
│   └── setup.yaml                         # Central YAML setup for cosmology, SPT, LPT, and collapse parameters
│
├── numcosmo_II/                           # Core Python package
│   ├── __init__.py
│   ├── boltzmann_map.py                   # CLASS & CAMB wrappers and P(k) sampling
│   ├── cosmology.py                       # FLRW background & linear growth ODE solvers
│   ├── gaussian_field.py                  # Gaussian random field generator
│   ├── lpt1.py                            # 1LPT / Zel'dovich approximation & grid mass assignment
│   ├── spherical_collapse.py              # Non-linear collapse models (A, B, C) & shooting threshold
│   └── spt_field.py                       # Standard Perturbation Theory (SPT) & statistics
│
├── references/                            # Academic papers and exercise guidelines
│   ├── EP2.pdf
│   └── Pace_Meyer_&_Bartelmann_(2017).pdf
│
├── tests/                                 # Automated unit & integration tests
│   ├── conftest.py
│   ├── test_boltzmann_map.py
│   ├── test_cosmology_growth.py
│   ├── test_gaussian_field.py
│   ├── test_lpt1.py
│   └── test_spt_field.py
│
├── pyproject.toml                         # Package installation & metadata
├── README.md                              # Project documentation
└── requirements.txt                       # Project dependencies
```

---

## Installation

1. Clone the repository and enter the project folder:
   cd EP_II/

2. Activate your virtual environment and install the package:
   pip install .

---

## Running Automated Tests

The test suite validates linear growth ODE limits, 1LPT displacement field calculations, spherical collapse equivalence across Models A, B, and C, shooting threshold convergence, and Boltzmann solver integrations.

Run all unit and integration tests with pytest:


pytest -v tests/

---

```python
import numpy as np
import numcosmo_II as nc

# 1. Initialize Cosmology & Linear Growth
cosmo = nc.FLRW(
    h=0.6737,
    Omega_m=0.315,
    Omega_lambda=0.685
)

a_grid = np.linspace(1e-4, 1.0, 1000)
growth_sol = cosmo.solve_growth(a=a_grid, y0=[1e-4, 1.0])
D_a, f_a = cosmo.get_growth_observables(growth_sol, a_eval=np.array([1.0]))

print(f"Growth Rate f(a=1.0): {f_a[0]:.4f}")

# 2. Spherical Collapse Threshold via Shooting Method
sc_model = nc.SphericalCollapse(cosmo)
threshold_finder = nc.spherical_collapse.CollapseThreshold(
    sc_model,
    model="C",
    ai=1e-4,
    delta_stop=1e4,
    f_i=f_a[0],
    n_points=1000
)

delta_i, delta_c = threshold_finder.find(z_c=0.0)
print(f"Initial Overdensity delta_i: {delta_i:.6e}")
print(f"Linear Collapse Threshold delta_c(z=0): {delta_c:.4f}")
```

* Complete analysis found at

  analysis/analysis.ipynb
  analysis/analysis_spherical_collapse.ipynb
  analysis/analysis.pdf


