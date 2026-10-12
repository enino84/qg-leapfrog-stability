# qg-leapfrog-stability

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22168803.svg)](https://doi.org/10.5281/zenodo.22168803)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**QG-Python: a reproducible 1.5-layer quasi-geostrophic modelling service
with spectral stability and filter sensitivity diagnostics**

This repository contains the QG-Python model, the analysis scripts, and the
shell drivers associated with the Data Services Article published in
*Geoscience Data Journal* (Royal Meteorological Society / Wiley):

> Nino-Ruiz, E. D. (2026). **QG-Python: A Reproducible 1.5-Layer
> Quasi-Geostrophic Modelling Service with Direct Sparse Solvers, Spectral
> Stability Diagnostics, and Filter Sensitivity Data**.
> *Geoscience Data Journal*. doi:[10.1002/gdj3.70107](https://doi.org/10.1002/gdj3.70107)

The archived snapshot of this repository **together with the simulation
outputs** is the citable version of record:
doi:[10.5281/zenodo.22168803](https://doi.org/10.5281/zenodo.22168803)

---

## Install

```bash
pip install qg-python          # model only (numpy, scipy)
pip install "qg-python[all]"   # + matplotlib and netCDF4 for the viewer and .nc output
```

```python
from qg_model import QGModel, QGParams

prm = QGParams(mrefin=6, tend=1000, dtout=20, scheme='leapfrog',
               ra_alpha=0.1, raw_filter=True, outfname='qg_run.npz')
psi, q = QGModel(prm).run('fourier', seed=42)
```

The same model is available on the command line as `qg-python` (run `qg-python --help`).
To reproduce the experiments of the paper, clone this repository and run `bash run_all.sh`.

---

## Overview

We implement the Leapfrog scheme with Robert-Asselin (RA) and
Robert-Asselin-Williams (RAW) filters in a 1.5-layer quasi-geostrophic (QG)
model, and carry out a systematic numerical analysis covering:

- Replacement of the original Fortran multigrid Helmholtz solver with a direct
  sparse LU factorization (SuperLU via SciPy), enabling unambiguous stability analysis
- Eigenspectrum of the linearized operator $J_\mathcal{F}(q_\mathrm{eq})$ at two
  grid resolutions (49×49 and 97×97), revealing resolution-dependent deviation
  from the classical von Neumann bound
- Maximum stable time step $\Delta t_{\max}$ as a function of grid spacing $h$
- First estimate of the maximum Lyapunov exponent for this model
  ($\sigma_1 = 0.0318$, $\tau_L = 31.5$ time units)
- Statistical uncertainty propagation: kinetic energy and enstrophy PDFs,
  energy spectra, and KL divergences across schemes
- Sensitivity of filter-induced dissipation to the RA parameter
  $\alpha \in \{0.05, 0.10, 0.20, 0.40\}$

The QG model is a faithful Python port of the QG-C model by
[Pavel Sakov (Bureau of Meteorology)](https://github.com/sakov/qg-c),
extended with the Leapfrog family of integrators.

---

## Repository structure

```
qg-leapfrog-stability/
│
├── qg_model.py                  # QG model — all time-stepping schemes
├── qg_viewer.py                 # Interactive field viewer (.npz files)
├── run_all.sh                   # Single script to reproduce all results
│
├── analysis/
│   ├── plot_style.py            # Shared publication style settings
│   ├── jacobian_spectrum.py     # Build J_F and compute eigenspectrum
│   ├── stability_regions.py     # Plot eigenspectrum + stability regions
│   ├── lyapunov.py              # Benettin algorithm for Lyapunov exponent
│   ├── uncertainty.py           # PDFs, power spectra, KL divergence
│   ├── plot_fields.py           # Spatial snapshots of streamfunction
│   └── plot_alpha_sensitivity.py  # RA filter sensitivity figures
│
├── experiments/
│   ├── run_reference.sh         # Spin-up + RK4 reference run
│   ├── run_leapfrog_long.sh     # LF+RA and LF+RAW main runs
│   ├── run_leapfrog_sweep.sh    # RA filter alpha sweep
│   └── run_analysis.sh          # Eigenspectrum + Lyapunov diagnostics
│
├── data/
│   ├── README.md                # Exact command line behind every run
│   ├── blowup_times.txt         # dt=1.0 instability onset, LF+RA / LF+RAW
│   ├── lyapunov_results.txt     # Benettin MLE summary
│   └── uncertainty_results.txt  # KE / enstrophy / KL divergence summary
│
├── paper/
│   └── README.md                # Pointer to the published article
│
├── figures/                     # Generated locally — not tracked by git
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
```

Install with:

```bash
pip install -r requirements.txt
```

No additional compiled extensions are required. The Helmholtz solver uses
`scipy.sparse.linalg.factorized` (SuperLU backend).

---

## Reproducing the results

All steps are collected in a single script. Run from the root of the repository:

```bash
bash run_all.sh
```

The script executes the following steps in order:

| Step | Description | Runtime (approx.) |
|------|-------------|-------------------|
| 1 | Spin-up from rest (mrefin=6, tend=100000, RK4, dt=1) | ~45 min |
| 2 | RK4 reference run (dt=1, t=100000→110000) | ~25 min |
| 3 | LF+RA main run (α=0.10, dt=0.5) | ~12 min |
| 4 | LF+RAW main run (α=0.10, dt=0.5) | ~12 min |
| 5 | Alpha sensitivity sweep (α=0.05, 0.20, 0.40) | ~36 min |
| 6 | Jacobian eigenspectrum, mrefin=6 (97×97) | ~2 min |
| 7a | Spin-up from rest, mrefin=5 (dt=0.1) | ~45 min |
| 7b | Jacobian eigenspectrum, mrefin=5 (49×49) | ~1 min |
| 8 | Lyapunov exponent (Benettin, 50000 steps) | ~90 min |
| 9 | All figures | ~1 min |

Total runtime on a modern laptop: approximately 4–5 hours.

> **Note:** The mrefin=5 spin-up (Step 7a) requires `dt=0.1` — larger time steps
> cause blow-up during spin-up at this resolution.

### Running individual steps

The pipeline is also split into four self-contained drivers under
`experiments/`, which can be run in isolation for partial reproduction:

```bash
bash experiments/run_reference.sh       # spin-up + RK4 reference
bash experiments/run_leapfrog_long.sh   # LF+RA and LF+RAW main runs
bash experiments/run_leapfrog_sweep.sh  # alpha sensitivity sweep
bash experiments/run_analysis.sh        # eigenspectrum + Lyapunov
```

Each command in `run_all.sh` is likewise self-contained. For example, to
recompute only the eigenspectrum:

```bash
python analysis/jacobian_spectrum.py --restart qg_zero.npz --neigs 200 --no-matrix --out jacobian_spectrum.npz
```

---

## Key results

| Quantity | Value |
|----------|-------|
| Grid (main experiments) | 97×97 (mrefin=6), $h=0.01042$ |
| Analytical $\Delta t_{\max}^{LF}$ (mrefin=6) | 0.090 |
| Numerical $\Delta t_{\max}^{LF}$ (mrefin=6) | 1.236 (factor **13.7×** larger) |
| Analytical $\Delta t_{\max}^{LF}$ (mrefin=5) | 0.362 |
| Numerical $\Delta t_{\max}^{LF}$ (mrefin=5) | 0.235 (factor **1.54×** smaller) |
| Maximum Lyapunov exponent $\sigma_1$ | 0.0318 |
| Lyapunov time $\tau_L$ | 31.5 time units |
| LF+RA kinetic energy bias (α=0.10) | −21% |
| LF+RA enstrophy bias (α=0.10) | −46% |
| LF+RAW kinetic energy bias | −2% |
| LF+RAW enstrophy bias | +4% |

---

## Time-stepping schemes

| Scheme | Order | RHS evals / step | Notes |
|--------|-------|-------------------|-------|
| `order1` | 1 | 1 | Explicit Euler |
| `order2` | 2 | 2 | Midpoint RK |
| `rk4` | 4 | 4 | Default in QG-C |
| `dp5` | 5 | 6 | Dormand-Prince |
| `leapfrog` | 2 | 1 | + RA or RAW filter |

The first Leapfrog step uses a single RK4 step as bootstrap to avoid
exciting the computational mode from the start.

### Leapfrog filter options

```bash
# Robert-Asselin filter (classic, Asselin 1972)
python qg_model.py --scheme leapfrog --ra-alpha 0.1 ...

# Robert-Asselin-Williams filter (less dissipative, Williams 2009)
python qg_model.py --scheme leapfrog --ra-alpha 0.1 --raw ...
```

---

## Data availability

The simulation output files (`.npz`) are several hundred megabytes and are
therefore not tracked by git. They are archived, together with a snapshot of
this repository, on Zenodo:

> **DOI**: [10.5281/zenodo.22168803](https://doi.org/10.5281/zenodo.22168803)
> **Record**: <https://doi.org/10.5281/zenodo.22168803>
> **Development version**: <https://github.com/enino84/qg-leapfrog-stability>

The Zenodo record is the authoritative citation target: it holds both the
tagged source snapshot (`qg-python-v1.0.zip`) and the simulation outputs. This
GitHub repository is the development mirror and tracks the source only.

The archive contains the equilibrium restart fields at `t = 100000` for both
resolutions, the eigenvalue clouds of the linearised operator, the Benettin
trace of the maximum Lyapunov exponent, and the time-resolved kinetic energy,
enstrophy and isotropic spectra for RK4, LF+RA and LF+RAW across
alpha in {0.05, 0.10, 0.20, 0.40}. `data/README.md` documents the exact
command line behind each file, so every output can also be regenerated from
scratch.

---

## Reference model

This work builds on:

> Sakov, P. (2024). *QG-C: A quasi-geostrophic model in C*.
> Bureau of Meteorology, Australia.
> https://github.com/sakov/qg-c

Key numerical methods:

> Arakawa, A. (1966). Computational design for long-term numerical integration
> of the equations of fluid motion.
> *Journal of Computational Physics*, 1(1), 119–143.

> Asselin, R. (1972). Frequency filter for time integrations.
> *Monthly Weather Review*, 100(6), 487–490.

> Williams, P. D. (2009). A proposed modification to the Robert-Asselin
> time filter. *Monthly Weather Review*, 137(8), 2538–2546.

> Li, X. S., & Demmel, J. W. (2003). SuperLU_DIST: A scalable distributed-memory
> sparse direct solver for unsymmetric linear systems.
> *ACM Transactions on Mathematical Software*, 29(2), 110–140.

---

## License

The source code in this repository is released under the MIT License — see
[LICENSE](LICENSE). The simulation output files distributed through the Zenodo
archive are released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Citation

Please cite both the article and the archived software release. Machine-readable
metadata is in [CITATION.cff](CITATION.cff).

```bibtex
@article{NinoRuiz2026qgpython,
  author  = {Nino-Ruiz, Elias D.},
  title   = {{QG-Python}: A Reproducible 1.5-Layer Quasi-Geostrophic Modelling
             Service with Direct Sparse Solvers, Spectral Stability Diagnostics,
             and Filter Sensitivity Data},
  journal = {Geoscience Data Journal},
  year    = {2026},
  doi     = {10.1002/gdj3.70107}
}

@software{NinoRuiz2026qgpythonsoftware,
  author    = {Nino-Ruiz, Elias D.},
  title     = {{QG-Python}: a reproducible 1.5-layer quasi-geostrophic modelling
               service with spectral stability and filter sensitivity diagnostics},
  version   = {1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22168803}
}
```

---

## A note on the enstrophy diagnostic

Enstrophy is reported throughout as the domain average of
`(nabla^2 psi)^2`, evaluated with the standard 5-point Laplacian on the
**interior nodes only**. This is the definition in
`analysis/uncertainty.py::enstrophy`, and it is the one every other script
imports.

The wall ring must be excluded from the average. `qg_model.py` imposes
homogeneous Dirichlet conditions `psi = 0` on all four walls, so at a wall node
the 5-point stencil collapses to `psi_adjacent / h^2`; with `h ~ 0.01` that is
`O(1e5)`, and squaring it lets a one-node ring — about 4% of the points —
dominate the domain average and inflate its variance by considerably more,
while leaving the *relative* biases between time-stepping schemes almost
unchanged.

`analysis/make_tables.py` regenerates the LaTeX rows of both tables in the
article directly from the `.npz` runs, so the statistical summary and the alpha
sweep cannot drift apart. Pass `--compare` to also print the wall-inclusive
average.
