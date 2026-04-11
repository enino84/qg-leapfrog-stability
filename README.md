# qg-leapfrog-stability

**Leapfrog time integration in a 1.5-layer quasi-geostrophic model:
stability analysis, uncertainty propagation, and filter-induced dissipation**

This repository contains the code and analysis scripts associated with the paper
submitted to *Ocean Modelling*.

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
├── data/
│   └── README.md                # Instructions to reproduce the .npz files
│
├── paper/
│   ├── manuscript.tex
│   └── references.bib
│
├── figures/                     # Generated locally — not tracked by git
├── requirements.txt
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

Each command in `run_all.sh` is self-contained and can be run independently.
For example, to recompute only the eigenspectrum:

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
| LF+RA enstrophy bias (α=0.10) | −45% |
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

The simulation output files (.npz) are archived on Zenodo:

> **DOI**: *(to be assigned upon submission)*

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

MIT — see [LICENSE](LICENSE).

---

## Citation

*(to be added upon acceptance)*