# qg-leapfrog-stability

**Leapfrog time integration in a 1.5-layer quasi-geostrophic model:
stability analysis, uncertainty propagation, and computational scaling**

This repository contains the code and analysis scripts associated with the paper
submitted to *Ocean Modelling*.

---

## Overview

We implement the Leapfrog scheme with Robert-Asselin (RA) and
Robert-Asselin-Williams (RAW) filters in a 1.5-layer quasi-geostrophic (QG)
model, and carry out a systematic numerical analysis covering:

- Spectral analysis of the linearised operator $J_\mathcal{F}$ and its
  dependence on grid resolution
- Stability regions in the complex plane for each time-stepping scheme
- Maximum stable time step $\Delta t_{\max}$ as a function of grid spacing $h$
- Lyapunov exponent estimation via the Benettin algorithm
- Statistical uncertainty propagation: energy PDFs, enstrophy PDFs, and
  power spectra across schemes
- Computational cost scaling with model dimension

The QG model is a faithful Python port of the QG-C model by
[Pavel Sakov (Bureau of Meteorology)](https://github.com/sakov/qg-c),
extended with the Leapfrog family of integrators.

---

## Repository structure

```
qg-leapfrog-stability/
│
├── qg_model.py              # QG model — all time-stepping schemes
├── qg_viewer.py             # Interactive field viewer (.npz files)
│
├── analysis/
│   ├── jacobian_spectrum.py     # Build J_F and compute eigenspectrum
│   ├── stability_regions.py     # Plot eigenspectrum + stability regions
│   ├── lyapunov.py              # Benettin algorithm for Lyapunov exponent
│   └── uncertainty.py           # PDFs, power spectra, KL divergence
│
├── experiments/
│   ├── run_reference.sh         # RK4 reference run from equilibrium state
│   ├── run_leapfrog_sweep.sh    # Stability sweep: vary dt and alpha
│   └── run_leapfrog_long.sh     # Long runs for statistical comparison
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

### Step 1 — Generate the equilibrium initial condition

The spin-up simulation (tend=100000, RK4, mrefin=6) must be run first.
Its final record serves as the initial condition for all subsequent experiments.

```bash
python qg_model.py \
    --mrefin 6 --tend 100000 --dt 1 --dtout 40 \
    --scheme rk4 --ic fourier --seed 42 \
    --out data/qg_spinup.npz
```

This takes approximately 10 minutes on a modern laptop (~9400 time steps/s).
The resulting file is **not** included in this repository but is archived on
Zenodo (see [Data availability](#data-availability)).

### Step 2 — Reference RK4 run

```bash
bash experiments/run_reference.sh
```

### Step 3 — Leapfrog stability sweep

```bash
bash experiments/run_leapfrog_sweep.sh
```

### Step 4 — Long Leapfrog runs for statistical analysis

```bash
bash experiments/run_leapfrog_long.sh
```

### Step 5 — Generate all figures

```bash
python analysis/jacobian_spectrum.py    --restart data/qg_spinup.npz
python analysis/stability_regions.py   --restart data/qg_spinup.npz
python analysis/lyapunov.py            --restart data/qg_spinup.npz
python analysis/uncertainty.py         --rk4 data/qg_rk4_ref.npz \
                                        --lf-ra data/qg_lf_ra.npz \
                                        --lf-raw data/qg_lf_raw.npz
```

All figures are saved to `figures/`.

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

---

## License

MIT — see [LICENSE](LICENSE).

---

## Citation

*(to be added upon publication)*
