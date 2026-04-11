#!/usr/bin/env bash
# =============================================================================
# run_all.sh
# =============================================================================
# Reproduce all simulations and figures for:
#   "Leapfrog time integration in a 1.5-layer quasi-geostrophic model:
#    stability analysis, uncertainty propagation, and filter-induced dissipation"
#
# Run from the ROOT of the repo (where qg_model.py lives):
#   bash run_all.sh
#
# Runtime estimates (mrefin=6, 97x97, modern laptop):
#   Step 1  ~45 min   (spin-up)
#   Step 2  ~25 min   (RK4 reference)
#   Step 3  ~12 min   (LF+RA, dt=0.5)
#   Step 4  ~12 min   (LF+RAW, dt=0.5)
#   Step 5  ~48 min   (alpha sensitivity: 4 runs x 12 min)
#   Step 6  ~2  min   (Jacobian + eigenspectrum, mrefin=6)
#   Step 7  ~2  min   (Jacobian + eigenspectrum, mrefin=5)
#   Step 8  ~90 min   (Lyapunov exponent)
#   Step 9  ~1  min   (all figures)
# =============================================================================

set -e   # stop on first error

echo ""
echo "============================================================"
echo "  QG Leapfrog paper — full reproduction script"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# STEP 1 — Spin-up from rest (generates equilibrium initial condition)
# ---------------------------------------------------------------------------
echo "=== Step 1: Spin-up (mrefin=6, tend=100000, RK4, dt=1) ==="
python qg_model.py \
    --mrefin 6 --tend 100000 --dt 1 --dtout 40 \
    --scheme rk4 --ic zero --save-q \
    --out qg_zero.npz

# ---------------------------------------------------------------------------
# STEP 2 — RK4 reference run (Section 6 reference)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: RK4 reference run (dt=1, t=100000->110000) ==="
python qg_model.py \
    --mrefin 6 --tend 110000 --dt 1 --dtout 40 \
    --scheme rk4 --ic restart --restart-file qg_zero.npz \
    --out qg_rk4_ref.npz

# ---------------------------------------------------------------------------
# STEP 3 — LF+RA main run (alpha=0.10, dt=0.5)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: LF+RA (alpha=0.10, dt=0.5) ==="
python qg_model.py \
    --mrefin 6 --tend 110000 --dt 0.5 --dtout 40 \
    --scheme leapfrog --ra-alpha 0.1 \
    --ic restart --restart-file qg_zero.npz \
    --out qg_lf_ra.npz

# ---------------------------------------------------------------------------
# STEP 4 — LF+RAW main run (alpha=0.10, dt=0.5)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: LF+RAW (alpha=0.10, dt=0.5) ==="
python qg_model.py \
    --mrefin 6 --tend 110000 --dt 0.5 --dtout 40 \
    --scheme leapfrog --ra-alpha 0.1 --raw \
    --ic restart --restart-file qg_zero.npz \
    --out qg_lf_raw.npz

# ---------------------------------------------------------------------------
# STEP 5 — Alpha sensitivity (LF+RA, 4 values)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 5: Alpha sensitivity sweep ==="

echo "  alpha=0.05..."
python qg_model.py \
    --mrefin 6 --tend 110000 --dt 0.5 --dtout 40 \
    --scheme leapfrog --ra-alpha 0.05 \
    --ic restart --restart-file qg_zero.npz \
    --out qg_lf_ra_a005.npz

echo "  alpha=0.20..."
python qg_model.py \
    --mrefin 6 --tend 110000 --dt 0.5 --dtout 40 \
    --scheme leapfrog --ra-alpha 0.20 \
    --ic restart --restart-file qg_zero.npz \
    --out qg_lf_ra_a020.npz

echo "  alpha=0.40..."
python qg_model.py \
    --mrefin 6 --tend 110000 --dt 0.5 --dtout 40 \
    --scheme leapfrog --ra-alpha 0.40 \
    --ic restart --restart-file qg_zero.npz \
    --out qg_lf_ra_a040.npz

# Note: alpha=0.10 is already in qg_lf_ra.npz (Step 3)

# ---------------------------------------------------------------------------
# STEP 6 — Jacobian eigenspectrum (mrefin=6)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 6: Jacobian eigenspectrum (mrefin=6, 97x97) ==="
python analysis/jacobian_spectrum.py \
    --restart qg_zero.npz \
    --neigs 200 \
    --no-matrix \
    --out jacobian_spectrum.npz

# ---------------------------------------------------------------------------
# STEP 7 — Spin-up and Jacobian eigenspectrum (mrefin=5)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 7a: Spin-up (mrefin=5, tend=100000, dt=0.1) ==="
python qg_model.py \
    --mrefin 5 --tend 100000 --dt 0.1 --dtout 40 \
    --scheme rk4 --ic zero --save-q \
    --out qg_zero_mrefin5.npz

echo ""
echo "=== Step 7b: Jacobian eigenspectrum (mrefin=5, 49x49) ==="
python analysis/jacobian_spectrum.py \
    --restart qg_zero_mrefin5.npz \
    --neigs 200 \
    --no-matrix \
    --out jacobian_spectrum_mrefin5.npz

# ---------------------------------------------------------------------------
# STEP 8 — Lyapunov exponent (Benettin algorithm)
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 8: Lyapunov exponent (nsteps=50000, renorm=100) ==="
python analysis/lyapunov.py \
    --restart qg_zero.npz \
    --nsteps 50000 \
    --renorm 100 \
    --dt 1.0 \
    --eps 1e-6 \
    --out lyapunov.npz \
    --plot

# ---------------------------------------------------------------------------
# STEP 9 — Generate all figures
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 9: Generate figures ==="

mkdir -p figures

python analysis/stability_regions.py \
    --spectrum jacobian_spectrum.npz \
    --dt 1.0 \
    --out figures/fig_stability_regions.png

python analysis/uncertainty.py \
    --rk4    qg_rk4_ref.npz \
    --lf-ra  qg_lf_ra.npz \
    --lf-raw qg_lf_raw.npz \
    --out    figures/

python analysis/plot_fields.py \
    --rk4    qg_rk4_ref.npz \
    --lf-ra  qg_lf_ra.npz \
    --lf-raw qg_lf_raw.npz \
    --zero   qg_zero.npz \
    --out    figures/

python analysis/plot_alpha_sensitivity.py \
    --rk4  qg_rk4_ref.npz \
    --a005 qg_lf_ra_a005.npz \
    --a010 qg_lf_ra.npz \
    --a020 qg_lf_ra_a020.npz \
    --a040 qg_lf_ra_a040.npz \
    --out  figures/

echo ""
echo "============================================================"
echo "  All steps completed. Figures saved to figures/"
echo "============================================================"