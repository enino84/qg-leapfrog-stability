#!/usr/bin/env bash
# =============================================================================
# run_analysis.sh, linearisation, eigenspectrum and Lyapunov diagnostics
# =============================================================================
# Requires an equilibrium restart file (default qg_zero.npz, produced by
# experiments/run_reference.sh). Override with:
#   RESTART=qg_zero_mrefin5.npz bash experiments/run_analysis.sh
#
# Run from the ROOT of the repository.
# Runtime: ~2 min (eigenspectrum) + ~90 min (Benettin) on one modern core.
# =============================================================================

set -e

RESTART="${RESTART:-qg_zero.npz}"

if [ ! -f "$RESTART" ]; then
    echo "ERROR: restart file '$RESTART' not found."
    echo "Run experiments/run_reference.sh first."
    exit 1
fi

echo "=== Step 1: Build J_F and compute eigenspectrum ==="
python analysis/jacobian_spectrum.py \
    --restart  "$RESTART"            \
    --record   -1                    \
    --neigs    200                   \
    --eps      1e-5                  \
    --no-matrix                      \
    --out      jacobian_spectrum.npz \
    --plot

echo ""
echo "=== Step 2: Maximum Lyapunov exponent (Benettin) ==="
python analysis/lyapunov.py \
    --restart "$RESTART" \
    --nsteps  50000      \
    --renorm  100        \
    --dt      1.0        \
    --eps     1e-6       \
    --out     lyapunov.npz \
    --plot
