#!/usr/bin/env bash
# =============================================================================
# run_reference.sh — spin-up from rest and RK4 reference integration
# =============================================================================
# Produces:
#   qg_zero.npz     equilibrium state at t = 100000 (restart for every other run)
#   qg_rk4_ref.npz  RK4 reference trajectory, t = 100000 -> 110000
#
# Run from the ROOT of the repository:
#   bash experiments/run_reference.sh
#
# Runtime: ~45 min (spin-up) + ~25 min (reference) on one modern core.
# =============================================================================

set -e

echo "=== Spin-up from rest (mrefin=6, tend=100000, RK4, dt=1) ==="
python qg_model.py \
    --mrefin 6 --tend 100000 --dt 1 --dtout 40 \
    --scheme rk4 --ic zero --save-q \
    --out qg_zero.npz

echo ""
echo "=== RK4 reference run (dt=1, t=100000->110000) ==="
python qg_model.py \
    --mrefin 6 --tend 110000 --dt 1 --dtout 40 \
    --scheme rk4 --ic restart --restart-file qg_zero.npz \
    --out qg_rk4_ref.npz
