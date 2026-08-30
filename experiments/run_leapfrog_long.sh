#!/usr/bin/env bash
# =============================================================================
# run_leapfrog_long.sh — main Leapfrog integrations (alpha = 0.10, dt = 0.5)
# =============================================================================
# Produces:
#   qg_lf_ra.npz   Leapfrog + Robert-Asselin filter
#   qg_lf_raw.npz  Leapfrog + Robert-Asselin-Williams filter
#
# Requires qg_zero.npz (see experiments/run_reference.sh).
#
# Run from the ROOT of the repository:
#   bash experiments/run_leapfrog_long.sh
#
# Runtime: ~12 min per run on one modern core.
# =============================================================================

set -e

RESTART="${RESTART:-qg_zero.npz}"

if [ ! -f "$RESTART" ]; then
    echo "ERROR: restart file '$RESTART' not found."
    echo "Run experiments/run_reference.sh first."
    exit 1
fi

echo "=== LF+RA (alpha=0.10, dt=0.5) ==="
python qg_model.py \
    --mrefin 6 --tend 110000 --dt 0.5 --dtout 40 \
    --scheme leapfrog --ra-alpha 0.1 \
    --ic restart --restart-file "$RESTART" \
    --out qg_lf_ra.npz

echo ""
echo "=== LF+RAW (alpha=0.10, dt=0.5) ==="
python qg_model.py \
    --mrefin 6 --tend 110000 --dt 0.5 --dtout 40 \
    --scheme leapfrog --ra-alpha 0.1 --raw \
    --ic restart --restart-file "$RESTART" \
    --out qg_lf_raw.npz
