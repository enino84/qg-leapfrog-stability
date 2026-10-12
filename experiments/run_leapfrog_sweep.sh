#!/usr/bin/env bash
# =============================================================================
# run_leapfrog_sweep.sh, LF+RA sensitivity to the filter parameter alpha
# =============================================================================
# Produces qg_lf_ra_a005.npz, qg_lf_ra_a020.npz, qg_lf_ra_a040.npz.
# The alpha = 0.10 member is qg_lf_ra.npz, produced by
# experiments/run_leapfrog_long.sh, it is not recomputed here.
#
# Requires qg_zero.npz (see experiments/run_reference.sh).
#
# Run from the ROOT of the repository:
#   bash experiments/run_leapfrog_sweep.sh
#
# Runtime: ~12 min per alpha value; the loop is trivially parallel.
# =============================================================================

set -e

RESTART="${RESTART:-qg_zero.npz}"

if [ ! -f "$RESTART" ]; then
    echo "ERROR: restart file '$RESTART' not found."
    echo "Run experiments/run_reference.sh first."
    exit 1
fi

for ALPHA in 0.05 0.20 0.40; do
    TAG=$(echo "$ALPHA" | tr -d '.')
    echo "=== LF+RA alpha=$ALPHA (dt=0.5) ==="
    python qg_model.py \
        --mrefin 6 --tend 110000 --dt 0.5 --dtout 40 \
        --scheme leapfrog --ra-alpha "$ALPHA" \
        --ic restart --restart-file "$RESTART" \
        --out "qg_lf_ra_a${TAG}.npz"
done
