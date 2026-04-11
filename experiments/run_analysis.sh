echo "=== Step 1: Build J_F and compute eigenspectrum ==="
python analysis/jacobian_spectrum.py \
    --restart  "$RESTART"            \
    --record   -1                    \
    --neigs    200                   \
    --eps      1e-5                  \
    --no-matrix                      \
    --out      jacobian_spectrum.npz \
    --plot