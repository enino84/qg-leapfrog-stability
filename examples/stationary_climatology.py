"""Build the climatology of the stationary preset: spin-up to t = 20000, then 200 snapshots every 250 units.
Output: results/qg_climatology_m<mrefin>.npy (200 x n_state, float32) and a summary line."""
import os, sys, time, numpy as np
from qgpython import stationary_model, api
mrefin = int(sys.argv[1]) if len(sys.argv) > 1 else 5
model = stationary_model(mrefin=mrefin); x = api.pack(model.q, model.psi); t0 = time.time()
x = api.propagate(model, x, [0.0, 20000.0]); print(f"spin-up done in {time.time()-t0:.0f}s, q std {api.unpack(model, x)[0].std():.4g}", flush=True)
snaps = [x.astype(np.float32)]
for k in range(199):
    x = api.propagate(model, x, [0.0, 250.0]); snaps.append(x.astype(np.float32))
    if k % 50 == 49: print(f"  {k+1} snapshots [{time.time()-t0:.0f}s]", flush=True)
S = np.array(snaps); os.makedirs("results", exist_ok=True); np.save(f"results/qg_climatology_m{mrefin}.npy", S)
q = S[:, :model.prm.m * model.prm.n]; a = q - q.mean(0)
print(f"saved {S.shape}; q std {q.std():.4g}; correlation of q 250 units apart {np.mean([np.corrcoef(a[i], a[i+1])[0,1] for i in range(len(a)-1)]):.2f}")
