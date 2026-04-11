"""
jacobian_spectrum.py
====================
Constructs the linearised operator J_F(q0) of the QG right-hand side
around an equilibrium state q0, computes its eigenspectrum, and saves
both the matrix and the eigenvalues for downstream analysis.

The Jacobian is built by finite differences of F(q, psi(q)):
    J_F[:, k] = (F(q0 + eps*e_k) - F(q0 - eps*e_k)) / (2*eps)
where e_k is the k-th unit vector in the interior-point space.

This requires 2*d Helmholtz solves (d = number of interior grid points),
which is the dominant cost. For mrefin=6 (d~9025) this takes ~5 min.

Usage
-----
    python jacobian_spectrum.py --restart qg_zero.npz [options]

Options
-------
    --restart FILE      equilibrium .npz file  (required)
    --record  INT       which record to use as q0 (default: -1, last)
    --neigs   INT       number of eigenvalues to compute (default: 200)
    --eps     FLOAT     finite-difference perturbation size (default: 1e-5)
    --out     FILE      output .npz with eigenvalues
                        (default: jacobian_spectrum.npz)
    --no-matrix         skip saving the full matrix (saves disk space)
    --plot              show eigenspectrum plot after computing

Output
------
    jacobian_spectrum.npz containing:
        eigenvalues      complex array (neigs,)
        eig_real         real parts
        eig_imag         imaginary parts
        lambda_real_max  max |Re(lambda)|  (dominant dissipation rate)
        lambda_imag_max  max |Im(lambda)|  (fastest wave frequency)
        d, m, n, h       grid info
        t_eq             time of the equilibrium state used
        physical params  rkb, rkh, rkh2, F, r, A, k
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, sys, os, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qg_model import QGModel, QGParams


# ── helpers ──────────────────────────────────────────────────────────────────

def interior_mask(m, n):
    mask = np.zeros((m, n), dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask

def q_to_vec(q, mask):
    return q[mask]

def vec_to_q(v, mask, m, n):
    q = np.zeros((m, n))
    q[mask] = v
    return q


# ── Jacobian by centred finite differences ────────────────────────────────────

def build_jacobian(model, q0, eps=1e-5, verbose=True):
    m, n = model.m, model.n
    mask = interior_mask(m, n)
    d    = int(mask.sum())

    J_dense = np.zeros((d, d), dtype=float)

    if verbose:
        print(f"  Building J_F ({d}x{d}) — {2*d} Helmholtz solves...", flush=True)
        t0 = time.time()

    for k in range(d):
        ek = np.zeros(d); ek[k] = 1.0

        qp = q0.copy(); qp[mask] += eps * ek
        Fp = model._rhs(qp, model._calc_psi(qp))

        qm = q0.copy(); qm[mask] -= eps * ek
        Fm = model._rhs(qm, model._calc_psi(qm))

        J_dense[:, k] = (q_to_vec(Fp, mask) - q_to_vec(Fm, mask)) / (2.0 * eps)

        if verbose and (k+1) % max(1, d//10) == 0:
            el  = time.time() - t0
            eta = el / (k+1) * (d - k - 1)
            print(f"    {100*(k+1)/d:5.1f}%  elapsed={el:.0f}s  ETA={eta:.0f}s",
                  flush=True)

    if verbose:
        print(f"  Done in {time.time()-t0:.1f}s")

    return sp.csr_matrix(J_dense)


# ── analytical bounds for the linear operator L_h ────────────────────────────

def analytical_bounds(prm):
    """
    Upper bounds on |lambda| from the purely linear part L_h (q0=0).
    Uses ||Delta_h|| <= 4/h^2 (spectral radius of the 5-point Laplacian).
    """
    h        = prm.lx / (prm.n - 1)
    lap_norm = 4.0 / (h * h)
    return {
        'rkh2 (biharmonic)': prm.rkh2  * lap_norm**2,
        '|rkh| (harmonic)' : abs(prm.rkh) * lap_norm,
        'rkb (Ekman)'      : prm.rkb   * lap_norm,
    }, h


# ── stability dt_max estimates ────────────────────────────────────────────────

def dt_max_leapfrog(lim_imag):
    """Leapfrog: |Im(lambda)| * dt <= 1."""
    return 1.0 / lim_imag if lim_imag > 0 else np.inf

def dt_max_rk4(lim_real, lim_imag):
    """
    RK4 stability region:
      imaginary axis: |Im(lambda)| * dt <= 2*sqrt(2) ~ 2.828
      real axis:      |Re(lambda)| * dt <= 2.785
    """
    candidates = {}
    if lim_imag > 0: candidates['imag'] = 2.0*np.sqrt(2) / lim_imag
    if lim_real > 0: candidates['real'] = 2.785          / lim_real
    if not candidates: return np.inf, 'none'
    key = min(candidates, key=candidates.get)
    return candidates[key], key


# ── main ─────────────────────────────────────────────────────────────────────

def run(args):
    # load equilibrium state
    print(f"\n  Loading '{args.restart}'  record={args.record}")
    data   = np.load(args.restart)
    psi0   = data['psi'][args.record].astype(float)
    t_eq   = float(data['t'][args.record])
    q0_raw = data['q'][args.record].astype(float) if 'q' in data else None

    m_grid, n_grid = psi0.shape
    mrefin = int(round(np.log2((m_grid - 1) / 3))) + 1

    prm = QGParams(
        mrefin=mrefin,
        lx   = float(data.get('lx',   1.0)),
        rkb  = float(data.get('rkb',  3e-4)),
        rkh  = float(data.get('rkh',  -5e-8)),
        rkh2 = float(data.get('rkh2', 1e-12)),
        f    = float(data.get('F',    1600.0)),
        r    = float(data.get('r',    1e-5)),
        a    = float(data.get('A',    6.29)),
        k    = float(data.get('k',    1.0)),
        scheme='rk4', verbose=False,
    )
    prm.__post_init__()
    print(f"  Grid: {prm.m}x{prm.n}   h={prm.lx/(prm.n-1):.5f}   t_eq={t_eq:.1f}")

    model     = QGModel(prm)
    model.psi = psi0
    if q0_raw is not None:
        model.q = q0_raw
    else:
        model._q_from_psi()
        print("  q not in file — reconstructed from psi.")
    q0 = model.q.copy()

    # ── analytical bounds ─────────────────────────────────────────────────
    print("\n  Analytical bounds on |lambda| (linear operator L_h, q0=0):")
    bounds, h = analytical_bounds(prm)
    for name, val in sorted(bounds.items(), key=lambda x: -x[1]):
        print(f"    {name:28s}: {val:.4e}")
    lam_lin = max(bounds.values())
    print(f"\n  Leapfrog dt_max (linear bound): {dt_max_leapfrog(lam_lin):.4f}")
    dt_rk4_lin, ax = dt_max_rk4(lam_lin, lam_lin)
    print(f"  RK4      dt_max (linear bound): {dt_rk4_lin:.4f}  [binding: {ax}]")

    # ── build Jacobian ────────────────────────────────────────────────────
    print()
    J = build_jacobian(model, q0, eps=args.eps, verbose=True)
    d = J.shape[0]
    print(f"\n  J: {J.shape}   nnz={J.nnz}   density={J.nnz/d**2*100:.2f}%")

    # ── eigenspectrum ─────────────────────────────────────────────────────
    neigs = min(args.neigs, d - 2)
    print(f"\n  Computing eigenvalues (ARPACK, k={neigs})...", flush=True)
    t0 = time.time()

    # largest real part  -> most unstable / least dissipative
    v_lr, _ = spla.eigs(J, k=neigs//2, which='LR', maxiter=10000, tol=1e-8)
    # largest imaginary part -> fastest oscillatory modes
    v_li, _ = spla.eigs(J, k=neigs//2, which='LI', maxiter=10000, tol=1e-8)

    eigenvalues = np.unique(np.concatenate([v_lr, v_li]))
    print(f"  Done in {time.time()-t0:.1f}s  ({len(eigenvalues)} unique eigs)")

    lam_real_max = np.max(np.abs(eigenvalues.real))
    lam_imag_max = np.max(np.abs(eigenvalues.imag))

    print(f"\n  Spectral summary (nonlinear state q0):")
    print(f"    max |Re(lambda)| = {lam_real_max:.4e}")
    print(f"    max |Im(lambda)| = {lam_imag_max:.4e}")
    print(f"    max |lambda|     = {np.abs(eigenvalues).max():.4e}")

    print(f"\n  Stability constraints (from computed spectrum):")
    print(f"    Leapfrog  dt_max = {dt_max_leapfrog(lam_imag_max):.4f}")
    dt_rk4_nl, ax = dt_max_rk4(lam_real_max, lam_imag_max)
    print(f"    RK4       dt_max = {dt_rk4_nl:.4f}  [binding: {ax}]")

    # ── save ──────────────────────────────────────────────────────────────
    kw = dict(
        eigenvalues     = eigenvalues,
        eig_real        = eigenvalues.real,
        eig_imag        = eigenvalues.imag,
        lambda_real_max = lam_real_max,
        lambda_imag_max = lam_imag_max,
        d=d, m=prm.m, n=prm.n, h=h, t_eq=t_eq,
        rkb=prm.rkb, rkh=prm.rkh, rkh2=prm.rkh2,
        F=prm.f, r=prm.r, A=prm.a, k=prm.k,
    )
    if not args.no_matrix:
        kw.update(J_data=J.data, J_indices=J.indices,
                  J_indptr=J.indptr, J_shape=np.array(J.shape))

    np.savez_compressed(args.out, **kw)
    print(f"\n  Output saved: {args.out}")

    if args.plot:
        _plot(eigenvalues, prm, h, args.out)


def _plot(eigenvalues, prm, h, outfile):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    for sp_ in ax.spines.values(): sp_.set_color('#30363d')
    ax.tick_params(colors='#8b949e')

    ax.scatter(eigenvalues.real, eigenvalues.imag,
               s=10, color='#58a6ff', alpha=0.75, linewidths=0,
               label=f'Eigenvalues ({len(eigenvalues)})')
    ax.axvline(0, color='#f78166', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='#f78166', lw=0.8, ls='--', alpha=0.5)

    ax.set_xlabel('Re($\\lambda$)', color='#8b949e')
    ax.set_ylabel('Im($\\lambda$)', color='#8b949e')
    ax.set_title(f'Eigenspectrum of $J_{{\\mathcal{{F}}}}(q_0)$  '
                 f'[h={h:.4f},  rkh2={prm.rkh2:.1e}]',
                 color='#e6edf3', fontsize=10)
    ax.legend(fontsize=8, labelcolor='#8b949e',
              facecolor='#21262d', edgecolor='#30363d')
    fig.tight_layout()

    figfile = outfile.replace('.npz', '_spectrum.png')
    fig.savefig(figfile, dpi=150, facecolor=fig.get_facecolor())
    print(f"  Figure saved: {figfile}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Build J_F and compute eigenspectrum for QG stability analysis')
    p.add_argument('--restart',   required=True, metavar='FILE')
    p.add_argument('--record',    type=int,   default=-1)
    p.add_argument('--neigs',     type=int,   default=200)
    p.add_argument('--eps',       type=float, default=1e-5)
    p.add_argument('--out',       default='jacobian_spectrum.npz')
    p.add_argument('--no-matrix', action='store_true')
    p.add_argument('--plot',      action='store_true')
    run(p.parse_args())