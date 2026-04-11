"""
lyapunov.py
===========
Estimates the maximum Lyapunov exponent (MLE) of the QG model using the
Benettin algorithm (Benettin et al. 1980).

Algorithm
---------
1. Start from equilibrium state q0 (from restart file).
2. Integrate the nonlinear model (reference trajectory) for nsteps.
3. In parallel, integrate the tangent linear model (TLM):
       dq'/dt = J_F(q(t)) * q'
   where J_F is approximated by a finite-difference perturbation:
       J_F(q) * q' ~ [F(q + eps*q') - F(q - eps*q')] / (2*eps)
4. Every renorm steps, record log(||q'||) and renormalise q' to unit norm.
5. The MLE is sigma_1 = (1/T) * sum_k log(||q'_k||).

Physical interpretation
-----------------------
- sigma_1 > 0  ->  chaotic system (trajectories diverge exponentially)
- Lyapunov time tau_L = 1/sigma_1  ->  time scale over which two initially
  close trajectories become uncorrelated
- For t >> tau_L, scheme comparison must be done statistically, not
  trajectory-by-trajectory

Usage
-----
    python analysis/lyapunov.py --restart qg_zero.npz [options]

Options
-------
    --restart FILE   equilibrium .npz  (required)
    --record  INT    which record to use as q0 (default: -1)
    --nsteps  INT    total integration steps (default: 50000)
    --renorm  INT    steps between renormalisations (default: 100)
    --dt      FLOAT  time step (default: 1.0)
    --eps     FLOAT  TLM perturbation size (default: 1e-6)
    --out     FILE   output .npz (default: lyapunov.npz)
    --plot           save convergence figure

Output
------
    lyapunov.npz:
        sigma1        MLE estimate (scalar)
        tau_L         Lyapunov time = 1/sigma1
        growth_rates  local growth rates at each renormalisation
        t_renorm      time at each renormalisation
        sigma_running running average of MLE (convergence check)
"punov_convergence.png    Running MLE estimate (convergence)

Usage
-----
    python analysis/lyapunov.py \
        --restart qg_zero.npz \
        --nsteps 50000 --renorm 100 --dt 1.0 --eps 1e-6 \
        --out lyapunov.npz --out-dir figures
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from plot_style import apply_style, style_ax, C_RK4, C_REF, C_MUTED
apply_style()

C_LOCAL = '#ff7f0e'   # amber for local rates


def load_model(restart, record, dt):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from qg_model import QGModel, QGParams
    data   = np.load(restart)
    psi0   = data['psi'][record].astype(float)
    t0     = float(data['t'][record])
    m_grid = psi0.shape[0]
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
        dt=dt, scheme='rk4', verbose=False,
    )
    prm.__post_init__()
    model = QGModel(prm)
    model.psi = psi0
    if 'q' in data:
        model.q = data['q'][record].astype(float)
    else:
        model._q_from_psi()
    model.t = t0
    print(f"  Grid: {prm.m}x{prm.n}   t0={t0:.1f}")
    return model


def tlm_step(model, q_ref, q_pert, eps):
    dt = model.prm.dt
    inv2eps = 1.0 / (2.0 * eps)

    def tlm_rhs(qr, qp):
        qrp = qr + eps*qp; qrm = qr - eps*qp
        return (model._rhs(qrp, model._calc_psi(qrp)) -
                model._rhs(qrm, model._calc_psi(qrm))) * inv2eps

    q0  = q_ref.copy()
    p1  = model._calc_psi(q0);            k1r = model._rhs(q0, p1)
    q2  = q0 + 0.5*dt*k1r
    p2  = model._calc_psi(q2);            k2r = model._rhs(q2, p2)
    q3  = q0 + 0.5*dt*k2r
    p3  = model._calc_psi(q3);            k3r = model._rhs(q3, p3)
    q4  = q0 + dt*k3r
    p4  = model._calc_psi(q4);            k4r = model._rhs(q4, p4)

    k1p = tlm_rhs(q0, q_pert)
    k2p = tlm_rhs(q2, q_pert + 0.5*dt*k1p)
    k3p = tlm_rhs(q3, q_pert + 0.5*dt*k2p)
    k4p = tlm_rhs(q4, q_pert + dt*k3p)
    return q_pert + (dt/6.0)*(k1p + 2*k2p + 2*k3p + k4p)


def benettin(model, nsteps, renorm, eps, verbose=True):
    dt  = model.prm.dt
    rng = np.random.default_rng(0)
    q_ref  = model.q.copy()
    q_pert = rng.standard_normal(q_ref.shape)
    q_pert[[0,-1],:] = 0; q_pert[:,[0,-1]] = 0
    q_pert /= np.linalg.norm(q_pert)

    growth_rates = []; t_renorm = []; sigma_running = []
    log_sum = 0.0; n_renorm = 0
    t0 = time.time()
    if verbose:
        print(f"  Benettin: {nsteps} steps, renorm every {renorm}...", flush=True)

    for step in range(1, nsteps+1):
        q_pert = tlm_step(model, q_ref, q_pert, eps)
        model._step_rk4()
        q_ref  = model.q.copy()

        if step % renorm == 0:
            norm = np.linalg.norm(q_pert) or 1e-300
            log_sum  += np.log(norm)
            q_pert   /= norm
            n_renorm += 1
            T         = n_renorm * renorm * dt
            sigma_est = log_sum / T
            growth_rates.append(np.log(norm) / (renorm*dt))
            t_renorm.append(model.t)
            sigma_running.append(sigma_est)
            if verbose and n_renorm % max(1,(nsteps//renorm)//10) == 0:
                el = time.time()-t0
                print(f"    {100*step/nsteps:5.1f}%  sigma1={sigma_est:.5f}  "
                      f"elapsed={el:.0f}s  ETA={el/step*(nsteps-step):.0f}s",
                      flush=True)

    sigma1 = log_sum / (n_renorm*renorm*dt)
    tau_L  = 1.0/sigma1 if sigma1 > 0 else np.inf
    if verbose: print(f"  Done in {time.time()-t0:.1f}s")
    return (sigma1, tau_L,
            np.array(growth_rates), np.array(t_renorm),
            np.array(sigma_running))


# ── figures ───────────────────────────────────────────────────────────────────

def fig_growth_rates(sigma1, growth_rates, t_renorm, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_renorm, growth_rates, color=C_LOCAL, lw=0.9,
            alpha=0.75, label='Local growth rate')
    ax.axhline(sigma1, color=C_REF, lw=1.8, ls='--',
               label=f'MLE  $\\sigma_1 = {sigma1:.5f}$')
    ax.axhline(0, color=C_MUTED, lw=0.7, ls=':')
    style_ax(ax, '$t$',
             '$\\log\\|\\mathbf{q}\'\\|\\ /\\ (n_{{\\mathrm{{renorm}}}}\\cdot\\Delta t)$',
             'Local Lyapunov growth rates (Benettin algorithm)')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


def fig_convergence(sigma1, tau_L, sigma_running, t_renorm, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_renorm, sigma_running, color=C_RK4, lw=1.8,
            label='Running $\\sigma_1(t)$')
    ax.axhline(sigma1, color=C_REF, lw=1.8, ls='--',
               label=f'Converged: $\\sigma_1 = {sigma1:.5f}$,  '
                     f'$\\tau_L = {tau_L:.1f}$')
    style_ax(ax, '$t$', '$\\sigma_1$',
             'Running MLE estimate — convergence of Benettin algorithm')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    # if lyapunov.npz already exists, just plot from it
    if args.npz and os.path.exists(args.npz):
        print(f"\n  Loading existing '{args.npz}'...")
        d = np.load(args.npz)
        sigma1        = float(d['sigma1'])
        tau_L         = float(d['tau_L'])
        growth_rates  = d['growth_rates']
        t_renorm      = d['t_renorm']
        sigma_running = d['sigma_running']
    else:
        print(f"\n  Loading '{args.restart}'  record={args.record}")
        model = load_model(args.restart, args.record, args.dt)
        sigma1, tau_L, growth_rates, t_renorm, sigma_running = benettin(
            model, args.nsteps, args.renorm, args.eps, verbose=True)

        print(f"\n  sigma_1 = {sigma1:.6f}")
        print(f"  tau_L   = {tau_L:.2f}")
        print(f"  t* (LF, dt=0.5) = {tau_L * np.log(4):.1f}")
        print(f"  t* (RK4, dt=1)  = {tau_L * np.log(1):.1f}")

        if args.npz:
            np.savez_compressed(args.npz,
                sigma1=sigma1, tau_L=tau_L,
                growth_rates=growth_rates, t_renorm=t_renorm,
                sigma_running=sigma_running,
                nsteps=args.nsteps, renorm=args.renorm,
                dt=args.dt, eps=args.eps)
            print(f"  Saved: {args.npz}")

    print("\n  Generating figures...")
    fig_growth_rates(sigma1, growth_rates, t_renorm,
                     f"{out}/fig_lyapunov_growth_rates.png")
    fig_convergence(sigma1, tau_L, sigma_running, t_renorm,
                    f"{out}/fig_lyapunov_convergence.png")
    print(f"\n  Done — 2 figures saved to '{out}/'")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--restart', default=None)
    p.add_argument('--record',  type=int,   default=-1)
    p.add_argument('--nsteps',  type=int,   default=50000)
    p.add_argument('--renorm',  type=int,   default=100)
    p.add_argument('--dt',      type=float, default=1.0)
    p.add_argument('--eps',     type=float, default=1e-6)
    p.add_argument('--npz',     default='lyapunov.npz',
                   help='Output/input .npz (if exists, skips simulation)')
    p.add_argument('--out-dir', default='figures')
    run(p.parse_args())