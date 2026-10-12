"""
uncertainty.py
==============
Post-process existing .npz runs and save one figure per diagnostic.

Output files (all in --out-dir):
  fig_ke_timeseries.png          KE vs time
  fig_ke_kde.png                 KDE of KE stationary distribution
  fig_enstrophy_timeseries.png   Enstrophy vs time
  fig_enstrophy_kde.png          KDE of enstrophy stationary distribution
  fig_energy_spectrum.png        Isotropic energy spectrum
  fig_trajectory_divergence.png  ||psi_LF - psi_RK4|| / ||psi_RK4||

Usage
-----
    python analysis/uncertainty.py \
        --rk4      qg_rk4_ref.npz \
        --lf-ra    qg_lf_ra.npz   \
        --lf-raw   qg_lf_raw.npz  \
        --lyapunov lyapunov.npz   \
        --out-dir  figures
"""

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from plot_style import (apply_style, style_ax,
                        C_RK4, C_RA, C_RAW, C_REF, C_MUTED,
                        SCHEME_LABELS, SCHEME_LS, SCHEME_LW)
apply_style()


# ── diagnostics ───────────────────────────────────────────────────────────────

def kinetic_energy(psi):
    return np.mean(psi**2, axis=(1, 2))


def enstrophy(psi, h):
    h2  = 1.0 / (h * h)
    lap = (psi[:, :-2, 1:-1] + psi[:, 2:, 1:-1] +
           psi[:, 1:-1, :-2] + psi[:, 1:-1, 2:] -
           4.0 * psi[:, 1:-1, 1:-1]) * h2
    return np.mean(lap**2, axis=(1, 2))


def mean_spectrum(psi_arr, lx=1.0, n_avg=50):
    def spectrum(psi_2d):
        m, n   = psi_2d.shape
        phat   = np.fft.fft2(psi_2d) / (m * n)
        power  = np.abs(phat)**2
        kx     = np.fft.fftfreq(n, d=lx/n)
        ky     = np.fft.fftfreq(m, d=lx/m)
        KX, KY = np.meshgrid(kx, ky)
        K      = np.sqrt(KX**2 + KY**2)
        k_max  = min(m, n) // 2
        k_bins = np.arange(1, k_max)
        E_k    = np.array([power[(K >= k-0.5) & (K < k+0.5)].sum()
                           for k in k_bins])
        return k_bins, E_k
    recs = psi_arr[-n_avg:]
    k_bins, E = spectrum(recs[0])
    for r in recs[1:]:
        _, Ei = spectrum(r); E += Ei
    return k_bins, E / len(recs)


def trajectory_distance(psi_a, psi_b):
    diff = psi_a - psi_b
    dist = np.sqrt(np.mean(diff**2,  axis=(1, 2)))
    norm = np.sqrt(np.mean(psi_a**2, axis=(1, 2)))
    return dist / (norm + 1e-300)


def kde_curve(data, n_pts=300):
    kde = gaussian_kde(data, bw_method='scott')
    lo  = data.min() - 0.05 * np.ptp(data)
    hi  = data.max() + 0.05 * np.ptp(data)
    x   = np.linspace(lo, hi, n_pts)
    return x, kde(x)


def save(fig, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def _schemes(t_rk4, V_rk4, t_ra, V_ra, t_raw, V_raw):
    yield t_rk4, V_rk4, C_RK4, 'RK4'
    yield t_ra,  V_ra,  C_RA,  'LF+RA'
    if V_raw is not None:
        yield t_raw, V_raw, C_RAW, 'LF+RAW'


# ── individual figures ────────────────────────────────────────────────────────

def fig_ke_timeseries(t_rk4, KE_rk4, t_ra, KE_ra, t_raw, KE_raw, tau_L, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for t, KE, col, key in _schemes(t_rk4, KE_rk4, t_ra, KE_ra, t_raw, KE_raw):
        ax.plot(t, KE, color=col, lw=SCHEME_LW[key],
                ls=SCHEME_LS[key], label=SCHEME_LABELS[key])
    if tau_L:
        ax.axvline(t_rk4[0] + tau_L, color=C_REF, lw=1.2, ls=':',
                   label=f'$\\tau_L = {tau_L:.1f}$')
    style_ax(ax, '$t$', '$\\langle\\psi^2\\rangle$',
             'Kinetic energy time series')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, out)


def fig_ke_kde(t_rk4, KE_rk4, t_ra, KE_ra, t_raw, KE_raw, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, KE, col, key in _schemes(t_rk4, KE_rk4, t_ra, KE_ra, t_raw, KE_raw):
        x, y = kde_curve(KE)
        ax.plot(x, y, color=col, lw=SCHEME_LW[key],
                ls=SCHEME_LS[key], label=SCHEME_LABELS[key])
        ax.fill_between(x, y, alpha=0.12, color=col)
    style_ax(ax, '$\\langle\\psi^2\\rangle$', 'Density',
             'Kinetic energy, stationary distribution (KDE)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, out)


def fig_enstrophy_timeseries(t_rk4, EN_rk4, t_ra, EN_ra, t_raw, EN_raw,
                              tau_L, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for t, EN, col, key in _schemes(t_rk4, EN_rk4, t_ra, EN_ra, t_raw, EN_raw):
        ax.plot(t, EN, color=col, lw=SCHEME_LW[key],
                ls=SCHEME_LS[key], label=SCHEME_LABELS[key])
    if tau_L:
        ax.axvline(t_rk4[0] + tau_L, color=C_REF, lw=1.2, ls=':',
                   label=f'$\\tau_L = {tau_L:.1f}$')
    style_ax(ax, '$t$', '$\\langle(\\nabla^2\\psi)^2\\rangle$',
             'Enstrophy time series')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, out)


def fig_enstrophy_kde(t_rk4, EN_rk4, t_ra, EN_ra, t_raw, EN_raw, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, EN, col, key in _schemes(t_rk4, EN_rk4, t_ra, EN_ra, t_raw, EN_raw):
        x, y = kde_curve(EN)
        ax.plot(x, y, color=col, lw=SCHEME_LW[key],
                ls=SCHEME_LS[key], label=SCHEME_LABELS[key])
        ax.fill_between(x, y, alpha=0.12, color=col)
    style_ax(ax, '$\\langle(\\nabla^2\\psi)^2\\rangle$', 'Density',
             'Enstrophy, stationary distribution (KDE)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, out)


def fig_energy_spectrum(psi_rk4, psi_ra, psi_raw, lx, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, psi in [('RK4', psi_rk4), ('LF+RA', psi_ra),
                     ('LF+RAW', psi_raw)]:
        if psi is None: continue
        col = {'RK4': C_RK4, 'LF+RA': C_RA, 'LF+RAW': C_RAW}[key]
        k, E = mean_spectrum(psi, lx)
        ax.loglog(k, E, color=col, lw=SCHEME_LW[key],
                  ls=SCHEME_LS[key], label=SCHEME_LABELS[key])
    # k^-3 reference
    k0, E0 = mean_spectrum(psi_rk4, lx)
    k_ref  = np.array([5, 40])
    ax.loglog(k_ref, E0[4] * (k_ref/k0[4])**(-3),
              color=C_MUTED, lw=1.2, ls='--', alpha=0.7,
              label='$k^{-3}$ reference')
    style_ax(ax, 'Wavenumber $k$', '$E(k)$',
             'Time-averaged isotropic energy spectrum')
    ax.legend()
    plt.tight_layout()
    save(fig, out)


def fig_trajectory_divergence(t_rk4, psi_rk4, t_ra, psi_ra,
                               t_raw, psi_raw, tau_L, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for t_lf, psi_lf, col, lbl, ls in [
        (t_ra,  psi_ra,  C_RA,  'LF + RA  vs  RK4',  '--'),
        (t_raw, psi_raw, C_RAW, 'LF + RAW  vs  RK4', ':'),
    ]:
        if psi_lf is None: continue
        nmin = min(len(t_rk4), len(t_lf))
        d    = trajectory_distance(psi_rk4[:nmin], psi_lf[:nmin])
        ax.semilogy(t_rk4[:nmin], d, color=col, lw=1.8, ls=ls, label=lbl)
    if tau_L:
        ax.axvline(t_rk4[0] + tau_L, color=C_REF, lw=1.4, ls=':',
                   label=f'$\\tau_L = {tau_L:.1f}$')
    ax.axhline(1.0, color=C_MUTED, lw=1.0, ls='--', alpha=0.6,
               label='O(1) threshold')
    style_ax(ax, '$t$',
             '$\\|\\psi_{\\mathrm{LF}}-\\psi_{\\mathrm{RK4}}\\|_2'
             '\\ /\\ \\|\\psi_{\\mathrm{RK4}}\\|_2$',
             'Relative trajectory distance (semi-log)')
    ax.legend()
    plt.tight_layout()
    save(fig, out)


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    def load(f):
        d = np.load(f)
        return d['t'].astype(float), d['psi'].astype(float)

    print("\n  Loading runs...")
    t_rk4, psi_rk4 = load(args.rk4)
    t_ra,  psi_ra  = load(args.lf_ra)
    t_raw = psi_raw = None
    if args.lf_raw:
        t_raw, psi_raw = load(args.lf_raw)

    tau_L = None
    if args.lyapunov:
        tau_L = float(np.load(args.lyapunov)['tau_L'])
        print(f"  tau_L = {tau_L:.2f}")

    h = 1.0 / (psi_rk4.shape[2] - 1)

    print("  Computing diagnostics...")
    KE_rk4 = kinetic_energy(psi_rk4)
    KE_ra  = kinetic_energy(psi_ra)
    KE_raw = kinetic_energy(psi_raw) if psi_raw is not None else None
    EN_rk4 = enstrophy(psi_rk4, h)
    EN_ra  = enstrophy(psi_ra,  h)
    EN_raw = enstrophy(psi_raw, h) if psi_raw is not None else None

    print(f"\n  {'Quantity':<22} {'RK4':>12} {'LF+RA':>12} {'LF+RAW':>12}")
    print(f"  {'':-<60}")
    for name, a, b, c in [
        ('mean KE',   KE_rk4.mean(), KE_ra.mean(),
         KE_raw.mean() if KE_raw is not None else np.nan),
        ('std KE',    KE_rk4.std(),  KE_ra.std(),
         KE_raw.std()  if KE_raw is not None else np.nan),
        ('mean Enst', EN_rk4.mean(), EN_ra.mean(),
         EN_raw.mean() if EN_raw is not None else np.nan),
        ('std Enst',  EN_rk4.std(),  EN_ra.std(),
         EN_raw.std()  if EN_raw is not None else np.nan),
    ]:
        print(f"  {name:<22} {a:12.4f} {b:12.4f} {c:12.4f}")

    print("\n  Generating figures...")
    p = out
    fig_ke_timeseries(t_rk4, KE_rk4, t_ra, KE_ra, t_raw, KE_raw, tau_L,
                      f"{p}/fig_ke_timeseries.png")
    fig_ke_kde(t_rk4, KE_rk4, t_ra, KE_ra, t_raw, KE_raw,
               f"{p}/fig_ke_kde.png")
    fig_enstrophy_timeseries(t_rk4, EN_rk4, t_ra, EN_ra, t_raw, EN_raw,
                              tau_L, f"{p}/fig_enstrophy_timeseries.png")
    fig_enstrophy_kde(t_rk4, EN_rk4, t_ra, EN_ra, t_raw, EN_raw,
                      f"{p}/fig_enstrophy_kde.png")
    fig_energy_spectrum(psi_rk4, psi_ra, psi_raw, 1.0,
                        f"{p}/fig_energy_spectrum.png")
    fig_trajectory_divergence(t_rk4, psi_rk4, t_ra, psi_ra,
                               t_raw, psi_raw, tau_L,
                               f"{p}/fig_trajectory_divergence.png")
    print(f"\n  Done, 6 figures saved to '{out}/'")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rk4',      required=True)
    p.add_argument('--lf-ra',    required=True)
    p.add_argument('--lf-raw',   default=None)
    p.add_argument('--lyapunov', default=None)
    p.add_argument('--out-dir',  default='figures')
    run(p.parse_args())