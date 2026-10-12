"""
stability_regions.py
====================
Reads jacobian_spectrum.npz and produces two independent figures:

  fig_eigenvalue_cloud.png       Eigenvalues in physical units (Re, Im)
  fig_stability_regions.png      Eigenvalues scaled by dt overlaid on
                                 stability regions of LF, RK4, dp5

Usage
-----
    python analysis/stability_regions.py \
        --spectrum jacobian_spectrum.npz \
        --dt 1.0 \
        --out-dir figures
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from plot_style import apply_style, style_ax, C_RK4, C_RA, C_RAW, C_REF, C_MUTED
apply_style()

C_DP5  = '#2ca02c'
C_LF   = '#d62728'
C_EIG  = '#ff7f0e'


# ── amplification factors ─────────────────────────────────────────────────────

def R_rk4(z): return 1 + z + z**2/2 + z**3/6 + z**4/24
def R_dp5(z): return 1 + z + z**2/2 + z**3/6 + z**4/24 + z**5/120

def stability_grid(amp_func, re_range=(-4, 0.6), im_range=(-4, 4), res=700):
    re = np.linspace(*re_range, res)
    im = np.linspace(*im_range, res)
    RE, IM = np.meshgrid(re, im)
    return RE, IM, np.abs(amp_func(RE + 1j*IM))


# ── load ──────────────────────────────────────────────────────────────────────

def load_spectrum(fname):
    data = np.load(fname, allow_pickle=False)
    try:
        eigs = data['eigenvalues']
        if not np.iscomplexobj(eigs):
            raise ValueError
    except Exception:
        eigs = data['eig_real'] + 1j * data['eig_imag']
    return (eigs,
            float(data['lambda_real_max']),
            float(data['lambda_imag_max']),
            float(data['h']),
            int(data['m']), int(data['n']),
            float(data['t_eq']))


# ── Figure 1: eigenvalue cloud in physical units ──────────────────────────────

def fig_eigenvalue_cloud(eigs, lam_real_max, lam_imag_max, h, m, n, t_eq,
                          dt, out):
    dt_lf  = 1.0 / lam_imag_max
    dt_rk4 = 2.0 * np.sqrt(2) / lam_imag_max

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(eigs.real, eigs.imag, s=14, color=C_EIG,
               alpha=0.8, linewidths=0, zorder=3,
               label=f'Eigenvalues of $J_{{\\mathcal{{F}}}}(q_0)$  '
                     f'({len(eigs)} computed)')

    ax.axhline( lam_imag_max, color=C_LF,  lw=1.3, ls='--',
                label=f'$|\\mathrm{{Im}}(\\lambda)|_{{\\max}} = {lam_imag_max:.3f}$'
                      f'  →  $\\Delta t_{{\\mathrm{{LF}}}}^{{\\max}} = {dt_lf:.2f}$')
    ax.axhline(-lam_imag_max, color=C_LF,  lw=1.3, ls='--')
    ax.axvline(-lam_real_max, color=C_RK4, lw=1.3, ls=':',
               label=f'$|\\mathrm{{Re}}(\\lambda)|_{{\\max}} = {lam_real_max:.3f}$'
                     f'  →  $\\Delta t_{{\\mathrm{{RK4}}}}^{{\\max}} = {dt_rk4:.2f}$')

    ax.text(0.03, 0.97,
            f'$\\Delta t_{{\\max}}^{{\\mathrm{{LF}}}}$  = {dt_lf:.3f}\n'
            f'$\\Delta t_{{\\max}}^{{\\mathrm{{RK4}}}}$ = {dt_rk4:.3f}\n'
            f'Linear bound underest.: {dt_lf/0.0904:.1f}$\\times$',
            transform=ax.transAxes, fontsize=10,
            va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='#cccccc',
                      boxstyle='round,pad=0.5', alpha=0.9))

    style_ax(ax,
             xlabel='Re($\\lambda$)',
             ylabel='Im($\\lambda$)',
             title=f'Eigenvalue cloud of $J_{{\\mathcal{{F}}}}(q_0)$ ,  '
                   f'QG {m}$\\times${n}  ($h={h:.4f}$,  $t_{{\\mathrm{{eq}}}}={t_eq:.0f}$)')
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2: stability regions with scaled eigenvalues ──────────────────────

def fig_stability_regions(eigs, lam_imag_max, h, m, n, t_eq, dt, out):
    z = eigs * dt
    dt_lf  = 1.0 / lam_imag_max
    dt_rk4 = 2.0 * np.sqrt(2) / lam_imag_max

    fig, ax = plt.subplots(figsize=(8, 7))

    RE, IM, Rrk4 = stability_grid(R_rk4)
    _,  _,  Rdp5 = stability_grid(R_dp5)

    ax.contourf(RE, IM, Rrk4, levels=[0, 1],
                colors=[C_RK4], alpha=0.15)
    ax.contour(RE, IM, Rrk4, levels=[1],
               colors=[C_RK4], linewidths=2.0,
               linestyles='-')
    ax.contour(RE, IM, Rdp5, levels=[1],
               colors=[C_DP5], linewidths=1.4,
               linestyles='--')

    # Leapfrog stability segment
    ax.plot([0, 0], [-1, 1], color=C_LF, lw=5,
            solid_capstyle='round', zorder=4)

    # scaled eigenvalues
    ax.scatter(z.real, z.imag, s=14, color=C_EIG,
               alpha=0.85, linewidths=0, zorder=5,
               label=f'$\\lambda \\cdot \\Delta t$  ({len(z)} eigs,  $\\Delta t={dt}$)')

    ax.text(-3.5,  0.5, 'RK4', color=C_RK4, fontsize=11, fontweight='bold')
    ax.text(-3.5,  3.4, 'dp5', color=C_DP5, fontsize=11, fontweight='bold')
    ax.text( 0.10, 0.8, 'LF',  color=C_LF,  fontsize=11, fontweight='bold')

    handles = [
        mpatches.Patch(color=C_RK4, alpha=0.4,
                       label='RK4 stable region'),
        plt.Line2D([0],[0], color=C_DP5, lw=1.5, ls='--',
                   label='dp5 boundary'),
        plt.Line2D([0],[0], color=C_LF, lw=4,
                   label='Leapfrog  $|\\mathrm{Im}(z)| \\leq 1$'),
        plt.Line2D([0],[0], color=C_EIG, marker='o', ms=5, ls='none',
                   label=f'$\\lambda \\cdot \\Delta t$  ({len(z)} eigs)'),
    ]

    style_ax(ax,
             xlabel='Re($z$) $= $ Re($\\lambda$) $\\cdot\\, \\Delta t$',
             ylabel='Im($z$) $= $ Im($\\lambda$) $\\cdot\\, \\Delta t$',
             title=f'Stability regions in the $z$-plane  ($\\Delta t = {dt}$,  '
                   f'QG {m}$\\times${n})')
    ax.set_xlim(-4, 0.6)
    ax.set_ylim(-4, 4)
    ax.legend(handles=handles, fontsize=9, loc='lower left')
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    print(f"\n  Loading '{args.spectrum}'...")
    eigs, lam_real_max, lam_imag_max, h, m, n, t_eq = load_spectrum(args.spectrum)

    dt_lf  = 1.0 / lam_imag_max
    dt_rk4 = 2.0 * np.sqrt(2) / lam_imag_max
    print(f"  max|Re(lambda)| = {lam_real_max:.4e}")
    print(f"  max|Im(lambda)| = {lam_imag_max:.4e}")
    print(f"  dt_max LF  = {dt_lf:.4f}")
    print(f"  dt_max RK4 = {dt_rk4:.4f}")
    print(f"  Underestimation factor: {dt_lf/0.0904:.1f}x")

    print("\n  Generating figures...")
    fig_eigenvalue_cloud(eigs, lam_real_max, lam_imag_max,
                         h, m, n, t_eq, args.dt,
                         f"{out}/fig_eigenvalue_cloud.png")
    fig_stability_regions(eigs, lam_imag_max,
                          h, m, n, t_eq, args.dt,
                          f"{out}/fig_stability_regions.png")
    print(f"\n  Done, 2 figures saved to '{out}/'")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spectrum', required=True)
    p.add_argument('--dt',      type=float, default=1.0)
    p.add_argument('--out-dir', default='figures')
    run(p.parse_args())