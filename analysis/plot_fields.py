"""
plot_fields.py
==============
Streamfunction snapshots, one independent figure per (scheme, time).

Output files in --out-dir:
  fig_fields_RK4_t{T}_sec2_initial_state.png
  fig_fields_LFRA_t{T}_sec2_initial_state.png
  fig_fields_LFRAW_t{T}_sec2_initial_state.png
  fig_fields_RK4_t{T}_sec4_mid_divergence.png
  ...  (9 files total for 3 schemes x 3 times)

In the paper, group them with \\subcaption per time instant:
  \\begin{figure}
    \\subcaption{$t=t_0$, initial state (all schemes identical)}
    \\includegraphics{fig_fields_RK4_t100000_sec2_initial_state}
    \\includegraphics{fig_fields_LFRA_t100000_sec2_initial_state}
    ...
  \\end{figure}

Usage
-----
    python analysis/plot_fields.py \
        --rk4    qg_rk4_ref.npz \
        --lf-ra  qg_lf_ra.npz  \
        --lf-raw qg_lf_raw.npz \
        --out-dir figures
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from plot_style import apply_style, C_RK4, C_RA, C_RAW
apply_style()

# (fraction_along_run, filename_tag, section_tag, title_hint)
TIMES = [
    (0.0, 'sec2_initial_state',
     'Initial state, all schemes identical'),
    (0.5, 'sec4_mid_divergence',
     'Mid-integration, trajectories diverged'),
    (1.0, 'sec6_final_state',
     'Final state, statistically decorrelated'),
]

SCHEMES = [
    ('RK4',     'RK4',   C_RK4),
    ('LF+RA',   'LFRA',  C_RA),
    ('LF+RAW',  'LFRAW', C_RAW),
]


def load(fname):
    d = np.load(fname)
    return d['t'].astype(float), d['psi'].astype(float)


def make_field_figure(psi_field, t_val, scheme_label, scheme_color,
                      clim, lx, title, out):
    m, n = psi_field.shape
    x = np.linspace(0, lx, n)
    y = np.linspace(0, lx, m)

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)

    im = ax.pcolormesh(x, y, psi_field,
                       cmap='RdBu_r', vmin=-clim, vmax=clim,
                       shading='auto', rasterized=True)
    ax.contour(x, y, psi_field,
               levels=10, colors='k', linewidths=0.4, alpha=0.4)

    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=9)

    for sp in ax.spines.values():
        sp.set_edgecolor(scheme_color)
        sp.set_linewidth(2.0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.06)
    cb  = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('$\\psi$', fontsize=10)

    plt.tight_layout()
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out}")


def run(args):
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    print("\n  Loading runs...")
    t_rk4,  psi_rk4  = load(args.rk4)
    t_ra,   psi_ra   = load(args.lf_ra)
    t_raw = psi_raw = None
    if args.lf_raw:
        t_raw, psi_raw = load(args.lf_raw)

    runs = [
        ('RK4',    'RK4',   C_RK4, t_rk4, psi_rk4),
        ('LF+RA',  'LFRA',  C_RA,  t_ra,  psi_ra),
    ]
    if psi_raw is not None:
        runs.append(('LF+RAW', 'LFRAW', C_RAW, t_raw, psi_raw))

    # global colour limit
    all_psi = np.concatenate([psi_rk4, psi_ra] +
                              ([psi_raw] if psi_raw is not None else []))
    clim = float(np.percentile(np.abs(all_psi), 98))
    lx   = 1.0

    t_start = t_rk4[0]
    t_end   = t_rk4[-1]

    print("  Generating figures...")
    for frac, time_tag, time_hint in TIMES:
        t_target = t_start + frac * (t_end - t_start)

        for scheme_label, scheme_tag, col, t_arr, psi_arr in runs:
            rec     = int(np.argmin(np.abs(t_arr - t_target)))
            t_actual = t_arr[rec]
            fname   = (f"fig_fields_{scheme_tag}"
                       f"_t{t_actual:.0f}_{time_tag}.png")
            title   = (f'{scheme_label} ,  $t = {t_actual:.0f}$\n'
                       f'{time_hint}')
            make_field_figure(psi_arr[rec], t_actual,
                              scheme_label, col, clim, lx,
                              title, os.path.join(out, fname))

    n_figs = len(TIMES) * len(runs)
    print(f"\n  Done, {n_figs} figures saved to '{out}/'")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rk4',     required=True)
    p.add_argument('--lf-ra',   required=True)
    p.add_argument('--lf-raw',  default=None)
    p.add_argument('--out-dir', default='figures')
    run(p.parse_args())