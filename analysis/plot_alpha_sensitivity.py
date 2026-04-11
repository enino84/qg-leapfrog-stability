"""
plot_alpha_sensitivity.py
Generate two independent figures:
  figures/fig_alpha_ke_bias.png
  figures/fig_alpha_enst_bias.png
Run from the repo root where the .npz files live.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── data ──────────────────────────────────────────────────────────────────────

files = {
    0.05: 'qg_lf_ra_a005.npz',
    0.10: 'qg_lf_ra.npz',
    0.20: 'qg_lf_ra_a020.npz',
    0.40: 'qg_lf_ra_a040.npz',
}
rk4_file = 'qg_rk4_ref.npz'


def compute_stats(psi, lx):
    nx = psi.shape[2]
    h  = lx / (nx - 1)
    ke   = np.mean(psi**2, axis=(1, 2))
    lap  = (np.roll(psi, -1, 1) + np.roll(psi, 1, 1) +
            np.roll(psi, -1, 2) + np.roll(psi, 1, 2) - 4*psi) / h**2
    enst = np.mean(lap**2, axis=(1, 2))
    return ke, enst


# RK4 reference
d          = np.load(rk4_file)
lx         = float(d['lx'])
ke0, enst0 = compute_stats(d['psi'], lx)
ke_rk4     = ke0.mean()
enst_rk4   = enst0.mean()

# LF+RA for each alpha
alphas    = sorted(files.keys())
ke_bias   = []
enst_bias = []

for a in alphas:
    d = np.load(files[a])
    ke, enst = compute_stats(d['psi'], lx)
    ke_bias.append((ke.mean()   - ke_rk4)   / ke_rk4   * 100)
    enst_bias.append((enst.mean() - enst_rk4) / enst_rk4 * 100)

alphas    = np.array(alphas)
ke_bias   = np.array(ke_bias)
enst_bias = np.array(enst_bias)

# ── shared style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.size':        13,
    'axes.titlesize':   14,
    'axes.labelsize':   13,
    'legend.fontsize':  11,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'figure.dpi':       150,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'figure.facecolor': 'white',
})

col    = '#d62728'
mstyle = dict(marker='o', markersize=8, linewidth=2, color=col, zorder=3)


def annotate_points(ax, xs, ys):
    for x, y in zip(xs, ys):
        ax.annotate(f'{y:.1f}%', xy=(x, y),
                    xytext=(0, -18), textcoords='offset points',
                    ha='center', fontsize=10, color=col)


# ── figure 1: KE bias ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.axhline(0, color='steelblue', lw=1.5, ls='--', label='RK4 (reference)')
ax.plot(alphas, ke_bias, **mstyle, label='LF+RA')
ax.fill_between(alphas, ke_bias, 0, color=col, alpha=0.08)
annotate_points(ax, alphas, ke_bias)

ax.set_xlabel(r'Filter parameter $\alpha$')
ax.set_ylabel('KE bias (%)')
ax.set_title(r'Kinetic energy bias vs $\alpha$  (LF+RA, $\Delta t=0.5$)')
ax.set_ylim(-30, 5)
ax.set_xticks(alphas)
ax.set_xticklabels([str(a) for a in alphas])
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
ax.legend(loc='lower right')
fig.tight_layout()

out1 = 'figures/fig_alpha_ke_bias.png'
fig.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out1}')
plt.close()

# ── figure 2: enstrophy bias ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.axhline(0, color='steelblue', lw=1.5, ls='--', label='RK4 (reference)')
ax.plot(alphas, enst_bias, **mstyle, label='LF+RA')
ax.fill_between(alphas, enst_bias, 0, color=col, alpha=0.08)
annotate_points(ax, alphas, enst_bias)

ax.set_xlabel(r'Filter parameter $\alpha$')
ax.set_ylabel('Enstrophy bias (%)')
ax.set_title(r'Enstrophy bias vs $\alpha$  (LF+RA, $\Delta t=0.5$)')
ax.set_ylim(-90, 5)
ax.set_xticks(alphas)
ax.set_xticklabels([str(a) for a in alphas])
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
ax.legend(loc='lower right')
fig.tight_layout()

out2 = 'figures/fig_alpha_enst_bias.png'
fig.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out2}')
plt.close()

print('Done.')