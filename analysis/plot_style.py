"""
plot_style.py
=============
Shared style constants and helpers for all QG paper figures.
Publication-ready: white background, large fonts, clean ticks.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── colour palette ────────────────────────────────────────────────────────────
C_RK4  = '#1f77b4'   # blue
C_RA   = '#d62728'   # red
C_RAW  = '#2ca02c'   # green
C_REF  = '#ff7f0e'   # orange  (tau_L, reference lines)
C_MUTED = '#666666'  # grey

SCHEME_COLORS  = {'RK4': C_RK4, 'LF+RA': C_RA, 'LF+RAW': C_RAW}
SCHEME_LABELS  = {'RK4': 'RK4  ($\\Delta t=1$)',
                  'LF+RA':  'LF + RA  ($\\Delta t=0.5,\\;\\alpha=0.1$)',
                  'LF+RAW': 'LF + RAW  ($\\Delta t=0.5,\\;\\alpha=0.1$)'}
SCHEME_LS      = {'RK4': '-', 'LF+RA': '--', 'LF+RAW': ':'}
SCHEME_LW      = {'RK4': 2.0, 'LF+RA': 1.8, 'LF+RAW': 1.8}

# ── rcParams for publication ──────────────────────────────────────────────────
PUB_RC = {
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.edgecolor':     '#333333',
    'axes.linewidth':     1.0,
    'axes.grid':          True,
    'grid.color':         '#dddddd',
    'grid.linewidth':     0.6,
    'axes.labelsize':     13,
    'axes.titlesize':     13,
    'axes.titlepad':      8,
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'xtick.major.size':   5,
    'ytick.major.size':   5,
    'xtick.minor.size':   3,
    'ytick.minor.size':   3,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'legend.fontsize':    10,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '#cccccc',
    'figure.titlesize':   14,
    'figure.titleweight': 'bold',
    'font.family':        'sans-serif',
    'font.size':          12,
    'lines.linewidth':    1.8,
    'savefig.dpi':        200,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
}


def apply_style():
    mpl.rcParams.update(PUB_RC)


def style_ax(ax, xlabel='', ylabel='', title=''):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', top=False, right=False)
