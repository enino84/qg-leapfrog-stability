"""
qg_viewer.py, Interactive viewer for QG-Python output (.npz)

Usage:
  python qg_viewer.py qg.npz
  python qg_viewer.py qg.npz --field psi      # or q
  python qg_viewer.py qg.npz --save anim.gif  # export animation

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
import argparse
import sys
import os


# ── colour maps ──────────────────────────────────────────────────────────────
CMAPS = ['RdBu_r', 'seismic', 'bwr', 'coolwarm', 'PuOr', 'PRGn']


def load(fname):
    data = np.load(fname)
    t    = data['t']
    psi  = data['psi'].astype(float)
    q    = data['q'].astype(float) if 'q' in data else None
    meta = {k: float(data[k]) for k in ('lx','dt','rkb','rkh','rkh2','F','r','A','k')
            if k in data}
    return t, psi, q, meta


def energy_series(psi, h):
    """Domain-averaged kinetic energy proxy: <psi^2>"""
    return np.mean(psi**2, axis=(1, 2))


def enstrophy_series(q):
    """Domain-averaged enstrophy: <q^2>"""
    return np.mean(q**2, axis=(1, 2)) if q is not None else None


# ── main viewer ──────────────────────────────────────────────────────────────

def run_viewer(fname, start_field='psi', save_to=None):
    t, psi, q, meta = load(fname)
    nrec, m, n = psi.shape
    h = meta.get('lx', 1.0) / (n - 1)

    has_q   = q is not None
    fields  = {'psi': psi, 'q': q} if has_q else {'psi': psi}
    field   = start_field if (start_field in fields) else 'psi'

    EK = energy_series(psi, h)
    EN = enstrophy_series(q)

    # ── figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor='#0d1117')
    fig.canvas.manager.set_window_title('QG-Python Viewer')

    # grid: left = main map (tall), right-top = energy, right-bottom = enstrophy
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[2.4, 1],
        height_ratios=[1, 1, 0.22],
        hspace=0.45, wspace=0.32,
        left=0.06, right=0.96, top=0.91, bottom=0.13,
    )

    ax_map  = fig.add_subplot(gs[0:2, 0])
    ax_ek   = fig.add_subplot(gs[0,   1])
    ax_en   = fig.add_subplot(gs[1,   1]) if has_q else None

    for ax in ([ax_map, ax_ek] + ([ax_en] if ax_en else [])):
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values():
            sp.set_color('#30363d')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')

    # ── initial plot ─────────────────────────────────────────────────────────
    F   = fields[field]
    lim = np.percentile(np.abs(F), 98)

    lx = meta.get('lx', 1.0)
    extent = [0, lx, 0, lx]

    im = ax_map.imshow(
        F[0], origin='lower', extent=extent,
        cmap=CMAPS[0], vmin=-lim, vmax=lim,
        interpolation='bilinear', aspect='equal',
    )
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(colors='#8b949e', labelsize=7)
    cbar.outline.set_color('#30363d')

    title = ax_map.set_title(
        f'{field}   t = {t[0]:.1f}',
        color='#e6edf3', fontsize=11, fontfamily='monospace', pad=8,
    )
    ax_map.set_xlabel('x', fontsize=9)
    ax_map.set_ylabel('y', fontsize=9)

    # contour overlay
    contours = [ax_map.contour(
        np.linspace(0, lx, n), np.linspace(0, lx, m),
        F[0], levels=12, colors='white', linewidths=0.25, alpha=0.25,
    )]

    # energy plot
    ax_ek.plot(t, EK, color='#58a6ff', lw=1.0, alpha=0.85)
    vline_ek = ax_ek.axvline(t[0], color='#f78166', lw=1.2, ls='--')
    ax_ek.set_title('KE  ⟨ψ²⟩', color='#e6edf3', fontsize=8, fontfamily='monospace')
    ax_ek.set_xlabel('t', fontsize=8); ax_ek.set_ylabel('')

    if ax_en and EN is not None:
        ax_en.plot(t, EN, color='#3fb950', lw=1.0, alpha=0.85)
        vline_en = ax_en.axvline(t[0], color='#f78166', lw=1.2, ls='--')
        ax_en.set_title('Enstrophy  ⟨q²⟩', color='#e6edf3', fontsize=8, fontfamily='monospace')
        ax_en.set_xlabel('t', fontsize=8)
    else:
        vline_en = None

    # meta text
    meta_str = '  '.join(f'{k}={v:.3g}' for k, v in list(meta.items())[:5])
    fig.text(0.5, 0.965, os.path.basename(fname),
             ha='center', color='#58a6ff', fontsize=9, fontfamily='monospace')
    fig.text(0.5, 0.945, meta_str,
             ha='center', color='#6e7681', fontsize=7.5, fontfamily='monospace')
    fig.text(0.06, 0.965, f'{m}×{n}  |  {nrec} records',
             color='#6e7681', fontsize=8, fontfamily='monospace')

    # ── widgets ──────────────────────────────────────────────────────────────
    # time slider
    ax_slider = fig.add_axes([0.10, 0.045, 0.60, 0.025], facecolor='#161b22')
    slider = Slider(ax_slider, '', 0, nrec - 1, valinit=0, valstep=1,
                    color='#58a6ff', initcolor='none')
    slider.label.set_color('#8b949e')
    slider.valtext.set_color('#8b949e')
    ax_slider.set_xlabel('record', color='#8b949e', fontsize=8, labelpad=-2)

    # play / pause button
    ax_btn = fig.add_axes([0.74, 0.038, 0.06, 0.038], facecolor='#21262d')
    btn_play = Button(ax_btn, '▶  Play', color='#21262d', hovercolor='#30363d')
    btn_play.label.set_color('#e6edf3')
    btn_play.label.set_fontsize(8)

    # field selector (only if q available)
    if has_q:
        ax_radio = fig.add_axes([0.82, 0.028, 0.10, 0.07], facecolor='#161b22')
        radio = RadioButtons(ax_radio, ('psi', 'q'), activecolor='#58a6ff')
        for lbl in radio.labels:
            lbl.set_color('#e6edf3'); lbl.set_fontsize(8)
    else:
        radio = None

    # ── state ────────────────────────────────────────────────────────────────
    state = {'rec': 0, 'playing': False, 'field': field, 'cmap_idx': 0}

    def update_frame(rec):
        rec = int(rec)
        state['rec'] = rec
        F   = fields[state['field']]
        lim = np.percentile(np.abs(F), 98)
        im.set_data(F[rec])
        im.set_clim(-lim, lim)
        title.set_text(f"{state['field']}   t = {t[rec]:.1f}")
        vline_ek.set_xdata([t[rec], t[rec]])
        if vline_en:
            vline_en.set_xdata([t[rec], t[rec]])
        # refresh contours (compatible with matplotlib >= 3.8)
        contours[0].remove()
        contours[0] = ax_map.contour(
            np.linspace(0, lx, n), np.linspace(0, lx, m),
            F[rec], levels=12, colors='white', linewidths=0.25, alpha=0.25,
        )
        fig.canvas.draw_idle()

    slider.on_changed(update_frame)

    # animation timer
    anim_obj = [None]

    def toggle_play(event):
        state['playing'] = not state['playing']
        btn_play.label.set_text('⏸  Pause' if state['playing'] else '▶  Play')
        if state['playing']:
            def step(frame):
                if not state['playing']:
                    return
                rec = (state['rec'] + 1) % nrec
                slider.set_val(rec)
            anim_obj[0] = animation.FuncAnimation(
                fig, step, interval=60, cache_frame_data=False)
        else:
            if anim_obj[0]:
                anim_obj[0].event_source.stop()
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    if radio:
        def switch_field(label):
            state['field'] = label
            update_frame(state['rec'])
        radio.on_clicked(switch_field)

    # keyboard shortcuts
    def on_key(event):
        if event.key == 'right':
            slider.set_val(min(state['rec'] + 1, nrec - 1))
        elif event.key == 'left':
            slider.set_val(max(state['rec'] - 1, 0))
        elif event.key == ' ':
            toggle_play(None)
        elif event.key == 'home':
            slider.set_val(0)
        elif event.key == 'end':
            slider.set_val(nrec - 1)
        elif event.key == 'c':
            state['cmap_idx'] = (state['cmap_idx'] + 1) % len(CMAPS)
            im.set_cmap(CMAPS[state['cmap_idx']])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)

    # ── export animation ─────────────────────────────────────────────────────
    if save_to:
        print(f"Saving animation to {save_to} ({nrec} frames)...")
        F   = fields[field]
        lim = np.percentile(np.abs(F), 98)

        fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor='#0d1117')
        ax2.set_facecolor('#161b22')
        im2 = ax2.imshow(F[0], origin='lower', extent=extent,
                         cmap=CMAPS[0], vmin=-lim, vmax=lim,
                         interpolation='bilinear', aspect='equal')
        ttl2 = ax2.set_title(f'{field}  t={t[0]:.1f}',
                              color='#e6edf3', fontfamily='monospace')
        fig2.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02)
        plt.tight_layout()

        def anim_func(i):
            im2.set_data(F[i])
            ttl2.set_text(f'{field}  t={t[i]:.1f}')
            return im2, ttl2

        ani = animation.FuncAnimation(fig2, anim_func, frames=nrec,
                                      interval=50, blit=True)
        ext = os.path.splitext(save_to)[1].lower()
        if ext == '.gif':
            ani.save(save_to, writer='pillow', fps=20, dpi=90)
        else:
            ani.save(save_to, fps=20, dpi=120)
        print("Done.")
        return

    # ── help overlay ─────────────────────────────────────────────────────────
    help_txt = 'Keys:  ←/→ step   Space play/pause   Home/End   C cycle colourmap'
    fig.text(0.10, 0.005, help_txt, color='#6e7681', fontsize=7.5, fontfamily='monospace')

    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interactive viewer for QG-Python .npz output')
    parser.add_argument('file', help='output file (.npz)')
    parser.add_argument('--field', default='psi', choices=['psi', 'q'],
        help='field to display at startup (default: psi)')
    parser.add_argument('--save', metavar='FILE',
        help='export animation to file (.gif or .mp4) instead of showing viewer')
    args = parser.parse_args()

    run_viewer(args.file, start_field=args.field, save_to=args.save)