"""Redraw the two FoM figures of this folder from data.json: the figures of
merit of each DESI layout down the linear-bias priors, from growth-tied
(one amplitude per group, b = B/D(z)) to one free bias per bin, under
model A (one-loop bias, notes/higher_order_bias) in its two forecast
variants: model A baseline (solid) and model A pessimistic (dashed), one
color per layout.

  baseline_ladder.pdf       - FoM(w0, wa), with the LSST Y1 3x2pt reference
  baseline_ladder_om_as.pdf - FoM(Omega_m, 10^9 A_s)

Each is one column of a two-column paper and stands alone (the LaTeX can
set them side by side). data.json is written by
EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb; it also holds the v1 linear-bias
and the exact-calibration FoMs of (w0, wa), which these figures do not draw.
Run from any folder: python plot_baseline_ladder.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Figure style of the whole notebook: the shared plotting functions never change
# rcParams, so these lines set every font, tick and grid default.
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

HERE = os.path.dirname(os.path.abspath(__file__))
# the line style of each drawn variant: the two forecast variants of model A
STYLES = {
    'A baseline': {'ls': '-', 'lw': 1.6, 'marker': 'o', 'ms': 4.5, 'label': 'model A baseline'},
    'A pessimistic': {'ls': '--', 'lw': 1.3, 'marker': 'D', 'ms': 4.0,
                      'label': 'model A pessimistic'},
}
# the LSST Y1 reference is dotted, so it never reads as a dashed model line
LSST_STYLE = {'color': '0.35', 'ls': ':', 'lw': 1.3}
# the names the figures print for the linear-bias priors (the data keys
# keep the notebook's names)
DISPLAY = {'tied': 'growth-tied', 'baseline': 'baseline',
           'pessimistic': 'pessimistic', 'independent': 'free'}
# one figure per figure of merit: data.json field, y label, output name
FIGURES = [('fom', r'FoM$(w_0, w_a)$, DESI $C_\ell^{gg}$ alone', 'baseline_ladder'),
           ('fom_om_as', r'FoM$(\Omega_\mathrm{m}, 10^9 A_\mathrm{s})$, DESI $C_\ell^{gg}$ alone',
            'baseline_ladder_om_as')]

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
rungs = [r['key'] for r in data['rungs']]
x = np.arange(len(rungs))
layouts = data['layouts']
lsst = data['lsst_y1_3x2pt_fom']
# k_max is the z = 0 value of the per-bin cut at k_max/D(z) (the caption
# explains the UV-fraction criterion and the shipped mask)
KMAX_TEXT = (rf'$k_\mathrm{{max}} = {data["kmax_h_per_Mpc"]:g}\,h\,\mathrm{{Mpc}}^{{-1}}$'
             r' at $z = 0$')


def line_points(ax):
    '''Display coordinates of points densely sampled along every line of ax,
    the LSST Y1 line included.'''
    pts = []
    for line in ax.get_lines():
        xd, yd = (np.asarray(v, dtype=float) for v in line.get_data())
        if line.get_transform() != ax.transData:
            # axhline: x in axes units, y in data units
            xs = np.linspace(*ax.get_xlim(), 400)
            pts.append(ax.transData.transform(np.column_stack([xs, np.full(400, yd[0])])))
            continue
        for i in range(len(xd) - 1):
            t = np.linspace(0.0, 1.0, 60)
            pts.append(ax.transData.transform(np.column_stack(
                [xd[i] + t*(xd[i + 1] - xd[i]), yd[i] + t*(yd[i + 1] - yd[i])])))
    return np.vstack(pts)


for field, ylabel, name in FIGURES:
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for k in ('baseline', 'pessimistic'):
        ax.axvspan(rungs.index(k) - 0.22, rungs.index(k) + 0.22, color='0.93', zorder=0)
    for lay in layouts:
        for variant, st in STYLES.items():
            y = np.array([lay[field][variant][k] for k in rungs])
            ax.plot(x, y, color=lay['color'], ls=st['ls'], lw=st['lw'], marker=st['marker'],
                    ms=st['ms'])
    top = max(lay[field][v][k] for lay in layouts for v in STYLES for k in rungs)
    with_lsst = field == 'fom'
    if with_lsst:
        ax.axhline(lsst, **LSST_STYLE)
        top = max(top, lsst)
    ax.set_xlim(-0.35, x[-1] + 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[r['key']] for r in data['rungs']], fontsize=9.5)
    ax.grid(False, axis='x')
    ax.set_ylabel(ylabel, fontsize=10)

    # the layouts by color, inside the panel in its empty upper right,
    # under the cut they share
    layout_legend = ax.legend(
        handles=[Line2D([], [], color=lay['color'], lw=2.0, label=lay['label'])
                 for lay in layouts],
        loc='upper right', frameon=False, fontsize=8, handlelength=1.2, labelspacing=0.2,
        handletextpad=0.5, borderaxespad=0.25, borderpad=0.2, title=KMAX_TEXT,
        title_fontsize=8)
    ax.add_artist(layout_legend)
    # the line styles above the panel, the same in both figures so their
    # panels line up side by side (the caption names the dotted LSST Y1
    # 3x2pt reference)
    style_handles = [Line2D([], [], color='0.25', ls=st['ls'], lw=st['lw'], marker=st['marker'],
                            ms=st['ms'], label=st['label']) for st in STYLES.values()]
    ax.legend(handles=style_handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=8.5, handlelength=2.4, columnspacing=1.0,
              handletextpad=0.5)

    # The legend inside the panel must never sit on the data. The y axis
    # gets the least headroom at which it clears every line (the LSST
    # reference included), so the axis is not stretched more than needed.
    for headroom in np.arange(1.12, 1.62, 0.02):
        ax.set_ylim(0.0, top*headroom)
        fig.canvas.draw()
        box = layout_legend.get_window_extent(fig.canvas.get_renderer()).expanded(1.03, 1.05)
        pts = line_points(ax)
        if not np.any((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1)
                      & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1)):
            break
    else:
        raise RuntimeError(f'{name}: the layout legend touches a line at every headroom')
    fig.savefig(os.path.join(HERE, f'{name}.pdf'))
    fig.savefig(os.path.join(HERE, f'{name}.png'), dpi=300)
    plt.close(fig)
