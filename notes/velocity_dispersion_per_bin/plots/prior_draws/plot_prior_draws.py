"""Redraw prior_draws.pdf from data.json: random truths of the per-bin
velocity-dispersion model on the Delta z = 0.02 layout.

One column, two stacked panels sharing both axes, one per residual prior:
residual A (10 percent of sigma_HOD, L = 0.2) on top, residual B (20
percent, BGS 30, L = 0.05) below.  Each panel draws three random truths
delta sigma_v,i = w_i sigma_v,g [s_g (zbar_i - zbar_g) + eps_i] in
percent of sigma_v,g (color + line width per draw; the same three seeds
in both panels, so the slopes repeat and only the residual changes).
Lines break between groups; the gray line is zero.
Run from any folder: python plot_prior_draws.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

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
DRAW_STYLE = [{'color': '#0072B2', 'lw': 2.0},
              {'color': '#E69F00', 'lw': 1.4},
              {'color': '#CC79A7', 'lw': 1.0}]
ZERO = {'color': '0.6', 'lw': 0.7}
PANELS = [('A', r'residual A ($L = 0.2$)'),
          ('B', r'residual B ($L = 0.05$)')]

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)


def line_points(ax):
    '''Display coordinates of points densely sampled along every drawn line of
    ax (reference lines included), inside the axes only.'''
    box = ax.get_window_extent()
    pts = [np.zeros((0, 2))]
    for line in ax.get_lines():
        xy = np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()])
        display_xy = line.get_transform().transform(xy)
        finite = np.all(np.isfinite(display_xy), axis=1)
        pts.append(display_xy[finite])
        t = np.linspace(0.0, 1.0, 60)[:, None]
        for i in range(len(display_xy) - 1):
            if finite[i] and finite[i + 1]:
                pts.append(display_xy[i] + t*(display_xy[i + 1] - display_xy[i]))
    pts = np.vstack(pts)
    inside = ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
              & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))
    return pts[inside]


def check_clear(fig, axes, artists):
    '''Raise when a label touches a drawn line or another label.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    for box, artist in zip(boxes, artists):
        padded = box.expanded(1.03, 1.08)
        for ax in axes:
            pts = line_points(ax)
            if np.any((pts[:, 0] > padded.x0) & (pts[:, 0] < padded.x1)
                      & (pts[:, 1] > padded.y0) & (pts[:, 1] < padded.y1)):
                raise RuntimeError(
                    f'prior_draws: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'prior_draws: artists {i} and {j} overlap')


fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.1), sharex=True, sharey=True,
                         gridspec_kw={'hspace': 0.06})
texts = []
for ax, (pname, plabel) in zip(axes, PANELS):
    ax.axhline(0.0, **ZERO)
    for st, draw in zip(DRAW_STYLE, data['priors'][pname]['draws']):
        for g in draw:
            ax.plot(g['zbar'], g['delta_pct'], **st)
    # the panel name as a right-hand rotated label, outside the data
    texts.append(ax.text(1.025, 0.5, plabel, transform=ax.transAxes,
                         fontsize=8, ha='left', va='center', rotation=90))
    ax.set_ylim(-18.5, 14.5)
    ax.yaxis.set_major_locator(FixedLocator([-15, -10, -5, 0, 5, 10]))
    ax.grid(False)
axes[-1].set_xlim(0.05, 1.65)
axes[-1].set_xlabel(r'bin mean redshift $\bar{z}_i$', fontsize=10)
fig.subplots_adjust(left=0.19)
sup = fig.supylabel(r'$\delta\sigma_{v,i}\ /\ \sigma_{v,g}$ [%]', fontsize=10,
                    x=0.015)
# the shared y label must clear every tick label
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
sup_box = sup.get_window_extent(renderer)
for ax in axes:
    for lab in ax.get_yticklabels():
        if lab.get_text() and sup_box.overlaps(lab.get_window_extent(renderer)):
            raise RuntimeError('prior_draws: the shared y label touches a tick')
check_clear(fig, list(axes), texts)
fig.savefig(os.path.join(HERE, 'prior_draws.pdf'))
fig.savefig(os.path.join(HERE, 'prior_draws.png'), dpi=300)
plt.close(fig)
