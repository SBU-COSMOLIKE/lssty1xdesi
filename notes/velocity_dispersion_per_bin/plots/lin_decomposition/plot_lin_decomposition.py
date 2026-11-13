"""Redraw lin_decomposition.pdf from data.json: each dispersion group's
sigma_v,g split into the linear flow sigma_lin(zbar_i) and the halo part
sigma_HOD,i = sqrt(sigma_v,g^2 - sigma_lin,i^2), per bin of the
Delta z = 0.02 layout.

One column, one panel against the bin mean redshift:
- black flat segments: the fiducial sigma_v,g of the seven groups;
- blue solid curve: sigma_lin(z), one smooth line (it depends on z only);
- orange segments with markers: sigma_HOD,i per group.  Their rise inside
  each group is the T_lin truth: the fiducial holds sigma_v flat while
  the linear part falls, so the implied halo part must rise.
Every group name sits just above its own sigma_v segment. Where two groups
share a redshift range (LRG3 and ELG1), the two names are shifted apart in
z, so each reads as the label of the segment right under it.
Run from any folder: python plot_lin_decomposition.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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
SV_STYLE = {'color': 'black', 'lw': 1.5, 'ls': '-'}
LIN_STYLE = {'color': '#0072B2', 'lw': 2.4, 'ls': (0, (6, 1.5))}
HOD_STYLE = {'color': '#E69F00', 'lw': 1.5, 'ls': '-', 'marker': 'o', 'ms': 2.2}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
groups = data['groups']


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
    '''Raise when a legend or label touches a drawn line or another one.'''
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
                    f'lin_decomposition: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'lin_decomposition: artists {i} and {j} overlap')


fig, ax = plt.subplots(figsize=(3.5, 2.9))
# sigma_lin(z): one smooth curve over all bins (z-sorted union)
z_all = np.sort(np.concatenate([g['zbar'] for g in groups]))
slin_all = np.interp(
    z_all, *map(np.array, zip(*sorted(
        (z, s) for g in groups for z, s in zip(g['zbar'], g['sigma_lin'])))))
ax.plot(z_all, slin_all, **LIN_STYLE)
texts = []
for g in groups:
    zb = np.array(g['zbar'])
    ax.plot([zb[0], zb[-1]], [g['sigma_v']]*2, **SV_STYLE)
    ax.plot(zb, g['sigma_hod'], **HOD_STYLE)
    # group name just above its own sigma_v segment; LRG3 and ELG1 share a
    # redshift range, so their names are shifted apart in z (LRG3 right,
    # ELG1 left) and ELG2's slightly left, so no label touches another
    # group's line
    name = g['name']
    dy, dx, va = 9.0, 0.0, 'bottom'
    if name == 'LRG3':
        dx = 0.08
    if name == 'ELG1':
        dx = -0.12
    if name == 'ELG2':
        dx = -0.03
    texts.append(ax.text(g['zbar_g'] + dx, g['sigma_v'] + dy, name,
                         fontsize=7, ha='center', va=va))
ax.set_xlim(0.05, 1.65)
ax.set_ylim(80.0, 440.0)
ax.yaxis.set_major_locator(FixedLocator([100, 200, 300, 400]))
ax.set_xlabel(r'bin mean redshift $\bar{z}_i$', fontsize=10)
ax.set_ylabel(r'velocity [km/s]', fontsize=10)
ax.grid(False)
legend = ax.legend(
    handles=[Line2D([], [], label=r'$\sigma_{v,g}$ (fiducial)', **SV_STYLE),
             Line2D([], [], label=r'$\sigma_{\rm lin}(\bar{z}_i)$', **LIN_STYLE),
             Line2D([], [], label=r'$\sigma_{\rm HOD,\,}{}_i$', **HOD_STYLE)],
    loc='lower left', fontsize=7.5, frameon=False, handlelength=2.6,
    labelspacing=0.3, borderaxespad=0.3)
check_clear(fig, [ax], [legend, *texts])
fig.savefig(os.path.join(HERE, 'lin_decomposition.pdf'))
fig.savefig(os.path.join(HERE, 'lin_decomposition.png'), dpi=300)
plt.close(fig)
