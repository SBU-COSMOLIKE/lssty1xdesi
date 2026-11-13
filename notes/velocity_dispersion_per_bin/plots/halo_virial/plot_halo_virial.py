"""Redraw halo_virial.pdf from data.json: the virial velocity
sigma_halo(M_200c, z) of the Evrard et al. (2008) calibration at the mean
redshifts of four dispersion groups, on log-log axes.

One column, one panel.  The four curves are a redshift sweep (color +
line width + dash pattern).  Filled circles mark the mean host masses of
the DESI LRG centrals and satellites on the z = 0.5 curve, open squares
the ELG centrals and satellites on the z = 0.95 curve; small text names
each marker.  data.json also holds the singular-isothermal-sphere
estimate at z = 0.5 (sis_z0p5), which is not drawn: it runs about 6
percent above the z = 0.5 curve and would be indistinguishable here.
Run from any folder: python plot_halo_virial.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

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
# the redshift sweep: color + line width + dash pattern per curve
CURVE_STYLE = [
    {'color': '#0072B2', 'lw': 2.4, 'ls': '-'},
    {'color': '#E69F00', 'lw': 1.8, 'ls': (0, (5, 1.5))},
    {'color': '#CC79A7', 'lw': 1.3, 'ls': (0, (3, 1.5))},
    {'color': '#56B4E9', 'lw': 0.9, 'ls': (0, (1.2, 1.2))},
]
MARKER = {'LRG': {'marker': 'o', 'mfc': 'black'},
          'ELG': {'marker': 's', 'mfc': 'white'}}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
log_m = np.array(data['log_m_h'])


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
                    f'halo_virial: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'halo_virial: artists {i} and {j} overlap')


fig, ax = plt.subplots(figsize=(3.5, 2.6))
for st, curve in zip(CURVE_STYLE, data['curves']):
    ax.plot(log_m, curve['sigma_halo'], color=st['color'], lw=st['lw'],
            ls=st['ls'], label=f"$z = {curve['z']:g}$ ({curve['name']})")
texts = []
for m in data['markers']:
    sample = 'LRG' if 'LRG' in m['label'] else 'ELG'
    mk = MARKER[sample]
    ax.plot([m['log_m']], [m['sigma_halo']], marker=mk['marker'], ms=5,
            mfc=mk['mfc'], mec='black', ls='none', zorder=5)
    dx, dy = (0.10, -0.14) if 'central' in m['label'] else (-0.10, 0.09)
    ha = 'left' if dx > 0 else 'right'
    short = ('cen.' if 'central' in m['label'] else 'sat.')
    texts.append(ax.text(m['log_m'] + dx, m['sigma_halo']*10**dy,
                         f'{sample} {short}', fontsize=7, ha=ha, va='center'))
ax.set_yscale('log')
ax.set_xlim(11.0, 15.0)
ax.set_ylim(60.0, 1500.0)
ax.yaxis.set_major_locator(FixedLocator([100, 300, 1000]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels(['100', '300', '1000'])
ax.set_xlabel(r'$\log_{10}\, M_{200c}\ [M_\odot/h]$', fontsize=10)
ax.set_ylabel(r'$\sigma_{\rm halo}$ [km/s]', fontsize=10)
ax.grid(False)
legend = ax.legend(loc='upper left', fontsize=7, frameon=False,
                   handlelength=2.6, labelspacing=0.3, borderaxespad=0.3)
check_clear(fig, [ax], [legend, *texts])
fig.savefig(os.path.join(HERE, 'halo_virial.pdf'))
fig.savefig(os.path.join(HERE, 'halo_virial.png'), dpi=300)
plt.close(fig)
