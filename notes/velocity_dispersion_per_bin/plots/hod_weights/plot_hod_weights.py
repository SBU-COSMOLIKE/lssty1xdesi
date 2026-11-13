"""Redraw hod_weights.pdf from data.json: the DESI LRG occupation and the
host masses of its centrals and satellites at z = 0.5.

One column, two stacked panels sharing the mass axis:
1. the mean occupations <N_c>(M) (blue, solid) and <N_s>(M) (orange,
   dashed) of the One-Percent xi fit at 0.4 < z < 0.6, log y;
2. the host-mass distributions p(ln M) proportional to dn/dlnM <N>(M)
   on a Sheth-Tormen mass function, each normalized to unit area, with
   the mean host mass of each population marked by a vertical tick.
Run from any folder: python plot_hod_weights.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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
CEN = {'color': '#0072B2', 'lw': 2.0, 'ls': '-'}
SAT = {'color': '#E69F00', 'lw': 1.4, 'ls': (0, (5, 1.5))}

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
                    f'hod_weights: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'hod_weights: artists {i} and {j} overlap')


fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.1), sharex=True,
                         gridspec_kw={'hspace': 0.06})
ax = axes[0]
ax.plot(log_m, np.clip(data['ncen'], 1e-4, None), label=r'$\langle N_c\rangle$', **CEN)
nsat = np.array(data['nsat'])
pos = nsat > 1e-4
ax.plot(log_m[pos], nsat[pos], label=r'$\langle N_s\rangle$', **SAT)
ax.set_yscale('log')
ax.set_ylim(1e-3, 60.0)
ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1, 10]))
ax.set_ylabel('occupation', fontsize=9)
ax.grid(False)
leg = ax.legend(loc='upper left', fontsize=8, frameon=False,
                handlelength=2.4, labelspacing=0.3, borderaxespad=0.4)

ax = axes[1]
ax.fill_between(log_m, data['p_cen'], color=CEN['color'], alpha=0.25, lw=0)
ax.fill_between(log_m, data['p_sat'], color=SAT['color'], alpha=0.25, lw=0)
ax.plot(log_m, data['p_cen'], **CEN)
ax.plot(log_m, data['p_sat'], **SAT)
for mean, st in [(data['mean_log_m_cen'], CEN), (data['mean_log_m_sat'], SAT)]:
    ax.axvline(mean, ymin=0.0, ymax=0.14, color=st['color'], lw=1.6)
texts = [ax.text(data['mean_log_m_cen'], 0.70, 'centrals', fontsize=8,
                 color=CEN['color'], ha='center'),
         ax.text(data['mean_log_m_sat'] + 0.55, 0.38, 'satellites', fontsize=8,
                 color=SAT['color'], ha='left')]
ax.set_ylim(0.0, 0.92)
ax.yaxis.set_major_locator(FixedLocator([0, 0.4, 0.8]))
ax.set_ylabel(r'host-mass p$(\ln M)$', fontsize=9)
ax.grid(False)
axes[1].set_xlim(11.5, 15.2)
axes[1].xaxis.set_minor_locator(NullLocator())
axes[1].set_xlabel(r'$\log_{10}\, M\ [M_\odot/h]$', fontsize=10)
fig.align_ylabels(axes)
check_clear(fig, list(axes), [leg, *texts])
fig.savefig(os.path.join(HERE, 'hod_weights.pdf'))
fig.savefig(os.path.join(HERE, 'hod_weights.png'), dpi=300)
plt.close(fig)
