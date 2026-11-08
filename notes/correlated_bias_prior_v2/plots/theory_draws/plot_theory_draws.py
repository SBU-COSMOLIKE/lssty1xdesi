"""Redraw theory_draws.pdf from data.json: random draws of the BGS bin
deviations from the exponential-kernel prior (left) and from the memory
prior, a free tilt plus the kernel on the residual (right), against the BGS
flux-limit drift. sigma_g is the prior width of one bin and L the
correlation length in redshift. data.json comes from
../make_theory_data.py. Run from any folder: python plot_theory_draws.py
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
DRAW_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b']


def data_points(ax):
    '''Display coordinates of every drawn data point of ax: the markers, and
    points sampled densely along every line and error bar.'''
    pts = [np.empty((0, 2))]
    t = np.linspace(0.0, 1.0, 50)[:, None]
    for line in ax.get_lines():
        xy = line.get_transform().transform(
            np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()]))
        pts.append(xy)
        if line.get_linestyle() not in ('None', 'none', ' ', ''):
            for a, b in zip(xy[:-1], xy[1:]):
                pts.append(a + t*(b - a))
    for coll in ax.collections:
        for seg in getattr(coll, 'get_segments', lambda: [])():
            xy = coll.get_transform().transform(np.asarray(seg, dtype=float))
            for a, b in zip(xy[:-1], xy[1:]):
                pts.append(a + t*(b - a))
    pts = np.vstack(pts)
    return pts[np.all(np.isfinite(pts), axis=1)]


def check_clear(fig, labels):
    '''Raise if a legend or text label touches a data point of any panel (a
    margin of 3 points around each label counts as touching).'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = 3.0*fig.dpi/72.0
    pts = np.vstack([data_points(ax) for ax in fig.axes])
    for name, artist in labels:
        box = artist.get_window_extent(renderer)
        hit = ((pts[:, 0] > box.x0 - pad) & (pts[:, 0] < box.x1 + pad)
               & (pts[:, 1] > box.y0 - pad) & (pts[:, 1] < box.y1 + pad))
        if np.any(hit):
            raise RuntimeError(f'{name} touches the data')


with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
z = np.array(data['z'])
kernel = data['kernel_prior']
memory = data['memory_prior']

# full width of a two-column article
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True, gridspec_kw={'wspace': 0.06})
for draw, color in zip(data['kernel_draws'], DRAW_COLORS):
    axes[0].plot(z, 100.0*np.array(draw), color=color, lw=1.2, marker='o', ms=2.5)
for draw, color in zip(data['memory_draws'], DRAW_COLORS):
    axes[1].plot(z, 100.0*np.array(draw), color=color, lw=1.2, marker='o', ms=2.5)
for ax in axes:
    ax.plot(z, 100.0*np.array(data['flux_limit']), color='black', lw=2.6, ls='-.')
    ax.axhline(0.0, color='0.2', lw=0.8)
    ax.set_xlabel(r'redshift $z$ (BGS, $\Delta z = 0.02$ bins)', fontsize=10)
axes[0].set_ylabel(r'$\delta_i$ [%], mean removed', fontsize=10)
axes[0].set_title(rf'kernel: $\sigma_g = {100*kernel["sigma_b"]:.0f}\%$, $L = {kernel["L"]:g}$',
                  fontsize=10.5)
axes[1].set_title(rf'memory: free tilt + $\sigma_g = {100*memory["sigma_b"]:.0f}\%$, '
                  rf'$L = {memory["L"]:g}$', fontsize=10.5)
handles = [Line2D([], [], color='black', lw=2.6, ls='-.', label=r'BGS flux limit $r < 19.5$'),
           Line2D([], [], color='0.4', lw=1.2, marker='o', ms=2.5, label='prior draws')]
legend = axes[1].legend(handles=handles, loc='lower right', frameon=False, fontsize=9.5)
check_clear(fig, [('legend', legend)])
fig.savefig(os.path.join(HERE, 'theory_draws.pdf'))
