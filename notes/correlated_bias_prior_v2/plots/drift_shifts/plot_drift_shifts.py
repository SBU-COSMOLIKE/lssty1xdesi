"""Redraw drift_shifts.pdf from data.json: the shift of a model A fit
(no tilts) when the true linear biases follow a known drift away from the
growth-tied shape (b2 and bs2 following b1 through their calibrations).
Left column: a conserved galaxy population in every group; right column:
the r < 19.5 flux limit in BGS. One row per scored parameter: (w0, wa) in
units of the fit's ellipse, Omega_m and A_s in their own marginalized
sigma. Inside a panel the bin layouts are the columns and the marker shape
is the linear-bias prior. data.json, written by
EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb, also holds the free prior (exactly
zero) and model A + tilts, which the figure leaves out (the caption
says so).
Run from any folder: python plot_drift_shifts.py
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
VARIANT = 'A'
PARAM_LABELS = {'w0wa': r'$(w_0, w_a)$', 'omegam': r'$\Omega_\mathrm{m}$',
                'As_1e9': r'$A_\mathrm{s}$'}
# one marker shape per drawn linear-bias prior, at its own offset inside the
# layout's column (the color is the layout's)
# The prior markers are the notes' prior markers (growth-tied star, baseline
# down-triangle, pessimistic X, free plus); circles, squares, diamonds and
# up-triangles are kept for the bin layouts. The star is drawn larger: at one
# size it looks smaller than the others.
PRIOR_MARKERS = {'tied': {'marker': '*', 'ms': 6.5, 'dx': -0.22, 'label': 'growth-tied'},
                 'baseline': {'marker': 'v', 'ms': 4.5, 'dx': 0.0, 'label': 'baseline'},
                 'pessimistic': {'marker': 'X', 'ms': 4.5, 'dx': 0.22,
                                 'label': 'pessimistic'}}
# one column of panels per known drift
DRIFT_TITLES = {'conserved': 'conserved population', 'flux_limit': r'BGS flux limit $r < 19.5$'}
TICK_LABELS = {'Delta d_com = 62 Mpc': '62\nMpc', 'Delta z = 0.02': '0.02',
               'Delta z = 0.05': '0.05', 'Delta z = 0.2': '0.2'}
# Shifts below FLOOR sit on it: the exact zeros of the baseline and
# pessimistic priors on Delta z = 0.2 (its groups hold one or two bins, so
# the free amplitude and tilt absorb any drift) and a few tiny A_s shifts.
FLOOR = 1.0e-3


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
layouts = data['layouts']
params = data['params']
drifts = data['drifts']
x = np.arange(len(layouts))
values = [lay['shift'][VARIANT][d][r][p] for lay in layouts for d in drifts
          for r in PRIOR_MARKERS for p in params]
# one y range for every panel
y_range = (FLOOR/2.0, max(values)*6.0)

# one column of a two-column article: a column of panels per drift, a row
# per scored parameter, all on one y range
fig, axes = plt.subplots(len(params), len(drifts), figsize=(3.5, 1.05*len(params)),
                         sharex=True, sharey=True,
                         gridspec_kw={'hspace': 0.06, 'wspace': 0.06})
for row, p in enumerate(params):
    for col, drift in enumerate(drifts):
        ax = axes[row, col]
        ax.axhline(data['shift_threshold'], color='0.1', ls='--', lw=1.0)
        for i, lay in enumerate(layouts):
            shift = lay['shift'][VARIANT][drift]
            for prior, st in PRIOR_MARKERS.items():
                ax.plot(i + st['dx'], max(shift[prior][p], FLOOR), ls='none',
                        marker=st['marker'], ms=st['ms'], color=lay['color'])
        ax.set_yscale('log')
        ax.set_ylim(*y_range)
        # labeled decades away from the edges, where the panels sit close
        ax.set_yticks([1e-2, 1e0])
        ax.grid(False, axis='x')
        if row == 0:
            ax.set_title(DRIFT_TITLES[drift], fontsize=8.5, pad=3)
    # the scored parameter on the right side, rotated like the y label, so it
    # can never sit on the data
    axes[row, -1].yaxis.set_label_position('right')
    axes[row, -1].set_ylabel(PARAM_LABELS[p], rotation=90, fontsize=10, labelpad=4)
for ax in axes[-1]:
    ax.set_xlim(-0.5, len(layouts) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([TICK_LABELS.get(lay['name'], lay['name']) for lay in layouts],
                       fontsize=8)
fig.supxlabel(r'bin layout: $\Delta d_\mathrm{com}$ or $\Delta z$', y=-0.05, fontsize=9)
fig.supylabel(r'shift [$\sigma$]', x=-0.02, fontsize=9.5)

# one key per encoding: the marker shape is the prior (the color is the
# layout, read off the x axis; the dashed sigma/3 line is named in the
# caption)
handles = [Line2D([], [], ls='none', marker=st['marker'], ms=st['ms'], color='0.3',
                  label=st['label']) for st in PRIOR_MARKERS.values()]
# the reference line (sigma/3) carries no legend entry: the caption names it
legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.94),
                    ncol=3, frameon=False, fontsize=8, handlelength=1.4, columnspacing=0.8,
                    handletextpad=0.3, title='linear-bias prior of the fit',
                    title_fontsize=8)
check_clear(fig, [('legend', legend)])
fig.savefig(os.path.join(HERE, 'drift_shifts.pdf'))
fig.savefig(os.path.join(HERE, 'drift_shifts.png'), dpi=300)
