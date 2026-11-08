"""Redraw calibration_drift.pdf from data.json: the shift of a model A
baseline fit when the calibrations themselves drift, the b2 and bs2
offsets changing linearly in redshift across each group (b1 at its
fiducial), under the four linear-bias priors. (w0, wa) is scored in units
of the fit's ellipse, Omega_m and A_s in their own marginalized sigma. One
wide panel per scored parameter, stacked, one column per bin layout.
data.json, written by EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb, also holds
model A pessimistic, whose tilts absorb this drift exactly by construction
(the caption says so).
Run from any folder: python plot_calibration_drift.py
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
VARIANT = 'A baseline'
PARAM_LABELS = {'w0wa': r'$(w_0, w_a)$', 'omegam': r'$\Omega_\mathrm{m}$',
                'As_1e9': r'$A_\mathrm{s}$'}
# every linear-bias prior at its own offset inside the layout's column
RUNG_MARKERS = {'tied': {'marker': 's', 'dx': -0.27, 'label': 'growth-tied'},
                'baseline': {'marker': 'o', 'dx': -0.09, 'label': 'baseline'},
                'pessimistic': {'marker': 'D', 'dx': 0.09, 'label': 'pessimistic'},
                'independent': {'marker': '^', 'dx': 0.27, 'label': 'free'}}
TICK_LABELS = {'Delta d_com = 62 Mpc': '$\\Delta d_\\mathrm{com}$\n62 Mpc',
               'Delta z = 0.02': '$\\Delta z$\n0.02', 'Delta z = 0.05': '$\\Delta z$\n0.05',
               'Delta z = 0.2': '$\\Delta z$\n0.2'}


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
x = np.arange(len(layouts))
values = [lay['shift'][VARIANT][r][p] for lay in layouts for r in RUNG_MARKERS for p in params]
# one y range for every panel
y_range = (min(values)/2.5, max(values)*5.0)

fig, axes = plt.subplots(len(params), 1, figsize=(3.5, 1.15*len(params)), sharex=True,
                         sharey=True, gridspec_kw={'hspace': 0.06})
for ax, p in zip(axes, params):
    ax.axhline(data['shift_threshold'], color='firebrick', ls='--', lw=1.0)
    for i, lay in enumerate(layouts):
        shift = lay['shift'][VARIANT]
        for rung, st in RUNG_MARKERS.items():
            ax.plot(i + st['dx'], shift[rung][p], ls='none', marker=st['marker'], ms=5.0,
                    color=lay['color'])
    ax.set_yscale('log')
    ax.set_ylim(*y_range)
    ax.set_yticks([1e-2, 1e-1, 1])
    ax.grid(False, axis='x')
    # the scored parameter on the right side, rotated like the y label, so
    # it can never sit on the data
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(PARAM_LABELS[p], rotation=90, fontsize=11, labelpad=5)
axes[-1].set_xlim(-0.5, len(layouts) - 0.5)
axes[-1].set_xticks(x)
axes[-1].set_xticklabels([TICK_LABELS.get(lay['name'], lay['name']) for lay in layouts],
                         fontsize=9)
fig.supylabel(r'shift [$\sigma$]', x=-0.04, fontsize=10)
handles = [Line2D([], [], ls='none', marker=st['marker'], ms=5.0, color='0.3',
                  label=st['label']) for st in RUNG_MARKERS.values()]
handles.append(Line2D([], [], color='firebrick', ls='--', lw=1.0, label=r'$\sigma/3$'))
legend = axes[0].legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                        ncol=5, frameon=False, fontsize=8.5, handlelength=1.4,
                        columnspacing=0.8, handletextpad=0.3, title='linear-bias prior',
                        title_fontsize=8.5)
check_clear(fig, [('legend', legend)])
fig.savefig(os.path.join(HERE, 'calibration_drift.pdf'))
fig.savefig(os.path.join(HERE, 'calibration_drift.png'), dpi=300)
