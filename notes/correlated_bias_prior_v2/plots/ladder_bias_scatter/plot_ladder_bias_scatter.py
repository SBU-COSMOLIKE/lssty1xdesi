"""Redraw ladder_bias_scatter.pdf from data.json: the (w0, wa) shift of
fits to random truths under model A (one-loop bias, notes/higher_order_bias),
in units of each fit's own ellipse. Each truth is a known drift of the
linear bias plus a residual drawn from the baseline prior (top panel) or the
pessimistic prior (bottom panel), with the higher-order biases following
b1; every linear-bias prior fits every truth. Filled markers: model A
baseline (one offset per group); open markers: model A pessimistic
(offsets plus tilts). Points: the mean shift over the draws; bars: the
16th-84th percentiles. data.json, written by
EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb, also holds the shifts of the
calibration drift, which plots/calibration_drift draws.
Run from any folder: python plot_ladder_bias_scatter.py
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
# shifts below this floor are drawn at the floor (the free prior absorbs any
# b1 truth exactly, and so do the baseline and pessimistic priors on the
# Delta z = 0.2 groups of one or two bins)
SHIFT_FLOOR = 1.0e-3
# horizontal offsets: one per layout, and a smaller one per variant
OFFSETS = [-0.27, -0.09, 0.09, 0.27]
VARIANTS = {'A baseline': {'dx': -0.035, 'filled': True},
            'A pessimistic': {'dx': 0.035, 'filled': False}}
# the names the figure prints for the linear-bias priors (the data keys
# keep the notebook's names)
DISPLAY = {'tied': 'growth-tied', 'baseline': 'baseline',
           'pessimistic': 'pessimistic', 'independent': 'free'}
MARKER_SIZE = 3.6


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
rungs = [r['key'] for r in data['rungs']]
x = np.arange(len(rungs))
truths = data['truths']

# one column of a two-column article: wide, short panels with a thin gap
fig, axes = plt.subplots(len(truths), 1, figsize=(3.5, 1.45*len(truths)), sharex=True,
                         sharey=True, gridspec_kw={'hspace': 0.06})
labels = []
for ax, truth in zip(axes, truths):
    for k in ('baseline', 'pessimistic'):
        ax.axvspan(rungs.index(k) - 0.4, rungs.index(k) + 0.4, color='0.93', zorder=0,
                   lw=0)
    for lay, dx in zip(data['layouts'], OFFSETS):
        for variant, st in VARIANTS.items():
            fits = [lay['fits'][variant][rung][truth] for rung in rungs]
            face = lay['color'] if st['filled'] else 'white'
            mean = np.maximum([s['mean'] for s in fits], SHIFT_FLOOR)
            low = np.maximum([s['p16'] for s in fits], SHIFT_FLOOR)
            high = np.maximum([s['p84'] for s in fits], SHIFT_FLOOR)
            ax.errorbar(x + dx + st['dx'], mean, yerr=[mean - low, high - mean], ls='none',
                        color=lay['color'], marker='o', ms=MARKER_SIZE, mfc=face,
                        mec=lay['color'], mew=0.9, capsize=1.5, elinewidth=0.8)
    ax.axhline(data['shift_threshold'], color='firebrick', lw=1.0, ls='--')
    ax.set_yscale('log')
    ax.set_ylim(0.5*SHIFT_FLOOR, 60.0)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1])
    ax.grid(False, axis='x')
    # the truth's residual prior on the right side, rotated like the y
    # label, so it can never sit on the data
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(f'truth: {truth}\nscatter', rotation=90, fontsize=8.5, labelpad=4)
    labels.append((f'sigma/3 label ({truth})',
                   ax.text(x[-1] + 0.47, 1.7*data['shift_threshold'], r'$\sigma/3$',
                           va='bottom', ha='right', fontsize=8.5, color='firebrick')))
axes[-1].set_xticks(x)
axes[-1].set_xticklabels([DISPLAY[r] for r in rungs], fontsize=8.5)
axes[-1].set_xlim(-0.5, x[-1] + 0.5)
axes[-1].set_xlabel('linear-bias prior of the fit', fontsize=9)
fig.supylabel(r'$(w_0, w_a)$ shift [$\sigma$]', x=-0.01, fontsize=9.5)

# Legend columns fill top to bottom, so the handles are ordered for the rows
# to read: (62 Mpc, 0.02, model A baseline) and (0.05, 0.2, model A
# pessimistic).
lay_handles = [Line2D([], [], ls='none', marker='o', ms=MARKER_SIZE + 0.6,
                      mfc=lay['color'], mec=lay['color'], label=lay['label'])
               for lay in data['layouts']]
var_handles = [Line2D([], [], ls='none', marker='o', ms=MARKER_SIZE + 0.6, mfc='0.3',
                      mec='0.3', label='model A baseline'),
               Line2D([], [], ls='none', marker='o', ms=MARKER_SIZE + 0.6, mfc='white',
                      mec='0.3', label='model A pessimistic')]
handles = [lay_handles[0], lay_handles[2], lay_handles[1], lay_handles[3]] + var_handles
legend = axes[0].legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                        ncol=3, frameon=False, fontsize=7.5, handlelength=1.0,
                        columnspacing=0.8, handletextpad=0.3, labelspacing=0.35)
labels.append(('legend', legend))
check_clear(fig, labels)
fig.savefig(os.path.join(HERE, 'ladder_bias_scatter.pdf'))
fig.savefig(os.path.join(HERE, 'ladder_bias_scatter.png'), dpi=300)
