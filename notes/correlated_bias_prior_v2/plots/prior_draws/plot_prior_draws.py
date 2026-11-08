"""Redraw prior_draws.pdf from data.json: per DESI group (columns), the known
bias drifts against random histories of the baseline prior (top row) and the
pessimistic prior (bottom row), each history sharing the drift's straight
line. Each column is divided by one constant delta_g, a round number just
above the column's largest |delta| (written in the column's top panel), so
every panel shares the same y axis and the panels sit close together.
data.json comes from ../make_theory_data.py.
Run from any folder: python plot_prior_draws.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

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
DRAW_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
LAW_STYLE = {'conserved': {'ls': '--', 'label': 'conserved population'},
             'flux_limit': {'ls': '-.', 'label': r'BGS flux limit $r < 19.5$'}}
ROWS = ['baseline', 'pessimistic']
# the round values a column's scale delta_g can take [%]: the smallest one at
# or above the column's largest |delta| is used
ROUND_SCALES = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
groups = data['groups']
priors = data['priors']


def column_scale(group):
    '''delta_g [%] of one group: the smallest ROUND_SCALES value at or above
    the largest |delta| of its drifts and of both priors' draws.'''
    values = [np.abs(np.array(v)).max() for v in group['laws'].values()]
    for case in ROWS:
        values += [np.abs(np.array(draw)).max() for draw in group['draws'][case]]
    largest = 100.0*max(values)
    return next(s for s in ROUND_SCALES if s >= largest)


def line_points(ax):
    '''Display coordinates of points densely sampled along every line of ax.'''
    pts = []
    for line in ax.get_lines():
        xd, yd = (np.asarray(v, dtype=float) for v in line.get_data())
        if line.get_transform() != ax.transData:
            # axhline: x in axes units, y in data units
            xs = np.linspace(*ax.get_xlim(), 200)
            pts.append(ax.transData.transform(np.column_stack([xs, np.full(200, yd[0])])))
            continue
        for i in range(len(xd) - 1):
            t = np.linspace(0.0, 1.0, 30)
            pts.append(ax.transData.transform(np.column_stack(
                [xd[i] + t*(xd[i + 1] - xd[i]), yd[i] + t*(yd[i + 1] - yd[i])])))
    return np.vstack(pts)


def touches(ax, artist, renderer):
    '''Whether the artist's box (with a small margin) covers any line point.'''
    box = artist.get_window_extent(renderer).expanded(1.1, 1.3)
    pts = line_points(ax)
    return bool(np.any((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1)
                       & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1)))


# the full text width of the note: one narrow column per group, the two
# priors stacked; a shared y axis lets the columns nearly touch
fig, axes = plt.subplots(2, len(groups), figsize=(6.4, 2.3), sharex='col', sharey=True,
                         gridspec_kw={'hspace': 0.06, 'wspace': 0.06})
scales = [column_scale(group) for group in groups]
for col, (group, scale) in enumerate(zip(groups, scales)):
    z = np.array(group['z'])
    for row, case in enumerate(ROWS):
        ax = axes[row, col]
        for draw, color in zip(group['draws'][case], DRAW_COLORS):
            ax.plot(z, 100.0*np.array(draw)/scale, color=color, lw=0.7, marker='o', ms=1.4,
                    alpha=0.85)
        for law, values in group['laws'].items():
            ax.plot(z, 100.0*np.array(values)/scale, color='black', lw=1.5,
                    ls=LAW_STYLE[law]['ls'])
        ax.axhline(0.0, color='0.3', lw=0.5)
        ax.tick_params(labelsize=6.5, length=2.5, pad=1.5)
        ax.grid(False, axis='x')
    # the outermost two round redshifts inside the column's range, so
    # neighboring columns never collide
    ticks = MaxNLocator(nbins=3, steps=[1, 2, 5, 10]).tick_values(z[0], z[-1])
    ticks = ticks[(ticks >= z[0]) & (ticks <= z[-1])]
    axes[-1, col].set_xticks(ticks[[0, -1]] if len(ticks) > 2 else ticks)
axes[0, 0].set_yticks([-1, 0, 1])
# the prior of each row on the right side, rotated like a y label
for row, case in enumerate(ROWS):
    ax = axes[row, -1]
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(f'{case}\n$L = {priors[case]["L"]:g}$', rotation=90, fontsize=8, labelpad=4)
fig.supylabel(r'$\delta(z)/\delta_g$', fontsize=9, x=0.07)
fig.supxlabel(r'redshift $z$', fontsize=9, y=0.0)

# Each column's group and scale inside its top panel, top left. The shared
# y axis gets the least headroom at which every label clears the data.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
labels = [axes[0, col].text(0.05, 0.95, f'{group["name"]}\n' + rf'$\delta_g = {scale}$%',
                            transform=axes[0, col].transAxes, ha='left', va='top',
                            fontsize=6.5, linespacing=1.1)
          for col, (group, scale) in enumerate(zip(groups, scales))]
for headroom in np.arange(1.05, 2.01, 0.05):
    axes[0, 0].set_ylim(-1.05, headroom)
    fig.canvas.draw()
    if not any(touches(axes[0, col], label, renderer) for col, label in enumerate(labels)):
        break
else:
    raise RuntimeError('a column label touches the data at every headroom')

handles = [Line2D([], [], color='black', lw=1.5, ls=LAW_STYLE[k]['ls'], label=LAW_STYLE[k]['label'])
           for k in ('conserved', 'flux_limit')]
handles.append(Line2D([], [], color='0.45', lw=0.7, marker='o', ms=1.4, label='prior draws'))
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, 0.93))
fig.savefig(os.path.join(HERE, 'prior_draws.pdf'))
fig.savefig(os.path.join(HERE, 'prior_draws.png'), dpi=300)
