"""Redraw summary_prior_draws.pdf from data.json: per DESI group, the known
bias drifts against random histories of the summary model's linear-bias
prior (the pessimistic prior of notes/correlated_bias_prior_v2), each
history sharing the drift's straight line. The figure names no prior: the
paper has one, and the caption gives its L.

One column of a two-column paper: one panel per group, two rows of three.
Each panel is divided by one constant delta_g, a round number just above
the group's largest |delta| (written in the panel), so every panel shares
the same y axis. Thick black lines: the known drifts (dashed, a conserved
population; dash-dot, the BGS flux limit). Thin lines: four draws, each
with its own color and marker, so a draw can be followed in black and
white. data.json is the pessimistic row of
notes/correlated_bias_prior_v2/plots/prior_draws/data.json, copied by
EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb.
Run from any folder: python plot_summary_prior_draws.py
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
# colorblind-safe colors (Okabe-Ito, without its green; no red or green),
# all distinct from the black drift lines
DRAW_COLORS = ['#0072B2', '#E69F00', '#56B4E9', '#CC79A7']
# each draw also has its own marker, so a draw can be followed through the
# crossings in black and white
DRAW_MARKERS = ['o', 's', '^', 'D']
# the marker size of the draws [pt]
DRAW_MARKER_SIZE = 2.3
LAW_STYLE = {'conserved': {'ls': '--', 'label': 'conserved population'},
             'flux_limit': {'ls': '-.', 'label': r'BGS flux limit $r < 19.5$'}}
# the round values a panel's scale delta_g can take [%]: the smallest one at
# or above the group's largest |delta| is used
ROUND_SCALES = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]
# the grid of panels: rows and columns
GRID = (2, 3)

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
groups = data['groups']


def panel_scale(group):
    '''delta_g [%] of one group: the smallest ROUND_SCALES value at or above
    the largest |delta| of its drifts and draws.'''
    values = [np.abs(np.array(v)).max() for v in group['laws'].values()]
    values += [np.abs(np.array(draw)).max() for draw in group['draws']]
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


def check_labels_off_ticks(fig, axes, labels, name):
    '''Raise when an axis label (a panel's own or a shared one) overlaps a
    visible tick label of any panel.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # the drawn tick labels: visible, with text, and inside the panel's span
    # (a locator also returns ticks beyond the view, which are not drawn)
    ticks = []
    for ax in axes:
        span = ax.get_window_extent(renderer)
        for tick in ax.get_xticklabels():
            box = tick.get_window_extent(renderer)
            center = 0.5*(box.x0 + box.x1)
            if tick.get_text() and tick.get_visible() and span.x0 - 1 <= center <= span.x1 + 1:
                ticks.append(tick)
        for tick in ax.get_yticklabels():
            box = tick.get_window_extent(renderer)
            center = 0.5*(box.y0 + box.y1)
            if tick.get_text() and tick.get_visible() and span.y0 - 1 <= center <= span.y1 + 1:
                ticks.append(tick)
    for label in labels:
        box = label.get_window_extent(renderer)
        for tick in ticks:
            if box.overlaps(tick.get_window_extent(renderer)):
                raise RuntimeError(f'{name}: the label "{label.get_text()}" overlaps the '
                                   f'tick label "{tick.get_text()}"')


def touches(ax, artist, renderer):
    '''Whether the artist's box (with a small margin) covers any line point.'''
    box = artist.get_window_extent(renderer).expanded(1.1, 1.3)
    pts = line_points(ax)
    return bool(np.any((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1)
                       & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1)))


# one column of a two-column article: two rows of three narrow panels on
# one shared y axis; each panel has its own redshift range
fig, axes = plt.subplots(*GRID, figsize=(3.5, 2.6), sharey=True,
                         gridspec_kw={'hspace': 0.32, 'wspace': 0.06})
flat = list(axes.ravel())
scales = [panel_scale(group) for group in groups]
for ax, group, scale in zip(flat, groups, scales):
    z = np.array(group['z'])
    for draw, color, marker in zip(group['draws'], DRAW_COLORS, DRAW_MARKERS):
        ax.plot(z, 100.0*np.array(draw)/scale, color=color, lw=0.7, marker=marker,
                ms=DRAW_MARKER_SIZE, alpha=0.85)
    for law, values in group['laws'].items():
        ax.plot(z, 100.0*np.array(values)/scale, color='black', lw=1.5,
                ls=LAW_STYLE[law]['ls'])
    ax.axhline(0.0, color='0.3', lw=0.5)
    ax.tick_params(labelsize=6.5, length=2.5, pad=1.5)
    ax.grid(False, axis='x')
    # the outermost two round redshifts inside the group's range, so
    # neighboring panels never collide
    ticks = MaxNLocator(nbins=3, steps=[1, 2, 5, 10]).tick_values(z[0], z[-1])
    ticks = ticks[(ticks >= z[0]) & (ticks <= z[-1])]
    ax.set_xticks(ticks[[0, -1]] if len(ticks) > 2 else ticks)
flat[0].set_yticks([-1, 0, 1])
supylabel = fig.supylabel(r'$\delta(z)/\delta_g$', fontsize=9, x=0.02)
# the shared x label just under the lowest tick label, so they never touch
fig.canvas.draw()
tick_bottom = min(label.get_window_extent(fig.canvas.get_renderer()).y0
                  for ax in axes[-1] for label in ax.get_xticklabels() if label.get_text())
tick_bottom = fig.transFigure.inverted().transform((0.0, tick_bottom))[1]
supxlabel = fig.supxlabel(r'redshift $z$', fontsize=9, y=tick_bottom - 0.01, va='top')

# Each group's name and scale inside its panel, top left. The shared y axis
# gets the least headroom at which every label clears the data.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
labels = [ax.text(0.05, 0.95, f'{group["name"]}\n' + rf'$\delta_g = {scale}$%',
                  transform=ax.transAxes, ha='left', va='top', fontsize=6.5,
                  linespacing=1.1)
          for ax, group, scale in zip(flat, groups, scales)]
for headroom in np.arange(1.05, 2.51, 0.05):
    flat[0].set_ylim(-1.05, headroom)
    fig.canvas.draw()
    if not any(touches(ax, label, renderer) for ax, label in zip(flat, labels)):
        break
else:
    raise RuntimeError('a panel label touches the data at every headroom')

handles = [Line2D([], [], color='black', lw=1.5, ls=LAW_STYLE[k]['ls'],
                  label=LAW_STYLE[k]['label'])
           for k in ('conserved', 'flux_limit')]
handles.append(Line2D([], [], color='0.45', lw=0.7, marker='o', ms=DRAW_MARKER_SIZE,
                      label='prior draws'))
fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5, 0.9), handlelength=2.4, columnspacing=1.0,
           labelspacing=0.3)
# the legend sits above the panels: it must not overlap any of them
fig.canvas.draw()
legend_box = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
if any(legend_box.overlaps(ax.get_window_extent()) for ax in flat):
    raise RuntimeError('summary_prior_draws: the legend overlaps a panel')
check_labels_off_ticks(fig, flat, [supylabel, supxlabel], 'summary_prior_draws')
fig.savefig(os.path.join(HERE, 'summary_prior_draws.pdf'))
fig.savefig(os.path.join(HERE, 'summary_prior_draws.png'), dpi=300)
plt.close(fig)
