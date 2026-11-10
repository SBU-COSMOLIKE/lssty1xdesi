"""Redraw summary_prior_draws.pdf from data.json: per DESI group, the known
bias drifts against random histories of the summary model's linear-bias
prior (the pessimistic prior of notes/correlated_bias_prior_v2), each
history sharing the drift's straight line. The figure names no prior: the
paper has one, and the caption gives its L.

One column of a two-column paper: one panel per group, two rows of three.
Each panel is divided by one scale, a round percentage just above the
group's largest |delta| (written in the panel as "scale 20%"), so every
panel shares the same y axis. Thick black lines: the known drifts (dashed,
a conserved population; dash-dot, the BGS flux limit). Thin lines: four
draws in neutral grays, each with its own shade, line width and marker, so
a draw can be followed in black and white and never reads as a bin layout
(the layouts own the Okabe-Ito colors and the markers o, s, D, ^ in the
other figures of the note). data.json is the pessimistic row of
notes/correlated_bias_prior_v2/plots/prior_draws/data.json, copied by
EXAMPLE_EVALUATE_FISHER_MODEL.ipynb.
Run from any folder: python plot_summary_prior_draws.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
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
# The four draws: a gray shade, a line width [pt] and a marker each, darkest
# and widest first, so a draw can be followed through the crossings in
# black and white. The grays keep the draws apart from the bin layouts,
# which own the Okabe-Ito colors in the other figures, and the markers are
# none of the layout markers (o, s, D, ^) or the markers of the untied
# versions (<, >, p, h, x) of this note. A star looks smaller than the other
# markers at the same size, so it gets a larger one [pt]. Every draw is thin
# and solid; the drifts are thick, black and dashed.
DRAW_STYLES = [
    {'color': '0.10', 'lw': 1.05, 'marker': 'v', 'ms': 2.8},
    {'color': '0.36', 'lw': 0.85, 'marker': 'X', 'ms': 2.8},
    {'color': '0.55', 'lw': 0.65, 'marker': 'P', 'ms': 2.8},
    {'color': '0.70', 'lw': 0.50, 'marker': '*', 'ms': 3.6},
]
LAW_STYLE = {'conserved': {'ls': '--', 'label': 'conserved population'},
             'flux_limit': {'ls': '-.', 'label': r'BGS flux limit $r < 19.5$'}}
# the round values a panel's scale can take [%]: the smallest one at or
# above the group's largest |delta| is used
ROUND_SCALES = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]
# the grid of panels: rows and columns
GRID = (2, 3)

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
groups = data['groups']


def panel_scale(group):
    '''The scale [%] of one group: the smallest ROUND_SCALES value at or
    above the largest |delta| of its drifts and draws.'''
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
    if len(group['draws']) > len(DRAW_STYLES):
        raise RuntimeError(f'summary_prior_draws: {group["name"]} has '
                           f'{len(group["draws"])} draws, DRAW_STYLES only {len(DRAW_STYLES)}')
    for draw, style in zip(group['draws'], DRAW_STYLES):
        ax.plot(z, 100.0*np.array(draw)/scale, mew=0.0, **style)
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
supylabel = fig.supylabel(r'$\delta(z)$ / scale', fontsize=9, x=0.02)
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
labels = [ax.text(0.05, 0.95, f'{group["name"]}, scale {scale}%',
                  transform=ax.transAxes, ha='left', va='top', fontsize=6.5)
          for ax, group, scale in zip(flat, groups, scales)]
for headroom in np.arange(1.05, 2.51, 0.05):
    flat[0].set_ylim(-1.05, headroom)
    fig.canvas.draw()
    if not any(touches(ax, label, renderer) for ax, label in zip(flat, labels)):
        break
else:
    raise RuntimeError('a panel label touches the data at every headroom')
# the one-line label must also fit inside its narrow panel
for ax, label in zip(flat, labels):
    box = label.get_window_extent(renderer)
    span = ax.get_window_extent(renderer)
    if box.x0 < span.x0 or box.x1 > span.x1 or box.y0 < span.y0 or box.y1 > span.y1:
        raise RuntimeError(f'summary_prior_draws: the label "{label.get_text()}" '
                           f'leaves its panel')

handles = [Line2D([], [], color='black', lw=1.5, ls=LAW_STYLE[k]['ls'],
                  label=LAW_STYLE[k]['label'])
           for k in ('conserved', 'flux_limit')]
# the draws' entry shows the four drawn styles side by side (one marker each)
handles.append(tuple(Line2D([], [], mew=0.0, **style)
                     for style in DRAW_STYLES))
legend_labels = [h.get_label() for h in handles[:-1]] + ['prior draws']
fig.legend(handles=handles, labels=legend_labels, loc='lower center', ncol=2,
           frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.9), handlelength=3.6,
           columnspacing=1.0, labelspacing=0.3,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
# the legend sits above the panels: it must not overlap any of them
fig.canvas.draw()
legend_box = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
if any(legend_box.overlaps(ax.get_window_extent()) for ax in flat):
    raise RuntimeError('summary_prior_draws: the legend overlaps a panel')
check_labels_off_ticks(fig, flat, [supylabel, supxlabel], 'summary_prior_draws')
fig.savefig(os.path.join(HERE, 'summary_prior_draws.pdf'))
fig.savefig(os.path.join(HERE, 'summary_prior_draws.png'), dpi=300)
plt.close(fig)
