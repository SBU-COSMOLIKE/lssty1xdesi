"""Redraw model_a_break.pdf from data.json: where the one-loop bias model A
(notes/higher_order_bias), with SN0 and SN2 per group, breaks when the truth
carries model B's radial counterterm alpha_2 or model A's own terms at the
edge of their one-sigma priors.

For each cut k_max, model A is fitted by Fisher linear response to six
fixed truths, each the same in every bin: alpha_2 = +-12.5 h^-2 Mpc^2,
and at model A's one-sigma widths SN2 = +-5 units and the b2 and bs2
offsets +-(1, 0.5). A fit is scored by its (w0, wa) shift in units of its
own ellipse and by its Omega_m and A_s shifts in their own sigma. The figure
draws the largest of these shifts over the six truths, against k_max: one
column of a two-column paper, three stacked panels (the baseline,
pessimistic and free linear-bias priors, named inside each panel) on one
shared logarithmic y axis, one color per bin layout, model A pessimistic
solid and model A baseline dashed. Model A breaks where its line crosses
sigma/3 (the red dotted line). k_max is the z = 0 value of a per-bin cut at
k_max/D(zbar_i).

data.json, written by EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb, also holds
every truth and score separately, the prior-mismatch diagnostic (the
offsets at model B's widths) and the random-truth statistics.
Run from any folder: python plot_model_a_break.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

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
# the line style of each model A variant
STYLES = {
    'A pessimistic': {'ls': '-', 'lw': 1.4, 'marker': 'o', 'ms': 3.0},
    'A baseline': {'ls': '--', 'lw': 1.3},
}
# the sigma/3 threshold, dotted so it never reads as a dashed model line
THRESHOLD_STYLE = {'color': 'firebrick', 'ls': ':', 'lw': 1.3}
PANEL_LABELS = {'baseline': 'baseline prior', 'pessimistic': 'pessimistic prior',
                'independent': 'free prior'}
# a shift that is exactly zero is drawn on this floor of the log axis
SHIFT_FLOOR = 1.0e-3
# the gap between the two legends above the top panel, in units of its height
LEGEND_GAP = 0.04
# The anchors a text may take inside a panel, in axes units (x, y,
# horizontal and vertical alignment), and the headroom factors of the y axis
# tried in turn until every panel name finds a corner free of lines.
CORNERS = {
    'upper left': (0.03, 0.94, 'left', 'top'),
    'upper right': (0.97, 0.94, 'right', 'top'),
    'lower right': (0.97, 0.06, 'right', 'bottom'),
    'lower left': (0.03, 0.06, 'left', 'bottom'),
}
# the panel names share one corner, the first free in every panel
NAME_CORNERS = ['lower right', 'upper left', 'upper right', 'lower left']
HEADROOM = [1.6, 2.2, 3.0, 4.5, 7.0, 10.0]

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax = np.array(data['kmax_h_per_Mpc'], dtype=float)
priors = data['priors']
layouts = data['layouts']
threshold = data['shift_threshold']


def as_array(values):
    '''A list of numbers and None as a float array, NaN for None (a gap).'''
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if value is not None:
            out[i] = max(float(value), SHIFT_FLOOR)
    return out


def plain_number(value, position):
    '''A tick label as a plain number: 0.01, 0.1, 1, 10 (position is unused).'''
    return f'{value:g}'


def line_points(ax):
    '''Display coordinates of points densely sampled along every drawn line of
    ax (the threshold included), inside the axes only.

    Each line's own transform maps its data to the display (an axhline keeps
    x in axes units), and the segments are sampled in display space, which is
    where a straight segment of a log axis is straight.'''
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


def touches_lines(box, axes):
    '''True when a display box, with a 3% / 8% margin, holds a point of a line.'''
    padded = box.expanded(1.03, 1.08)
    for ax in axes:
        pts = line_points(ax)
        hit = ((pts[:, 0] > padded.x0) & (pts[:, 0] < padded.x1)
               & (pts[:, 1] > padded.y0) & (pts[:, 1] < padded.y1))
        if np.any(hit):
            return True
    return False


def place_text(fig, axes, ax, text, corner, keep_clear):
    '''Put a text in one corner of a panel; None (text removed) when it
    touches a line or overlaps one of the artists keep_clear.'''
    renderer = fig.canvas.get_renderer()
    x, y, ha, va = CORNERS[corner]
    candidate = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5)
    fig.canvas.draw()
    box = candidate.get_window_extent(renderer)
    clear = not touches_lines(box, axes)
    for other in keep_clear:
        if box.overlaps(other.get_window_extent(renderer)):
            clear = False
    if not clear:
        candidate.remove()
        return None
    return candidate


def place_panel_names(fig, axes, names, keep_clear):
    '''Put every panel's name in one corner shared by all panels, the first
    of NAME_CORNERS free in every panel; None when no corner is.'''
    for corner in NAME_CORNERS:
        placed = []
        for ax, name in zip(axes, names):
            artist = place_text(fig, axes, ax, name, corner, keep_clear)
            if artist is None:
                break
            placed.append(artist)
        if len(placed) == len(names):
            return placed
        for artist in placed:
            artist.remove()
    return None


def check_clear(fig, axes, artists, name):
    '''Raise when a legend or label touches a drawn line or another one.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    for box, artist in zip(boxes, artists):
        if touches_lines(box, axes):
            raise RuntimeError(f'{name}: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'{name}: {type(artists[i]).__name__} {i} overlaps '
                                   f'{type(artists[j]).__name__} {j}')


fig, axes = plt.subplots(len(priors), 1, figsize=(3.5, 1.1*len(priors)), sharex=True,
                         sharey=True, gridspec_kw={'hspace': 0.06})
drawn = [threshold]
for ax, prior in zip(axes, priors):
    for lay in layouts:
        for variant, style in STYLES.items():
            y = as_array(lay['worst'][variant][prior])
            ax.plot(kmax, y, color=lay['color'], **style)
            drawn += [float(v) for v in y[np.isfinite(y)]]
    ax.axhline(threshold, **THRESHOLD_STYLE)
    ax.grid(False, axis='x')
axes[0].set_yscale('log')
# a label on every decade (the default locator skips decades on a short
# panel), as plain numbers, no minor labels
axes[0].yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
axes[0].yaxis.set_major_formatter(FuncFormatter(plain_number))
axes[0].yaxis.set_minor_formatter(NullFormatter())
axes[-1].set_xlim(kmax[0] - 0.01, kmax[-1] + 0.01)
axes[-1].set_xlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$')
supylabel = fig.supylabel(r'largest shift of $(w_0, w_a)$, $\Omega_\mathrm{m}$, '
                          r'$A_\mathrm{s}$ [$\sigma$]', x=-0.03, fontsize=9.5)

# the two legends above the top panel: the layouts by color, and above them
# the line styles (markers as drawn)
layout_legend = axes[0].legend(
    handles=[Line2D([], [], color=lay['color'], lw=2.0, label=lay['label'])
             for lay in layouts],
    loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8,
    handlelength=1.4, columnspacing=1.2, handletextpad=0.5, labelspacing=0.25,
    borderaxespad=0.2)
axes[0].add_artist(layout_legend)
fig.canvas.draw()
top = axes[0].transAxes.inverted().transform(
    layout_legend.get_window_extent(fig.canvas.get_renderer()).p1)[1]
style_handles = [Line2D([], [], color='0.25', label='model ' + variant, **style)
                 for variant, style in STYLES.items()]
style_handles.append(Line2D([], [], label=r'$\sigma/3$', **THRESHOLD_STYLE))
style_legend = axes[0].legend(handles=style_handles, loc='lower center',
                              bbox_to_anchor=(0.5, top + LEGEND_GAP), ncol=3,
                              frameon=False, fontsize=8, handlelength=2.4,
                              columnspacing=1.0, handletextpad=0.5, borderaxespad=0.0)

# each prior's name inside its panel, in one corner shared by the panels;
# the y axis gains headroom until the names clear the lines
texts = None
for headroom in HEADROOM:
    axes[0].set_ylim(min(drawn)/1.6, max(drawn)*headroom)
    texts = place_panel_names(fig, axes, [PANEL_LABELS[prior] for prior in priors],
                              [layout_legend, style_legend])
    if texts is not None:
        break
if texts is None:
    raise RuntimeError('model_a_break: the panel names find no corner free of lines')
check_clear(fig, axes, [layout_legend, style_legend, supylabel, *texts], 'model_a_break')
fig.savefig(os.path.join(HERE, 'model_a_break.pdf'))
fig.savefig(os.path.join(HERE, 'model_a_break.png'), dpi=300)
plt.close(fig)
