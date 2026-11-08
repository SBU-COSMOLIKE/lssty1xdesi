"""Redraw summary_fom_kmax.pdf from data.json: the figures of merit of the
summary model, model B + tilts on the pessimistic linear-bias prior
(notes/higher_order_bias, notes/correlated_bias_prior_v2), DESI C_ell^gg
alone, against the scale cut k_max. Built from a copy of
notes/correlated_bias_prior_v2/plots/fom_kmax_models/plot_fom_kmax_models.py.

One column of a two-column paper, two stacked panels: FoM(w0, wa) on top,
FoM(Omega_m, 10^9 A_s) below, each on its own logarithmic y axis (two
quantities). Each bin layout has a color, a line width and a marker at
every cut, so the figure reads in black and white. The LSST Y1 3x2pt
reference is a gray long-dashed line in a panel whose highest point it
exceeds by at most a factor 1.5, and a text inside the panel otherwise; it
has no legend entry (the caption names it). k_max is the z = 0 value of a
per-bin cut at k_max/D(zbar_i); cuts up to KMAX_MAX are drawn. A cut that
leaves no data (Delta z = 0.02 at 0.05) has no point.

data.json is written by EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb.
Run from any folder: python plot_summary_fom_kmax.py
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
# Each layout has a line width and a marker as well as its color, so the
# layouts stay apart in black and white; the marker sits at every cut.
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
    'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
    'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
    'Delta z = 0.2': {'lw': 0.9, 'marker': '^'},
}
# the marker size [pt]
MARKER_SIZE = 3.5
# the LSST Y1 reference: gray long dashes
LSST_STYLE = {'color': '0.5', 'ls': (0, (7, 3)), 'lw': 1.2}
# the sigma/3 threshold: black and dotted, apart from every layout line
THRESHOLD_STYLE = {'color': '0.1', 'ls': ':', 'lw': 1.3}
# the reference is drawn in a panel only when it sits at most this factor
# above the panel's highest point; farther up it would stretch the axis
REFERENCE_SPAN = 1.5
# the largest cut drawn [h/Mpc at z = 0]; data.json may hold larger ones
KMAX_MAX = 0.2
# the k_max labels compare floats from data.json
KMAX_TOLERANCE = 1.0e-9
# a shift that is exactly zero is drawn on this floor of the log axis
SHIFT_FLOOR = 1.0e-3
# The anchors a text may take inside a panel, in axes units (x, y,
# horizontal and vertical alignment), and the headroom factors of a log
# axis tried in turn until every text finds a corner free of lines.
CORNERS = {
    'upper left': (0.04, 0.93, 'left', 'top'),
    'upper right': (0.96, 0.93, 'right', 'top'),
    'lower right': (0.96, 0.07, 'right', 'bottom'),
    'lower left': (0.04, 0.07, 'left', 'bottom'),
}
HEADROOM = [1.6, 2.2, 3.0, 4.5, 7.0, 10.0]
# the height of one panel [in]
PANEL_HEIGHT = 1.3
# the panels: data.json field, LSST Y1 field, y label
PANELS = [('fom', 'lsst_y1_3x2pt_fom', r'FoM$(w_0, w_a)$'),
          ('fom_om_as', 'lsst_y1_3x2pt_fom_om_as', r'FoM$(\Omega_\mathrm{m}, 10^9 A_\mathrm{s})$')]

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax_all = np.array(data['kmax_h_per_Mpc'], dtype=float)
# the cuts drawn: KMAX_MAX and below
DRAWN_CUTS = kmax_all <= KMAX_MAX + KMAX_TOLERANCE
kmax = kmax_all[DRAWN_CUTS]
layouts = data['layouts']


def as_array(values, floor=None):
    '''A list of numbers and None, one per cut of data.json, as a float array
    of the drawn cuts, NaN for None (a gap); values below floor are raised
    to it.'''
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if value is not None:
            out[i] = float(value) if floor is None else max(float(value), floor)
    return out[DRAWN_CUTS]


def plain_number(value, position):
    '''A tick label as a plain number: 0.01, 0.1, 1, 10 (position is unused).'''
    return f'{value:g}'


def line_points(ax):
    '''Display coordinates of points densely sampled along every drawn line of
    ax (the reference lines included), inside the axes only.

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
    candidate = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8)
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


def check_clear(fig, axes, artists):
    '''Raise when a legend or label touches a drawn line or another one.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    for box, artist in zip(boxes, artists):
        if touches_lines(box, axes):
            raise RuntimeError(f'summary_fom_kmax: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'summary_fom_kmax: {type(artists[i]).__name__} {i} '
                                   f'overlaps {type(artists[j]).__name__} {j}')


fig, axes = plt.subplots(len(PANELS), 1, figsize=(3.5, PANEL_HEIGHT*len(PANELS)),
                         sharex=True, gridspec_kw={'hspace': 0.06})
flat = list(axes)
texts = []
for ax, (field, lsst_field, ylabel) in zip(axes, PANELS):
    values = []
    for lay in layouts:
        st = LAYOUT_STYLE[lay['name']]
        y = as_array(lay[field])
        ax.plot(kmax, y, color=lay['color'], lw=st['lw'], marker=st['marker'],
                ms=MARKER_SIZE)
        values += [float(v) for v in y[np.isfinite(y)]]
    # the reference: a line when it does not stretch the axis, a text
    # inside the panel otherwise
    lsst = data[lsst_field]
    high = max(values)
    text = None
    if lsst <= REFERENCE_SPAN*high:
        ax.axhline(lsst, **LSST_STYLE)
        high = max(high, lsst)
    else:
        text = f'LSST Y1 3x2pt: {lsst:.0f}'
    ax.set_yscale('log')
    # a label on every decade, as plain numbers, no minor labels
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(False, axis='x')
    placed = None
    for headroom in HEADROOM:
        ax.set_ylim(min(values)/1.6, high*headroom)
        if text is None:
            break
        for corner in ('upper left', 'lower right'):
            placed = place_text(fig, flat, ax, text, corner, texts)
            if placed is not None:
                break
        if placed is not None:
            break
    if text is not None:
        if placed is None:
            raise RuntimeError('summary_fom_kmax: the LSST text finds no corner free of lines')
        texts.append(placed)
axes[-1].set_xlim(kmax[0] - 0.01, kmax[-1] + 0.01)
axes[-1].set_xticks([0.05, 0.1, 0.15, 0.2])
axes[-1].set_xticklabels(['0.05', '0.1', '0.15', '0.2'])
axes[-1].set_xlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$', fontsize=10)
fig.align_ylabels(axes)

# the layouts above the top panel (color, line width and marker)
layout_legend = axes[0].legend(
    handles=[Line2D([], [], color=lay['color'], lw=LAYOUT_STYLE[lay['name']]['lw'],
                    marker=LAYOUT_STYLE[lay['name']]['marker'], ms=MARKER_SIZE,
                    label=lay['label']) for lay in layouts],
    loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.2, columnspacing=1.2, handletextpad=0.5, labelspacing=0.25,
    borderaxespad=0.2, borderpad=0.1)
check_clear(fig, flat, [layout_legend, *texts, axes[-1].xaxis.label,
                        *[ax.yaxis.label for ax in axes]])
check_labels_off_ticks(fig, flat, [axes[-1].xaxis.label, *[ax.yaxis.label for ax in axes]],
                       'summary_fom_kmax')
fig.savefig(os.path.join(HERE, 'summary_fom_kmax.pdf'))
fig.savefig(os.path.join(HERE, 'summary_fom_kmax.png'), dpi=300)
plt.close(fig)
