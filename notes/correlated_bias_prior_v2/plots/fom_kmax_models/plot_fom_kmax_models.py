"""Redraw fom_kmax_models.pdf from data.json: two scenarios of the DESI
C_ell^gg forecast with the one-loop bias models A and B of
notes/higher_order_bias, against the scale cut k_max.

One full-width figure, two rows by three columns:
  rows    - the two scenarios: the baseline linear-bias prior with model A
            and model B, and the pessimistic prior with model A + tilts and
            model B + tilts (the free prior is not drawn; data.json keeps
            it); each row is named on the right of column (c);
  columns - (a) FoM(w0, wa), (b) FoM(Omega_m, 10^9 A_s), (c) the largest
            shift of a fit over its truths and over (w0, wa) (in its
            ellipse), Omega_m and A_s (in their own sigma), with sigma/3.
Each bin layout has a color, a line width and a marker (at every cut), so
the figure reads in black and white; model A is solid with filled markers,
model B dashed with open markers. In
(a) and (b) model A is drawn up to its break point in that row (the largest
cut below sigma/3 in (c)) and model B at every cut; in (c) both are drawn at
every cut, so each crossing is visible. Each column has one logarithmic y
axis for both rows. Cuts run to k_max = 0.2 h/Mpc (KMAX_MAX). The LSST Y1
3x2pt reference, named in the caption, is a gray long-dashed line in a panel
of (a) or (b) whose highest point it exceeds by at most a factor 1.5, and a
text inside the first such panel of the column otherwise. k_max is the z = 0
value of a per-bin cut at k_max/D(zbar_i).

data.json, written by EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb, also holds the
shift of every truth and score, the truth that sets each break point, and
both models' FoMs on every prior.
Run from any folder: python plot_fom_kmax_models.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
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
# the figure's width [in]: the full text width of a two-column paper
FIG_WIDTH = 6.8
# the height of one row of panels [in]
ROW_HEIGHT = 1.55
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
# The two models, labelled "model A" and "model B"; the row names say which
# variant each row draws (no tilts, or + tilts): solid with filled markers, dashed with open
# markers. The dash and gap lengths are in points: matplotlib scales a dash
# pattern by the line width, so dividing by it keeps the dashes visible on
# the thinnest line.
DASH = (4.5, 2.5)
MODEL_STYLE = {'A': {'dashed': False, 'open': False, 'label': 'model A'},
               'B': {'dashed': True, 'open': True, 'label': 'model B'}}
# the LSST Y1 reference: gray long dashes, apart from model B's short dashes
LSST_STYLE = {'color': '0.5', 'ls': (0, (7, 3)), 'lw': 1.2}
# the sigma/3 threshold: black and dotted, apart from every model line
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
# the gap between the two legends above the figure, in figure units: the
# legends stack with almost no space between them
LEGEND_GAP = 0.002
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
# the columns: data.json field (None: the largest shift), column title
COLUMNS = [('fom', r'(a) FoM$(w_0, w_a)$'),
           ('fom_om_as', r'(b) FoM$(\Omega_\mathrm{m}, 10^9 A_\mathrm{s})$'),
           (None, r'(c) largest shift [$\sigma$]')]
# the row names, rotated on the right of column (c): the linear-bias prior,
# and "+ tilts" on the row that draws model A + tilts and model B + tilts
ROW_LABELS = {'baseline': 'baseline prior', 'pessimistic': 'pessimistic prior\n+ tilts'}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax_all = np.array(data['kmax_h_per_Mpc'], dtype=float)
# the cuts drawn: KMAX_MAX and below
DRAWN_CUTS = kmax_all <= KMAX_MAX + KMAX_TOLERANCE
kmax = kmax_all[DRAWN_CUTS]
layouts = data['layouts']
scenarios = data['scenarios']
threshold = data['shift_threshold']
LSST = {'fom': data['lsst_y1_3x2pt_fom'], 'fom_om_as': data['lsst_y1_3x2pt_fom_om_as']}


def as_array(values, floor=None):
    '''A list of numbers and None, one per cut of data.json, as a float array
    of the drawn cuts, NaN for None (a gap); values below floor are raised
    to it.'''
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if value is not None:
            out[i] = float(value) if floor is None else max(float(value), floor)
    return out[DRAWN_CUTS]


def model_a_fom(lay, field, spec):
    '''Model A's FoM in one scenario up to its break point (NaN beyond).'''
    y = as_array(lay[field][spec['A']][spec['prior']])
    cut = lay['break'][spec['A']][spec['prior']]['kmax']
    if cut is None:
        # the smallest fitted cut already fails: no model A line
        y[:] = np.nan
    else:
        y[kmax > cut + KMAX_TOLERANCE] = np.nan
    return y


def line_style(lw, marker, model, color):
    '''The plot keywords of a line of width lw and marker for one model.'''
    st = MODEL_STYLE[model]
    ls = (0, (DASH[0]/lw, DASH[1]/lw)) if st['dashed'] else '-'
    return {'color': color, 'ls': ls, 'lw': lw, 'marker': marker, 'ms': MARKER_SIZE,
            'mfc': 'white' if st['open'] else color, 'mec': color}


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


def check_clear(fig, axes, artists):
    '''Raise when a legend or label touches a drawn line or another one.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    for box, artist in zip(boxes, artists):
        if touches_lines(box, axes):
            raise RuntimeError(f'fom_kmax_models: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'fom_kmax_models: {type(artists[i]).__name__} {i} '
                                   f'overlaps {type(artists[j]).__name__} {j}')


rows = list(scenarios)
fig, axes = plt.subplots(len(rows), len(COLUMNS),
                         figsize=(FIG_WIDTH, ROW_HEIGHT*len(rows)), sharex=True,
                         sharey='col', gridspec_kw={'hspace': 0.08, 'wspace': 0.28})
flat = list(axes.ravel())
# the y values drawn in each column, and in each FoM panel (for the reference)
column_values = {c: [] for c in range(len(COLUMNS))}
panel_high = {}
for r, scenario in enumerate(rows):
    spec = scenarios[scenario]
    for c, (field, title) in enumerate(COLUMNS):
        ax = axes[r, c]
        values = []
        for lay in layouts:
            if field is None:
                y_a = as_array(lay['worst'][spec['A']][spec['prior']], floor=SHIFT_FLOOR)
                y_b = as_array(lay['worst'][spec['B']][spec['prior']], floor=SHIFT_FLOOR)
            else:
                y_a = model_a_fom(lay, field, spec)
                y_b = as_array(lay[field][spec['B']][spec['prior']])
            st = LAYOUT_STYLE[lay['name']]
            ax.plot(kmax, y_a, **line_style(st['lw'], st['marker'], 'A', lay['color']))
            ax.plot(kmax, y_b, **line_style(st['lw'], st['marker'], 'B', lay['color']))
            for y in (y_a, y_b):
                values += [float(v) for v in y[np.isfinite(y)]]
        if field is None:
            ax.axhline(threshold, **THRESHOLD_STYLE)
            values.append(threshold)
        column_values[c] += values
        panel_high[(r, c)] = max(values)
        ax.grid(False, axis='x')
        if r == 0:
            ax.set_title(title, fontsize=9.5)
    # the scenario's name on the right of column (c), rotated like a y label
    axes[r, -1].yaxis.set_label_position('right')
    axes[r, -1].set_ylabel(ROW_LABELS[scenario], rotation=90, fontsize=9, labelpad=6,
                           linespacing=1.0)

# the LSST Y1 reference in (a) and (b): a line in every panel it does not
# stretch; its value as text in the first panel of the column without it
texts_needed = []
for c, (field, title) in enumerate(COLUMNS):
    if field is None:
        continue
    lsst = LSST[field]
    without = []
    for r in range(len(rows)):
        if lsst <= REFERENCE_SPAN*panel_high[(r, c)]:
            axes[r, c].axhline(lsst, **LSST_STYLE)
            column_values[c].append(lsst)
        else:
            without.append(r)
    if without:
        texts_needed.append((c, axes[without[0], c], f'LSST Y1 3x2pt: {lsst:.0f}'))

for c in range(len(COLUMNS)):
    axes[0, c].set_yscale('log')
    # a label on every decade (the default locator skips decades on a short
    # panel), as plain numbers, no minor labels
    axes[0, c].yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    axes[0, c].yaxis.set_major_formatter(FuncFormatter(plain_number))
    axes[0, c].yaxis.set_minor_formatter(NullFormatter())
    axes[0, c].set_ylim(min(column_values[c])/1.6, max(column_values[c])*1.6)
    axes[-1, c].set_xlim(kmax[0] - 0.01, kmax[-1] + 0.01)
    axes[-1, c].set_xticks([0.05, 0.1, 0.15, 0.2])
    axes[-1, c].set_xticklabels(['0.05', '0.1', '0.15', '0.2'])
# the shared x label just under the lowest tick label of the bottom row
fig.canvas.draw()
tick_bottom = min(label.get_window_extent(fig.canvas.get_renderer()).y0
                  for ax in axes[-1] for label in ax.get_xticklabels())
tick_bottom = fig.transFigure.inverted().transform((0.0, tick_bottom))[1]
supxlabel = fig.supxlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$', fontsize=10,
                          y=tick_bottom - 0.01, va='top')

# the reference texts, in an upper corner, with more headroom in their
# column until they clear the lines
texts = []
for c, ax, text in texts_needed:
    placed = None
    for headroom in HEADROOM:
        axes[0, c].set_ylim(min(column_values[c])/1.6, max(column_values[c])*headroom)
        for corner in ('upper left', 'upper right'):
            placed = place_text(fig, flat, ax, text, corner, texts)
            if placed is not None:
                break
        if placed is not None:
            break
    if placed is None:
        raise RuntimeError('fom_kmax_models: the LSST text finds no corner free of lines')
    texts.append(placed)

# the two legends above the figure: the layouts (color, line width and
# marker), and above them the two models, each shown by its four drawn lines
# side by side (solid with filled markers, dashed with open ones). The
# reference lines (LSST Y1 3x2pt, sigma/3) carry no legend entry: the
# caption names them.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
title_top = max(ax.title.get_window_extent(renderer).y1 for ax in axes[0])
title_top = fig.transFigure.inverted().transform((0.0, title_top))[1]
layout_legend = fig.legend(
    handles=[Line2D([], [], label=lay['label'],
                    **line_style(LAYOUT_STYLE[lay['name']]['lw'],
                                 LAYOUT_STYLE[lay['name']]['marker'], 'A', lay['color']))
             for lay in layouts],
    loc='lower center', bbox_to_anchor=(0.5, title_top + LEGEND_GAP), ncol=len(layouts),
    frameon=False, fontsize=8.5, handlelength=2.4, columnspacing=1.6, handletextpad=0.5,
    borderaxespad=0.0, borderpad=0.1)
fig.canvas.draw()
legend_top = fig.transFigure.inverted().transform(
    (0.0, layout_legend.get_window_extent(renderer).y1))[1]
model_handles = [tuple(Line2D([], [], **line_style(LAYOUT_STYLE[lay['name']]['lw'],
                                                   LAYOUT_STYLE[lay['name']]['marker'], m,
                                                   lay['color']))
                       for lay in layouts) for m in ('A', 'B')]
style_legend = fig.legend(
    handles=model_handles, labels=[MODEL_STYLE[m]['label'] for m in ('A', 'B')],
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
    loc='lower center', bbox_to_anchor=(0.5, legend_top + LEGEND_GAP), ncol=2,
    frameon=False, fontsize=8.5, handlelength=8.0, columnspacing=2.0, handletextpad=0.5,
    borderaxespad=0.0, borderpad=0.1)
check_clear(fig, flat, [layout_legend, style_legend, supxlabel, *texts,
                        *[label for ax in axes[-1] for label in ax.get_xticklabels()],
                        *[ax.title for ax in axes[0]],
                        *[axes[r, -1].yaxis.label for r in range(len(rows))]])
fig.savefig(os.path.join(HERE, 'fom_kmax_models.pdf'))
fig.savefig(os.path.join(HERE, 'fom_kmax_models.png'), dpi=300)
plt.close(fig)
