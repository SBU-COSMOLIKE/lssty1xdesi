"""Redraw summary_shifts_kmax.pdf from data.json: the largest shift of the
summary model, model B + tilts on the pessimistic linear-bias prior, over
fixed truths it models with a prior, each at one prior width (alpha_2 =
+-0.5/k_max^2 h^-2 Mpc^2, SN2 = +-5 units, the b2 and bs2 offsets at +-1
sigma of model B's widths), or absorbs (the calibration drift), against the
scale cut k_max. Built from a copy of
notes/correlated_bias_prior_v2/plots/fom_kmax_models/plot_fom_kmax_models.py
(its column (c)).

One column of a two-column paper, three stacked panels on one shared
linear y axis, from 0 to the round value just above the largest shift: the
(w0, wa) shift in units of the fit's ellipse, and the Omega_m and A_s shifts
in their own marginalized sigma, each the largest over the truths. Each bin
layout has a color, a line width and a marker at every cut. Every truth lies
inside the model's prior (the tilts absorb the drift), and a truth drawn from
the prior may move the fit by about 1 sigma without making its error bars
dishonest, so the shifts are reported and no threshold is drawn. k_max is
the z = 0 value of a per-bin cut at k_max/D(zbar_i); cuts up to KMAX_MAX are
drawn.

data.json, written by EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb, also holds every
truth's shift.
Run from any folder: python plot_summary_shifts_kmax.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

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
# the reference is drawn in a panel only when it sits at most this factor
# above the panel's highest point; farther up it would stretch the axis
REFERENCE_SPAN = 1.5
# the largest cut drawn [h/Mpc at z = 0]; data.json may hold larger ones
KMAX_MAX = 0.2
# the k_max labels compare floats from data.json
KMAX_TOLERANCE = 1.0e-9
# the round values the top of the shared y axis can take [sigma]: the
# smallest one at or above Y_MARGIN times the largest shift is used, so the
# highest marker stays inside the panel
ROUND_TOPS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
Y_MARGIN = 1.1
# the spacing of the labeled and of the unlabeled y ticks [sigma]; above
# Y_WIDE_TOP the labels step by Y_MAJOR_WIDE, so a short panel shows at
# most four of them
Y_MAJOR = 0.2
Y_MAJOR_WIDE = 0.5
Y_WIDE_TOP = 0.8
Y_MINOR = 0.1
# the height of one panel [in]
PANEL_HEIGHT = 1.15
# the panels, top to bottom: data.json score and its name, on the right side
PANELS = [('w0wa', r'$(w_0, w_a)$'), ('omegam', r'$\Omega_\mathrm{m}$'),
          ('As_1e9', r'$A_\mathrm{s}$')]

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
    '''A tick label as a plain number: 0, 0.2, 0.4 (position is unused).'''
    return f'{value:g}'


def line_points(ax, skip=()):
    '''Display coordinates of points densely sampled along every drawn line of
    ax (the reference lines included), inside the axes only; the lines in
    skip are left out.

    Each line's own transform maps its data to the display (an axhline keeps
    x in axes units), and the segments are sampled in display space, where a
    segment is straight on any axis scale.'''
    box = ax.get_window_extent()
    pts = [np.zeros((0, 2))]
    for line in ax.get_lines():
        if line in skip:
            continue
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


def touches_lines(box, axes, skip=()):
    '''True when a display box, with a 3% / 8% margin, holds a point of a line
    (the lines in skip left out).'''
    padded = box.expanded(1.03, 1.08)
    for ax in axes:
        pts = line_points(ax, skip)
        hit = ((pts[:, 0] > padded.x0) & (pts[:, 0] < padded.x1)
               & (pts[:, 1] > padded.y0) & (pts[:, 1] < padded.y1))
        if np.any(hit):
            return True
    return False


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
            raise RuntimeError(f'summary_shifts_kmax: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'summary_shifts_kmax: {type(artists[i]).__name__} {i} '
                                   f'overlaps {type(artists[j]).__name__} {j}')


# one column of a two-column paper: wide, short panels with a thin gap, one
# shared y range
fig, axes = plt.subplots(len(PANELS), 1, figsize=(3.5, PANEL_HEIGHT*len(PANELS)),
                         sharex=True, sharey=True, gridspec_kw={'hspace': 0.06})
flat = list(axes)
values = []
for ax, (score, name) in zip(axes, PANELS):
    for lay in layouts:
        st = LAYOUT_STYLE[lay['name']]
        y = as_array(lay['largest'][score])
        # unclipped, so a marker near a shift of 0 shows whole on the axis
        ax.plot(kmax, y, color=lay['color'], lw=st['lw'], marker=st['marker'],
                ms=MARKER_SIZE, clip_on=False)
        values += [float(v) for v in y[np.isfinite(y)]]
    ax.grid(False, axis='x')
    # the score's name on the right side, rotated like a y label, so it can
    # never sit on the data
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(name, rotation=90, fontsize=10, labelpad=5)
# a linear axis from 0, so the shifts read directly; the tick labels as
# plain numbers (0, 0.2, 0.4, ...)
y_top = next(top for top in ROUND_TOPS if top >= Y_MARGIN*max(values))
axes[0].set_ylim(0.0, y_top)
y_major = Y_MAJOR_WIDE if y_top > Y_WIDE_TOP else Y_MAJOR
axes[0].yaxis.set_major_locator(MultipleLocator(y_major))
axes[0].yaxis.set_minor_locator(MultipleLocator(Y_MINOR))
# the top of the axis stays unlabeled: the panels are stacked with a thin
# gap, and its label would touch the 0 of the panel above
axes[0].yaxis.set_major_formatter(FuncFormatter(
    lambda value, position: '' if value > y_top - KMAX_TOLERANCE
    else plain_number(value, position)))
axes[-1].set_xlim(kmax[0] - 0.01, kmax[-1] + 0.01)
axes[-1].set_xticks([0.05, 0.1, 0.15, 0.2])
axes[-1].set_xticklabels(['0.05', '0.1', '0.15', '0.2'])
axes[-1].set_xlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$', fontsize=10)
# the shared y label just left of the widest tick label, so they never touch
fig.canvas.draw()
tick_labels = [label for ax in axes for label in ax.get_yticklabels() if label.get_text()]
tick_left = min(label.get_window_extent(fig.canvas.get_renderer()).x0
                for label in tick_labels)
tick_left = fig.transFigure.inverted().transform((tick_left, 0.0))[0]
supylabel = fig.supylabel(r'largest shift [$\sigma$]', x=tick_left - 0.01, ha='right', fontsize=10)

# the layouts above the top panel (color, line width and marker)
layout_legend = axes[0].legend(
    handles=[Line2D([], [], color=lay['color'], lw=LAYOUT_STYLE[lay['name']]['lw'],
                    marker=LAYOUT_STYLE[lay['name']]['marker'], ms=MARKER_SIZE,
                    label=lay['label']) for lay in layouts],
    loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.2, columnspacing=1.2, handletextpad=0.5, labelspacing=0.25,
    borderaxespad=0.2, borderpad=0.1)
check_clear(fig, flat, [layout_legend, supylabel, axes[-1].xaxis.label,
                        *[ax.yaxis.label for ax in axes]])
check_labels_off_ticks(fig, flat, [supylabel, axes[-1].xaxis.label,
                                   *[ax.yaxis.label for ax in axes]],
                       'summary_shifts_kmax')
fig.savefig(os.path.join(HERE, 'summary_shifts_kmax.pdf'))
fig.savefig(os.path.join(HERE, 'summary_shifts_kmax.png'), dpi=300)
plt.close(fig)
