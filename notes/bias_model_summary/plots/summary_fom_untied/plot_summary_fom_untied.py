"""Redraw summary_fom_untied.pdf from data.json: the figures of merit of
DESI C_ell^gg alone when the bias parameters are untied per bin, relative
to the summary model (the paper model: model B + tilts without alpha_2, the
sigma_v prior widths 6x those of the exploration fits, on the pessimistic
linear-bias prior), against the scale cut k_max. Built from a copy of
../summary_fom_free_bins/plot_summary_fom_free_bins.py.

Every version keeps the model's sigma_v widths and no alpha_2. The first four
versions free b1 per bin (flat prior) and give Delta_2, Delta_s, alpha_0, SN0
and SN2 one value per bin. The group width is the
width the model gives a group parameter, and N is the number of bins of the
group (the figure writes no symbol for the width: "w" would read as the
dark-energy w0, wa):
  per bin, group width       - no group offsets, no tilts;
  per bin, group width
    x sqrt(N)                - the group's mean keeps the group width;
  per bin, x sqrt(N)
    + group tilt             - plus a flat tilt in redshift per group, as the
                               model has (SN0 and SN2 never tilt);
  per bin, on top of the
    model                    - per-bin values of the group width added to
                               the model's group offsets and tilts;
  correlated residuals on top
    of the model             - the model plus a per-bin residual of Delta_2,
                               Delta_s and alpha_0 (not SN0, SN2),
                               Gaussian and correlated in redshift over
                               L = 0.05 with the group width, charging only
                               the deviation from the group's line (the
                               nested version is this with L -> 0, no
                               projection); b1 keeps the model's prior;
  one constant value per wide
    sample                   - one value per group of every bias parameter,
                               b1 constant in redshift (not B_g/D(z)), with
                               the model's priors on the group values, no
                               tilts and no residuals.
A version is drawn when data.json holds it, so a version added by a later
notebook run needs only its entry in VERSIONS; a version of data.json that
VERSIONS lacks raises an error, since its label and style cannot be guessed.

One full-width figure of three rows by four columns, the bin layouts, each
in its layout's color. The first two rows are the FoM ratios, FoM(w0, wa)
and FoM(Omega_m, 10^9 A_s), each with one shared logarithmic y axis and one
y label centered on the two rows. The third row is sqrt(<chi^2>/n_dof) of
(w0, wa) for every version, under the known drifts plus the pessimistic
scatter (for the constant version also the fiducial's own evolution within
each sample, which a constant cannot follow), on a linear axis from 0 with
its own y label: 1 means honest error bars, and a version with b1 free per
bin sits at 1 by construction, so those lines lie on top of each other there.
The versions differ by line style, line width and marker (never by color
alone; the markers differ from the layout markers of the other figures). The
thin gray line marks a ratio of 1 and an honest 1 in the third row; it has
no legend entry (the caption names it). k_max is the z = 0 value of a per-bin cut at
k_max/D(zbar_i); cuts from KMAX_MIN to KMAX_MAX are drawn. data.json also
holds k_max = 0.05, where the model's own FoM(w0, wa) is below 0.1 and the
ratios would dominate the eye; it is not drawn. A cut that leaves no data
has no point.

data.json, written by EXAMPLE_EVALUATE_FISHER_MODEL.ipynb, also holds the
FoMs of the model and of every version, and per version (the model as
'model') the statistic of every score, the coverage of the 68.3% region and
the largest shift over the fixed truths.
Run from any folder: python plot_summary_fom_untied.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

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
ROW_HEIGHT = 1.45
# The versions, in the order of the narrative: data.json key, legend label
# and line style. Each has its own line style, width and marker, so the
# versions read in black and white within a column of one color; the markers
# are none of the layout markers (o, s, D, ^) or prior markers (*, v, X, P)
# of the other figures. The dash and gap lengths are in points: matplotlib
# scales a dash pattern by the line width, so dividing by it keeps them fixed.
# The labels write no symbol for the group width ("w" would read as the
# dark-energy w0, wa). Adding a version is one entry here; it is drawn once
# data.json holds its key.
VERSIONS = [
    ('gauss', 'per bin, group width',
     {'lw': 1.0, 'dash': (1.2, 1.6), 'marker': '<', 'open': True}),
    ('sqrtn', r'per bin, group width $\times\sqrt{N}$',
     {'lw': 1.4, 'dash': (5.0, 1.8, 1.2, 1.8), 'marker': '>', 'open': False}),
    ('sqrtn_tilt', r'per bin, $\times\sqrt{N}$ + group tilt',
     {'lw': 1.8, 'dash': (4.5, 2.2), 'marker': 'p', 'open': False}),
    ('nested', 'per bin, on top of the model',
     {'lw': 2.2, 'dash': None, 'marker': 'h', 'open': True}),
    ('correlated_residuals', 'correlated residuals on top of the model',
     {'lw': 1.2, 'dash': (7.0, 1.6, 1.2, 1.6, 1.2, 1.6), 'marker': 'x', 'open': False}),
    ('constant_per_sample', 'one constant value per wide sample',
     {'lw': 2.6, 'dash': (8.0, 2.6), 'marker': '+', 'open': False}),
]
MARKER_SIZE = 4.0
# the line at a ratio of 1 (no change): thin and gray
UNITY_STYLE = {'color': '0.55', 'ls': '-', 'lw': 0.8}
# the rows: data.json field and the figure of merit's name, on the right side
ROWS = [('ratio_fom', r'FoM$(w_0, w_a)$'),
        ('ratio_fom_om_as', r'FoM$(\Omega_\mathrm{m}, 10^9 A_\mathrm{s})$')]
# the shared y label
YLABEL = 'FoM ratio: per bin / model'
# the smallest and largest cuts drawn [h/Mpc at z = 0]; data.json also holds
# 0.05 (the model's own FoM(w0, wa) is below 0.1 there) and may hold larger
# cuts
KMAX_MIN = 0.075
KMAX_MAX = 0.2
# the k_max labels compare floats from data.json
KMAX_TOLERANCE = 1.0e-9
# The y axis is logarithmic: the constant-per-sample version reaches ratios
# near 10 while the others stay between 0.3 and 2.3, and on a log axis a ratio
# and its inverse sit at the same distance from 1. The margin of each row's
# range is a fraction of its span in decades; the labeled ticks are those of
# Y_TICKS inside it.
Y_MARGIN = 0.06
Y_TICKS = [0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax_all = np.array(data['kmax_h_per_Mpc'], dtype=float)
# the cuts drawn: KMAX_MIN to KMAX_MAX
DRAWN_CUTS = ((kmax_all >= KMAX_MIN - KMAX_TOLERANCE)
              & (kmax_all <= KMAX_MAX + KMAX_TOLERANCE))
kmax = kmax_all[DRAWN_CUTS]
layouts = data['layouts']


def present_versions():
    '''The VERSIONS entries that data.json holds, in VERSIONS order.

    A version counts as present when every layout holds it in every row; one
    that only some layouts or rows hold is an incomplete export and raises.
    A version of data.json that VERSIONS lacks also raises.'''
    known = [key for key, label, style in VERSIONS]
    held = [set(lay[field]) for lay in layouts for field, name in ROWS]
    unknown = sorted(set().union(*held) - set(known))
    if unknown:
        raise RuntimeError(f'summary_fom_untied: data.json holds the versions {unknown}, '
                           f'which VERSIONS lacks: add their label and style')
    partial = sorted(set().union(*held) - set.intersection(*held))
    if partial:
        raise RuntimeError(f'summary_fom_untied: the versions {partial} are missing '
                           f'from some layouts or rows of data.json')
    return [entry for entry in VERSIONS if entry[0] in held[0]]


drawn_versions = present_versions()


def version_style(style, color):
    '''The plot keywords of one version in a column of one color.'''
    ls = '-' if style['dash'] is None else (0, tuple(d/style['lw'] for d in style['dash']))
    return {'color': color, 'ls': ls, 'lw': style['lw'], 'marker': style['marker'],
            'ms': MARKER_SIZE, 'mfc': 'white' if style['open'] else color, 'mec': color}


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
            raise RuntimeError(f'summary_fom_untied: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'summary_fom_untied: {type(artists[i]).__name__} {i} '
                                   f'overlaps {type(artists[j]).__name__} {j}')


# The third row: sqrt(<chi^2>/n_dof) of (w0, wa) for every version, under the
# known drifts plus the pessimistic scatter (data.json's eta, keyed by
# version); the gray line at 1 marks honest error bars. A linear axis from 0:
# its top lies ETA_HEADROOM above the largest value, and its labeled ticks are
# those of ETA_TICKS below the top (1 always among them).
ETA_FIELD = 'eta'
ETA_SCORE = 'w0wa'
ETA_TICKS = [1, 2, 4, 6, 8, 10, 15, 20]
ETA_HEADROOM = 0.12
ETA_NAME = r'$(w_0, w_a)$'
ETA_YLABEL = r'$\sqrt{\langle\chi^2\rangle/n_\mathrm{dof}}$'
n_rows = len(ROWS) + 1
fig, axes = plt.subplots(n_rows, len(layouts), figsize=(FIG_WIDTH, ROW_HEIGHT*n_rows),
                         sharex=True, sharey='row', gridspec_kw={'hspace': 0.08, 'wspace': 0.08})
flat = list(axes.ravel())
for r, (field, name) in enumerate(ROWS):
    values = [1.0]
    for c, lay in enumerate(layouts):
        ax = axes[r, c]
        ax.axhline(1.0, **UNITY_STYLE)
        for key, label, style in drawn_versions:
            y = as_array(lay[field][key])
            ax.plot(kmax, y, **version_style(style, lay['color']))
            values += [float(v) for v in y[np.isfinite(y)]]
        ax.grid(False, axis='x')
        if r == 0:
            ax.set_title(lay['label'], fontsize=9.5, color='black')
    decades = np.log10(max(values)/min(values))
    low = min(values)/10.0**(Y_MARGIN*decades)
    high = max(values)*10.0**(Y_MARGIN*decades)
    axes[r, 0].set_yscale('log')
    axes[r, 0].set_ylim(low, high)
    axes[r, 0].yaxis.set_major_locator(FixedLocator([t for t in Y_TICKS if low < t < high]))
    axes[r, 0].yaxis.set_major_formatter(FuncFormatter(plain_number))
    axes[r, 0].yaxis.set_minor_locator(NullLocator())
    # the figure of merit's name on the right of the last column
    axes[r, -1].yaxis.set_label_position('right')
    axes[r, -1].set_ylabel(name, rotation=90, fontsize=9, labelpad=5)
r3 = len(ROWS)
eta_values = [1.0]
for c, lay in enumerate(layouts):
    ax = axes[r3, c]
    ax.axhline(1.0, **UNITY_STYLE)
    for key, label, style in drawn_versions:
        y = as_array(lay[ETA_FIELD][key][ETA_SCORE])
        ax.plot(kmax, y, **version_style(style, lay['color']))
        eta_values += [float(v) for v in y[np.isfinite(y)]]
    ax.grid(False, axis='x')
eta_top = max(eta_values)*(1.0 + ETA_HEADROOM)
axes[r3, 0].set_ylim(0.0, eta_top)
axes[r3, 0].yaxis.set_major_locator(FixedLocator([t for t in ETA_TICKS if t < eta_top]))
axes[r3, 0].yaxis.set_major_formatter(FuncFormatter(plain_number))
axes[r3, 0].yaxis.set_minor_locator(NullLocator())
axes[r3, -1].yaxis.set_label_position('right')
axes[r3, -1].set_ylabel(ETA_NAME, rotation=90, fontsize=9, labelpad=5)
# labels at 0.1 and 0.2 only, so the labels of neighboring columns never
# meet, and an unlabeled tick at every other cut drawn
for ax in axes[-1]:
    ax.set_xlim(kmax[0] - 0.01, kmax[-1] + 0.01)
    ax.set_xticks([0.1, 0.2])
    ax.set_xticklabels(['0.1', '0.2'])
    ax.set_xticks([0.075, 0.125, 0.15, 0.175], minor=True)
# the shared x label just under the lowest tick label
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
tick_bottom = min(label.get_window_extent(renderer).y0
                  for ax in axes[-1] for label in ax.get_xticklabels() if label.get_text())
tick_bottom = fig.transFigure.inverted().transform((0.0, tick_bottom))[1]
supxlabel = fig.supxlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$', fontsize=10,
                          y=tick_bottom - 0.01, va='top')
# the shared y label just left of the widest tick label
tick_left = min(label.get_window_extent(renderer).x0
                for ax in axes[:, 0] for label in ax.get_yticklabels() if label.get_text())
tick_left = fig.transFigure.inverted().transform((tick_left, 0.0))[0]
to_fig = fig.transFigure.inverted()
def row_center(rows):
    '''The vertical center, in figure units, of the rows given.'''
    top = axes[rows[0], 0].get_window_extent(renderer).y1
    bottom = axes[rows[-1], 0].get_window_extent(renderer).y0
    return to_fig.transform((0.0, 0.5*(top + bottom)))[1]
supylabel = fig.text(tick_left - 0.01, row_center([0, len(ROWS) - 1]), YLABEL,
                     rotation=90, ha='right', va='center', fontsize=10)
etalabel = fig.text(tick_left - 0.01, row_center([r3]), ETA_YLABEL,
                    rotation=90, ha='right', va='center', fontsize=10)

# the versions above the titles, in dark gray (the color is the column's
# layout), with the line style, width and marker drawn
fig.canvas.draw()
title_top = max(ax.title.get_window_extent(renderer).y1 for ax in axes[0])
title_top = fig.transFigure.inverted().transform((0.0, title_top))[1]
legend = fig.legend(
    handles=[Line2D([], [], label=label, **version_style(style, '0.25'))
             for key, label, style in drawn_versions],
    loc='lower center', bbox_to_anchor=(0.5, title_top + 0.005), ncol=2, frameon=False,
    fontsize=8.5, handlelength=3.4, columnspacing=1.8, handletextpad=0.5,
    labelspacing=0.3, borderaxespad=0.0, borderpad=0.1)
labels = [supxlabel, supylabel, etalabel, *[ax.title for ax in axes[0]],
          *[axes[r, -1].yaxis.label for r in range(n_rows)]]
check_clear(fig, flat, [legend, *labels])
check_labels_off_ticks(fig, flat, labels, 'summary_fom_untied')
# the tick labels of neighboring columns must not meet either
fig.canvas.draw()
tick_boxes = [label.get_window_extent(fig.canvas.get_renderer()) for ax in axes[-1]
              for label in ax.get_xticklabels() if label.get_text()]
for i in range(len(tick_boxes)):
    for j in range(i + 1, len(tick_boxes)):
        if tick_boxes[i].overlaps(tick_boxes[j]):
            raise RuntimeError('summary_fom_untied: two x tick labels overlap')
fig.savefig(os.path.join(HERE, 'summary_fom_untied.pdf'))
fig.savefig(os.path.join(HERE, 'summary_fom_untied.png'), dpi=300)
plt.close(fig)
