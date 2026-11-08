"""Redraw the two figures of this folder from data.json: the figure of merit
of DESI C_ell^gg alone against the scale cut k_max, under the one-loop bias
models A and B of notes/higher_order_bias.

  fom_kmax_models.pdf       - FoM(w0, wa)
  fom_kmax_models_om_as.pdf - FoM(Omega_m, 10^9 A_s)

Each figure is one column of a two-column paper: three stacked panels, one
per linear-bias prior that passes the accuracy test (baseline, pessimistic,
free), named inside the panel, on one shared y axis (logarithmic when the
values span decades), one color per bin layout. Model A pessimistic (solid,
circles) is drawn only up to its break point: the largest cut at which
model A fits to truths with alpha_2 = +-12.5 h^-2 Mpc^2, SN2 = +-5 units or
the b2 and bs2 offsets at model A's one-sigma widths shift (w0, wa), Omega_m
and A_s by less than sigma/3. Model B pessimistic (dashed) is drawn at every
cut. The LSST Y1 3x2pt reference is a gray long-dashed line in every panel
whose highest point it exceeds by at most a factor 1.5; otherwise it is not
drawn, and a text inside the first such panel gives its value. k_max is the
z = 0 value of a per-bin cut at k_max/D(zbar_i).

data.json, written by EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb, also holds the
baseline variant of each model and the model A FoMs beyond its break, which
these figures do not draw.
Run from any folder: python plot_fom_kmax_models.py
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
# The drawn variant of each model. The break point is that of model A
# pessimistic, so the model A line is the FoM of the variant that passed;
# model B pessimistic is its like-for-like extension.
DRAWN_A = 'A pessimistic'
DRAWN_B = 'B pessimistic'
STYLE_A = {'ls': '-', 'lw': 1.4, 'marker': 'o', 'ms': 3.0}
STYLE_B = {'ls': '--', 'lw': 1.3}
# the LSST Y1 reference, the style of the v2 note's FoM figure: gray long
# dashes, apart from model B's short dashes
LSST_STYLE = {'color': '0.5', 'ls': (0, (7, 3)), 'lw': 1.2}
# the reference is drawn in a panel only when it sits at most this factor
# above the panel's highest point; farther up it would stretch the axis
REFERENCE_SPAN = 1.5
# the panel names, inside each panel (the data keys keep the notebook's
# names; 'independent' is the free prior)
PANEL_LABELS = {'baseline': 'baseline prior', 'pessimistic': 'pessimistic prior',
                'independent': 'free prior'}
# one figure per figure of merit: data.json field, LSST Y1 reference field,
# y label, output name
FIGURES = [('fom', 'lsst_y1_3x2pt_fom', r'FoM$(w_0, w_a)$, DESI $C_\ell^{gg}$ alone',
            'fom_kmax_models'),
           ('fom_om_as', 'lsst_y1_3x2pt_fom_om_as',
            r'FoM$(\Omega_\mathrm{m}, 10^9 A_\mathrm{s})$, DESI $C_\ell^{gg}$ alone',
            'fom_kmax_models_om_as')]
# the y axis is logarithmic when the drawn values span more than this factor
LOG_SPAN = 30.0
# the k_max labels compare floats from data.json: a break at 0.1 keeps the
# cut 0.1 and drops 0.125
KMAX_TOLERANCE = 1.0e-9
# the gap between the two legends above the top panel, in units of its height
LEGEND_GAP = 0.04
# The anchors a text may take inside a panel, in axes units (x, y,
# horizontal and vertical alignment), and the headroom factors of the y axis
# tried in turn until every text finds a corner free of lines.
CORNERS = {
    'upper left': (0.03, 0.94, 'left', 'top'),
    'upper right': (0.97, 0.94, 'right', 'top'),
    'lower right': (0.97, 0.06, 'right', 'bottom'),
    'lower left': (0.03, 0.06, 'left', 'bottom'),
}
# the panel names share one corner, the first free in every panel
NAME_CORNERS = ['lower right', 'upper left', 'upper right', 'lower left']
LOG_HEADROOM = [1.6, 2.2, 3.0, 4.5, 7.0, 10.0]
LINEAR_HEADROOM = [1.1, 1.25, 1.4, 1.6, 1.8, 2.0]

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax = np.array(data['kmax_h_per_Mpc'], dtype=float)
priors = data['priors']
layouts = data['layouts']


def as_array(values):
    '''A list of numbers and None as a float array, NaN for None (a gap).'''
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if value is not None:
            out[i] = float(value)
    return out


def model_a_curve(lay, field, prior):
    '''The model A FoM of one layout and prior up to its break point (NaN beyond).'''
    y = as_array(lay[field][DRAWN_A][prior])
    cut = lay['model_a_break'][DRAWN_A][prior]['kmax']
    if cut is None:
        # the smallest fitted cut already fails: no model A line
        y[:] = np.nan
    else:
        y[kmax > cut + KMAX_TOLERANCE] = np.nan
    return y


def plain_number(value, position):
    '''A tick label as a plain number: 0.1, 1, 10 (position is unused).'''
    return f'{value:g}'


def variant_label(variant):
    '''"A pessimistic" -> "model A pessimistic".'''
    return 'model ' + variant


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


for field, lsst_field, ylabel, name in FIGURES:
    lsst = data[lsst_field]
    fig, axes = plt.subplots(len(priors), 1, figsize=(3.5, 1.1*len(priors)), sharex=True,
                             sharey=True, gridspec_kw={'hspace': 0.06})
    drawn = []
    panel_high = {}
    for ax, prior in zip(axes, priors):
        values = []
        for lay in layouts:
            y_a = model_a_curve(lay, field, prior)
            y_b = as_array(lay[field][DRAWN_B][prior])
            ax.plot(kmax, y_a, color=lay['color'], **STYLE_A)
            ax.plot(kmax, y_b, color=lay['color'], **STYLE_B)
            for y in (y_a, y_b):
                values += [float(v) for v in y[np.isfinite(y)]]
        panel_high[prior] = max(values)
        drawn += values
        ax.grid(False, axis='x')
    # the reference line in every panel it does not stretch; its value as
    # text in the first panel without it
    with_line = [prior for prior in priors if lsst <= REFERENCE_SPAN*panel_high[prior]]
    without_line = [prior for prior in priors if prior not in with_line]
    for ax, prior in zip(axes, priors):
        if prior in with_line:
            ax.axhline(lsst, **LSST_STYLE)
            drawn.append(lsst)
    low, high = min(drawn), max(drawn)
    log_axis = high/low > LOG_SPAN
    if log_axis:
        axes[0].set_yscale('log')
        # a label on every decade (the default locator skips decades on a
        # short panel), as plain numbers (0.1, 1, 10), no minor labels
        axes[0].yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        axes[0].yaxis.set_major_formatter(FuncFormatter(plain_number))
        axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[-1].set_xlim(kmax[0] - 0.01, kmax[-1] + 0.01)
    axes[-1].set_xlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$')
    supylabel = fig.supylabel(ylabel, x=-0.03, fontsize=10)

    # the two legends above the top panel: the layouts by color, and above
    # them the line styles drawn (markers as drawn)
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
    style_handles = [
        Line2D([], [], color='0.25', label=variant_label(DRAWN_A), **STYLE_A),
        Line2D([], [], color='0.25', label=variant_label(DRAWN_B), **STYLE_B),
    ]
    if with_line:
        style_handles.append(Line2D([], [], label='LSST Y1 3x2pt', **LSST_STYLE))
    style_legend = axes[0].legend(handles=style_handles, loc='lower center',
                                  bbox_to_anchor=(0.5, top + LEGEND_GAP), ncol=2,
                                  frameon=False, fontsize=8, handlelength=2.4,
                                  columnspacing=1.2, handletextpad=0.5, labelspacing=0.25,
                                  borderaxespad=0.0)

    # the texts inside the panels: each prior's name, in one corner shared by
    # the panels, and the reference's value where its line is not drawn, in
    # an upper corner under the legends; the y axis gains headroom until
    # every text clears the lines
    legends = [layout_legend, style_legend]
    names = [PANEL_LABELS[prior] for prior in priors]
    texts = None
    for headroom in (LOG_HEADROOM if log_axis else LINEAR_HEADROOM):
        if log_axis:
            axes[0].set_ylim(low/1.6, high*headroom)
        else:
            axes[0].set_ylim(0.0, high*headroom)
        placed = place_panel_names(fig, axes, names, legends)
        if placed is None:
            continue
        if without_line:
            ax = axes[priors.index(without_line[0])]
            reference = None
            for corner in ('upper right', 'upper left'):
                reference = place_text(fig, axes, ax, f'LSST Y1 3x2pt: {lsst:g}', corner,
                                       [*legends, *placed])
                if reference is not None:
                    break
            if reference is None:
                for artist in placed:
                    artist.remove()
                continue
            placed.append(reference)
        texts = placed
        break
    if texts is None:
        raise RuntimeError(f'{name}: a panel text finds no corner free of lines')
    check_clear(fig, axes, [layout_legend, style_legend, supylabel, *texts], name)
    fig.savefig(os.path.join(HERE, f'{name}.pdf'))
    fig.savefig(os.path.join(HERE, f'{name}.png'), dpi=300)
    plt.close(fig)
