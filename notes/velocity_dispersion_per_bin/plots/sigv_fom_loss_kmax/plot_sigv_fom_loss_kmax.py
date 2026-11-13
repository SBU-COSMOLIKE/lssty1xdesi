"""Redraw sigv_fom_loss_kmax.pdf from data.json: what the per-bin freedom of
the velocity dispersions costs, as the FoM loss 100 (1 - FoM/FoM_V0) in
percent against today's model V0 (one dispersion per group), against the
scale cut k_max.

One column, two stacked panels sharing both axes, named on the right:
FoM(w0, wa) and FoM(Omega_m, 10^9 A_s). Two models: a free halo-part slope
per group at the baseline width (V1, dashed) and every bin's dispersion
free (V4, the bound, dash-dotted). Each bin layout has a color, a line
width and a marker (at every cut), so the figure reads in black and white.
The y axis is logarithmic and the same in both panels. V1 on the Delta z =
0.2 layout, whose loss stays below 10^-4 percent, is not drawn; data.json
keeps it and the other variants (the slope widths 0.5 and 2.0, residuals
A and B, the white term).
k_max is the z = 0 value of a per-bin cut at k_max/D(zbar_i).
Run from any folder: python plot_sigv_fom_loss_kmax.py
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
# the figure's width [in]: one column of a two-column paper
FIG_WIDTH = 3.5
# the height of one panel [in]
PANEL_HEIGHT = 1.3
# Each layout has a line width and a marker as well as its color, so the
# layouts stay apart in black and white; the marker sits at every cut.
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
    'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
    'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
    'Delta z = 0.2': {'lw': 0.9, 'marker': '^'},
}
# the models: data.json key, line style (a dash pattern visible at the
# thinnest width), legend label, and the layouts drawn (V1 on Delta z = 0.2
# loses less than 10^-4 percent and is not drawn)
VARIANTS = [('V1 1.0', (0, (4, 1.6)), r'V1: + halo-part slope',
             ['Delta d_com = 62 Mpc', 'Delta z = 0.02', 'Delta z = 0.05']),
            ('V4', (0, (6, 1.5, 1.5, 1.5)), r'V4: every bin free',
             ['Delta d_com = 62 Mpc', 'Delta z = 0.02', 'Delta z = 0.05', 'Delta z = 0.2'])]
# the marker size [pt]
MARKER_SIZE = 3.5
# the panels, top to bottom: data.json key and the name on the right
PANELS = [('fom', r'FoM$(w_0, w_a)$'),
          ('fom_om_as', r'FoM$(\Omega_\mathrm{m}, A_\mathrm{s})$')]
# the margin of the log y range below the lowest and above the highest point
Y_MARGIN = 1.6
# the gap between the two legends, and between the lower legend and the top
# panel, in figure units
LEGEND_GAP = 0.004

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax = np.array(data['kmax_h_per_Mpc'], dtype=float)


def plain_number(value, position):
    '''A tick label as a plain number: 0.001, 0.01, 10 (position is unused).'''
    return f'{value:g}'


def as_array(values):
    '''A list of numbers and None, one per cut, as a float array with NaN for
    None (a gap in the line).'''
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if value is not None:
            out[i] = float(value)
    return out


def line_style(lw, marker, color, ls):
    '''The plot keywords of a line of width lw and style ls with filled
    markers.'''
    return {'color': color, 'ls': ls, 'lw': lw, 'marker': marker, 'ms': MARKER_SIZE,
            'mfc': color, 'mec': color}


def line_points(ax):
    '''Display coordinates of points densely sampled along every drawn line of
    ax, inside the axes only (each line's own transform maps its data to the
    display, and the segments are sampled in display space).'''
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


def shown_yticklabels(fig, axes):
    '''The y tick labels drawn on the page: a log locator also returns ticks
    outside the panel's y range, whose labels matplotlib never draws.'''
    fig.canvas.draw()
    shown = []
    for ax in axes:
        low, high = ax.get_ylim()
        for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
            if label.get_text() and low <= tick <= high:
                shown.append(label)
    return shown


def check_clear(fig, axes, artists):
    '''Raise when a legend or label touches a drawn line or another one.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    for box, artist in zip(boxes, artists):
        if touches_lines(box, axes):
            raise RuntimeError(f'sigv_fom_loss_kmax: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'sigv_fom_loss_kmax: {type(artists[i]).__name__} {i} '
                                   f'overlaps {type(artists[j]).__name__} {j}')


layouts = data['layouts']
fig, axes = plt.subplots(len(PANELS), 1, figsize=(FIG_WIDTH, PANEL_HEIGHT*len(PANELS)),
                         sharex=True, sharey=True, gridspec_kw={'hspace': 0.06})
values = []
for ax, (key, name) in zip(axes, PANELS):
    for variant, ls, _, drawn in VARIANTS:
        for lay in layouts:
            if lay['name'] not in drawn:
                continue
            st = LAYOUT_STYLE[lay['name']]
            y = as_array(lay['loss_percent'][variant][key])
            ax.plot(kmax, y, **line_style(st['lw'], st['marker'], lay['color'], ls))
            values += [float(v) for v in y[np.isfinite(y)]]
    ax.grid(False, axis='x')
    # the FoM's name on the right, rotated like a y label
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(name, rotation=90, fontsize=9, labelpad=6)

axes[0].set_yscale('log')
# a label on every decade, as plain numbers, no minor labels
axes[0].yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
axes[0].yaxis.set_major_formatter(FuncFormatter(plain_number))
axes[0].yaxis.set_minor_formatter(NullFormatter())
axes[0].set_ylim(min(values)/Y_MARGIN, max(values)*Y_MARGIN)
axes[-1].set_xlim(kmax[0] - 0.01, kmax[-1] + 0.01)
axes[-1].set_xticks([0.05, 0.1, 0.15, 0.2])
axes[-1].set_xticklabels(['0.05', '0.1', '0.15', '0.2'])
axes[-1].set_xlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$', fontsize=10)
# the shared y label just left of the widest y tick label
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
tick_left = min(label.get_window_extent(renderer).x0
                for label in shown_yticklabels(fig, axes))
tick_left = fig.transFigure.inverted().transform((tick_left, 0.0))[0]
supylabel = fig.supylabel('FoM loss against V0 [percent]', fontsize=10,
                          x=tick_left - 0.015, ha='right')

# two legends above the top panel, close together: the layouts (color, line
# width and marker) just above it, the models (line style, in black) above
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
top = axes[0].get_window_extent(renderer).y1
top = fig.transFigure.inverted().transform((0.0, top))[1]
# Four layouts do not fit on one line of a column: two lines of two, in the
# project's order read row by row. A legend fills its columns first, so the
# handles go in the order first, third, second, fourth.
layout_handles = []
for lay in layouts:
    layout_handles.append(Line2D([], [], label=lay['label'],
                                 **line_style(LAYOUT_STYLE[lay['name']]['lw'],
                                              LAYOUT_STYLE[lay['name']]['marker'],
                                              lay['color'], '-')))
layout_legend = fig.legend(
    handles=[layout_handles[0], layout_handles[2], layout_handles[1], layout_handles[3]],
    loc='lower center', bbox_to_anchor=(0.5, top + LEGEND_GAP), ncol=2,
    frameon=False, fontsize=8, handlelength=2.0, columnspacing=1.6, handletextpad=0.4,
    labelspacing=0.35, borderaxespad=0.0, borderpad=0.1)
fig.canvas.draw()
layout_top = layout_legend.get_window_extent(renderer).y1
layout_top = fig.transFigure.inverted().transform((0.0, layout_top))[1]
variant_legend = fig.legend(
    handles=[Line2D([], [], color='black', ls=ls, lw=1.3, label=label)
             for _, ls, label, _ in VARIANTS],
    loc='lower center', bbox_to_anchor=(0.5, layout_top + LEGEND_GAP), ncol=len(VARIANTS),
    frameon=False, fontsize=8, handlelength=2.6, columnspacing=1.2, handletextpad=0.4,
    borderaxespad=0.0, borderpad=0.1)
check_clear(fig, list(axes), [layout_legend, variant_legend, supylabel,
                              axes[-1].xaxis.label, *[ax.yaxis.label for ax in axes],
                              *[label for label in axes[-1].get_xticklabels()],
                              *shown_yticklabels(fig, axes)])
fig.savefig(os.path.join(HERE, 'sigv_fom_loss_kmax.pdf'))
fig.savefig(os.path.join(HERE, 'sigv_fom_loss_kmax.png'), dpi=300)
plt.close(fig)
