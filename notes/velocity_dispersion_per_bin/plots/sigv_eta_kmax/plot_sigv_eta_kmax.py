"""Redraw sigv_eta_kmax.pdf from data.json: how well each fit's quoted error
bar covers where the fit actually lands, measured by eta, when the true
velocity dispersions vary bin by bin (the truth T_lin + B: the linear
flow's own redshift dependence plus random halo-part slopes and residual-B
wiggles), against the scale cut k_max.

One column, three stacked panels sharing both axes, one per scored
parameter, named on the right: (w0, wa), Omega_m and A_s. Two models:
today's model with one velocity dispersion per group (V0, solid) and the
model with a free halo-part slope per group at the baseline width (V1,
dashed). Each bin layout has a color, a line width and a marker (at every
cut), so the figure reads in black and white. The y axis is linear and the
same in all panels. The Delta z = 0.2 layout, whose eta differs from 1 by
less than 10^-5, is not drawn (data.json keeps it, and every other truth).
k_max is the z = 0 value of a per-bin cut at k_max/D(zbar_i).
Run from any folder: python plot_sigv_eta_kmax.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
PANEL_HEIGHT = 1.15
# Each layout has a line width and a marker as well as its color, so the
# layouts stay apart in black and white; the marker sits at every cut.
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
    'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
    'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
    'Delta z = 0.2': {'lw': 0.9, 'marker': '^'},
}
# the layouts drawn; Delta z = 0.2 (eta within 10^-5 of 1) is not
DRAWN_LAYOUTS = ['Delta d_com = 62 Mpc', 'Delta z = 0.02', 'Delta z = 0.05']
# the models: data.json key, line style (a dash pattern visible at the
# thinnest width), legend label
VARIANTS = [('V0', '-', r'V0: one $\sigma_v$ per group'),
            ('V1 1.0', (0, (4, 1.6)), r'V1: + halo-part slope')]
# the marker size [pt]
MARKER_SIZE = 3.5
# the panels, top to bottom: data.json key and the name on the right
PANELS = [('w0wa', r'$(w_0, w_a)$'),
          ('omegam', r'$\Omega_\mathrm{m}$'),
          ('As', r'$A_\mathrm{s}$')]
# the margin of the linear y range, as a fraction of the drawn span
Y_MARGIN = 0.10
# the gap between the two legends, and between the lower legend and the top
# panel, in figure units
LEGEND_GAP = 0.004

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax = np.array(data['kmax_h_per_Mpc'], dtype=float)
truth = data['truth']


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
    '''The y tick labels drawn on the page: a locator may also return ticks
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
            raise RuntimeError(f'sigv_eta_kmax: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'sigv_eta_kmax: {type(artists[i]).__name__} {i} '
                                   f'overlaps {type(artists[j]).__name__} {j}')


layouts = [lay for lay in data['layouts'] if lay['name'] in DRAWN_LAYOUTS]
fig, axes = plt.subplots(len(PANELS), 1, figsize=(FIG_WIDTH, PANEL_HEIGHT*len(PANELS)),
                         sharex=True, sharey=True, gridspec_kw={'hspace': 0.06})
values = []
for ax, (key, name) in zip(axes, PANELS):
    for variant, ls, _ in VARIANTS:
        for lay in layouts:
            st = LAYOUT_STYLE[lay['name']]
            y = as_array(lay['eta'][variant][truth][key])
            ax.plot(kmax, y, **line_style(st['lw'], st['marker'], lay['color'], ls))
            values += [float(v) for v in y[np.isfinite(y)]]
    ax.grid(False, axis='x')
    # the parameter's name on the right, rotated like a y label
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(name, rotation=90, fontsize=9, labelpad=6)

# the y range fitted to the drawn values (1 included), with a small margin
low = min(values + [1.0])
high = max(values)
span = high - low
axes[0].set_ylim(low - Y_MARGIN*span, high + Y_MARGIN*span)
axes[0].set_yticks([1.0, 1.02, 1.04, 1.06])
axes[0].set_yticklabels(['1', '1.02', '1.04', '1.06'])
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
supylabel = fig.supylabel(r'$\eta$ under $T_\mathrm{lin}$ + B', fontsize=10,
                          x=tick_left - 0.015, ha='right')

# two legends above the top panel, close together: the layouts (color, line
# width and marker) just above it, the models (line style, in black) above
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
top = axes[0].get_window_extent(renderer).y1
top = fig.transFigure.inverted().transform((0.0, top))[1]
layout_legend = fig.legend(
    handles=[Line2D([], [], label=lay['label'],
                    **line_style(LAYOUT_STYLE[lay['name']]['lw'],
                                 LAYOUT_STYLE[lay['name']]['marker'], lay['color'], '-'))
             for lay in layouts],
    loc='lower center', bbox_to_anchor=(0.5, top + LEGEND_GAP), ncol=len(layouts),
    frameon=False, fontsize=8, handlelength=2.0, columnspacing=1.0, handletextpad=0.4,
    borderaxespad=0.0, borderpad=0.1)
fig.canvas.draw()
layout_top = layout_legend.get_window_extent(renderer).y1
layout_top = fig.transFigure.inverted().transform((0.0, layout_top))[1]
variant_legend = fig.legend(
    handles=[Line2D([], [], color='black', ls=ls, lw=1.3, label=label)
             for _, ls, label in VARIANTS],
    loc='lower center', bbox_to_anchor=(0.5, layout_top + LEGEND_GAP), ncol=len(VARIANTS),
    frameon=False, fontsize=8, handlelength=2.6, columnspacing=1.2, handletextpad=0.4,
    borderaxespad=0.0, borderpad=0.1)
check_clear(fig, list(axes), [layout_legend, variant_legend, supylabel,
                              axes[-1].xaxis.label, *[ax.yaxis.label for ax in axes],
                              *[label for label in axes[-1].get_xticklabels()],
                              *shown_yticklabels(fig, axes)])
fig.savefig(os.path.join(HERE, 'sigv_eta_kmax.pdf'))
fig.savefig(os.path.join(HERE, 'sigv_eta_kmax.png'), dpi=300)
plt.close(fig)
