"""Redraw mag_shift_kmax.pdf from data.json: how far the paper model, which
sets the magnification coefficient C to zero, moves the fit when the data
carry lensing magnification, against the scale cut k_max.

A full-width grid: rows are the scored parameters, named on the right
((w0, wa) in units of its error ellipse, sqrt(dp^T C^-1 dp) of the 2x2
marginalized covariance; Omega_m and A_s in units of their own sigma);
columns are the three ELG coefficients (C_ELG = 0, +2, -2, with BGS 2 and
LRG 3 in every column). Each bin layout has a color, a line width and a
marker; the paper pair list is solid, the maximal (pairmax) list dashed. The
y axis is logarithmic and the same in every panel. Every nuisance parameter
is free. k_max is the z = 0 value of a per-bin cut at k_max/D(zbar_i).
Run from any folder: python plot_mag_shift_kmax.py
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
# the figure's width [in]: the full width of a two-column page
FIG_WIDTH = 6.4
# the height of one panel row [in]
PANEL_HEIGHT = 1.2
# Each layout has a line width and a marker as well as its color, so the
# layouts stay apart in black and white; the marker sits at every cut.
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
    'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
    'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
    'Delta z = 0.2': {'lw': 0.9, 'marker': '^'},
}
# the pair lists: the paper list solid, the maximal list dashed (a dash
# pattern still visible at the thinnest width)
LIST_STYLE = {'paper': '-', 'pairmax': (0, (3.0, 1.6))}
LIST_LABEL = {'paper': 'paper pair list', 'pairmax': 'maximal pair list'}
MARKER_SIZE = 3.0
# the rows, top to bottom: data.json key and the name on the right
ROWS = [('w0wa', r'$(w_0, w_a)$'),
        ('omegam', r'$\Omega_\mathrm{m}$'),
        ('As', r'$A_\mathrm{s}$')]
# the columns: data.json coefficient set and its title
COLUMNS = [('fiducial', r'$C_\mathrm{ELG} = 0$'),
           ('elg_plus', r'$C_\mathrm{ELG} = +2$'),
           ('elg_minus', r'$C_\mathrm{ELG} = -2$')]
# the margin of the log y range below the lowest and above the highest point
Y_MARGIN = 1.5
# the vertical gap [figure fraction] between the panels and each legend row
LEGEND_GAP = 0.004

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax = np.array(data['kmax_h_per_Mpc'], dtype=float)


def as_array(values):
    '''A list of numbers and None, one per cut, as a float array with NaN for
    None (a gap in the line).'''
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if value is not None:
            out[i] = float(value)
    return out


def line_style(lw, marker, color, ls):
    '''The plot keywords of a line of width lw and style ls with filled markers.'''
    return {'color': color, 'ls': ls, 'lw': lw, 'marker': marker, 'ms': MARKER_SIZE,
            'mfc': color, 'mec': color}


def plain_number(value, position):
    '''A tick label as a plain number: 0.01, 0.1, 1 (position is unused).'''
    return f'{value:g}'


def line_points(ax):
    '''Display coordinates of points densely sampled along every drawn line of
    ax, inside the axes only (segments sampled in display space, where a
    straight segment of a log axis is straight).'''
    box = ax.get_window_extent()
    pts = [np.zeros((0, 2))]
    for line in ax.get_lines():
        xy = np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()])
        display_xy = line.get_transform().transform(xy)
        finite = np.all(np.isfinite(display_xy), axis=1)
        pts.append(display_xy[finite])
        # 60 samples per segment: at figure resolution a label box cannot
        # slip between two consecutive samples of a drawn segment
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
    '''The y tick labels drawn on the page (a log locator also returns ticks
    outside the y range, whose labels are never drawn).'''
    fig.canvas.draw()
    shown = []
    for ax in axes:
        low, high = ax.get_ylim()
        for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
            if label.get_text() and low <= tick <= high:
                shown.append(label)
    return shown


def check_clear(fig, axes, artists):
    '''Raise when a legend, title or label touches a drawn line or another one.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    for box, artist in zip(boxes, artists):
        if touches_lines(box, axes):
            raise RuntimeError(f'mag_shift_kmax: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'mag_shift_kmax: {type(artists[i]).__name__} {i} '
                                   f'overlaps {type(artists[j]).__name__} {j}')


fig, axes = plt.subplots(len(ROWS), len(COLUMNS),
                         figsize=(FIG_WIDTH, PANEL_HEIGHT*len(ROWS)),
                         sharex=True, sharey=True,
                         gridspec_kw={'hspace': 0.06, 'wspace': 0.06})
values = []
for r, (key, name) in enumerate(ROWS):
    for c, (column, title) in enumerate(COLUMNS):
        ax = axes[r, c]
        for lay in data['layouts']:
            st = LAYOUT_STYLE[lay['name']]
            for lst, ls in LIST_STYLE.items():
                y = as_array(lay[lst][column][key])
                ax.plot(kmax, y, **line_style(st['lw'], st['marker'], lay['color'], ls))
                # collect every finite positive value drawn, for the shared
                # log y range set after the loop (a log axis cannot show 0)
                values += [float(v) for v in y[np.isfinite(y)] if v > 0]
        ax.grid(False, axis='x')
        if r == 0:
            ax.set_title(title, fontsize=9, pad=3)
        if c == len(COLUMNS) - 1:
            ax.yaxis.set_label_position('right')
            ax.set_ylabel(name, rotation=90, fontsize=9, labelpad=6)

axes[0, 0].set_yscale('log')
axes[0, 0].yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
axes[0, 0].yaxis.set_major_formatter(FuncFormatter(plain_number))
axes[0, 0].yaxis.set_minor_formatter(NullFormatter())
axes[0, 0].set_ylim(min(values)/Y_MARGIN, max(values)*Y_MARGIN)
for c in range(len(COLUMNS)):
    axes[-1, c].set_xlim(kmax[0] - 0.01, kmax[-1] + 0.01)
    axes[-1, c].set_xticks([0.05, 0.1, 0.15, 0.2])
    axes[-1, c].set_xticklabels(['0.05', '0.1', '0.15', '0.2'])
fig.supxlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$', fontsize=10, y=0.0)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
first_column = [axes[r, 0] for r in range(len(ROWS))]
tick_left = min(label.get_window_extent(renderer).x0
                for label in shown_yticklabels(fig, first_column))
tick_left = fig.transFigure.inverted().transform((tick_left, 0.0))[0]
supylabel = fig.supylabel(r'shift [$\sigma$]', fontsize=10,
                          x=tick_left - 0.01, ha='right')

# the legend above the top row: the layouts, then the two pair lists
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
top = max(ax.title.get_window_extent(renderer).y1 for ax in axes[0])
top = fig.transFigure.inverted().transform((0.0, top))[1]
layout_handles = [Line2D([], [], label=lay['label'],
                         **line_style(LAYOUT_STYLE[lay['name']]['lw'],
                                      LAYOUT_STYLE[lay['name']]['marker'], lay['color'], '-'))
                  for lay in data['layouts']]
list_handles = [Line2D([], [], color='black', lw=1.3, ls=ls, label=LIST_LABEL[lst])
                for lst, ls in LIST_STYLE.items()]
legend_lists = fig.legend(handles=list_handles, loc='lower center',
                          bbox_to_anchor=(0.5, top + LEGEND_GAP), ncol=2, frameon=False,
                          fontsize=8, handlelength=2.6, columnspacing=1.4,
                          handletextpad=0.4, borderaxespad=0.0, borderpad=0.1)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
top2 = fig.transFigure.inverted().transform(
    (0.0, legend_lists.get_window_extent(renderer).y1))[1]
legend_layouts = fig.legend(handles=layout_handles, loc='lower center',
                            bbox_to_anchor=(0.5, top2 + LEGEND_GAP), ncol=4, frameon=False,
                            fontsize=8, handlelength=2.0, columnspacing=1.0,
                            handletextpad=0.4, borderaxespad=0.0, borderpad=0.1)
flat = list(axes.ravel())
check_clear(fig, flat, [legend_layouts, legend_lists, supylabel,
                        *[ax.title for ax in axes[0]],
                        *[axes[r, -1].yaxis.label for r in range(len(ROWS))],
                        *shown_yticklabels(fig, first_column)])
fig.savefig(os.path.join(HERE, 'mag_shift_kmax.pdf'))
fig.savefig(os.path.join(HERE, 'mag_shift_kmax.png'), dpi=300)
plt.close(fig)
