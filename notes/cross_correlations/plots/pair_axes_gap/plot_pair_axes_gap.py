"""Redraw pair_axes_gap.pdf from data.json: what the figures of merit gain when cross
pairs are added to the current data vector along one axis (data.json "axis"):

- gap: every pair whose n(z) ranges are at most g bin widths apart is
  measured (g = 0 is the current list), at the fiducial sigma_v;
- sigv: every pair whose Fingers-of-God-blurred kernels overlap by
  O_ij >= 1e-6 when every sigma_v is s km/s is measured; s only selects the
  pairs, and the model keeps the fiducial sigma_v ("fid." on the axis).

A full-width grid of a two-column paper: rows k_max = 0.075, 0.1 and 0.2
h/Mpc (named on the right), columns FoM(w0, wa) and FoM(Omega_m, 10^9 A_s)
(named above), each over the same FoM of the current data vector. Every
panel shares one y range. The four layouts each have a color, a line width
and a marker; solid lines with filled markers fit sigma_v, dashed lines with
open markers hold it at the fiducial. Gray: one. The model drawn is MODEL.
k_max is the z = 0 value of a per-bin cut at k_max/D(zbar_i).
Run from any folder: python plot_pair_axes_gap.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

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
NAME = os.path.basename(HERE)
# the model drawn: 'paper' (the paper's model) or 'reference' (the model of
# the pair_threshold figure); data.json holds both
MODEL = 'paper'
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
    'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
    'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
    'Delta z = 0.2': {'lw': 0.9, 'marker': '^'},
}
MARKER_SIZE = 3.5
VARIANT_STYLE = {'free': {'ls': '-', 'filled': True,
                          'label': r'$\sigma_v$ fitted'},
                 'fixed': {'ls': (0, (4, 2)), 'filled': False,
                           'label': r'$\sigma_v$ held at the fiducial'}}
REFERENCE = {'color': '0.6', 'lw': 0.8}
PANEL_HEIGHT = 1.15
FIG_WIDTH = 6.4
LEGEND_GAP = 0.006
# the columns: data.json key and the name written above the top panel
COLUMNS = [('fom_ratio', r'$\mathrm{FoM}(w_0, w_a)$'),
           ('fom_om_as_ratio', r'$\mathrm{FoM}(\Omega_\mathrm{m}, 10^9 A_\mathrm{s})$')]
# one y range for every panel, a round step above and below the data, the
# ticks kept off the panel edges so the stacked panels' labels never touch
Y_STEP = 0.2
Y_FLOOR_STEP = 0.1
Y_PAD = 0.06
X_MARGIN = 0.07  # of the x range, on each side

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
cuts = data['kmax_h_per_Mpc']


def line_points(ax):
    '''Display coordinates of points densely sampled along every drawn line of
    ax (reference lines included), inside the axes only.'''
    box = ax.get_window_extent()
    pts = [np.zeros((0, 2))]
    for line in ax.get_lines():
        xy = np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()])
        display_xy = line.get_transform().transform(xy)
        finite = np.all(np.isfinite(display_xy), axis=1)
        pts.append(display_xy[finite])
        steps = np.linspace(0.0, 1.0, 60)[:, None]
        for i in range(len(display_xy) - 1):
            if finite[i] and finite[i + 1]:
                pts.append(display_xy[i] + steps*(display_xy[i + 1] - display_xy[i]))
    pts = np.vstack(pts)
    inside = ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
              & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))
    return pts[inside]


def touches_lines(box, axes):
    padded = box.expanded(1.03, 1.08)
    for ax in axes:
        pts = line_points(ax)
        if np.any((pts[:, 0] > padded.x0) & (pts[:, 0] < padded.x1)
                  & (pts[:, 1] > padded.y0) & (pts[:, 1] < padded.y1)):
            return True
    return False


def check_clear(fig, axes, artists):
    '''Raise when a legend or label touches a drawn line or another one, or
    when two tick labels touch.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    for box, artist in zip(boxes, artists):
        if touches_lines(box, axes):
            raise RuntimeError(f'{NAME}: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'{NAME}: {type(artists[i]).__name__} {i} overlaps '
                                   f'{type(artists[j]).__name__} {j}')
    shown = [*[label for ax in axes for label in ax.get_yticklabels()],
             *[label for ax in axes for label in ax.get_xticklabels()]]
    ticks = [label.get_window_extent(renderer) for label in shown
             if label.get_text() and label.get_visible()]
    for i in range(len(ticks)):
        for j in range(i + 1, len(ticks)):
            if ticks[i].overlaps(ticks[j]):
                raise RuntimeError(f'{NAME}: two tick labels touch')


fig, grid = plt.subplots(len(cuts), len(COLUMNS),
                         figsize=(FIG_WIDTH, PANEL_HEIGHT*len(cuts)),
                         sharex=True, sharey=True,
                         gridspec_kw={'hspace': 0.06, 'wspace': 0.06})
values = [1.0]
for row, cut in enumerate(cuts):
    for col, (key, _) in enumerate(COLUMNS):
        ax = grid[row, col]
        ax.axhline(1.0, **REFERENCE)
        for lay in data['layouts']:
            st = LAYOUT_STYLE[lay['name']]
            x = [p['x'] for p in lay['points']]
            for variant, vs in VARIANT_STYLE.items():
                y = [p[cut][MODEL][variant][key] for p in lay['points']]
                face = lay['color'] if vs['filled'] else 'white'
                ax.plot(x, y, color=lay['color'], lw=st['lw'], ls=vs['ls'],
                        marker=st['marker'], ms=MARKER_SIZE, mfc=face, clip_on=False)
                values += y
        ax.grid(False)
    # the row's k_max on the right side, rotated like a y label
    right = grid[row, -1]
    right.yaxis.set_label_position('right')
    right.set_ylabel(rf'$k_\mathrm{{max}} = {cut}$', rotation=90, fontsize=9, labelpad=5)
titles = [grid[0, col].set_title(name, fontsize=9, pad=4)
          for col, (_, name) in enumerate(COLUMNS)]

y_lo = np.floor(min(values)/Y_FLOOR_STEP)*Y_FLOOR_STEP
y_hi = np.ceil(max(values)/Y_STEP)*Y_STEP
y_ticks = [v for v in np.round(np.arange(0.0, y_hi + 1e-9, Y_STEP), 6) if v >= y_lo - 1e-9]
x_ticks = data['x_ticks']
x_span = x_ticks[-1] - x_ticks[0]
for ax in grid.flat:
    ax.set_ylim(y_lo - Y_PAD, y_hi + Y_PAD)
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.set_xlim(x_ticks[0] - X_MARGIN*x_span, x_ticks[-1] + X_MARGIN*x_span)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_minor_locator(NullLocator())
for ax in grid[-1]:
    ax.set_xticklabels(data['x_ticklabels'])
xlabel = fig.supxlabel(data['xlabel'], fontsize=10)

style_legend = fig.legend(
    handles=[Line2D([], [], color='0.25', lw=1.3, ls=vs['ls'], marker='o', ms=MARKER_SIZE,
                    mfc='0.25' if vs['filled'] else 'white', label=vs['label'])
             for vs in VARIANT_STYLE.values()],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.6, columnspacing=1.6, handletextpad=0.5, borderaxespad=0.0)
layout_legend = fig.legend(
    handles=[Line2D([], [], color=lay['color'], lw=LAYOUT_STYLE[lay['name']]['lw'],
                    marker=LAYOUT_STYLE[lay['name']]['marker'], ms=MARKER_SIZE,
                    label=lay['label']) for lay in data['layouts']],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=8,
    handlelength=2.2, columnspacing=1.6, handletextpad=0.5, borderaxespad=0.0)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
top = max(t.get_window_extent(renderer).y1 for t in titles)
top = inv.transform((0.0, top))[1]
layout_legend.set_bbox_to_anchor((0.5, top + LEGEND_GAP), transform=fig.transFigure)
fig.canvas.draw()
layout_top = inv.transform((0.0, layout_legend.get_window_extent(renderer).y1))[1]
style_legend.set_bbox_to_anchor((0.5, layout_top + 0.5*LEGEND_GAP), transform=fig.transFigure)
# the shared y label just left of the widest tick label
fig.canvas.draw()
tick_left = min(label.get_window_extent(renderer).x0 for ax in grid[:, 0]
                for label in ax.get_yticklabels() if label.get_text())
tick_left = inv.transform((tick_left, 0.0))[0]
supylabel = fig.supylabel(r'$\mathrm{FoM}/\mathrm{FoM}_\mathrm{current}$', x=tick_left - 0.008,
                          ha='right', fontsize=10)
check_clear(fig, list(grid.flat), [style_legend, layout_legend, supylabel, xlabel, *titles,
                                   *[ax.yaxis.label for ax in grid[:, -1]]])
fig.savefig(os.path.join(HERE, f'{NAME}.pdf'))
fig.savefig(os.path.join(HERE, f'{NAME}.png'), dpi=300)
plt.close(fig)
