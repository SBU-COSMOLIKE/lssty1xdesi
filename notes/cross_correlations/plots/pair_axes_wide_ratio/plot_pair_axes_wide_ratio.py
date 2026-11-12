"""Redraw pair_axes_wide_ratio.pdf from data.json: what thin bins gain over
wide bins, and how that gain grows as cross pairs beyond the current list
are measured.

A full-width grid: rows FoM(w0, wa) and FoM(Omega_m, 10^9 A_s) (named on
the right), columns k_max = 0.075, 0.1 and 0.2 h/Mpc (named above). Each
panel draws, against g (every pair whose raw n(z) ranges are at most g bin
widths apart is measured; g = 0 is the current list), the figure of merit
of the fiducial model with sigma_v fitted over the same figure of merit of
the Delta z = 0.2 layout on its current list at the same cut. Delta z = 0.2
itself, within 2% of one by construction, is not drawn (its ratios are in
data.json). The y axis is logarithmic (the ratios span a few to a few tens)
and shared by every panel. Color, line width and marker: the layout. k_max
is the z = 0 value of a per-bin cut at k_max/D(zbar_i).
Run from any folder: python plot_pair_axes_wide_ratio.py
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
NAME = 'pair_axes_wide_ratio'
SV = 'sigma_v free'
KEY = 'fom_ratio'
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
    'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
    'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
    'Delta z = 0.2': {'lw': 0.9, 'marker': '^'},
}
ROWS = [('fom_ratio_to_wide', r'$\mathrm{FoM}(w_0, w_a)$'),
        ('fom_om_as_ratio_to_wide', r'$\mathrm{FoM}(\Omega_\mathrm{m}, 10^9 A_\mathrm{s})$')]
Y_TICKS = [1, 2, 3, 5, 10, 20, 30, 50, 100]
# the wide-bin reference, the denominator of every ratio, is not drawn
NOT_DRAWN = ('Delta z = 0.2',)
REFERENCE = {'color': '0.6', 'lw': 0.8}
MARKER_SIZE = 3.5
PANEL_HEIGHT = 1.35
FIG_WIDTH = 6.4
LEGEND_GAP = 0.006
Y_PAD = 1.12  # factor of headroom on the log axis, below and above

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
cuts = data['kmax_h_per_Mpc']
layouts = [lay for lay in data['layouts'] if lay['name'] not in NOT_DRAWN]


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
    shown = [label for ax in axes for label in (*ax.get_yticklabels(), *ax.get_xticklabels())]
    ticks = [label.get_window_extent(renderer) for label in shown
             if label.get_text() and label.get_visible()]
    for i in range(len(ticks)):
        for j in range(i + 1, len(ticks)):
            if ticks[i].overlaps(ticks[j]):
                raise RuntimeError(f'{NAME}: two tick labels touch')



fig, grid = plt.subplots(len(ROWS), len(cuts), figsize=(FIG_WIDTH, PANEL_HEIGHT*len(ROWS)),
                         sharex=True, sharey=True, gridspec_kw={'hspace': 0.06, 'wspace': 0.06})
values = []
g_max = 0
for row, (key, label) in enumerate(ROWS):
    for col, cut in enumerate(cuts):
        ax = grid[row, col]
        for lay in layouts:
            st = LAYOUT_STYLE[lay['name']]
            g = [p['g'] for p in lay['points']]
            g_max = max(g_max, max(g))
            y = [p[cut][key] for p in lay['points']]
            ax.plot(g, y, color=lay['color'], lw=st['lw'], marker=st['marker'], ms=MARKER_SIZE,
                    clip_on=False)
            values += y
        ax.grid(False)
    right = grid[row, -1]
    right.yaxis.set_label_position('right')
    right.set_ylabel(label, rotation=90, fontsize=9, labelpad=5)
titles = [grid[0, col].set_title(rf'$k_\mathrm{{max}} = {cut}$', fontsize=9, pad=4)
          for col, cut in enumerate(cuts)]

y_lo, y_hi = min(values)/Y_PAD, max(values)*Y_PAD
ticks = [t for t in Y_TICKS if y_lo <= t <= y_hi]
for ax in grid.flat:
    ax.set_yscale('log')
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xlim(-0.25, g_max + 0.25)
    ax.xaxis.set_major_locator(FixedLocator(list(range(g_max + 1))))
    ax.xaxis.set_minor_locator(NullLocator())
fig.canvas.draw()
tick_bottom = min(t.get_window_extent(fig.canvas.get_renderer()).y0 for ax in grid[-1]
                  for t in ax.get_xticklabels() if t.get_text())
tick_bottom = fig.transFigure.inverted().transform((0.0, tick_bottom))[1]
xlabel = fig.supxlabel(r'$g$: pairs measured up to $g$ bin widths apart', fontsize=10,
                       y=tick_bottom - 0.015, va='top')

legend = fig.legend(
    handles=[Line2D([], [], color=lay['color'], lw=LAYOUT_STYLE[lay['name']]['lw'],
                    marker=LAYOUT_STYLE[lay['name']]['marker'], ms=MARKER_SIZE,
                    label=lay['label']) for lay in layouts],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=8,
    handlelength=2.2, columnspacing=1.6, handletextpad=0.5, borderaxespad=0.0)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
top = max(t.get_window_extent(renderer).y1 for t in titles)
legend.set_bbox_to_anchor((0.5, inv.transform((0.0, top))[1] + LEGEND_GAP),
                          transform=fig.transFigure)
fig.canvas.draw()
tick_left = min(t.get_window_extent(renderer).x0 for ax in grid[:, 0]
                for t in ax.get_yticklabels() if t.get_text())
tick_left = inv.transform((tick_left, 0.0))[0]
supylabel = fig.supylabel(r'$\mathrm{FoM}/\mathrm{FoM}(\Delta z = 0.2,\ \mathrm{current})$',
                          x=tick_left - 0.008, ha='right', fontsize=10)
check_clear(fig, list(grid.flat), [legend, supylabel, xlabel, *titles,
                                   *[ax.yaxis.label for ax in grid[:, -1]],
                                   *[t for ax in grid[-1] for t in ax.get_xticklabels()
                                     if t.get_text()]])
fig.savefig(os.path.join(HERE, f'{NAME}.pdf'))
fig.savefig(os.path.join(HERE, f'{NAME}.png'), dpi=300)
plt.close(fig)
