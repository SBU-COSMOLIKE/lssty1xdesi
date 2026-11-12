"""Redraw pair_axes_attribution_wide.pdf from data.json: the attribution fits
of plots/pair_axes_attribution, measured against wide bins.

A full-width grid: rows are the three layouts whose gain is large (named on
the right), columns k_max = 0.075, 0.1 and 0.2 h/Mpc (named above). Each
panel draws, against g (pairs measured up to g bin widths apart; g = 0 is
the current list), FoM(w0, wa) of the fiducial model with sigma_v fitted
over the same figure of merit of the Delta z = 0.2 layout on its current
list, each curve with the same block of nuisance parameters held at the
fiducial in numerator and denominator: none (every parameter free), the
correlated residuals of the per-bin linear biases b1, the second-order and
tidal biases b2 and bs2 with their tilts, the level and slope of each
group's b1 amplitude, or every nuisance parameter. The y axis is
logarithmic and shared by every panel. Each curve has a color, a line style,
a line width and a marker. FoM(Omega_m, 10^9 A_s) is in data.json. k_max
is the z = 0 value of a per-bin cut at k_max/D(zbar_i).
Run from any folder: python plot_pair_axes_attribution.py
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
NAME = 'pair_axes_attribution_wide'
SV = 'sigma_v free'
KEY = 'fom_ratio_to_wide'
CURVES = [  # block, label, color, line style, width, marker (drawn in this order)
    ('b1 residuals fixed (level, slope free)', r'$b_1$ residuals fixed', '#56B4E9', '-', 3.4, 'v'),
    ('all free', 'every parameter free', 'black', '-', 1.6, 'o'),
    ('b2 bs2 fixed', r'$b_2$, $b_{s^2}$ fixed', '#0072B2', (0, (5, 1.5, 1, 1.5)), 1.4, 'D'),
    ('b1 level and slope fixed (residuals free)', r'$b_1$ level and slope fixed', '#D55E00',
     (0, (4, 2)), 1.6, 's'),
    ('all nuisance fixed', 'every nuisance parameter fixed', '#E69F00', (0, (7, 2)), 2.2, '^'),
]
REFERENCE = {'color': '0.6', 'lw': 0.8}
MARKER_SIZE = 3.5
PANEL_HEIGHT = 1.15
FIG_WIDTH = 6.4
LEGEND_GAP = 0.006
Y_TICKS = [1, 2, 5, 10, 20, 50, 100]
Y_PAD = 1.12  # factor of headroom on the log axis, below and above

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
cuts = data['kmax_h_per_Mpc']
layouts = data['layouts']


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


fig, grid = plt.subplots(len(layouts), len(cuts), figsize=(FIG_WIDTH, PANEL_HEIGHT*len(layouts)),
                         sharex=True, sharey=True, gridspec_kw={'hspace': 0.06, 'wspace': 0.06})
values = []
g_max = 0
for row, lay in enumerate(layouts):
    g = [p['g'] for p in lay['points']]
    g_max = max(g_max, max(g))
    for col, cut in enumerate(cuts):
        ax = grid[row, col]
        for block, _, color, ls, lw, marker in CURVES:
            y = [p[cut][block][KEY] for p in lay['points']]
            ax.plot(g, y, color=color, ls=ls, lw=lw, marker=marker, ms=MARKER_SIZE, clip_on=False)
            values += y
        ax.grid(False)
    right = grid[row, -1]
    right.yaxis.set_label_position('right')
    right.set_ylabel(lay['label'], rotation=90, fontsize=9, labelpad=5)
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
xlabel = fig.supxlabel(r'$g$: pairs measured up to $g$ bin widths apart', fontsize=10)

legend = fig.legend(
    handles=[Line2D([], [], color=color, ls=ls, lw=lw, marker=marker, ms=MARKER_SIZE, label=label)
             for _, label, color, ls, lw, marker in CURVES],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=8,
    handlelength=3.0, columnspacing=1.6, handletextpad=0.5, labelspacing=0.3,
    borderaxespad=0.0)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
top = max(t.get_window_extent(renderer).y1 for t in titles)
legend.set_bbox_to_anchor((0.5, inv.transform((0.0, top))[1] + LEGEND_GAP),
                          transform=fig.transFigure)
fig.canvas.draw()
tick_left = min(label.get_window_extent(renderer).x0 for ax in grid[:, 0]
                for label in ax.get_yticklabels() if label.get_text())
tick_left = inv.transform((tick_left, 0.0))[0]
supylabel = fig.supylabel(r'$\mathrm{FoM}(w_0, w_a)/\mathrm{FoM}(\Delta z = 0.2,\ \mathrm{current})$',
                          x=tick_left - 0.008, ha='right', fontsize=10)
check_clear(fig, list(grid.flat), [legend, supylabel, xlabel, *titles,
                                   *[ax.yaxis.label for ax in grid[:, -1]]])
fig.savefig(os.path.join(HERE, f'{NAME}.pdf'))
fig.savefig(os.path.join(HERE, f'{NAME}.png'), dpi=300)
plt.close(fig)
