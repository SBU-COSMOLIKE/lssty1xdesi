"""Redraw summary_pair_wide_ratio.pdf from data.json: what thin bins gain over
wide bins, and how that gain grows as cross pairs beyond the current list
are measured.

One column of a two-column paper, two stacked panels (named on the right):
FoM(w0, wa) and FoM(Omega_m, 10^9 A_s) of the fiducial model with sigma_v
fitted, for each thin layout along the gap axis (every pair whose raw n(z)
ranges are at most g bin widths apart; g = 0 is the current list), over the
same figure of merit of the Delta z = 0.2 layout on its current list at the
same cut. The y axis is logarithmic and shared by both panels. Color, line
width and marker: the layout; solid lines with filled markers: k_max = 0.2
h/Mpc; dashed with open markers: 0.075. k_max is the z = 0 value of a
per-bin cut at k_max/D(zbar_i).
Run from any folder: python plot_summary_pair_wide_ratio.py
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
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
    'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
    'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
    'Delta z = 0.2': {'lw': 0.9, 'marker': '^'},
}
MARKER_SIZE = 3.5
CUT_STYLE = {'0.2': {'ls': '-', 'filled': True, 'label': r'$k_\mathrm{max} = 0.2$'},
             '0.075': {'ls': (0, (4, 2)), 'filled': False,
                       'label': r'$k_\mathrm{max} = 0.075$'}}
NAME = 'summary_pair_wide_ratio'
PANEL_HEIGHT = 1.3
FIG_WIDTH = 3.5
LEGEND_GAP = 0.006
ROWS = [('fom_ratio_to_wide', r'$(w_0, w_a)$'),
        ('fom_om_as_ratio_to_wide', r'$(\Omega_\mathrm{m}, 10^9 A_\mathrm{s})$')]
Y_TICKS = [1, 2, 3, 5, 10, 20, 30, 50, 100]
Y_PAD = 1.12  # factor of headroom on the log axis, below and above

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
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
    shown = [*[label for ax in axes for label in ax.get_yticklabels()],
             *[label for ax in axes for label in ax.get_xticklabels()]]
    ticks = [label.get_window_extent(renderer) for label in shown
             if label.get_text() and label.get_visible()]
    for i in range(len(ticks)):
        for j in range(i + 1, len(ticks)):
            if ticks[i].overlaps(ticks[j]):
                raise RuntimeError(f'{NAME}: two tick labels touch')




fig, axes = plt.subplots(len(ROWS), 1, figsize=(FIG_WIDTH, PANEL_HEIGHT*len(ROWS)),
                         sharex=True, sharey=True, gridspec_kw={'hspace': 0.06})
values = []
g_max = 0
for ax, (key, label) in zip(axes, ROWS):
    for lay in layouts:
        st = LAYOUT_STYLE[lay['name']]
        g_max = max(g_max, max(lay['g']))
        for cut, cs in CUT_STYLE.items():
            y = lay[cut][key]
            ax.plot(lay['g'], y, color=lay['color'], lw=st['lw'], ls=cs['ls'],
                    marker=st['marker'], ms=MARKER_SIZE,
                    mfc=lay['color'] if cs['filled'] else 'white', clip_on=False)
            values += y
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(label, rotation=90, fontsize=9, labelpad=5)
    ax.grid(False)
y_lo, y_hi = min(values)/Y_PAD, max(values)*Y_PAD
ticks = [t for t in Y_TICKS if y_lo <= t <= y_hi]
for ax in axes:
    ax.set_yscale('log')
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xlim(-0.25, g_max + 0.25)
    ax.xaxis.set_major_locator(FixedLocator(list(range(g_max + 1))))
    ax.xaxis.set_minor_locator(NullLocator())
axes[-1].set_xlabel(r'$g$: pairs measured up to $g$ bin widths apart', fontsize=9)

cut_legend = fig.legend(
    handles=[Line2D([], [], color='0.25', lw=1.3, ls=cs['ls'], marker='o', ms=MARKER_SIZE,
                    mfc='0.25' if cs['filled'] else 'white', label=cs['label'])
             for cs in CUT_STYLE.values()],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.6, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0)
layout_legend = fig.legend(
    handles=[Line2D([], [], color=lay['color'], lw=LAYOUT_STYLE[lay['name']]['lw'],
                    marker=LAYOUT_STYLE[lay['name']]['marker'], ms=MARKER_SIZE,
                    label=lay['label']) for lay in layouts],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=8,
    handlelength=2.2, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
top = inv.transform((0.0, axes[0].get_window_extent(renderer).y1))[1]
layout_legend.set_bbox_to_anchor((0.5, top + LEGEND_GAP), transform=fig.transFigure)
fig.canvas.draw()
layout_top = inv.transform((0.0, layout_legend.get_window_extent(renderer).y1))[1]
cut_legend.set_bbox_to_anchor((0.5, layout_top + 0.5*LEGEND_GAP), transform=fig.transFigure)
fig.canvas.draw()
tick_left = min(t.get_window_extent(renderer).x0 for ax in axes
                for t in ax.get_yticklabels() if t.get_text())
tick_left = inv.transform((tick_left, 0.0))[0]
supylabel = fig.supylabel(r'$\mathrm{FoM}/\mathrm{FoM}(\Delta z = 0.2,\ \mathrm{current})$',
                          x=tick_left - 0.01, ha='right', fontsize=9)
check_clear(fig, list(axes), [cut_legend, layout_legend, supylabel, axes[-1].xaxis.label,
                              *[ax.yaxis.label for ax in axes]])
fig.savefig(os.path.join(HERE, f'{NAME}.pdf'))
fig.savefig(os.path.join(HERE, f'{NAME}.png'), dpi=300)
plt.close(fig)
