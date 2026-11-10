"""Redraw fog_pair_overlap.pdf from data.json: how far the Fingers-of-God
pair rule of the DESI data vector holds when the velocity dispersions
sigma_v move away from their fiducials.

One column of a two-column paper, one panel with a log y axis: the largest
blurred-kernel overlap O_ij = int n_i n_j dz / sqrt(int n_i^2 dz int n_j^2 dz)
over the pairs of bins outside the layout's pair list, against the offset
k of every sigma_v group from its fiducial, in units of its prior width
Delta sigma_v times a width factor s: sigma_v = sigma_v^fid + k s Delta
sigma_v. s = 1 (the current priors) solid with filled markers, s = 4 (the
order of the widths that let sigma_v absorb a DESI-size alpha_2) dashed with
open markers; markers at k = 0, 1, 2, 3. Dotted: the rule's threshold
O_ij = 1e-6, above which a pair belongs in the data vector. Overlaps below
FLOOR (zero included: kernels that do not meet) are drawn on the floor.
Each layout has a color, a line width and a marker.
Run from any folder: python plot_fog_pair_overlap.py
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
# the width factors drawn: line style and filled or open markers
S_STYLE = {1: {'ls': '-', 'filled': True}, 4: {'ls': (0, (4, 2)), 'filled': False}}
K_MAX = 3.0
K_MARKERS = [0, 1, 2, 3]
# overlaps below FLOOR (zero included) sit on the floor of the log axis
FLOOR = 1.0e-12
Y_TOP = 1.0
THRESHOLD_STYLE = {'color': 'black', 'ls': ':', 'lw': 1.0}
LEGEND_GAP = 0.004

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
t = np.array(data['t'], dtype=float)


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
        steps = np.linspace(0.0, 1.0, 30)[:, None]
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
            raise RuntimeError(f'fog_pair_overlap: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'fog_pair_overlap: {type(artists[i]).__name__} {i} overlaps '
                                   f'{type(artists[j]).__name__} {j}')
    shown = [*axes[0].get_yticklabels(), *axes[0].get_xticklabels()]
    ticks = [label.get_window_extent(renderer) for label in shown
             if label.get_text() and label.get_visible()]
    for i in range(len(ticks)):
        for j in range(i + 1, len(ticks)):
            if ticks[i].overlaps(ticks[j]):
                raise RuntimeError('fog_pair_overlap: two tick labels touch')


fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.axhline(data['overlap_min'], **THRESHOLD_STYLE)
for lay in data['layouts']:
    st = LAYOUT_STYLE[lay['name']]
    values = np.maximum(np.array(lay['max_unlisted'], dtype=float), FLOOR)
    for s, ss in S_STYLE.items():
        keep = t <= K_MAX*s + 1.0e-9
        x = t[keep]/s
        y = values[keep]
        ax.plot(x, y, color=lay['color'], lw=st['lw'], ls=ss['ls'], clip_on=False)
        at = [int(np.argmin(np.abs(x - k))) for k in K_MARKERS]
        ax.plot(x[at], y[at], color=lay['color'], ls='none', marker=st['marker'],
                ms=MARKER_SIZE, mfc=lay['color'] if ss['filled'] else 'white', clip_on=False)
ax.set_yscale('log')
ax.set_ylim(FLOOR, Y_TOP)
ax.yaxis.set_major_locator(FixedLocator([1e-12, 1e-9, 1e-6, 1e-3, 1.0]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels([r'$\leq 10^{-12}$', r'$10^{-9}$', r'$10^{-6}$', r'$10^{-3}$', r'$1$'])
ax.set_xlim(-0.08, K_MAX + 0.08)
ax.xaxis.set_major_locator(FixedLocator(K_MARKERS))
ax.set_xlabel(r'$k$: $\sigma_v = \sigma_v^\mathrm{fid} + k\,s\,\Delta\sigma_v$', fontsize=10)
ax.set_ylabel(r'largest unlisted $O_{ij}$', fontsize=10)
ax.grid(False)

style_legend = fig.legend(
    handles=[Line2D([], [], color='0.25', lw=1.3, ls=S_STYLE[1]['ls'], marker='o',
                    ms=MARKER_SIZE, label=r'$s = 1$'),
             Line2D([], [], color='0.25', lw=1.3, ls=S_STYLE[4]['ls'], marker='o',
                    ms=MARKER_SIZE, mfc='white', label=r'$s = 4$')],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.6, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0)
layout_legend = fig.legend(
    handles=[Line2D([], [], color=lay['color'], lw=LAYOUT_STYLE[lay['name']]['lw'],
                    marker=LAYOUT_STYLE[lay['name']]['marker'], ms=MARKER_SIZE,
                    label=lay['label']) for lay in data['layouts']],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.2, columnspacing=1.2, handletextpad=0.5, labelspacing=0.25,
    borderaxespad=0.0)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
top = inv.transform((0.0, ax.get_window_extent(renderer).y1))[1]
layout_legend.set_bbox_to_anchor((0.5, top + LEGEND_GAP), transform=fig.transFigure)
fig.canvas.draw()
layout_top = inv.transform((0.0, layout_legend.get_window_extent(renderer).y1))[1]
style_legend.set_bbox_to_anchor((0.5, layout_top + 0.5*LEGEND_GAP), transform=fig.transFigure)
check_clear(fig, [ax], [style_legend, layout_legend, ax.xaxis.label, ax.yaxis.label])
fig.savefig(os.path.join(HERE, 'fog_pair_overlap.pdf'))
fig.savefig(os.path.join(HERE, 'fog_pair_overlap.png'), dpi=300)
plt.close(fig)
