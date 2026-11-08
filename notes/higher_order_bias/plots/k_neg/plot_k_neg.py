"""Redraw k_neg.pdf from data.json: k_neg, the wavenumber at which the
one-loop galaxy power spectrum P_gg in CosmoLike's subtracted convention first
turns negative, against Delta_2, the offset of the quadratic bias b_2 from the
Lazeyras et al. fit (arXiv:1511.01096), for two BGS redshifts and one ELG
redshift.

The subtraction of the low-k constants of the quadratic templates leaves the
high-k constant P_inf = -(sigma_4/2) D^4 (b_2 + 2 b_{s^2}/3)^2 <= 0 (sigma_4
the low-k constant, D the growth factor, b_{s^2} the tidal bias on
co-evolution), so P_gg turns negative where P_inf outweighs the decaying
positive terms. Where b_2 + 2 b_{s^2}/3 vanishes, near Delta_2 = +0.9, the
constant vanishes and k_neg runs off the top of the panel. data.json, written
by ../make_figure_data.py, holds the curves (null where P_gg stays positive
to 40 h/Mpc), the biases and the values at Delta_2 = -1, -0.5, 0, 0.5, 1.
Run from any folder: python plot_k_neg.py
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
# one color per tracer type, the same in every figure of the note; the two BGS
# redshifts differ by line style
TRACER_COLORS = {'BGS': '#1b9e77', 'LRG': '#d95f02', 'ELG': '#7570b3'}
LINE_STYLES = {0.1: '-', 0.3: '--', 1.1: '-'}
# k_neg spans almost two decades before it runs off near the zero of
# b_2 + 2 b_{s^2}/3, so the y axis is logarithmic
Y_RANGE = (0.1, 10.0)
# a label may not come closer to a drawn line or marker than this, in points
CLEARANCE_PT = 1.5


def drawn_points(fig, ax):
    '''Return the display coordinates of what ax draws.

    Lines are straight between their display-space vertices, so each segment
    is sampled linearly there; a reference line (axhline) goes through its own
    blended transform the same way. Points outside the axes are not drawn and
    are dropped.

    Returns:
      (line_points [n, 2], markers): markers is a list of (center [2], radius),
      both in display pixels
    '''
    box = ax.get_window_extent()
    line_points = []
    markers = []
    for line in ax.get_lines():
        line_xy = np.asarray(line.get_xydata(), dtype=float)
        vertices = line.get_transform().transform(line_xy)
        finite = np.all(np.isfinite(vertices), axis=1)
        if line.get_linestyle() not in ('None', 'none', '', ' '):
            fraction = np.linspace(0.0, 1.0, 40)[:, None]
            for i in range(len(vertices) - 1):
                if finite[i] and finite[i + 1]:
                    step = vertices[i + 1] - vertices[i]
                    line_points.append(vertices[i] + fraction*step)
        if line.get_marker() not in ('None', 'none', '', ' ', None):
            radius = 0.5*line.get_markersize()*fig.dpi/72.0
            for center in vertices[finite]:
                markers.append((center, radius))
    if len(line_points) > 0:
        points = np.vstack(line_points)
    else:
        points = np.zeros((0, 2))
    inside = ((points[:, 0] >= box.x0) & (points[:, 0] <= box.x1)
              & (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1))
    return points[inside], markers


def assert_labels_clear(fig, labels):
    '''Raise when a label or legend touches a drawn line, reference line or marker.

    Arguments:
      fig    = the figure, after every artist is added
      labels = list of (name, artist) of the in-figure texts and legends
    Raises:
      RuntimeError naming the label and the panel it touches
    '''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = CLEARANCE_PT*fig.dpi/72.0
    for name, artist in labels:
        box = artist.get_window_extent(renderer)
        x0, x1 = box.x0 - pad, box.x1 + pad
        y0, y1 = box.y0 - pad, box.y1 + pad
        for i_ax, ax in enumerate(fig.axes):
            points, markers = drawn_points(fig, ax)
            hit = ((points[:, 0] > x0) & (points[:, 0] < x1)
                   & (points[:, 1] > y0) & (points[:, 1] < y1))
            if np.any(hit):
                raise RuntimeError(f'the label "{name}" touches a line in panel {i_ax}')
            for center, radius in markers:
                nearest_x = min(max(center[0], x0), x1)
                nearest_y = min(max(center[1], y0), y1)
                if np.hypot(center[0] - nearest_x, center[1] - nearest_y) < radius:
                    raise RuntimeError(f'the label "{name}" touches a marker in '
                                       f'panel {i_ax}')


with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
delta2 = np.array(data['delta2'])

fig, ax = plt.subplots(figsize=(3.5, 2.3))
handles = []
for item in data['bins']:
    # null marks an offset at which P_gg stays positive on the whole grid
    k_neg = []
    for value in item['k_neg_h_per_mpc']:
        if value is None:
            k_neg.append(np.nan)
        else:
            k_neg.append(value)
    style = {'color': TRACER_COLORS[item['name']], 'ls': LINE_STYLES[item['z']],
             'lw': 1.5}
    ax.plot(delta2, k_neg, **style)
    label = f"{item['name']}, $z = {item['z']:g}$, $b_1 = {item['b1']:.2f}$"
    handles.append(Line2D([], [], label=label, **style))
ax.set_yscale('log')
ax.set_ylim(*Y_RANGE)
ax.set_yticks([0.1, 0.3, 1, 3, 10])
ax.set_yticklabels(['0.1', '0.3', '1', '3', '10'])
ax.set_xlim(delta2[0], delta2[-1])
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_xticklabels(['$-1$', '$-0.5$', '0', '0.5', '1'])
ax.set_xlabel(r'$\Delta_2 = b_2 - b_2^\mathrm{fit}(b_1)$', fontsize=10)
ax.set_ylabel(r'$k_\mathrm{neg}$ [$h\,\mathrm{Mpc}^{-1}$]', fontsize=10)
# the bins in the empty upper left: there b_2 sits below the calibration and
# every k_neg is small
legend = ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=8,
                   handlelength=1.8, handletextpad=0.5, labelspacing=0.35,
                   borderaxespad=0.4)
assert_labels_clear(fig, [('legend', legend)])
fig.savefig(os.path.join(HERE, 'k_neg.pdf'))
fig.savefig(os.path.join(HERE, 'k_neg.png'), dpi=300)
