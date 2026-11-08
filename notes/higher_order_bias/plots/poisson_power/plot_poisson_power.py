"""Redraw poisson_power.pdf from data.json: the Poisson power 1/nbar of the
project's thin-bin densities, in percent of the linear-bias model
b_1^2 P_mm (b_1 the linear bias, P_mm the halofit matter power), against the
wavenumber k, for the four DESI-like tracers of the term_sizes figure.

nbar is the tracer number density of the thin bin nearest each tracer
redshift. The covariance carries this power as noise; it sets the scale of
the stochastic amplitude SN_0, whose model A prior width is twice it.
data.json, written by ../make_figure_data.py, also holds nbar, 1/nbar and the
values at the wavenumbers the note's text reads.
Run from any folder: python plot_poisson_power.py
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
# one color per tracer type, the same in every figure of the note; the two LRG
# redshifts differ by line style
TRACER_COLORS = {'BGS-like': '#1b9e77', 'LRG-like': '#d95f02', 'ELG-like': '#7570b3'}
LINE_STYLES = {0.3: '-', 0.5: '-', 0.8: '--', 1.3: '-'}
# the values span 2.5 decades, so the y axis is logarithmic
Y_RANGE = (0.2, 400.0)
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
k = np.array(data['k_h_per_mpc'])
tracers = data['tracers']

fig, ax = plt.subplots(figsize=(3.5, 2.1))
handles = []
for tracer in tracers:
    style = {'color': TRACER_COLORS[tracer['name']], 'ls': LINE_STYLES[tracer['z']],
             'lw': 1.5}
    ax.plot(k, tracer['percent_of_b1sq_pmm'], **style)
    label = f"{tracer['name']}, $z = {tracer['z']:g}$"
    handles.append(Line2D([], [], label=label, **style))
ax.set_yscale('log')
ax.set_ylim(*Y_RANGE)
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(['1', '10', '100'])
ax.set_xlim(k[0], k[-1])
ax.set_xticks([0.05, 0.1, 0.15, 0.2])
ax.set_xticklabels(['0.05', '0.1', '0.15', '0.2'])
ax.set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]', fontsize=10)
ax.set_ylabel(r'$(1/\bar{n})\,/\,b_1^2 P_\mathrm{mm}$  [%]', fontsize=10)
legend = ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
                   frameon=False, fontsize=8, handlelength=2.0, columnspacing=1.2,
                   handletextpad=0.5, labelspacing=0.35)
assert_labels_clear(fig, [('legend', legend)])
fig.savefig(os.path.join(HERE, 'poisson_power.pdf'))
fig.savefig(os.path.join(HERE, 'poisson_power.png'), dpi=300)
