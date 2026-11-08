"""Redraw draft_bias_check.pdf from data.json: the draft paper's linear-bias
column read as the DESI DR1 sampling combination b_1 sigma_8(z_eff), with b_1
the Eulerian linear bias and sigma_8(z_eff) the rms linear overdensity in
8 Mpc/h spheres at the chunk's DESI DR1 effective redshift.

Drawn against redshift z: the project fiducial b_1 = B_g/D(z) over each
chunk (B_g the chunk amplitude, D the linear growth factor), the draft value
as printed (open markers) and the draft value divided by sigma_8(z_eff)
(filled markers). The draft has no ELG1 row. data.json, written by
../make_figure_data.py, holds sigma_8(z_eff) and the ratios of the implied
to the project bias.
Run from any folder: python plot_draft_bias_check.py
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
PROJECT_STYLE = {'color': '0.55', 'lw': 1.6}
DIVIDED_STYLE = {'marker': 'o', 'ms': 5.0, 'ls': 'none', 'color': 'black'}
PRINTED_STYLE = {'marker': 'o', 'ms': 5.0, 'ls': 'none', 'color': 'black',
                 'markerfacecolor': 'none', 'markeredgewidth': 1.1}
# the DESI chunk edges, which label the redshift axis
Z_TICKS = [0.1, 0.4, 0.6, 0.8, 1.1, 1.6]
# the chunk name sits this far below its open marker, in points
NAME_OFFSET_PT = 6.0
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
chunks = data['chunks']

fig, ax = plt.subplots(figsize=(3.5, 2.3))
labels = []
for chunk in chunks:
    ax.plot(chunk['z'], chunk['project_b1_curve'], **PROJECT_STYLE)
    ax.plot(chunk['z_eff'], chunk['draft_value_printed'], **PRINTED_STYLE)
    ax.plot(chunk['z_eff'], chunk['draft_value_over_sigma8'], **DIVIDED_STYLE)
    # the chunk name under its open marker, offset in points so it keeps its
    # distance at any figure size
    text = ax.annotate(chunk['name'], xy=(chunk['z_eff'], chunk['draft_value_printed']),
                       xytext=(0.0, -NAME_OFFSET_PT), textcoords='offset points',
                       ha='center', va='top', fontsize=8)
    labels.append((chunk['name'], text))
ax.set_xlim(0.05, 1.65)
ax.set_xticks(Z_TICKS)
ax.set_xticklabels([f'{z:g}' for z in Z_TICKS])
ax.set_ylim(0.3, 3.0)
ax.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5])
ax.grid(False, axis='x')
ax.set_xlabel(r'$z$', fontsize=10)
ax.set_ylabel(r'linear bias $b_1$', fontsize=10)

handles = [
    Line2D([], [], label=r'project $B_g/D(z)$', **PROJECT_STYLE),
    Line2D([], [], label=r'draft value / $\sigma_8(z_\mathrm{eff})$', **DIVIDED_STYLE),
    Line2D([], [], label='draft value as printed', **PRINTED_STYLE),
]
legend = ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
                   frameon=False, fontsize=8, handlelength=1.8, columnspacing=1.2,
                   handletextpad=0.5, labelspacing=0.35)
labels.append(('legend', legend))
assert_labels_clear(fig, labels)
fig.savefig(os.path.join(HERE, 'draft_bias_check.pdf'))
fig.savefig(os.path.join(HERE, 'draft_bias_check.png'), dpi=300)
