"""Redraw coevolution_biases.pdf from data.json: the fiducial galaxy biases of
the six DESI chunks against redshift z, in the CCL convention. One wide
panel per bias, stacked, all on one y range:

  b_1      = B_g/D(z), the linear bias, B_g the chunk amplitude and D the
             linear growth factor (D(0) = 1);
  b_2      = the Lazeyras et al. fit (arXiv:1511.01096),
             0.412 - 2.143 b_1 + 0.929 b_1^2 + 0.008 b_1^3;
  b_{s^2}  = the co-evolution tidal bias, -(4/7)(b_1 - 1);
  b_3nl    = the co-evolution third-order bias, (32/315)(b_1 - 1).

Each chunk is a curve over its redshift range, colored by tracer, with a
marker at its DESI DR1 effective redshift. data.json, written by
../make_figure_data.py, holds every curve, the values at the effective
redshifts and b_{s^2} at the chunk edges.
Run from any folder: python plot_coevolution_biases.py
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
# one color per tracer type, the same in every figure of the note
# colorblind-safe colors (Okabe-Ito, without its green; no red or green)
TRACER_COLORS = {'bgs': '#0072B2', 'lrg': '#E69F00', 'elg': '#CC79A7'}
TRACER_NAMES = {'bgs': 'BGS', 'lrg': 'LRG', 'elg': 'ELG'}
# each tracer also has its own line style and marker shape, so the curves
# read in black and white
TRACER_STYLE = {'bgs': {'ls': '-', 'marker': 'o'}, 'lrg': {'ls': '--', 'marker': 's'},
                'elg': {'ls': '-.', 'marker': 'D'}}
# the panels, top to bottom: data.json key and the bias name on the right side
PANELS = [('b1', r'$b_1$'), ('b2', r'$b_2$'), ('bs2', r'$b_{s^2}$'),
          ('b3nl', r'$b_{3\mathrm{nl}}$')]
CURVE_STYLE = {'lw': 1.6}
MARKER_STYLE = {'ms': 4.0, 'ls': 'none'}
ZERO_LINE = {'color': '0.6', 'lw': 0.6}
# the DESI chunk edges, which label the redshift axis
Z_TICKS = [0.1, 0.4, 0.6, 0.8, 1.1, 1.6]
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


def assert_ylabel_clear(fig, ylabel):
    '''Raise when the shared y label overlaps a y tick label of any panel.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_box = ylabel.get_window_extent(renderer)
    for i_ax, ax in enumerate(fig.axes):
        for tick_label in ax.get_yticklabels():
            if tick_label.get_text() == '':
                continue
            if label_box.overlaps(tick_label.get_window_extent(renderer)):
                raise RuntimeError(f'the shared y label overlaps the tick label '
                                   f'"{tick_label.get_text()}" of panel {i_ax}')


with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
chunks = data['chunks']

# one y range for every panel, from the extremes of every drawn curve
lowest = 0.0
highest = 0.0
for chunk in chunks:
    for key, _ in PANELS:
        lowest = min(lowest, min(chunk[key]))
        highest = max(highest, max(chunk[key]))
margin = 0.08*(highest - lowest)
y_range = (lowest - margin, highest + margin)

# one column of a two-column article: wide, short panels with a thin gap
fig, axes = plt.subplots(len(PANELS), 1, figsize=(3.5, 1.1*len(PANELS)), sharex=True,
                         sharey=True, gridspec_kw={'hspace': 0.06})
for ax, (key, name) in zip(axes, PANELS):
    ax.axhline(0.0, **ZERO_LINE)
    for chunk in chunks:
        color = TRACER_COLORS[chunk['sample']]
        style = TRACER_STYLE[chunk['sample']]
        ax.plot(chunk['z'], chunk[key], color=color, ls=style['ls'], **CURVE_STYLE)
        ax.plot(chunk['z_eff'], chunk['at_z_eff'][key], color=color, marker=style['marker'],
                **MARKER_STYLE)
    ax.set_ylim(*y_range)
    ax.set_yticks([-1, 0, 1, 2, 3])
    ax.tick_params(axis='y', labelsize=8.5)
    ax.grid(False, axis='x')
    # the bias on the right side, rotated like the y label, so it can never
    # sit on the data
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(name, rotation=90, fontsize=11, labelpad=5)
axes[-1].set_xlim(0.05, 1.65)
axes[-1].set_xticks(Z_TICKS)
axes[-1].set_xticklabels([f'{z:g}' for z in Z_TICKS])
axes[-1].set_xlabel(r'$z$', fontsize=10)
ylabel = fig.supylabel('fiducial bias, CCL convention', x=-0.03, fontsize=10)

handles = []
for sample in ('bgs', 'lrg', 'elg'):
    handles.append(Line2D([], [], color=TRACER_COLORS[sample], label=TRACER_NAMES[sample],
                          ls=TRACER_STYLE[sample]['ls'], marker=TRACER_STYLE[sample]['marker'],
                          ms=4.0, **CURVE_STYLE))
legend = axes[0].legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                        ncol=3, frameon=False, fontsize=8.5, handlelength=2.0,
                        columnspacing=1.4, handletextpad=0.5)
assert_labels_clear(fig, [('legend', legend)])
assert_ylabel_clear(fig, ylabel)
fig.savefig(os.path.join(HERE, 'coevolution_biases.pdf'))
fig.savefig(os.path.join(HERE, 'coevolution_biases.png'), dpi=300)
