"""Redraw term_sizes.pdf from data.json: each one-loop term of the galaxy
power spectrum P_gg in percent of the linear-bias model b_1^2 P_mm (P_mm the
halofit matter power), against the wavenumber k, for four DESI-like tracers
at the co-evolution biases. One wide panel per tracer, stacked.

Drawn per panel: b_1 b_2 P_d1d2, b_2^2/4 P_d2d2, the sum of the three tidal
terms (b_1 b_{s^2} P_d1s2 + b_2 b_{s^2}/2 P_d2s2 + b_{s^2}^2/4 P_s2s2),
b_1 b_3nl sigma_3^2 P_lin, the loop sum of all six, and the counterterm
alpha_0 k^2 P_lin at the DESI DR1 one-sigma width alpha_0 = 12.5 (Mpc/h)^2.
Here b_1, b_2, b_{s^2} and b_3nl are the linear, quadratic, tidal and
third-order biases and P_XY the FAST-PT templates. data.json, written by
../make_figure_data.py, also holds each tidal term on its own and the values
at the wavenumbers the note's text reads.
Run from any folder: python plot_term_sizes.py
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
# the drawn terms: data.json key, legend label, line style; the three tidal
# terms are drawn as their sum to keep each panel to six curves
TERMS = [
    ('b1b2', r'$b_1 b_2 P_{\delta_1\delta_2}$', {'color': '#1f77b4', 'lw': 1.3}),
    ('b2b2', r'$\frac{1}{4}b_2^2 P_{\delta_2\delta_2}$', {'color': '#ff7f0e', 'lw': 1.3}),
    ('tidal', r'$b_{s^2}$ terms', {'color': '#2ca02c', 'lw': 1.3}),
    ('b1b3nl', r'$b_1 b_{3\mathrm{nl}}\sigma_3^2 P_\mathrm{lin}$',
     {'color': '#9467bd', 'lw': 1.3}),
    ('loop_sum', 'loop sum', {'color': 'black', 'lw': 2.0}),
    ('counterterm', r'$\alpha_0 k^2 P_\mathrm{lin}$',
     {'color': '0.45', 'lw': 1.3, 'ls': '--'}),
]
ZERO_LINE = {'color': '0.6', 'lw': 0.6}
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
k = np.array(data['k_h_per_mpc'])
tracers = data['tracers']

# one y range for every panel, symmetric about zero, from the largest drawn term
largest = 0.0
for tracer in tracers:
    for key, _, _ in TERMS:
        largest = max(largest, float(np.max(np.abs(tracer['percent_of_b1sq_pmm'][key]))))
y_limit = 1.12*largest

# one column of a two-column article: wide, short panels with a thin gap
fig, axes = plt.subplots(len(tracers), 1, figsize=(3.5, 1.15*len(tracers)), sharex=True,
                         sharey=True, gridspec_kw={'hspace': 0.06})
labels = []
for ax, tracer in zip(axes, tracers):
    ax.axhline(0.0, **ZERO_LINE)
    for key, _, style in TERMS:
        ax.plot(k, tracer['percent_of_b1sq_pmm'][key], **style)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_yticks([-10, 0, 10])
    # the tracer inside the panel, in its upper left, where every term is
    # still small
    name = f"{tracer['name']}, $z = {tracer['z']:g}$"
    text = ax.text(0.03, 0.92, name, transform=ax.transAxes, ha='left', va='top',
                   fontsize=8.5)
    labels.append((name, text))
axes[-1].set_xlim(k[0], k[-1])
axes[-1].set_xticks([0.05, 0.1, 0.15, 0.2])
axes[-1].set_xticklabels(['0.05', '0.1', '0.15', '0.2'])
axes[-1].set_xlabel(r'$k$ [$h\,\mathrm{Mpc}^{-1}$]', fontsize=10)
ylabel = fig.supylabel(r'term / $b_1^2 P_\mathrm{mm}$  [%]', x=-0.05, fontsize=10)

handles = []
for _, label, style in TERMS:
    handles.append(Line2D([], [], label=label, **style))
legend = axes[0].legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                        ncol=3, frameon=False, fontsize=8, handlelength=1.8,
                        columnspacing=0.9, handletextpad=0.4, labelspacing=0.35)
labels.append(('legend', legend))
assert_labels_clear(fig, labels)
assert_ylabel_clear(fig, ylabel)
fig.savefig(os.path.join(HERE, 'term_sizes.pdf'))
fig.savefig(os.path.join(HERE, 'term_sizes.png'), dpi=300)
