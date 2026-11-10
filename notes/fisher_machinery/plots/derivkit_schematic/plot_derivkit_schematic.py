"""Redraw derivkit_schematic.pdf from data.json: a toy that shows how
DerivKit's fit lowers a derivative's sensitivity to numerical noise, next to
the five-point stencil and a two-point central difference (data.json:
make_derivkit_schematic_data.py in this folder; the numbers are a toy, not a
measurement).

One column of a two-column paper, two panels:

(a) One typical noise realization (data.json: how it was chosen). Against t/theta0 = (theta - theta0)/theta0, the
    sampled values minus the true tangent at theta0, in units of the noise
    sigma: the samples of each estimator (central difference at +-2%, the
    stencil at +-2%, +-4%, DerivKit at the center and +-1, 1.5, 2.25, 3.375,
    5%) and each estimator's slope as a line through the center. Gray: the
    true curve minus its tangent (the tangent is the zero line).
(b) Over many realizations, (estimate - true)/true in percent per
    estimator: box from the 16th to the 84th percentile, whiskers from the
    2.5th to the 97.5th, filled marker at the mean, open marker at the
    noise-free value (the bias of the cubic term), and a cross at the error
    on a smooth deterministic ripple added to m without noise, which every
    estimator follows alike.

Each estimator has a color, a line style, a line width and a marker.
Run from any folder: python plot_derivkit_schematic.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator

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
# the estimators, top to bottom in panel (b): name, color, line style, width, marker
STYLE = {
    'derivkit': {'label': 'DerivKit', 'color': '#0072B2', 'ls': '-', 'lw': 2.0,
                 'marker': 'o', 'ms': 3.4},
    'five_point': {'label': 'five-point stencil', 'color': '#E69F00', 'ls': (0, (4, 2)),
                   'lw': 1.5, 'marker': 's', 'ms': 3.6},
    'central': {'label': 'central difference', 'color': '#CC79A7', 'ls': (0, (1.2, 1.4)),
                'lw': 1.2, 'marker': '^', 'ms': 4.2},
}
GRAY = '0.55'
BOX_HALF = 0.22

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
toy = data['toy']


def line_points(ax):
    '''Display coordinates of points densely sampled along every drawn line of
    ax (reference lines included) and of every patch edge, inside the axes.'''
    box = ax.get_window_extent()
    pts = [np.zeros((0, 2))]
    paths = [(line.get_transform(),
              np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()]))
             for line in ax.get_lines()]
    for patch in ax.patches:
        paths.append((patch.get_transform(), patch.get_path().vertices))
    for transform, xy in paths:
        display_xy = transform.transform(xy)
        finite = np.all(np.isfinite(display_xy), axis=1)
        pts.append(display_xy[finite])
        steps = np.linspace(0.0, 1.0, 40)[:, None]
        for i in range(len(display_xy) - 1):
            if finite[i] and finite[i + 1]:
                pts.append(display_xy[i] + steps*(display_xy[i + 1] - display_xy[i]))
    pts = np.vstack(pts)
    inside = ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
              & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))
    return pts[inside]


def touches_lines(bbox, axes):
    padded = bbox.expanded(1.03, 1.08)
    for ax in axes:
        pts = line_points(ax)
        if np.any((pts[:, 0] > padded.x0) & (pts[:, 0] < padded.x1)
                  & (pts[:, 1] > padded.y0) & (pts[:, 1] < padded.y1)):
            return True
    return False


def place(fig, ax, make, corners, avoid=()):
    '''Create an artist with make(corner) at the first corner where it touches
    no line of ax and no artist in avoid; raise when none is free.'''
    for corner in corners:
        artist = make(corner)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = artist.get_window_extent(renderer)
        if not touches_lines(box, [ax]) and not any(
                box.overlaps(other.get_window_extent(renderer)) for other in avoid):
            return artist
        artist.remove()
    raise RuntimeError('derivkit_schematic: no free place for a label')


def check_clear(fig, axes, artists):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer) for a in artists]
    for b, a in zip(boxes, artists):
        if touches_lines(b, axes):
            raise RuntimeError(f'derivkit_schematic: {type(a).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'derivkit_schematic: {type(artists[i]).__name__} {i} '
                                   f'overlaps {type(artists[j]).__name__} {j}')


fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.5, 3.6),
                                 gridspec_kw={'height_ratios': [1.55, 1.0], 'hspace': 0.42})
sigma, theta0, s0 = toy['sigma'], toy['theta0'], toy['s0']

# (a) one realization, values minus the true tangent, in units of sigma
t = np.array(data['curve']['t'])
ax_a.axhline(0.0, color=GRAY, lw=0.8, ls=(0, (2, 2)))
ax_a.plot(100*t/theta0, np.array(data['curve']['m_minus_tangent'])/sigma, color=GRAY, lw=1.0)
for name in ('central', 'five_point', 'derivkit'):
    st = STYLE[name]
    e = data['example'][name]
    th = np.array(e['theta'])
    resid = (np.array(e['m']) - 1.0 - s0*(th - theta0))/sigma
    ax_a.plot(100*(th - theta0)/theta0, resid, ls='none', marker=st['marker'], ms=st['ms'] + 0.6,
              color=st['color'], mfc='white' if name == 'derivkit' else st['color'],
              clip_on=False, zorder=3)
    ax_a.plot(100*t/theta0, (e['slope'] - s0)*t/sigma, color=st['color'], ls=st['ls'],
              lw=st['lw'], zorder=2)
ax_a.set_xlim(-5.6, 5.6)
ax_a.set_ylim(-3.4, 4.4)
ax_a.set_xlabel(r'$(\theta - \theta_0)/\theta_0$ [%]', fontsize=9)
ax_a.set_ylabel(r'(sample $-$ tangent)$/\sigma$', fontsize=9)
ax_a.xaxis.set_major_locator(FixedLocator([-4, -2, 0, 2, 4]))
ax_a.grid(False)

# (b) the spread over realizations, in percent of the true slope
rows = ['central', 'five_point', 'derivkit']
for y, name in enumerate(rows):
    st = STYLE[name]
    d = data['distribution'][name]
    p = {k: 100*v for k, v in d['percentiles'].items()}
    ax_b.plot([p['2.5'], p['16']], [y, y], color=st['color'], lw=1.0)
    ax_b.plot([p['84'], p['97.5']], [y, y], color=st['color'], lw=1.0)
    ax_b.add_patch(Rectangle((p['16'], y - BOX_HALF), p['84'] - p['16'], 2*BOX_HALF,
                             facecolor='white', edgecolor=st['color'], lw=st['lw']))
    ax_b.plot([p['50'], p['50']], [y - BOX_HALF, y + BOX_HALF], color=st['color'], lw=st['lw'])
    ax_b.plot([100*d['mean']], [y], ls='none', marker=st['marker'], ms=st['ms'] + 1.0,
              color=st['color'], zorder=3)
    ax_b.plot([100*d['curvature_bias']], [y + 2*BOX_HALF], ls='none', marker=st['marker'],
              ms=st['ms'] + 1.0, color=st['color'], mfc='white', zorder=3, clip_on=False)
    ax_b.plot([100*d['deterministic_error']], [y + 2*BOX_HALF], ls='none', marker='x',
              ms=st['ms'] + 1.5, mew=1.2, color=st['color'], zorder=3, clip_on=False)
ax_b.axvline(0.0, color=GRAY, lw=0.8, ls=(0, (2, 2)))
ax_b.set_yticks(range(len(rows)))
ax_b.set_yticklabels([STYLE[n]['label'] for n in rows], fontsize=8)
ax_b.set_ylim(-0.6, len(rows) - 0.25)
ax_b.set_xlim(-11.0, 11.0)
ax_b.set_xlabel(r'(estimate $-$ true slope)$/$true slope [%]', fontsize=9)
ax_b.tick_params(axis='y', length=0)
ax_b.grid(False)

# the estimators: one legend above panel (a)
legend_a = fig.legend(
    handles=[Line2D([], [], color=STYLE[n]['color'], ls=STYLE[n]['ls'], lw=STYLE[n]['lw'],
                    marker=STYLE[n]['marker'], ms=STYLE[n]['ms'] + 0.6,
                    mfc='white' if n == 'derivkit' else STYLE[n]['color'],
                    label=STYLE[n]['label']) for n in ('derivkit', 'five_point', 'central')],
    loc='lower center', bbox_to_anchor=(0.55, 0.0), ncol=3, frameon=False, fontsize=7.5,
    handlelength=2.4, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.0)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
top = inv.transform((0.0, ax_a.get_window_extent(renderer).y1))[1]
legend_a.set_bbox_to_anchor((0.55, top + 0.006), transform=fig.transFigure)
# the markers of panel (b), in a free corner of it
marker_handles = [Line2D([], [], color='0.25', ls='none', marker='o', ms=4.0, label='mean'),
                  Line2D([], [], color='0.25', ls='none', marker='o', ms=4.0, mfc='white',
                         label='no noise'),
                  Line2D([], [], color='0.25', ls='none', marker='x', ms=5.0, mew=1.2,
                         label='smooth ripple')]
name_a = place(fig, ax_a, lambda c: ax_a.text(*c, '(a)', transform=ax_a.transAxes,
                                              ha='left' if c[0] < 0.5 else 'right',
                                              va='top' if c[1] > 0.5 else 'bottom', fontsize=9),
               [(0.03, 0.95), (0.97, 0.95), (0.03, 0.05), (0.97, 0.05)])
name_b = place(fig, ax_b, lambda c: ax_b.text(*c, '(b)', transform=ax_b.transAxes,
                                              ha='left' if c[0] < 0.5 else 'right',
                                              va='top' if c[1] > 0.5 else 'bottom', fontsize=9),
               [(0.03, 0.95), (0.97, 0.95), (0.03, 0.05), (0.97, 0.05)])
legend_b = place(fig, ax_b, lambda corner: ax_b.legend(
    handles=marker_handles, loc=corner, frameon=False, fontsize=7, handletextpad=0.2,
    labelspacing=0.15, borderaxespad=0.3, borderpad=0.1),
    ['lower right', 'lower left', 'upper right', 'upper left'], avoid=[name_b])
check_clear(fig, [ax_a, ax_b], [legend_a, legend_b, name_a, name_b, ax_a.xaxis.label,
                                ax_a.yaxis.label, ax_b.xaxis.label])
fig.savefig(os.path.join(HERE, 'derivkit_schematic.pdf'))
fig.savefig(os.path.join(HERE, 'derivkit_schematic.png'), dpi=300)
plt.close(fig)
