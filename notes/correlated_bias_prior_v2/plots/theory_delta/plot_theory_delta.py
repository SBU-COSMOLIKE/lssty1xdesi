"""Redraw theory_delta.pdf from data.json: the fractional deviations from
the growth-tied shape B_g/D of a conserved population (every group) and of
the BGS flux limit, with the best straight line through each (top), and
what the line leaves (bottom). data.json comes from ../make_theory_data.py.
Run from any folder: python plot_theory_delta.py
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
# colorblind-safe colors (Okabe-Ito, without its green; no red or green);
# ELG2 shares LRG1's orange, far from it in redshift and named on the curve
GROUP_COLORS = {'BGS': '#0072B2', 'LRG1': '#E69F00', 'LRG2': '#56B4E9',
                'LRG3': '#CC79A7', 'ELG1': '#000000', 'ELG2': '#E69F00'}
LAW_STYLE = {'conserved': ('--', 'o'), 'flux_limit': ('-.', 's')}
# heights of the group labels (percent); LRG3 and ELG1 share a redshift
# range, so ELG1 sits one row lower
LABEL_HEIGHT = {'ELG1': -14.5, 'LRG3': -12.0}


def data_points(ax):
    '''Display coordinates of every drawn data point of ax: the markers, and
    points sampled densely along every line and error bar.'''
    pts = [np.empty((0, 2))]
    t = np.linspace(0.0, 1.0, 50)[:, None]
    for line in ax.get_lines():
        xy = line.get_transform().transform(
            np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()]))
        pts.append(xy)
        if line.get_linestyle() not in ('None', 'none', ' ', ''):
            for a, b in zip(xy[:-1], xy[1:]):
                pts.append(a + t*(b - a))
    for coll in ax.collections:
        for seg in getattr(coll, 'get_segments', lambda: [])():
            xy = coll.get_transform().transform(np.asarray(seg, dtype=float))
            for a, b in zip(xy[:-1], xy[1:]):
                pts.append(a + t*(b - a))
    pts = np.vstack(pts)
    return pts[np.all(np.isfinite(pts), axis=1)]


def check_clear(fig, labels):
    '''Raise if a legend or text label touches a data point of any panel (a
    margin of 3 points around each label counts as touching).'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = 3.0*fig.dpi/72.0
    pts = np.vstack([data_points(ax) for ax in fig.axes])
    for name, artist in labels:
        box = artist.get_window_extent(renderer)
        hit = ((pts[:, 0] > box.x0 - pad) & (pts[:, 0] < box.x1 + pad)
               & (pts[:, 1] > box.y0 - pad) & (pts[:, 1] < box.y1 + pad))
        if np.any(hit):
            raise RuntimeError(f'{name} touches the data')


with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)

# full width of a two-column article
fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True,
                                  gridspec_kw={'height_ratios': [2.0, 1.0], 'hspace': 0.08})
labeled = set()
labels = []
for curve in data['curves']:
    color = GROUP_COLORS[curve['group']]
    style, marker = LAW_STYLE[curve['law']]
    z = np.array(curve['z'])
    top.plot(z, 100.0*np.array(curve['line']), color='0.45', lw=1.0)
    top.plot(z, 100.0*np.array(curve['delta']), color=color, lw=1.8, ls=style,
             marker=marker, ms=5)
    bottom.plot(z, 100.0*np.array(curve['residual']), color=color, lw=1.8, ls=style,
                marker=marker, ms=5)
    if curve['group'] not in labeled:
        labeled.add(curve['group'])
        labels.append((curve['group'],
                       top.text(z[-1] + 0.015, LABEL_HEIGHT.get(curve['group'], -13.2),
                                curve['group'], color=color, fontsize=10, ha='right')))
top.axhline(0.0, color='0.2', lw=0.8)
bottom.axhline(0.0, color='0.2', lw=0.8)
top.set_ylabel(r'$\delta(z) = b\,D/B_g - 1$ [%]')
bottom.set_ylabel('left by the line [%]')
bottom.set_xlabel(r'redshift $z$')
top.set_ylim(-16.0, 18.0)
bottom.set_ylim(-4.0, 4.0)
handles = [
    Line2D([], [], color='0.25', lw=1.8, ls='--', marker='o', ms=5,
           label='conserved population'),
    Line2D([], [], color='0.25', lw=1.8, ls='-.', marker='s', ms=5,
           label=r'BGS flux limit $r < 19.5$'),
    Line2D([], [], color='0.45', lw=1.0, label='best straight line per group'),
]
legend = top.legend(handles=handles, loc='upper right', frameon=False, fontsize=10)
check_clear(fig, [('legend', legend)] + labels)
fig.savefig(os.path.join(HERE, 'theory_delta.pdf'))
