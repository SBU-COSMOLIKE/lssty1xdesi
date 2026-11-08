"""Redraw theory_bz.pdf from data.json: the linear bias b(z) of every DESI
group under the growth-tied shape B_g/D(z) (solid), a conserved population
(dashed) and, for BGS, the r < 19.5 flux limit (dash-dot). Each group is
labeled at the first of a few candidate places that clears every curve.
data.json comes from ../make_theory_data.py. Run from any folder:
python plot_theory_bz.py
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


def place_label(fig, ax, group, color, others):
    '''Put the group's name at the first candidate place that clears every
    curve, the legend and the labels placed before it: right of the curve's
    end, below or above its middle or a quarter of the way in from either
    end, left of its start.'''
    z, b = np.asarray(group['z']), np.asarray(group['b_tied'])
    curves = np.vstack([b, group['b_conserved']] + ([group['b_flux_limit']]
                                                   if 'b_flux_limit' in group else []))
    candidates = [(z[-1] + 0.015, b[-1], 'left', 'center')]
    for k in (len(z)//2, 3*len(z)//4, len(z)//4):
        candidates += [(z[k], curves[:, k].min() - 0.1, 'center', 'top'),
                       (z[k], curves[:, k].max() + 0.1, 'center', 'bottom')]
    candidates.append((z[0] - 0.015, b[0], 'right', 'center'))
    renderer = fig.canvas.get_renderer()
    for zx, by, ha, va in candidates:
        text = ax.text(zx, by, group['name'], color=color, fontsize=10, ha=ha, va=va)
        try:
            check_clear(fig, [(group['name'], text)])
        except RuntimeError:
            text.remove()
            continue
        box = text.get_window_extent(renderer)
        if any(box.overlaps(o.get_window_extent(renderer)) for _, o in others):
            text.remove()
            continue
        return text
    raise RuntimeError(f'{group["name"]}: no label place clears the curves')


with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)

# full width of a two-column article
fig, ax = plt.subplots(figsize=(7.0, 4.4))
for group in data['groups']:
    color = GROUP_COLORS[group['name']]
    ax.plot(group['z'], group['b_tied'], color=color, lw=2.0)
    ax.plot(group['z'], group['b_conserved'], color=color, lw=2.0, ls='--')
    if 'b_flux_limit' in group:
        ax.plot(group['z'], group['b_flux_limit'], color=color, lw=2.0, ls='-.')
ax.set_xlabel(r'redshift $z$')
ax.set_ylabel(r'linear bias $b(z)$')
ax.set_xlim(0.05, 1.75)
handles = [
    Line2D([], [], color='0.25', lw=2.0, label=r'growth-tied: $B_g/D(z)$'),
    Line2D([], [], color='0.25', lw=2.0, ls='--',
           label=r'conserved: $1 + (b_c - 1)\,D(z_c)/D(z)$'),
    Line2D([], [], color='0.25', lw=2.0, ls='-.', label=r'BGS flux limit $r < 19.5$'),
]
legend = ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=10)
# the least headroom above the curves at which the legend clears them all
low, high = ax.get_ylim()
for headroom in np.arange(0.0, 0.62, 0.02):
    ax.set_ylim(low, high + headroom*(high - low))
    try:
        check_clear(fig, [('legend', legend)])
        break
    except RuntimeError:
        continue
else:
    raise RuntimeError('the legend touches a curve at every headroom')
placed = [('legend', legend)]
for group in data['groups']:
    text = place_label(fig, ax, group, GROUP_COLORS[group['name']], placed)
    placed.append((group['name'], text))
check_clear(fig, placed)
fig.savefig(os.path.join(HERE, 'theory_bz.pdf'))
