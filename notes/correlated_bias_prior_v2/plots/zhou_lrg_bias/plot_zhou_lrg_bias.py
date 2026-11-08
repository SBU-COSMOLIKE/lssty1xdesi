"""Redraw zhou_lrg_bias.pdf from data.json: the measured large-scale bias of
DESI-like luminous red galaxies (Zhou et al. 2021, arXiv:2001.06018), shown
as its fractional residual around the constant clustering amplitude
b = B/D(z), B = 1.5, with D the linear growth factor of flat LambdaCDM
(Omega_m = 0.3, D(0) = 1) at each photometric bin's center. The residual is
derived here from the measured biases; horizontal bars mark the bin edges.
data.json holds the measured inputs, transcribed by hand from the paper.
Run from any folder: python plot_zhou_lrg_bias.py
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


def growth_factor(z, omega_m):
    '''Linear growth factor of flat LambdaCDM, normalized to D(0) = 1:
    D(a) is proportional to H(a) times the integral of da'/(a' H(a'))^3 from
    0 to a, evaluated with the trapezoid rule on a fine grid in a.'''
    def unnormalized(a_end):
        a = np.linspace(1.0e-6, a_end, 200001)
        e = np.sqrt(omega_m/a**3 + 1.0 - omega_m)
        f = 1.0/(a*e)**3
        integral = np.sum(0.5*(f[1:] + f[:-1])*np.diff(a))
        return np.sqrt(omega_m/a_end**3 + 1.0 - omega_m)*integral
    return unnormalized(1.0/(1.0 + z))/unnormalized(1.0)


with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
B = data['clustering_amplitude_B']
z_lo = np.array([row['z_lo'] for row in data['bins']])
z_hi = np.array([row['z_hi'] for row in data['bins']])
z_c = 0.5*(z_lo + z_hi)
b = np.array([row['b'] for row in data['bins']])
d = np.array([growth_factor(z, data['omega_matter']) for z in z_c])
# residual around b = B/D in percent, and the bias errors carried through
residual = 100.0*(b*d/B - 1.0)
err_up = 100.0*np.array([row['b_err_up'] for row in data['bins']])*d/B
err_down = 100.0*np.array([row['b_err_down'] for row in data['bins']])*d/B

# one column of a two-column article
fig, ax = plt.subplots(figsize=(3.5, 1.7))
ax.axhline(0.0, color='0.3', ls='--', lw=1.0)
ax.errorbar(z_c, residual, yerr=[err_down, err_up], xerr=[z_c - z_lo, z_hi - z_c], ls='none',
            marker='o', ms=4.0, color='C0', capsize=1.5, elinewidth=0.9)
ax.set_xlabel(r'photometric redshift $z$', fontsize=9)
ax.set_ylabel(r'$b\,D(z)/1.5 - 1$ [%]', fontsize=9)
ax.set_xlim(0.38, 0.96)
ax.set_ylim(-6.0, 15.0)
ax.grid(False, axis='x')
legend = ax.legend(handles=[Line2D([], [], color='0.3', ls='--', lw=1.0,
                                   label=r'constant clustering amplitude, $b = 1.5/D$'),
                            Line2D([], [], ls='none', marker='o', ms=4.0, color='C0',
                                   label='Zhou et al. (2021)')],
                   loc='upper left', frameon=False, fontsize=7.5, handlelength=1.6,
                   labelspacing=0.3)
check_clear(fig, [('legend', legend)])
fig.savefig(os.path.join(HERE, 'zhou_lrg_bias.pdf'))
fig.savefig(os.path.join(HERE, 'zhou_lrg_bias.png'), dpi=300)
