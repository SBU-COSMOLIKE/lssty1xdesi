"""Redraw limber_real.pdf from data.json: the correlation coefficient of one
Delta z = 0.02 bin (bin 56, ELG1, zbar = 0.93) with its neighbour, the bin
two apart and the bin three apart, from the forecast's exact (non-Limber)
spectra and from cosmolike's Limber approximation of the same model.

One column of a two-column paper, three stacked panels (named on the
right): bins 56 x 57 (neighbours), 56 x 58 (two apart) and 56 x 59 (three
apart). Each draws, against ell,
r_ell = C_ell^ij/sqrt(C_ell^ii C_ell^jj) of the signal: exact (black
solid, circles) and Limber (vermillion dashed, squares). The Limber density
term of two bins whose kernels do not overlap is zero; cosmolike's Limber
redshift-space term samples each kernel at chi and chi + chi/ell, so at low
ell it still couples bins that far apart. Vertical dotted lines: the last
band each pair keeps under the UV-fraction cut at k_max = 0.1 (thin) and
0.2 h/Mpc (thick), z = 0 values of the per-bin cut at k_max/D(zbar_i).
Every panel shares one y range. Gray: zero.
Run from any folder: python plot_limber_real.py
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
NAME = 'limber_real'
FIG_WIDTH = 3.5
PANEL_HEIGHT = 1.15
LEGEND_GAP = 0.006
REFERENCE = {'color': '0.6', 'lw': 0.8}
CURVES = [('r_exact', 'exact (non-Limber)', 'black', '-', 1.6, 'o'),
          ('r_limber', 'Limber', '#D55E00', (0, (4, 2)), 1.3, 's')]
CUT_LINES = {'0.1': {'lw': 0.7}, '0.2': {'lw': 1.4}}
ELL_MAX = 360
MARK_EVERY = 3
MARKER_SIZE = 3.0
NAMES = {1: 'neighbours', 2: 'two apart', 3: 'three apart'}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)


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
        steps = np.linspace(0.0, 1.0, 20)[:, None]
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
            raise RuntimeError(f'{NAME}: {type(artist).__name__} '
                               f'{getattr(artist, "get_text", lambda: "")()!r} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'{NAME}: {type(artists[i]).__name__} {i} overlaps '
                                   f'{type(artists[j]).__name__} {j}')


def place_text(fig, ax, text, corners):
    '''Put text in the first corner of ax that touches no drawn line.'''
    for x, y, ha, va in corners:
        t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=9)
        fig.canvas.draw()
        if not touches_lines(t.get_window_extent(fig.canvas.get_renderer()), [ax]):
            return t
        t.remove()
    raise RuntimeError(f'{NAME}: no free corner for {text!r}')



ell = np.asarray(data['ell'])
show = ell <= ELL_MAX
fig, axes = plt.subplots(len(data['curves']), 1, figsize=(FIG_WIDTH, PANEL_HEIGHT*len(data['curves'])),
                         sharex=True, sharey=True, gridspec_kw={'hspace': 0.06})
values = [0.0]
for ax, c in zip(axes, data['curves']):
    ax.axhline(0.0, **REFERENCE)
    for key, label, color, ls, lw, marker in CURVES:
        y = np.asarray(c[key])[show]
        ax.plot(ell[show], y, color=color, ls=ls, lw=lw, marker=marker, ms=MARKER_SIZE,
                markevery=MARK_EVERY)
        values += list(y)
    for cut, style in CUT_LINES.items():
        kept = np.asarray(c['kept'][cut])
        if kept.any():
            ax.axvline(ell[np.nonzero(kept)[0][-1]], color='0.35', ls=':', **style)
    i, j = c['bins']
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(f'{NAMES[c["apart"]]}, {i}$\\times${j}', rotation=90, fontsize=7.5,
                  labelpad=4)
    ax.grid(False)
pad = 0.08*(max(values) - min(values))
for ax in axes:
    ax.set_ylim(min(values) - pad, max(values) + pad)
    ax.set_xlim(ell[show].min() - 5, ell[show].max() + 5)
    ax.yaxis.set_major_locator(FixedLocator([-0.1, 0.0, 0.1, 0.2]))
    ax.tick_params(labelsize=8)
axes[-1].set_xlabel(r'$\ell$', fontsize=9)
legend = fig.legend(
    handles=[Line2D([], [], color=color, ls=ls, lw=lw, marker=marker, ms=MARKER_SIZE, label=label)
             for _, label, color, ls, lw, marker in CURVES],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.6, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
legend.set_bbox_to_anchor((0.5, inv.transform((0.0, axes[0].get_window_extent(renderer).y1))[1]
                           + LEGEND_GAP), transform=fig.transFigure)
fig.canvas.draw()
tick_left = min(t.get_window_extent(renderer).x0 for ax in axes for t in ax.get_yticklabels()
                if t.get_text())
supylabel = fig.supylabel(r'$r_\ell = C_\ell^{ij}/\sqrt{C_\ell^{ii} C_\ell^{jj}}$',
                          x=inv.transform((tick_left, 0.0))[0] - 0.01, ha='right', fontsize=9)
check_clear(fig, list(axes), [legend, supylabel, axes[-1].xaxis.label,
                              *[ax.yaxis.label for ax in axes]])
fig.savefig(os.path.join(HERE, f'{NAME}.pdf'))
fig.savefig(os.path.join(HERE, f'{NAME}.png'), dpi=300)
plt.close(fig)
