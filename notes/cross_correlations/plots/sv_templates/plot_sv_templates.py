"""Redraw sv_templates.pdf from data.json: why fitting the velocity
dispersion sigma_v is costly in the auto spectra of thin bins, and why the
cross spectra of bins two apart release it.

A full-width strip of two rows and three columns: one Delta z = 0.02 ELG
bin's auto spectrum, its cross spectrum with the next bin (in the current
pair list) and its cross spectrum with the bin two apart (added at gap
g = 1). Every curve is in units of the band's error sigma_ell (diagonal of
the Gaussian covariance), against the band center ell, on the bands kept at
k_max = 0.1 h/Mpc (z = 0 value of a per-bin cut at k_max/D(zbar_i)).

- Top row: the sigma_v template sigma_ell^-1 dC_ell/d ln sigma_v, and the
  b_1 and w0 templates each multiplied by the single coefficient that best
  matches the sigma_v template over the kept bands of all 90 auto spectra.
  Where the three curves coincide, the data cannot tell sigma_v from b_1 or
  w0.
- Bottom row: the leftover r_ell = sigma_v template - (c_b1 b_1 template +
  c_w0 w0 template), with the two coefficients fitted jointly over the same
  auto bands and applied unchanged to every column. A leftover far from zero
  is a pattern that no change of b_1 and w0 can mimic: it is what lets the
  fit separate sigma_v from them.

Each row shares one y axis; a column whose curves are smaller is multiplied
by the round factor written in its panels. Each curve has a color, a line
style, a line width and a marker.
Run from any folder: python plot_sv_templates.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator

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
NAME = 'sv_templates'
CURVES = [  # data.json key, color, line style, width, marker
    ('sigma_v', '#0072B2', (0, (4, 2)), 1.8, 's'),
    ('b1', '#D55E00', '-', 1.3, 'o'),
    ('w0', '#CC79A7', (0, (5, 1.5, 1, 1.5)), 1.1, 'D'),
]
LEFTOVER = ('black', '-', 1.3, 'd')
REFERENCE = {'color': '0.6', 'lw': 0.8}
MARKER_SIZE = 3.0
FACTORS = [1, 2, 5, 10, 20, 50, 100, 200, 500]
PANEL_HEIGHT = 1.35
FIG_WIDTH = 6.4
LEGEND_GAP = 0.006
Y_PAD = 0.10

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
single = data['match_single']
joint = data['match_joint']
spectra = data['spectra']


def curves_of(sp):
    t = {key: np.asarray(sp[f't_{key}']) for key, *_ in CURVES}
    top = {'sigma_v': t['sigma_v'], 'b1': single['b1']*t['b1'], 'w0': single['w0']*t['w0']}
    leftover = t['sigma_v'] - joint['b1']*t['b1'] - joint['w0']*t['w0']
    return top, leftover


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
        steps = np.linspace(0.0, 1.0, 60)[:, None]
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
                               f'"{getattr(artist, "get_text", lambda: "")()}" touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'{NAME}: {type(artists[i]).__name__} {i} overlaps '
                                   f'{type(artists[j]).__name__} {j}')
    shown = [label for ax in axes for label in (*ax.get_yticklabels(), *ax.get_xticklabels())]
    ticks = [label.get_window_extent(renderer) for label in shown
             if label.get_text() and label.get_visible()]
    for i in range(len(ticks)):
        for j in range(i + 1, len(ticks)):
            if ticks[i].overlaps(ticks[j]):
                raise RuntimeError(f'{NAME}: two tick labels touch')


# one round factor per column, so that every column fills the auto column's range
tops, lefts = zip(*[curves_of(sp) for sp in spectra])
reach = [max(max(np.abs(v).max() for v in top.values()), np.abs(left).max())
         for top, left in zip(tops, lefts)]
factors = []
for r in reach:
    f = max([x for x in FACTORS if x*r <= 1.05*reach[0]] or [1])
    factors.append(f)

fig, grid = plt.subplots(2, len(spectra), figsize=(FIG_WIDTH, 2*PANEL_HEIGHT),
                         sharex='col', gridspec_kw={'hspace': 0.06, 'wspace': 0.06})
top_values, left_values = [0.0], [0.0]
for col, (sp, top, left, f) in enumerate(zip(spectra, tops, lefts, factors)):
    ell = np.asarray(sp['ell'])
    ax = grid[0, col]
    ax.axhline(0.0, **REFERENCE)
    for key, color, ls, lw, marker in CURVES:
        ax.plot(ell, f*top[key], color=color, ls=ls, lw=lw, marker=marker, ms=MARKER_SIZE,
                clip_on=False)
        top_values += list(f*top[key])
    ax = grid[1, col]
    ax.axhline(0.0, **REFERENCE)
    color, ls, lw, marker = LEFTOVER
    ax.plot(ell, f*left, color=color, ls=ls, lw=lw, marker=marker, ms=MARKER_SIZE, clip_on=False)
    left_values += list(f*left)
    for ax in grid[:, col]:
        ax.grid(False)
        ax.set_xlim(ell.min() - 0.08*np.ptp(ell), ell.max() + 0.08*np.ptp(ell))
        # ticks kept off the column edges, so neighbouring columns' labels never touch
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True, prune='both'))
    grid[1, col].set_xlabel(r'$\ell$', fontsize=10)

for row, values in enumerate([top_values, left_values]):
    lo, hi = min(values), max(values)
    pad = Y_PAD*(hi - lo)
    for col, ax in enumerate(grid[row]):
        ax.set_ylim(lo - pad, hi + pad)
        # ticks kept off the panel edges, so the stacked rows' labels never touch
        ticks = MaxNLocator(4).tick_values(lo - pad, hi + pad)
        margin = 0.12*(hi - lo + 2*pad)
        ax.yaxis.set_major_locator(FixedLocator(
            [v for v in ticks if lo - pad + margin <= v <= hi + pad - margin]))
        if col > 0:
            ax.tick_params(labelleft=False)

b0, b1_ = spectra[0]['bins']
titles = []
for col, sp in enumerate(spectra):
    i, j = sp['bins']
    if i == j:
        text = rf'{sp["sample"]} auto, bin {i} ($\bar z = {sp["zbar"][0]:.2f}$)'
    else:
        text = rf'{sp["name"]}: bins {i}$\times${j}'
    titles.append(grid[0, col].set_title(text, fontsize=9, pad=4))

# the round factors, inside the scaled panels, in the free corner farthest from the data
texts = []
for col, f in enumerate(factors):
    if f == 1:
        continue
    for ax in grid[:, col]:
        placed = None
        for (x, y, ha, va) in [(0.97, 0.94, 'right', 'top'), (0.03, 0.94, 'left', 'top'),
                               (0.97, 0.06, 'right', 'bottom'), (0.03, 0.06, 'left', 'bottom')]:
            t = ax.text(x, y, rf'$\times{f}$', transform=ax.transAxes, ha=ha, va=va, fontsize=9)
            fig.canvas.draw()
            if not touches_lines(t.get_window_extent(fig.canvas.get_renderer()), [ax]):
                placed = t
                break
            t.remove()
        if placed is None:
            raise RuntimeError(f'{NAME}: no free corner for the factor of column {col}')
        texts.append(placed)

legend = fig.legend(
    handles=[Line2D([], [], color=CURVES[0][1], ls=CURVES[0][2], lw=CURVES[0][3],
                    marker=CURVES[0][4], ms=MARKER_SIZE,
                    label=r'$\sigma_v$: $\sigma_\ell^{-1}\,\partial C_\ell/\partial \ln \sigma_v$'),
             Line2D([], [], color=CURVES[1][1], ls=CURVES[1][2], lw=CURVES[1][3],
                    marker=CURVES[1][4], ms=MARKER_SIZE,
                    label=r'$b_1$, matched on the autos'),
             Line2D([], [], color=CURVES[2][1], ls=CURVES[2][2], lw=CURVES[2][3],
                    marker=CURVES[2][4], ms=MARKER_SIZE,
                    label=r'$w_0$, matched on the autos'),
             Line2D([], [], color=LEFTOVER[0], ls=LEFTOVER[1], lw=LEFTOVER[2],
                    marker=LEFTOVER[3], ms=MARKER_SIZE,
                    label=r'leftover $r_\ell$ (lower row)')],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.8, columnspacing=1.6, handletextpad=0.5, labelspacing=0.3,
    borderaxespad=0.0)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
top = max(t.get_window_extent(renderer).y1 for t in titles)
legend.set_bbox_to_anchor((0.5, inv.transform((0.0, top))[1] + LEGEND_GAP),
                          transform=fig.transFigure)
ylabels = [grid[0, 0].set_ylabel(r'$\sigma_\ell^{-1}\,\partial C_\ell/\partial p$', fontsize=10),
           grid[1, 0].set_ylabel(r'$\sigma_\ell^{-1}\,r_\ell$', fontsize=10)]
check_clear(fig, list(grid.flat), [legend, *titles, *texts, *ylabels,
                                   *[ax.xaxis.label for ax in grid[1]]])
fig.savefig(os.path.join(HERE, f'{NAME}.pdf'))
fig.savefig(os.path.join(HERE, f'{NAME}.png'), dpi=300)
plt.close(fig)
print('column factors', factors)
