"""Redraw limber_toy.pdf from data.json: why two redshift shells with no
overlap have a cross spectrum, and why the Limber approximation sets it to
zero (a flat-sky toy: two top-hat shells two bin widths apart).

A full-width strip of three panels, each with its own axes:

(a) the two shells (gray bands) along the line of sight, and three
    line-of-sight modes cos(k_par chi): k_par = 0 and 2 pi/d take the same
    value in both shells (they make the shells correlated), k_par = pi/d
    has a crest in one and a trough in the other (anticorrelated);
(b) the power each line-of-sight mode k_par carries at a given ell,
    P(sqrt(k_perp^2 + k_par^2)), over the power P(k_perp) that the Limber
    approximation gives every mode (the constant gray line at one); shaded:
    the k_par where the shells' weight cos(k_par d) sinc^2(k_par w/2) is
    negative (crest in one shell, trough in the other). With equal power
    the positive and negative weights cancel exactly; at ell = 20 the
    shaded modes carry more power (the spectrum still rises with k) and the
    cross spectrum is negative; at ell = 200 the ratio stays near one;
(c) the correlation coefficient r_ell = C_ij/sqrt(C_ii C_jj) against ell
    for shells that touch, two apart and three apart (color, line width
    and marker), in real space (solid) and with the Kaiser redshift-space
    term (dashed); the Limber value is zero for all of them (gray line).

Run from any folder: python plot_limber_toy.py
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
NAME = 'limber_toy'
FIG_WIDTH = 6.4
PANEL_HEIGHT = 2.0
LEGEND_GAP = 0.01
REFERENCE = {'color': '0.6', 'lw': 0.8}
SHELL_FILL = '0.88'
MODE_STYLE = [  # k_par in units of pi/d: color, line style, width
    (0, 'black', '-', 1.6),
    (1, '#D55E00', (0, (4, 2)), 1.4),
    (2, '#0072B2', (0, (5, 1.5, 1, 1.5)), 1.2),
]
INTEGRAND_STYLE = {'limber': ('0.45', '-', 2.6),
                   20: ('#D55E00', (0, (4, 2)), 1.4),
                   200: ('#0072B2', (0, (5, 1.5, 1, 1.5)), 1.2)}
GAP_STYLE = {0: ('#0072B2', 2.0, 'o'), 1: ('#E69F00', 1.5, 's'), 2: ('#CC79A7', 1.1, 'D')}
GAP_LABEL = {0: 'touching', 1: 'two apart', 2: 'three apart'}
VARIANT_STYLE = {0.0: '-', 'kaiser': (0, (4, 2))}
MARK_EVERY = 12
MARKER_SIZE = 3.0

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


fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, PANEL_HEIGHT),
                         gridspec_kw={'wspace': 0.5, 'width_ratios': [1.0, 1.0, 1.15]})

# (a) the shells and the line-of-sight modes
ax = axes[0]
x = np.asarray(data['shell_x_Mpc'])
for lo, hi in data['shells'].values():
    ax.axvspan(lo, hi, color=SHELL_FILL, lw=0, zorder=0)
for (m, color, ls, lw), mode in zip(MODE_STYLE, data['modes']):
    ax.plot(x, mode['value'], color=color, ls=ls, lw=lw)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-1.55, 1.55)
ax.yaxis.set_major_locator(FixedLocator([-1, 0, 1]))
ax.set_xlabel(r'$\chi - \chi_i$ [Mpc]', fontsize=9)
ax.set_ylabel(r'$\cos(k_\parallel \chi)$', fontsize=9)
ax.grid(False)

# (b) the power each line-of-sight mode carries, over the power Limber gives it
ax = axes[1]
k = np.asarray(data['k_par'])
for lo, hi in data['negative_weight']:
    ax.axvspan(lo, hi, color=SHELL_FILL, lw=0, zorder=0)
color, ls, lw = INTEGRAND_STYLE['limber']
ax.plot(k, np.ones_like(k), color=color, ls=ls, lw=lw)
ratios = []
for it in data['integrand']:
    color, ls, lw = INTEGRAND_STYLE[it['ell']]
    ax.plot(k, it['power_ratio'], color=color, ls=ls, lw=lw)
    ratios += it['power_ratio']
ax.set_xlim(0.0, k.max())
ax.set_ylim(0.0, 1.12*max(ratios))
ax.xaxis.set_major_locator(FixedLocator([0.0, 0.05, 0.1]))
ax.set_xlabel(r'$k_\parallel$ [Mpc$^{-1}$]', fontsize=9)
ax.set_ylabel(r'$P(\sqrt{k_\perp^2 + k_\parallel^2})/P(k_\perp)$', fontsize=9)
ax.grid(False)

# (c) the correlation coefficient against ell
ax = axes[2]
ell = np.asarray(data['ell'])
ax.axhline(0.0, **REFERENCE)
for c in data['curves']:
    color, lw, marker = GAP_STYLE[c['gap']]
    ls = VARIANT_STYLE[0.0] if c['f'] == 0.0 else VARIANT_STYLE['kaiser']
    ax.plot(ell, c['r'], color=color, lw=lw, ls=ls, marker=marker, ms=MARKER_SIZE,
            markevery=MARK_EVERY, mfc=color if c['f'] == 0.0 else 'white')
ax.set_xscale('log')
ax.set_xlim(ell.min(), ell.max())
ax.xaxis.set_major_locator(FixedLocator([2, 10, 100]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(['2', '10', '100'])
ax.set_xlabel(r'$\ell$', fontsize=9)
ax.set_ylabel(r'$r_\ell$', fontsize=9)
ax.grid(False)

for a in axes:
    a.tick_params(labelsize=8)
labels = [place_text(fig, a, f'({s})', [(0.04, 0.96, 'left', 'top'), (0.96, 0.96, 'right', 'top'),
                                         (0.04, 0.04, 'left', 'bottom'), (0.96, 0.04, 'right', 'bottom')])
          for a, s in zip(axes, 'abc')]

# one legend row per panel, above it
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
top = inv.transform((0.0, max(a.get_window_extent(renderer).y1 for a in axes)))[1]
legends = []
handles = [
    [Line2D([], [], color=c, ls=ls, lw=lw, label=lab) for (m, c, ls, lw), lab in
     zip(MODE_STYLE, [r'$k_\parallel = 0$', r'$k_\parallel = \pi/d$', r'$k_\parallel = 2\pi/d$'])],
    [Line2D([], [], color=INTEGRAND_STYLE['limber'][0], ls=INTEGRAND_STYLE['limber'][1],
            lw=INTEGRAND_STYLE['limber'][2], label='Limber'),
     Line2D([], [], color=INTEGRAND_STYLE[20][0], ls=INTEGRAND_STYLE[20][1],
            lw=INTEGRAND_STYLE[20][2], label=r'$\ell = 20$'),
     Line2D([], [], color=INTEGRAND_STYLE[200][0], ls=INTEGRAND_STYLE[200][1],
            lw=INTEGRAND_STYLE[200][2], label=r'$\ell = 200$')],
    [*[Line2D([], [], color=GAP_STYLE[g][0], lw=GAP_STYLE[g][1], marker=GAP_STYLE[g][2],
              ms=MARKER_SIZE, label=GAP_LABEL[g]) for g in (0, 1, 2)],
     Line2D([], [], color='0.25', lw=1.2, ls='-', label='real space'),
     Line2D([], [], color='0.25', lw=1.2, ls=VARIANT_STYLE['kaiser'], label='with Kaiser')],
]
for a, h in zip(axes, handles):
    box = a.get_window_extent(renderer)
    xc = inv.transform(((box.x0 + box.x1)/2, 0.0))[0]
    legends.append(fig.legend(handles=h, loc='lower center', bbox_to_anchor=(xc, top + LEGEND_GAP),
                              ncol=1 if len(h) <= 3 else 2, frameon=False, fontsize=7.5,
                              handlelength=2.4, handletextpad=0.4, columnspacing=0.8,
                              labelspacing=0.25, borderaxespad=0.0))
check_clear(fig, list(axes), [*legends, *labels, *[a.xaxis.label for a in axes],
                              *[a.yaxis.label for a in axes]])
fig.savefig(os.path.join(HERE, f'{NAME}.pdf'))
fig.savefig(os.path.join(HERE, f'{NAME}.png'), dpi=300)
plt.close(fig)
