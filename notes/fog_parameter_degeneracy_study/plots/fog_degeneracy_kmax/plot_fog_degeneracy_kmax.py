"""Redraw fog_degeneracy_kmax.pdf from data.json: how far the alpha_2
counterterm and the sigma_v (Fingers-of-God) dispersions describe the same
signal in the DESI band powers, against the scale cut k_max, for the four
bin layouts.

One column of a two-column paper, four stacked panels sharing the k_max
axis, each with its own y axis:

1. 100 (1 - f), the share of an alpha_2 group template's information
   (C^-1 metric, on the bands the cut keeps) that no combination of the
   sigma_v templates reproduces, for the group where that share is largest;
2. the largest |correlation| between an alpha_2 parameter and a sigma_v in
   the posterior of model B + tilts on the pessimistic linear-bias prior;
3. FoM_{alpha_2 = 0}/FoM: FoM(w0, wa) of that model with alpha_2 and its
   tilts fixed at 0, over FoM with alpha_2 free: sigma_v on its Gaussian
   prior (solid) or flat (dashed);
4. delta_{sigma_v free}/delta_{sigma_v fixed}, with delta the (w0, wa) shift
   of model A + tilts on the pessimistic prior (no alpha_2) fitted to the
   truth alpha_2 = 12.5 h^-2 Mpc^2 in every group: sigma_v free on its
   Gaussian prior (solid) or flat (dashed), over sigma_v fixed at the
   fiducial.

Gray lines: 1 in panels 3 and 4. Each layout has a color, a line width and
a marker; the line style tells the sigma_v prior. A cut that keeps no band
(Delta z = 0.02 at 0.05) leaves a gap. k_max is the z = 0 value of a
per-bin cut at k_max/D(zbar_i).
Run from any folder: python plot_fog_degeneracy_kmax.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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
# Each layout has a line width and a marker as well as its color
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
    'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
    'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
    'Delta z = 0.2': {'lw': 0.9, 'marker': '^'},
}
MARKER_SIZE = 3.5
# sigma_v on its Gaussian prior: solid; flat: dashed with open markers
PRIOR_LS = '-'
FLAT_LS = (0, (4, 2))
REFERENCE = {'color': '0.6', 'lw': 0.8}
PANEL_HEIGHT = 1.15
KMAX_MAX = 0.2
KMAX_TOLERANCE = 1.0e-9
# the panels: (y label, y limits, y ticks, curves); a curve is
# (data.json key, sub-key or None, scale, line style, filled marker)
PANELS = [
    (r'$100\,(1 - f)$', (-0.6, 5.0), [0, 2, 4],
     [('one_minus_f_least_redundant_group', None, 100.0, PRIOR_LS, True)]),
    (r'max $|\rho(\alpha_2, \sigma_v)|$', (-0.1, 1.05), [0, 0.5, 1],
     [('rho_max', None, 1.0, PRIOR_LS, True)]),
    (r'$\mathrm{FoM}_{\alpha_2 = 0}/\mathrm{FoM}$', (0.55, 2.95), [1, 2],
     [('fom_ratio_alpha2_fixed', None, 1.0, PRIOR_LS, True),
      ('fom_ratio_alpha2_fixed_flat_sigma_v', None, 1.0, FLAT_LS, False)]),
    (r'$\delta_{\sigma_v\,\mathrm{free}}/\delta_{\sigma_v\,\mathrm{fixed}}$', (-0.1, 1.12), [0, 0.5, 1],
     [('shift_ratio_sigma_v_prior', 'w0wa', 1.0, PRIOR_LS, True),
      ('shift_ratio_sigma_v_flat', 'w0wa', 1.0, FLAT_LS, False)]),
]
REFERENCE_AT_ONE = {2, 3}
# the gap between the top panel and the layout legend, and between the two
# legends [figure fraction]
LEGEND_GAP = 0.004

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax_all = np.array(data['kmax_h_per_Mpc'], dtype=float)
DRAWN = kmax_all <= KMAX_MAX + KMAX_TOLERANCE
kmax = kmax_all[DRAWN]


def as_array(values, scale):
    out = np.array([np.nan if v is None else scale*float(v) for v in values])
    return out[DRAWN]


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
        t = np.linspace(0.0, 1.0, 60)[:, None]
        for i in range(len(display_xy) - 1):
            if finite[i] and finite[i + 1]:
                pts.append(display_xy[i] + t*(display_xy[i + 1] - display_xy[i]))
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
            raise RuntimeError(f'fog_degeneracy_kmax: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'fog_degeneracy_kmax: {type(artists[i]).__name__} {i} '
                                   f'overlaps {type(artists[j]).__name__} {j}')
    ticks = [label.get_window_extent(renderer) for ax in axes
             for label in ax.get_yticklabels() if label.get_text() and label.get_visible()]
    for i in range(len(ticks)):
        for j in range(i + 1, len(ticks)):
            if ticks[i].overlaps(ticks[j]):
                raise RuntimeError('fog_degeneracy_kmax: two y tick labels touch')


fig, axes = plt.subplots(len(PANELS), 1, figsize=(3.5, PANEL_HEIGHT*len(PANELS)),
                         sharex=True, gridspec_kw={'hspace': 0.06})
for p, (ax, (ylabel, ylim, yticks, curves)) in enumerate(zip(axes, PANELS)):
    if p in REFERENCE_AT_ONE:
        ax.axhline(1.0, **REFERENCE)
    for lay in data['layouts']:
        st = LAYOUT_STYLE[lay['name']]
        for key, sub, scale, ls, filled in curves:
            values = lay[key] if sub is None else lay[key][sub]
            ax.plot(kmax, as_array(values, scale), color=lay['color'], lw=st['lw'], ls=ls,
                    marker=st['marker'], ms=MARKER_SIZE,
                    mfc=lay['color'] if filled else 'white', clip_on=False)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(False, axis='x')
fig.align_ylabels(axes)
axes[-1].set_xlim(kmax[0] - 0.01, kmax[-1] + 0.01)
axes[-1].set_xticks([0.05, 0.1, 0.15, 0.2])
axes[-1].set_xticklabels(['0.05', '0.1', '0.15', '0.2'])
axes[-1].set_xlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$', fontsize=10)

# the line styles above, the layouts below them, both above the top panel
style_legend = fig.legend(
    handles=[Line2D([], [], color='0.25', lw=1.3, ls=PRIOR_LS, marker='o', ms=MARKER_SIZE,
                    label=r'$\sigma_v$ on its Gaussian prior'),
             Line2D([], [], color='0.25', lw=1.3, ls=FLAT_LS, marker='o', ms=MARKER_SIZE,
                    mfc='white', label=r'$\sigma_v$ flat')],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.6, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0)
layout_legend = fig.legend(
    handles=[Line2D([], [], color=lay['color'], lw=LAYOUT_STYLE[lay['name']]['lw'],
                    marker=LAYOUT_STYLE[lay['name']]['marker'], ms=MARKER_SIZE,
                    label=lay['label']) for lay in data['layouts']],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.2, columnspacing=1.2, handletextpad=0.5, labelspacing=0.25,
    borderaxespad=0.0)
# stack the two legends just above the top panel, close together
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
top = inv.transform((0.0, axes[0].get_window_extent(renderer).y1))[1]
layout_legend.set_bbox_to_anchor((0.5, top + LEGEND_GAP), transform=fig.transFigure)
fig.canvas.draw()
layout_top = inv.transform((0.0, layout_legend.get_window_extent(renderer).y1))[1]
style_legend.set_bbox_to_anchor((0.5, layout_top + 0.5*LEGEND_GAP), transform=fig.transFigure)
check_clear(fig, list(axes), [style_legend, layout_legend, axes[-1].xaxis.label,
                              *[ax.yaxis.label for ax in axes]])
fig.savefig(os.path.join(HERE, 'fog_degeneracy_kmax.pdf'))
fig.savefig(os.path.join(HERE, 'fog_degeneracy_kmax.png'), dpi=300)
plt.close(fig)
