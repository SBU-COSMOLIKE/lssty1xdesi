"""Redraw fog_sigv_width.pdf from data.json: how wide the Gaussian prior of
the velocity dispersions sigma_v must be for them to absorb a DESI-size
alpha_2 = +12.5 h^-2 Mpc^2 in a fit without alpha_2 (model B + tilts on the
pessimistic linear-bias prior with alpha_2 and its tilts fixed at 0; at s = 6
it is the paper model), and what that fit's FoM is then.

One column of a two-column paper, two stacked panels against s, the factor
on all seven sigma_v prior widths, on a log axis; the flat sigma_v (s = inf)
are drawn as separate points at the right edge, past a gap:

1. delta_max: the largest shift of the fit, over the (w0, wa) shift in its
   ellipse and the Omega_m and A_s shifts in their own sigma; dotted: sigma/3;
2. FoM_{alpha_0 only}/FoM: FoM(w0, wa) of that fit (alpha_0 the only
   counterterm) over the reference model's (model B + tilts, alpha_2 free,
   sigma_v at s = 1); gray: 1.

The four layouts at k_max = 0.075 (solid, filled markers) and 0.2 (dashed,
open markers); each layout has a color, a line width and a marker. k_max is
the z = 0 value of a per-bin cut at k_max/D(zbar_i). data.json also holds
every cut, the three shifts, FoM(Omega_m, 10^9 A_s), the variants that widen
only the LRG or only the ELG sigma_v, and the passing s.
Run from any folder: python plot_fog_sigv_width.py
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
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
    'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
    'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
    'Delta z = 0.2': {'lw': 0.9, 'marker': '^'},
}
MARKER_SIZE = 3.5
# the cuts drawn: solid with filled markers, dashed with open ones
CUT_STYLE = {'0.075': {'ls': '-', 'filled': True},
             '0.2': {'ls': (0, (4, 2)), 'filled': False}}
# where the flat sigma_v sit on the log axis, past a gap after the last scanned s
X_FLAT = 25.0
X_GAP = 15.0
REFERENCE = {'color': '0.6', 'lw': 0.8}
THRESHOLD_STYLE = {'color': 'black', 'ls': ':', 'lw': 1.0}
PANEL_HEIGHT = 1.35
LEGEND_GAP = 0.004
# the panels: (data.json key, its flat value, y label, y limits, y ticks, log y)
PANELS = [('max_shift', 'max_shift_flat', r'$\delta_\mathrm{max}$ [$\sigma$]',
           (-0.08, 1.75), [0, 0.5, 1, 1.5], False),
          ('fom_ratio', 'fom_ratio_flat',
           r'$\mathrm{FoM}_{\alpha_0\,\mathrm{only}}/\mathrm{FoM}$', (0.6, 3.0), [1, 2],
           False)]

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
s_finite = np.array(data['s_finite'], dtype=float)


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
            raise RuntimeError(f'fog_sigv_width: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'fog_sigv_width: {type(artists[i]).__name__} {i} overlaps '
                                   f'{type(artists[j]).__name__} {j}')
    # the drawn tick labels: every panel's y labels and the bottom panel's x
    # labels (the shared x axis hides the others)
    shown = [*[label for ax in axes for label in ax.get_yticklabels()],
             *axes[-1].get_xticklabels()]
    ticks = [label.get_window_extent(renderer) for label in shown
             if label.get_text() and label.get_visible()]
    for i in range(len(ticks)):
        for j in range(i + 1, len(ticks)):
            if ticks[i].overlaps(ticks[j]):
                raise RuntimeError('fog_sigv_width: two tick labels touch')


fig, axes = plt.subplots(len(PANELS), 1, figsize=(3.5, PANEL_HEIGHT*len(PANELS)),
                         sharex=True, gridspec_kw={'hspace': 0.06})
for p, (ax, (key, key_flat, ylabel, ylim, yticks, logy)) in enumerate(zip(axes, PANELS)):
    if p == 0:
        ax.axhline(data['shift_threshold'], **THRESHOLD_STYLE)
    else:
        ax.axhline(1.0, **REFERENCE)
    # the gap between the scanned widths and the flat sigma_v
    # (short of the frame, so the legend above stays clear of it)
    ax.axvline(X_GAP, ymin=0.04, ymax=0.96, color='0.75', lw=0.6)
    for lay in data['layouts']:
        st = LAYOUT_STYLE[lay['name']]
        for cut, cs in CUT_STYLE.items():
            e = lay['all_sigma_v'][cut]
            if e is None:
                continue
            face = lay['color'] if cs['filled'] else 'white'
            ax.plot(s_finite, e[key], color=lay['color'], lw=st['lw'], ls=cs['ls'],
                    marker=st['marker'], ms=MARKER_SIZE, mfc=face, clip_on=False)
            ax.plot([X_FLAT], [e[key_flat]], color=lay['color'], ls='none',
                    marker=st['marker'], ms=MARKER_SIZE + 0.5, mfc=face, clip_on=False)
    ax.set_xscale('log')
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(False)
fig.align_ylabels(axes)
axes[-1].set_xlim(0.85, 32.0)
axes[-1].xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, X_FLAT]))
axes[-1].xaxis.set_minor_locator(NullLocator())
axes[-1].set_xticklabels(['1', '2', '5', '10', r'$\infty$'])
axes[-1].set_xlabel(r'$s$, the factor on every $\sigma_v$ prior width', fontsize=10)

style_legend = fig.legend(
    handles=[Line2D([], [], color='0.25', lw=1.3, ls=CUT_STYLE['0.075']['ls'], marker='o',
                    ms=MARKER_SIZE, label=r'$k_\mathrm{max} = 0.075$'),
             Line2D([], [], color='0.25', lw=1.3, ls=CUT_STYLE['0.2']['ls'], marker='o',
                    ms=MARKER_SIZE, mfc='white', label=r'$k_\mathrm{max} = 0.2$')],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.6, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0)
layout_legend = fig.legend(
    handles=[Line2D([], [], color=lay['color'], lw=LAYOUT_STYLE[lay['name']]['lw'],
                    marker=LAYOUT_STYLE[lay['name']]['marker'], ms=MARKER_SIZE,
                    label=lay['label']) for lay in data['layouts']],
    loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.2, columnspacing=1.2, handletextpad=0.5, labelspacing=0.25,
    borderaxespad=0.0)
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
fig.savefig(os.path.join(HERE, 'fog_sigv_width.pdf'))
fig.savefig(os.path.join(HERE, 'fog_sigv_width.png'), dpi=300)
plt.close(fig)
