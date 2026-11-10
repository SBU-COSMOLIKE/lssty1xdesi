"""Redraw pair_threshold.pdf from data.json: what model B + tilts on the
pessimistic linear-bias prior (alpha_2 free, the exploration sigma_v widths;
the reference model of the Fingers-of-God study) keeps of its figures of merit
when the data vector holds only the cross pairs whose Fingers-of-God-blurred
kernels overlap by at least O_min at the fiducial sigma_v.

One column of a two-column paper, two stacked panels against O_min on a log
axis, with "all" (every listed pair, the current data vector) as a separate
point at the left, past a thin gray separator:

1. FoM(w0, wa) at O_min over FoM with every listed pair (FoM_all);
2. the same for FoM(Omega_m, 10^9 A_s).
The y axis is shared; each panel's name is on its right.

The four layouts at k_max = 0.2 (solid, filled markers) and 0.075 (dashed,
open markers); each layout has a color, a line width and a marker. Gray: 1.
k_max is the z = 0 value of a per-bin cut at k_max/D(zbar_i).
Run from any folder: python plot_pair_threshold.py
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
CUT_STYLE = {'0.2': {'ls': '-', 'filled': True},
             '0.075': {'ls': (0, (4, 2)), 'filled': False}}
# where "all" sits on the log axis, past a separator
X_ALL = 1.0e-8
X_GAP = 1.0e-7
REFERENCE = {'color': '0.6', 'lw': 0.8}
PANEL_HEIGHT = 1.3
LEGEND_GAP = 0.004
# the panels: data.json key and its name, on the right side
PANELS = [('fom_ratio', r'$(w_0, w_a)$'),
          ('fom_om_as_ratio', r'$(\Omega_\mathrm{m}, 10^9 A_\mathrm{s})$')]
# one y range for both panels, a round step above and below the data, the
# ticks kept off the panel edges so the stacked panels' labels never touch
Y_STEP = 0.2
Y_FLOOR_STEP = 0.1
Y_PAD = 0.06

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
o_min = np.array(data['o_min'], dtype=float)


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
            raise RuntimeError(f'pair_threshold: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'pair_threshold: {type(artists[i]).__name__} {i} overlaps '
                                   f'{type(artists[j]).__name__} {j}')
    shown = [*[label for ax in axes for label in ax.get_yticklabels()],
             *axes[-1].get_xticklabels()]
    ticks = [label.get_window_extent(renderer) for label in shown
             if label.get_text() and label.get_visible()]
    for i in range(len(ticks)):
        for j in range(i + 1, len(ticks)):
            if ticks[i].overlaps(ticks[j]):
                raise RuntimeError('pair_threshold: two tick labels touch')


fig, axes = plt.subplots(len(PANELS), 1, figsize=(3.5, PANEL_HEIGHT*len(PANELS)),
                         sharex=True, gridspec_kw={'hspace': 0.06})
all_values = []
for ax, (key, ylabel) in zip(axes, PANELS):
    ax.axhline(1.0, **REFERENCE)
    ax.axvline(X_GAP, ymin=0.04, ymax=0.96, color='0.75', lw=0.6)
    values = [1.0]
    for lay in data['layouts']:
        st = LAYOUT_STYLE[lay['name']]
        for cut, cs in CUT_STYLE.items():
            per = lay['scan'].get(cut)
            if per is None:
                continue
            face = lay['color'] if cs['filled'] else 'white'
            y = [per[f'{o:.0e}'].get(key, np.nan) for o in o_min]
            ax.plot(o_min, y, color=lay['color'], lw=st['lw'], ls=cs['ls'],
                    marker=st['marker'], ms=MARKER_SIZE, mfc=face, clip_on=False)
            ax.plot([X_ALL], [per['current'][key]], color=lay['color'], ls='none',
                    marker=st['marker'], ms=MARKER_SIZE + 0.5, mfc=face, clip_on=False)
            values += [v for v in y if np.isfinite(v)]
    all_values.extend(values)
    ax.set_xscale('log')
    # the panel's name on the right side, rotated like a y label
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(ylabel, rotation=90, fontsize=9, labelpad=5)
    ax.grid(False)
y_lo = np.floor(min(all_values)/Y_FLOOR_STEP)*Y_FLOOR_STEP
y_hi = np.ceil(max(all_values)/Y_STEP)*Y_STEP
ticks = [v for v in np.round(np.arange(0.0, y_hi + 1e-9, Y_STEP), 6) if v >= y_lo - 1e-9]
for ax in axes:
    ax.set_ylim(y_lo - Y_PAD, y_hi + Y_PAD)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
axes[-1].set_xlim(4.0e-9, 3.0e-2)
axes[-1].xaxis.set_major_locator(FixedLocator([X_ALL, 1e-6, 1e-4, 1e-3, 1e-2]))
axes[-1].xaxis.set_minor_locator(NullLocator())
axes[-1].set_xticklabels(['all', r'$10^{-6}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$'])
axes[-1].set_xlabel(r'$O_\mathrm{min}$: cross pairs kept if $O_{ij} \geq O_\mathrm{min}$',
                    fontsize=10)

style_legend = fig.legend(
    handles=[Line2D([], [], color='0.25', lw=1.3, ls=CUT_STYLE['0.2']['ls'], marker='o',
                    ms=MARKER_SIZE, label=r'$k_\mathrm{max} = 0.2$'),
             Line2D([], [], color='0.25', lw=1.3, ls=CUT_STYLE['0.075']['ls'], marker='o',
                    ms=MARKER_SIZE, mfc='white', label=r'$k_\mathrm{max} = 0.075$')],
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
# the shared y label just left of the widest tick label
fig.canvas.draw()
tick_left = min(label.get_window_extent(renderer).x0 for ax in axes
                for label in ax.get_yticklabels() if label.get_text())
tick_left = inv.transform((tick_left, 0.0))[0]
supylabel = fig.supylabel(r'$\mathrm{FoM}/\mathrm{FoM}_\mathrm{all}$', x=tick_left - 0.01,
                          ha='right', fontsize=10)
check_clear(fig, list(axes), [style_legend, layout_legend, supylabel, axes[-1].xaxis.label,
                              *[ax.yaxis.label for ax in axes]])
fig.savefig(os.path.join(HERE, 'pair_threshold.pdf'))
fig.savefig(os.path.join(HERE, 'pair_threshold.png'), dpi=300)
plt.close(fig)
