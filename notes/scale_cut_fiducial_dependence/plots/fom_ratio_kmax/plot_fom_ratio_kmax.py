"""Redraw fom_ratio_kmax.pdf from data.json: how much the figures of merit of
the paper model move when its UV-fraction scale cut is computed on the paper
model's own fiducial instead of the linear-bias fiducial.

The paper model is model B + tilts without alpha_2 (notes/higher_order_bias),
with the seven sigma_v prior widths 6 times those of the exploration fits, on
the pessimistic linear-bias prior, DESI C_ell^gg alone. Each panel draws, per bin layout, the ratio of a figure
of merit with the paper-fiducial cut to the same figure of merit with the
linear-bias cut: FoM(w0, wa) on top, FoM(Omega_m, 10^9 A_s) below, both on
one linear y axis (the same quantity). A gray dotted line marks a ratio of 1;
it has no legend entry. Each layout has a color, a line width and a marker at
every cut, so the figure reads in black and white. k_max is the z = 0 value
of a per-bin cut at k_max/D(zbar_i), intersected with the shipped mask; cuts
up to KMAX_MAX are drawn, and a cut without a fit has no point.

data.json is written by make_scale_cut_fiducial_data.py in this note's plots/
folder. Run from any folder: python plot_fom_ratio_kmax.py
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
NAME = 'fom_ratio_kmax'
# Each layout has a line width and a marker as well as its color, so the
# layouts stay apart in black and white; the marker sits at every cut.
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
    'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
    'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
    'Delta z = 0.2': {'lw': 0.9, 'marker': '^'},
}
MARKER_SIZE = 3.5
# the ratio of 1: gray and dotted, no legend entry (the caption names it)
UNITY_STYLE = {'color': '0.5', 'ls': ':', 'lw': 1.0}
# the largest cut drawn [h/Mpc at z = 0]; data.json may hold larger ones
KMAX_MAX = 0.2
KMAX_TOLERANCE = 1.0e-9
# the height of one panel [in]
PANEL_HEIGHT = 1.2
# the y margin around the data, as a fraction of the data span, and the
# extra headroom tried in turn until the panel names clear every line
MARGIN = 0.12
HEADROOM = [0.0, 0.15, 0.3, 0.5, 0.8, 1.2]
# the panels: data.json field and the panel name written inside it
PANELS = [('fom_ratio', r'FoM$(w_0, w_a)$'),
          ('fom_om_as_ratio', r'FoM$(\Omega_\mathrm{m}, 10^9 A_\mathrm{s})$')]
CORNERS = {
    'upper left': (0.04, 0.93, 'left', 'top'),
    'upper right': (0.96, 0.93, 'right', 'top'),
    'lower right': (0.96, 0.07, 'right', 'bottom'),
    'lower left': (0.04, 0.07, 'left', 'bottom'),
}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax_all = np.array(data['kmax_h_per_Mpc'], dtype=float)
DRAWN_CUTS = kmax_all <= KMAX_MAX + KMAX_TOLERANCE
kmax = kmax_all[DRAWN_CUTS]
layouts = data['layouts']


def as_array(values):
    '''A list of numbers and None, one per cut of data.json, as a float array
    of the drawn cuts, NaN for None (a gap).'''
    out = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if value is not None:
            out[i] = float(value)
    return out[DRAWN_CUTS]


def line_points(ax):
    '''Display coordinates of points densely sampled along every drawn line of
    ax (the reference line included), inside the axes only.'''
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
    '''True when a display box, with a 3% / 8% margin, holds a point of a line.'''
    padded = box.expanded(1.03, 1.08)
    for ax in axes:
        pts = line_points(ax)
        hit = ((pts[:, 0] > padded.x0) & (pts[:, 0] < padded.x1)
               & (pts[:, 1] > padded.y0) & (pts[:, 1] < padded.y1))
        if np.any(hit):
            return True
    return False


def place_text(fig, axes, ax, text, corner):
    '''Put a text in one corner of a panel; None (text removed) when it
    touches a line.'''
    x, y, ha, va = CORNERS[corner]
    candidate = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5)
    fig.canvas.draw()
    box = candidate.get_window_extent(fig.canvas.get_renderer())
    if touches_lines(box, axes):
        candidate.remove()
        return None
    return candidate


def check_clear(fig, axes, artists):
    '''Raise when a legend or label touches a drawn line or another one.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    for box, artist in zip(boxes, artists):
        if touches_lines(box, axes):
            raise RuntimeError(f'{NAME}: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'{NAME}: {type(artists[i]).__name__} {i} overlaps '
                                   f'{type(artists[j]).__name__} {j}')


fig, axes = plt.subplots(len(PANELS), 1, figsize=(3.5, PANEL_HEIGHT*len(PANELS)),
                         sharex=True, gridspec_kw={'hspace': 0.06})
flat = list(axes)
values = [1.0]
for ax, (field, name) in zip(axes, PANELS):
    ax.axhline(1.0, **UNITY_STYLE)
    for lay in layouts:
        st = LAYOUT_STYLE[lay['name']]
        y = as_array(lay[field])
        ax.plot(kmax, y, color=lay['color'], lw=st['lw'], marker=st['marker'],
                ms=MARKER_SIZE)
        values += [float(v) for v in y[np.isfinite(y)]]
    ax.grid(False, axis='x')
low, high = min(values), max(values)
span = high - low
texts = []
for headroom in HEADROOM:
    for ax in axes:
        ax.set_ylim(low - MARGIN*span, high + (MARGIN + headroom)*span)
    for text in texts:
        text.remove()
    texts = []
    for ax, (field, name) in zip(axes, PANELS):
        for corner in ('upper left', 'upper right', 'lower left', 'lower right'):
            placed = place_text(fig, flat, ax, name, corner)
            if placed is not None:
                texts.append(placed)
                break
    if len(texts) == len(PANELS):
        break
if len(texts) != len(PANELS):
    raise RuntimeError(f'{NAME}: a panel name finds no corner free of lines')
axes[-1].set_xlim(kmax[0] - 0.01, kmax[-1] + 0.01)
axes[-1].set_xticks([0.05, 0.1, 0.15, 0.2])
axes[-1].set_xticklabels(['0.05', '0.1', '0.15', '0.2'])
axes[-1].set_xlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$', fontsize=10)
ylabel = fig.supylabel('paper-fiducial cut / linear-bias cut', fontsize=9.5, x=-0.02)

layout_legend = axes[0].legend(
    handles=[Line2D([], [], color=lay['color'], lw=LAYOUT_STYLE[lay['name']]['lw'],
                    marker=LAYOUT_STYLE[lay['name']]['marker'], ms=MARKER_SIZE,
                    label=lay['label']) for lay in layouts],
    loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8,
    handlelength=2.2, columnspacing=1.2, handletextpad=0.5, labelspacing=0.25,
    borderaxespad=0.2, borderpad=0.1)
check_clear(fig, flat, [layout_legend, *texts, axes[-1].xaxis.label])
fig.savefig(os.path.join(HERE, NAME + '.pdf'))
fig.savefig(os.path.join(HERE, NAME + '.png'), dpi=300)
plt.close(fig)
