"""Redraw the two FoM figures of this folder from data.json: the figures of
merit of each DESI layout on the linear-bias priors that pass the
accuracy test (baseline, pessimistic, free), under model A (one-loop
bias, notes/higher_order_bias) in its two variants: model A (solid, filled
markers) and model A + tilts (dashed, open markers). Each layout has its own color, line width and marker, so the
figures read in black and white.
The growth-tied prior is not drawn: the accuracy test shows it biases the
cosmology by many sigma, so its FoM means nothing (data.json keeps it).

  baseline_ladder.pdf       - FoM(w0, wa)
  baseline_ladder_om_as.pdf - FoM(Omega_m, 10^9 A_s)

Each is one column of a two-column paper and stands alone (the LaTeX can
set them side by side). The LSST Y1 3x2pt reference of each figure of merit
is drawn as a line when it is within reach of the DESI points, and is
otherwise quoted as a text comment inside the panel. data.json is written by
EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb; it also holds the v1 linear-bias
and the exact-calibration FoMs of (w0, wa), which these figures do not draw.
Run from any folder: python plot_baseline_ladder.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

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
# Each layout has a line width and a marker as well as its color, so the
# layouts stay apart in black and white; the marker sits at each prior.
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
    'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
    'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
    'Delta z = 0.2': {'lw': 0.9, 'marker': '^'},
}
# the marker size [pt]
MARKER_SIZE = 4.0
# The two variants of model A: solid with filled markers, dashed
# with open markers. The dash and gap lengths are in points: matplotlib
# scales a dash pattern by the line width, so dividing by it keeps the
# dashes visible on the thinnest line.
DASH = (4.5, 2.5)
STYLES = {
    'A': {'dashed': False, 'open': False, 'label': 'model A'},
    'A tilts': {'dashed': True, 'open': True, 'label': 'model A + tilts'},
}


def line_style(lay, variant, color):
    '''The plot keywords of one layout (lay: its name) and one variant.'''
    st = LAYOUT_STYLE[lay]
    ls = (0, (DASH[0]/st['lw'], DASH[1]/st['lw'])) if STYLES[variant]['dashed'] else '-'
    face = 'white' if STYLES[variant]['open'] else color
    return {'color': color, 'ls': ls, 'lw': st['lw'], 'marker': st['marker'],
            'ms': MARKER_SIZE, 'mfc': face, 'mec': color}


# the LSST Y1 reference: a gray line with long dashes, distinct from the
# dashed model A + tilts lines
LSST_STYLE = {'color': '0.5', 'ls': (0, (7, 3)), 'lw': 1.2}
# a reference more than this factor above the highest DESI point is not
# drawn: the axis would be stretched for one line, so a text comment inside
# the panel quotes it instead
LSST_OFF_SCALE = 1.5
# the names the figures print for the linear-bias priors (the data keys
# keep the notebook's names)
DISPLAY = {'tied': 'growth-tied', 'baseline': 'baseline',
           'pessimistic': 'pessimistic', 'independent': 'free'}
# one figure per figure of merit: data.json field, y label, the data.json
# key of its LSST Y1 3x2pt reference, output name
FIGURES = [('fom', r'FoM$(w_0, w_a)$, DESI $C_\ell^{gg}$ alone', 'lsst_y1_3x2pt_fom',
            'baseline_ladder'),
           ('fom_om_as', r'FoM$(\Omega_\mathrm{m}, 10^9 A_\mathrm{s})$, DESI $C_\ell^{gg}$ alone',
            'lsst_y1_3x2pt_fom_om_as', 'baseline_ladder_om_as')]

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
# the priors that pass the accuracy test (growth-tied fails it, so its FoM
# is not drawn)
rungs = [r['key'] for r in data['rungs'] if r['key'] != 'tied']
x = np.arange(len(rungs))
layouts = data['layouts']
# k_max is the z = 0 value of the per-bin cut at k_max/D(z) (the caption
# explains the UV-fraction criterion and the shipped mask)
KMAX_TEXT = (rf'$k_\mathrm{{max}} = {data["kmax_h_per_Mpc"]:g}\,h\,\mathrm{{Mpc}}^{{-1}}$'
             r' at $z = 0$')


def line_points(ax):
    '''Display coordinates of points densely sampled along every line of ax,
    the LSST Y1 line included.'''
    pts = []
    for line in ax.get_lines():
        xd, yd = (np.asarray(v, dtype=float) for v in line.get_data())
        if line.get_transform() != ax.transData:
            # axhline: x in axes units, y in data units
            xs = np.linspace(*ax.get_xlim(), 400)
            pts.append(ax.transData.transform(np.column_stack([xs, np.full(400, yd[0])])))
            continue
        for i in range(len(xd) - 1):
            t = np.linspace(0.0, 1.0, 60)
            pts.append(ax.transData.transform(np.column_stack(
                [xd[i] + t*(xd[i + 1] - xd[i]), yd[i] + t*(yd[i + 1] - yd[i])])))
    return np.vstack(pts)


for field, ylabel, lsst_key, name in FIGURES:
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for lay in layouts:
        for variant in STYLES:
            y = np.array([lay[field][variant][k] for k in rungs])
            ax.plot(x, y, **line_style(lay['name'], variant, lay['color']))
    top = max(lay[field][v][k] for lay in layouts for v in STYLES for k in rungs)
    lsst = data.get(lsst_key)
    lsst_text = None
    if lsst is not None and lsst <= LSST_OFF_SCALE*top:
        ax.axhline(lsst, **LSST_STYLE)
        top = max(top, lsst)
    elif lsst is not None:
        # far above every DESI point: quoted inside the panel instead
        lsst_text = f'LSST Y1 3x2pt: {lsst:.0f}'
    ax.set_xlim(-0.35, x[-1] + 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[k] for k in rungs], fontsize=9.5)
    ax.grid(False, axis='x')
    ax.set_ylabel(ylabel, fontsize=10)

    # the layouts (color, line width and marker), inside the panel in its
    # empty upper right, under the cut they share
    layout_legend = ax.legend(
        handles=[Line2D([], [], label=lay['label'],
                        **line_style(lay['name'], 'A', lay['color']))
                 for lay in layouts],
        loc='upper right', ncol=2, frameon=False, fontsize=8, handlelength=2.0,
        labelspacing=0.2, handletextpad=0.5, columnspacing=0.9, borderaxespad=0.25,
        borderpad=0.2, title=KMAX_TEXT, title_fontsize=8)
    ax.add_artist(layout_legend)
    # the two variants above the panel, the same in both figures so their
    # panels line up side by side; each is shown by its four drawn lines side
    # by side (solid with filled markers, dashed with open ones)
    style_handles = [tuple(Line2D([], [], **line_style(lay['name'], variant, lay['color']))
                           for lay in layouts) for variant in STYLES]
    ax.legend(handles=style_handles, labels=[st['label'] for st in STYLES.values()],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
              loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=8.5, handlelength=6.0, columnspacing=1.2,
              handletextpad=0.5)

    # the off-scale LSST Y1 value, as a comment just under the layout legend
    note = None
    if lsst_text is not None:
        note = ax.text(0.0, 0.0, lsst_text, fontsize=8, color='0.35', ha='right', va='top')

    # The legend (and the comment) inside the panel must never sit on the
    # data. The y axis gets the least headroom at which they clear every
    # line (a drawn LSST reference included), so the axis is not stretched
    # more than needed.
    for headroom in np.arange(1.12, 2.02, 0.02):
        ax.set_ylim(0.0, top*headroom)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = layout_legend.get_window_extent(renderer)
        if note is not None:
            # right-aligned with the legend, just below it (axes coordinates)
            corner = ax.transAxes.inverted().transform((box.x1, box.y0))
            note.set_transform(ax.transAxes)
            note.set_position((corner[0] - 0.01, corner[1] - 0.01))
            fig.canvas.draw()
            box = Bbox.union([box, note.get_window_extent(renderer)])
        box = box.expanded(1.03, 1.05)
        pts = line_points(ax)
        if not np.any((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1)
                      & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1)):
            break
    else:
        raise RuntimeError(f'{name}: the layout legend touches a line at every headroom')
    fig.savefig(os.path.join(HERE, f'{name}.pdf'))
    fig.savefig(os.path.join(HERE, f'{name}.png'), dpi=300)
    plt.close(fig)
