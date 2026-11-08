"""Redraw fom_kmax.pdf and fom_kmax.png from data.json: the dark-energy
figure of merit of a linear-bias fit to the DESI band powers C_ell^gg
alone, against the scale cut k_max.

FoM(w0, wa) = [det Cov(w0, wa)]^(-1/2) is the inverse area of the fit's
marginalized (w0, wa) error ellipse: a larger FoM is a smaller ellipse.
k_max is the z = 0 value of a per-bin cut: bin i keeps a band power
while less than 5% of it comes from wavenumbers above k_max/D(zbar_i),
with D the linear growth factor and zbar_i the bin's mean redshift.

The figure is one column of a two-column paper: one panel per bin
layout, stacked, all on the same logarithmic y range. Each panel draws
the linear-bias priors that pass the accuracy test of the linear-bias
priors (baseline, pessimistic, free). The growth-tied prior fails that
test, so its FoM is not drawn; data.json keeps it. A gray dashed line
marks the LSST Y1 3x2pt FoM(w0, wa) for scale.

data.json holds the fom entries of ../fob_kmax/data.json, which
EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb writes; a cut that leaves no
data has no fit and is stored as null.
Run from any folder: python plot_fom_kmax.py
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
# The priors drawn, with the colors of the shift figure fob_kmax so a prior
# keeps one color across the note. Drawn in this order: on Delta z = 0.2
# the three priors coincide (every group holds one or two bins), and the
# free line, drawn last, covers the other two.
PRIOR_STYLE = {
    'baseline': {'color': 'C0', 'label': 'baseline prior'},
    'pessimistic': {'color': 'C3', 'label': 'pessimistic prior'},
    'independent': {'color': 'C2', 'label': 'free'},
}
# the LSST Y1 reference: gray with long dashes, so it reads as a reference
# and not as one more prior
LSST_STYLE = {
    'color': '0.5',
    'ls': (0, (7, 3)),
    'lw': 1.1,
}
# One y range for every panel, in FoM units: the smallest drawn FoM is
# 0.036 (free prior, 62 Mpc, k_max = 0.05) and the largest about 38, so
# 0.02 to 100 holds every curve and the LSST Y1 line with little white
# space.
Y_RANGE = (0.02, 100.0)
# Grow each label's and the legend's box by this factor before the
# overlap test, so that a label grazing a line also counts as touching it.
BOX_MARGIN = 1.06


def line_points(ax):
    """Return display coordinates of points densely sampled along every line of ax.

    Straight segments between data points are sampled in display space,
    which is where a log axis draws them straight. An axhline spans the
    axes in x (axes units) and sits at one y value (data units), so it is
    sampled across the x limits instead.

    Arguments:
      ax = a matplotlib Axes whose lines are already drawn.

    Returns:
      numpy array of shape (n, 2): x and y in display pixels.
    """
    pts = []
    for line in ax.get_lines():
        xd, yd = (np.asarray(v, dtype=float) for v in line.get_data())
        if line.get_transform() != ax.transData:
            xs = np.linspace(*ax.get_xlim(), 400)
            ys = np.full(400, yd[0])
            pts.append(ax.transData.transform(np.column_stack([xs, ys])))
            continue
        # a null FoM (no fit) is NaN: those points and their segments are
        # not drawn, so they are skipped here too
        disp = ax.transData.transform(np.column_stack([xd, yd]))
        for i in range(len(xd) - 1):
            if not np.all(np.isfinite(disp[i:i + 2])):
                continue
            t = np.linspace(0.0, 1.0, 60)[:, None]
            pts.append(disp[i] + t*(disp[i + 1] - disp[i]))
    return np.vstack(pts)


def check_clear(fig, artists, axes):
    """Raise when a text label or the legend touches any drawn line.

    The check runs on the rendered figure, in display pixels: every line
    of every panel (the reference line included) is sampled densely, and
    no sample may fall inside the slightly grown bounding box of any
    label or legend. A visual check misses a line that grazes a label;
    this test does not.

    Arguments:
      fig = the figure, already laid out;
      artists = the text labels and legends that must stay off the data;
      axes = the panels whose lines are tested.

    Raises:
      RuntimeError naming the artist that touches a line; the fix is to
      move that artist or widen Y_RANGE.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = np.vstack([line_points(ax) for ax in axes])
    for artist in artists:
        box = artist.get_window_extent(renderer).expanded(BOX_MARGIN, BOX_MARGIN)
        inside = ((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1)
                  & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1))
        if np.any(inside):
            raise RuntimeError(f'fom_kmax: {artist} touches a drawn line; move it '
                               'or widen Y_RANGE')


with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax = np.array(data['kmax_h_per_Mpc'], dtype=float)
layouts = data['layouts']
lsst = data['lsst_y1_3x2pt_fom']

fig, axes = plt.subplots(len(layouts), 1, figsize=(3.5, 1.1*len(layouts)), sharex=True,
                         sharey=True, gridspec_kw={'hspace': 0.06})
labels = []
for ax, lay in zip(axes, layouts):
    ax.axhline(lsst, **LSST_STYLE)
    for key, st in PRIOR_STYLE.items():
        # null (no fit at that cut) becomes NaN, which matplotlib leaves
        # out of the line
        fom = np.array([np.nan if v is None else v for v in lay['fom'][key]], dtype=float)
        ax.plot(kmax, fom, color=st['color'], lw=1.4, marker='o', ms=2.8)
    ax.set_yscale('log')
    ax.set_ylim(*Y_RANGE)
    # no tick at the top edge, where its label would reach into the gap
    # between two panels
    ax.set_yticks([1e-1, 1e0, 1e1])
    # the layout's name in the panel's lower right: the FoM grows with
    # k_max, so every curve is high there
    labels.append(ax.text(0.97, 0.07, lay['label'], transform=ax.transAxes, ha='right',
                          va='bottom', fontsize=8.5))
axes[-1].set_xlim(kmax[0] - 0.01, kmax[-1] + 0.01)
axes[-1].set_xlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$', fontsize=10)
fig.supylabel(r'FoM$(w_0, w_a)$, DESI $C_\ell^{gg}$ alone', x=-0.02, fontsize=10)

handles = [Line2D([], [], color=st['color'], lw=1.4, marker='o', ms=2.8, label=st['label'])
           for st in PRIOR_STYLE.values()]
handles.append(Line2D([], [], label='LSST Y1 3x2pt', **LSST_STYLE))
# two rows above the top panel, where no data are drawn
legend = axes[0].legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                        ncol=2, frameon=False, fontsize=8, handlelength=2.2,
                        columnspacing=1.2, handletextpad=0.5, labelspacing=0.3)
check_clear(fig, labels + [legend], axes)
fig.savefig(os.path.join(HERE, 'fom_kmax.pdf'))
fig.savefig(os.path.join(HERE, 'fom_kmax.png'), dpi=300)
plt.close(fig)
