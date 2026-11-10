"""Redraw fog_templates.pdf from data.json: the alpha_2 and sigma_v
derivative templates of three DESI spectra of the Delta d_com = 62 Mpc
layout, and what is left of the alpha_2 template after its best combination
of the sigma_v templates (after Fig. 1 of arXiv:1706.09412).

A full-width strip of two rows and three columns (a BGS auto spectrum, an
LRG auto spectrum and the LRG cross spectrum of that bin with the next one).
Every curve is in units of the band-power error sigma_ell = sqrt(C_ii) of
the Gaussian covariance, against the band center ell, on the bands the
shipped mask keeps. Each parameter p is taken in units of its prior width
Delta (Delta, not sigma: sigma_v is itself a dispersion), as in Fig. 1 of
arXiv:1706.09412.

- Top row: sigma_ell^-1 dC_ell/dp for p = alpha_2/Delta alpha_2 (alpha_2 of
  the bin's group, Delta alpha_2 = 0.5/k_max^2 at k_max = 0.2 h/Mpc) and for
  p = sigma_v/Delta sigma_v (the bin's velocity dispersion and its prior
  width).
- Bottom row: sigma_ell^-1 r_ell, with
      r_ell = dC_ell/d(alpha_2/Delta alpha_2)
              - sum_g c_g dC_ell/d(sigma_v,g/Delta sigma_v,g),
  the leftover after the best combination c of all seven sigma_v, fitted
  over the whole data vector in the C^-1 metric on the bands a cut keeps,
  drawn on those bands only, for the cuts k_max = 0.2 and 0.075 h/Mpc.
- Vertical dotted lines: where the shown spectrum's kept bands end at
  k_max = 0.075 (thin) and 0.2 (thick); at 0.2 the shipped mask ends them
  first. k_max is the z = 0 value of a per-bin cut at k_max/D(zbar_i).

Each row shares one y axis; the BGS column, larger than the others, is
divided by a round constant written inside its panels. Each curve has a
color, a line style, a line width and a marker.

data.json also holds the Delta z = 0.2 layout (not drawn: its wide bins
barely see alpha_2). Run from any folder: python plot_fog_templates.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

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
# the layout drawn
LAYOUT = 'Delta d_com = 62 Mpc'
# the alpha_2 template is scaled by its prior width at this cut [h/Mpc at z = 0]
ALPHA2_CUT = '0.2'
# the curves: color, line style, width and marker
STYLE = {
    'alpha2': {'color': '#D55E00', 'ls': '-', 'lw': 1.8, 'marker': 'o', 'ms': 3.0},
    'sigma_v': {'color': '#0072B2', 'ls': (0, (4, 2)), 'lw': 1.3, 'marker': 's', 'ms': 3.0},
    '0.2': {'color': 'black', 'ls': '-', 'lw': 1.6, 'marker': 'D', 'ms': 3.0},
    '0.075': {'color': '#CC79A7', 'ls': (0, (3, 1.5)), 'lw': 1.2, 'marker': '^', 'ms': 3.8},
}
# the cut lines: width per cut (dotted, gray)
CUT_LINE_WIDTH = {'0.075': 0.8, '0.2': 1.7}
CUT_LINE_COLOR = '0.45'
# the y headroom of each row beyond its largest value, and the panel size [in]
Y_MARGIN = 1.12
PANEL_HEIGHT = 1.2
FIG_WIDTH = 6.4
# the round constants a column can be divided by, so that every column of a
# row fills a similar part of the shared y axis
DIVISORS = [1, 2, 5, 10]
COLUMN_SPAN = 1.8
# no x tick within this fraction of the axis length from a panel's right edge
TICK_EDGE = 0.04

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
layout = next(lay for lay in data['layouts'] if lay['name'] == LAYOUT)
panels = layout['panels']


def column_title(p):
    if len(p['bins']) == 1:
        return f"{p['kind']}, bin {p['bins'][0]} " + rf"($\bar z = {p['zbar'][0]:.2f}$)"
    return (f"{p['kind']}, bins {p['bins'][0]}" + r'$\times$' + f"{p['bins'][1]} "
            + rf"($\bar z = {p['zbar'][0]:.2f}, {p['zbar'][1]:.2f}$)")


def curves(p):
    '''The drawn values of one column: {name: (ell, y)} in units of sigma_ell.'''
    ell = np.array(p['ell'])
    sigma = np.array(p['sigma'])
    shipped = np.array(p['shipped_mask'])
    w_a2 = p['cuts'][ALPHA2_CUT]['alpha2_prior_width']
    out = {'alpha2': (ell[shipped], (np.array(p['dC_dalpha2'])*w_a2/sigma)[shipped]),
           'sigma_v': (ell[shipped], (np.array(p['dC_dsigma_v'])
                                      * p['sigma_v_prior_width_km_s']/sigma)[shipped])}
    for cut in ('0.2', '0.075'):
        kept = np.array(p['cuts'][cut]['kept'])
        resid = np.array(p['cuts'][cut]['residual_per_unit_alpha2'])*w_a2/sigma
        out[cut] = (ell[kept], resid[kept])
    return out


def line_points(ax, skip=()):
    '''Display coordinates of points densely sampled along every drawn line of
    ax (reference lines included), inside the axes only.'''
    box = ax.get_window_extent()
    pts = [np.zeros((0, 2))]
    for line in ax.get_lines():
        if line in skip:
            continue
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
    padded = box.expanded(1.04, 1.12)
    for ax in axes:
        pts = line_points(ax)
        if np.any((pts[:, 0] > padded.x0) & (pts[:, 0] < padded.x1)
                  & (pts[:, 1] > padded.y0) & (pts[:, 1] < padded.y1)):
            return True
    return False


def place_text(fig, ax, text, axes):
    '''Put a short text in the first corner of ax that touches no line;
    raise when none is free.'''
    fig.canvas.draw()
    for x, y, ha, va in ((0.04, 0.94, 'left', 'top'), (0.96, 0.94, 'right', 'top'),
                         (0.04, 0.06, 'left', 'bottom'), (0.96, 0.06, 'right', 'bottom')):
        artist = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8)
        fig.canvas.draw()
        if not touches_lines(artist.get_window_extent(fig.canvas.get_renderer()), axes):
            return artist
        artist.remove()
    raise RuntimeError(f'fog_templates: no free corner for "{text}"')


def check_clear(fig, axes, artists, outside=()):
    '''Raise when a legend or label touches a drawn line or another one. The
    artists in outside (the column titles, above the frames) are only checked
    against the other artists: the cut lines run up to the frame's top edge.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = list(artists) + list(outside)
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    for box, artist in zip(boxes, artists):
        if artist in outside:
            for ax in axes:
                frame = ax.get_window_extent(renderer)
                if (box.x1 > frame.x0 and box.x0 < frame.x1 and box.y0 < frame.y1 - 0.5
                        and box.y1 > frame.y0):
                    raise RuntimeError('fog_templates: a column title overlaps a panel')
            continue
        if touches_lines(box, axes):
            raise RuntimeError(f'fog_templates: {type(artist).__name__} touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(f'fog_templates: {type(artists[i]).__name__} {i} overlaps '
                                   f'{type(artists[j]).__name__} {j}')


values = [curves(p) for p in panels]
# one divisor per column (the same in both rows): the smallest round value
# that brings the column's largest value within COLUMN_SPAN of the other
# columns' (top row decides)
top_max = [max(np.abs(v['alpha2'][1]).max(), np.abs(v['sigma_v'][1]).max()) for v in values]
divisor = []
for col, peak in enumerate(top_max):
    others = min(top_max)
    divisor.append(next(d for d in DIVISORS if peak/d <= COLUMN_SPAN*others))

fig, axes = plt.subplots(2, len(panels), figsize=(FIG_WIDTH, 2*PANEL_HEIGHT),
                         sharey='row', gridspec_kw={'hspace': 0.06, 'wspace': 0.06})
rows = {0: ('alpha2', 'sigma_v'), 1: ('0.2', '0.075')}
row_values = {0: [], 1: []}
for col, (p, v) in enumerate(zip(panels, values)):
    for row, names in rows.items():
        ax = axes[row, col]
        ax.axhline(0.0, color='0.6', lw=0.6)
        for name in names:
            x, y = v[name]
            st = STYLE[name]
            ax.plot(x, y/divisor[col], color=st['color'], ls=st['ls'], lw=st['lw'],
                    marker=st['marker'], ms=st['ms'], clip_on=False)
            row_values[row] += list(np.abs(y/divisor[col]))
        for cut, width in CUT_LINE_WIDTH.items():
            edge = p['cuts'][cut]['cut_ell']
            if edge is not None:
                ax.axvline(edge, color=CUT_LINE_COLOR, ls=':', lw=width)
        ax.grid(False)
        ell = np.array(p['ell'])[np.array(p['shipped_mask'])]
        half_band = 0.5*(p['ell'][1] - p['ell'][0])
        x_hi = ell.max() + 2*half_band
        ax.set_xlim(0.0, x_hi)
        # round ticks; the panels nearly touch, so a column after the first
        # drops its 0 and no tick sits within TICK_EDGE of the right edge
        ticks = MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(0.0, x_hi)
        ticks = [x for x in ticks if (x > 0.0 or col == 0) and x <= (1.0 - TICK_EDGE)*x_hi]
        ax.set_xticks(ticks)
    axes[0, col].set_title(column_title(p), fontsize=8.5, pad=3)
    axes[0, col].tick_params(labelbottom=False)
for row in rows:
    top = Y_MARGIN*max(row_values[row])
    axes[row, 0].set_ylim(-top, top)
axes[0, 0].set_ylabel(r'$\sigma_\ell^{-1}\,dC_\ell/dp$', fontsize=10)
axes[1, 0].set_ylabel(r'$\sigma_\ell^{-1}\,r_\ell$', fontsize=10)
for ax in axes[1]:
    ax.set_xlabel(r'$\ell$', fontsize=10)
fig.align_ylabels(axes[:, 0])

texts = []
for col in range(len(panels)):
    if divisor[col] != 1:
        for row in rows:
            texts.append(place_text(fig, axes[row, col], rf'$\times 1/{divisor[col]}$',
                                    [axes[row, col]]))
labels = {'alpha2': r'$p = \alpha_2/\Delta\alpha_2$ ($\Delta\alpha_2$ at $k_\mathrm{max} = 0.2$)',
          'sigma_v': r'$p = \sigma_v/\Delta\sigma_v$',
          '0.2': r'$r_\ell$, $\sigma_v$ fitted at $k_\mathrm{max} = 0.2$',
          '0.075': r'$r_\ell$, $\sigma_v$ fitted at $k_\mathrm{max} = 0.075$'}
handles = [Line2D([], [], color=STYLE[n]['color'], ls=STYLE[n]['ls'], lw=STYLE[n]['lw'],
                  marker=STYLE[n]['marker'], ms=STYLE[n]['ms'], label=labels[n])
           for n in ('alpha2', 'sigma_v', '0.2', '0.075')]
legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.995), ncol=2,
                    frameon=False, fontsize=8, handlelength=2.6, columnspacing=1.5,
                    handletextpad=0.5, labelspacing=0.3, borderaxespad=0.0)
flat = list(axes.ravel())
# the x tick labels of neighboring columns must not touch
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
tick_boxes = [label.get_window_extent(renderer) for ax in axes[1]
              for label in ax.get_xticklabels() if label.get_text()]
for i in range(len(tick_boxes)):
    for j in range(i + 1, len(tick_boxes)):
        if tick_boxes[i].overlaps(tick_boxes[j]):
            raise RuntimeError('fog_templates: two x tick labels touch')
check_clear(fig, flat, [legend, *texts, *[ax.xaxis.label for ax in axes[1]],
                        axes[0, 0].yaxis.label, axes[1, 0].yaxis.label],
            outside=[ax.title for ax in axes[0]])
fig.savefig(os.path.join(HERE, 'fog_templates.pdf'))
fig.savefig(os.path.join(HERE, 'fog_templates.png'), dpi=300)
plt.close(fig)
