"""Redraw growth_tied_kmax.pdf from data.json: how well the quoted error bars
of the paper's higher-order model cover the truth when its linear bias is
growth-tied, against the same model on the correlated prior, at every cut.

The fit is model B + tilts. Growth-tied: every bin of a DESI group follows
the group amplitude, b_i = B_g/D(zbar_i), with no tilt and no residual (one
bias per group). Correlated prior: the pessimistic correlated prior of the
note. The truths carry the known bias drifts (a conserved population; the
r < 19.5 flux limit in BGS) plus a residual drawn from the pessimistic
prior. The statistic, eta = sqrt(<chi2>/n_dof), is the distance of the
recovered parameters from the truth in units of the fit's own covariance
((w0, wa): n_dof = 2; Omega_m or A_s: n_dof = 1), averaged exactly over the
random truths and the data noise; 1 means honest error bars. Three stacked
panels, one per scored parameter, on one shared logarithmic y axis; the
bin layouts in color, line width and marker, growth-tied solid, the
correlated prior thin dashed. data.json is written by
EXAMPLE_EVALUATE_FISHER_MODEL.ipynb (section 9).
Run from any folder: python plot_growth_tied_kmax.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

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
PARAM_LABELS = {'w0wa': r'$(w_0, w_a)$', 'omegam': r'$\Omega_\mathrm{m}$',
                'As_1e9': r'$A_\mathrm{s}$'}
# the layouts: color from data.json, line width and marker from the
# project's figure rules (paper-figures skill)
LAYOUT_STYLE = {'Delta d_com = 62 Mpc': {'lw': 2.4, 'marker': 'o'},
                'Delta z = 0.02': {'lw': 1.8, 'marker': 's'},
                'Delta z = 0.05': {'lw': 1.3, 'marker': 'D'},
                'Delta z = 0.2': {'lw': 0.9, 'marker': '^'}}
MARKER_SIZE = 3.5
# the two fits: growth-tied solid at the layout's width, the correlated
# prior thin and dashed, with open markers
FITS = {'growth_tied': {'label': 'growth-tied', 'dashed': False},
        'correlated': {'label': 'correlated prior', 'dashed': True}}
THIN = 0.8
# the candidate y ticks (one log axis shared by the panels)
TICK_CANDIDATES = [0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
# the gap between the two stacked legends, in units of the top panel's height
LEGEND_GAP = 0.03
DASH = (4.0, 2.0)
# the y headroom above the highest point is grown in these steps until the
# legends and labels clear every line
HEADROOM = [1.3, 1.6, 2.2, 3.0]


def line_style(lay, fit):
    '''The plot keywords of one layout's line for one fit.'''
    st = LAYOUT_STYLE[lay['name']]
    if FITS[fit]['dashed']:
        return {'color': lay['color'], 'lw': THIN, 'ls': (0, DASH), 'marker': st['marker'],
                'ms': MARKER_SIZE, 'mfc': 'white', 'mec': lay['color'], 'mew': 0.8}
    return {'color': lay['color'], 'lw': st['lw'], 'ls': '-', 'marker': st['marker'],
            'ms': MARKER_SIZE, 'mfc': lay['color'], 'mec': lay['color']}


def as_array(values):
    '''A list with None (no fit at that cut) as a float array with NaN.'''
    out = []
    for v in values:
        out.append(np.nan if v is None else float(v))
    return np.array(out)


def line_points(ax):
    '''Display coordinates of points densely sampled along every drawn line of
    ax (the reference line included), inside the axes only. Segments are
    sampled in display space, where a segment of a log axis is straight.'''
    box = ax.get_window_extent()
    pts = [np.zeros((0, 2))]
    t = np.linspace(0.0, 1.0, 60)[:, None]
    for line in ax.get_lines():
        xy = np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()])
        display_xy = line.get_transform().transform(xy)
        finite = np.all(np.isfinite(display_xy), axis=1)
        pts.append(display_xy[finite])
        for i in range(len(display_xy) - 1):
            if finite[i] and finite[i + 1]:
                pts.append(display_xy[i] + t*(display_xy[i + 1] - display_xy[i]))
    pts = np.vstack(pts)
    inside = ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
              & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))
    return pts[inside]


def touches(box, axes):
    '''True when a display box, with a margin of 3 points, holds a line point.'''
    pad = 3.0*axes[0].figure.dpi/72.0
    for ax in axes:
        pts = line_points(ax)
        hit = ((pts[:, 0] > box.x0 - pad) & (pts[:, 0] < box.x1 + pad)
               & (pts[:, 1] > box.y0 - pad) & (pts[:, 1] < box.y1 + pad))
        if np.any(hit):
            return True
    return False


def check_clear(fig, axes, artists):
    '''Raise when a legend, a label or a tick label touches a drawn line, or
    when two of them overlap.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for name, artist in artists:
        box = artist.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        if touches(box, axes):
            raise RuntimeError(f'growth_tied_kmax: {name} touches a line')
        boxes.append((name, box))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                raise RuntimeError(f'growth_tied_kmax: {boxes[i][0]} overlaps {boxes[j][0]}')


def plain_number(value, position):
    '''A tick label as a plain number: 1, 3, 10 (position is unused).'''
    return f'{value:g}'


with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
layouts = data['layouts']
params = data['params']
kmax = np.asarray(data['kmax_h_per_Mpc'], dtype=float)
values = []
for lay in layouts:
    for fit in FITS:
        for p in params:
            y = as_array(lay['eta'][fit][p])
            values += [float(v) for v in y[np.isfinite(y)]]
low = min(min(values), 1.0)/1.3
high = max(values)

for headroom in HEADROOM:
    # one column of a two-column article: three wide, short panels with a
    # thin gap and one shared y range
    fig, axes = plt.subplots(len(params), 1, figsize=(3.5, 1.15*len(params)), sharex=True,
                             sharey=True, gridspec_kw={'hspace': 0.06})
    for ax, p in zip(axes, params):
        # the reference eta = 1 (honest error bars) carries no legend entry:
        # the caption names it
        ax.axhline(1.0, color='0.5', ls='--', lw=0.9)
        for lay in layouts:
            for fit in FITS:
                ax.plot(kmax, as_array(lay['eta'][fit][p]), **line_style(lay, fit))
        ax.set_yscale('log')
        ax.set_ylim(low, high*headroom)
        # labelled ticks only away from the panel edges, where the labels of
        # two stacked panels would meet
        ticks = []
        for tick in TICK_CANDIDATES:
            if low*1.25 < tick < high*headroom/1.25:
                ticks.append(tick)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(False, axis='x')
        # the scored parameter on the right side, rotated like the y label
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(PARAM_LABELS[p], rotation=90, fontsize=11, labelpad=5)
    axes[-1].set_xlim(kmax.min() - 0.01, kmax.max() + 0.01)
    axes[-1].set_xticks([0.05, 0.1, 0.15, 0.2])
    axes[-1].set_xticklabels(['0.05', '0.1', '0.15', '0.2'])
    axes[-1].xaxis.set_minor_locator(NullLocator())
    axes[-1].set_xlabel(r'$k_\mathrm{max}$ [$h\,\mathrm{Mpc}^{-1}$] at $z = 0$', fontsize=10)
    supylabel = fig.supylabel(r'$\sqrt{\langle\chi^2\rangle/n_\mathrm{dof}}$', x=-0.02,
                              fontsize=10)
    fit_handles = []
    for fit in FITS:
        fit_handles.append(Line2D([], [], label=FITS[fit]['label'],
                                  **line_style({'name': 'Delta z = 0.02', 'color': '0.3'}, fit)))
    layout_handles = []
    for lay in layouts:
        layout_handles.append(Line2D([], [], label=lay['label'], **line_style(lay, 'growth_tied')))
    # the legends sit above the top panel, stacked close together: the
    # layouts first, the two fits above them
    legend_layouts = axes[0].legend(handles=layout_handles, loc='lower center',
                                    bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                                    fontsize=7.5, handlelength=2.4, columnspacing=1.2,
                                    handletextpad=0.4, borderaxespad=0.2, borderpad=0.1)
    axes[0].add_artist(legend_layouts)
    fig.canvas.draw()
    top = axes[0].transAxes.inverted().transform(
        (0.0, legend_layouts.get_window_extent(fig.canvas.get_renderer()).y1))[1]
    legend_fits = axes[0].legend(handles=fit_handles, loc='lower center',
                                 bbox_to_anchor=(0.5, top + LEGEND_GAP), ncol=2, frameon=False,
                                 fontsize=7.5, handlelength=2.8, columnspacing=1.6,
                                 handletextpad=0.4, borderaxespad=0.0, borderpad=0.1)
    artists = [('layout legend', legend_layouts), ('fit legend', legend_fits),
               ('x label', axes[-1].xaxis.label)]
    for k, ax in enumerate(axes):
        artists.append((f'panel {k} label', ax.yaxis.label))
        for t, label in enumerate(ax.get_yticklabels()):
            if label.get_visible() and label.get_text():
                artists.append((f'panel {k} y tick {t}', label))
    for t, label in enumerate(axes[-1].get_xticklabels()):
        artists.append((f'x tick {t}', label))
    try:
        check_clear(fig, list(axes), artists)
    except RuntimeError as error:
        plt.close(fig)
        last_error = error
        continue
    break
else:
    raise RuntimeError(f'growth_tied_kmax: no headroom in {HEADROOM} clears the legends '
                       f'({last_error})')
fig.savefig(os.path.join(HERE, 'growth_tied_kmax.pdf'))
fig.savefig(os.path.join(HERE, 'growth_tied_kmax.png'), dpi=300)
plt.close(fig)
