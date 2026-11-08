"""Redraw baseline_ladder.pdf from data.json: the FoM(w0, wa) of each DESI
layout down the linear-bias priors, from growth-tied (one amplitude per
group, b = B/D(z)) to one free bias per bin, under model A (one-loop bias,
notes/higher_order_bias) in its two forecast variants, model A baseline
and model A pessimistic. One wide panel per layout, stacked with a thin
gap, all on the same y range, for one column of a two-column paper.
data.json is written by EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb; it also
holds the v1 linear-bias and the exact-calibration FoMs, which this figure
does not draw.
Run from any folder: python plot_baseline_ladder.py
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
# the line style of each drawn variant: the two forecast variants of model A
STYLES = {
    'A baseline': {'ls': '-', 'lw': 1.7, 'marker': 'o', 'ms': 5.0, 'label': 'model A baseline'},
    'A pessimistic': {'ls': '--', 'lw': 1.4, 'marker': 'D', 'ms': 4.5,
                      'label': 'model A pessimistic'},
}
# the names the figure prints for the linear-bias priors (the data keys
# keep the notebook's names)
DISPLAY = {'tied': 'growth-tied', 'baseline': 'baseline',
           'pessimistic': 'pessimistic', 'independent': 'free'}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
rungs = [r['key'] for r in data['rungs']]
x = np.arange(len(rungs))
layouts = data['layouts']
lsst = data['lsst_y1_3x2pt_fom']

# One y range for every panel, so heights compare across layouts, with
# room at the top for the layout label. Labeled ticks only at decades well
# inside the range, where no label runs into the next panel.
values = [v for lay in layouts for name in STYLES for v in lay['fom'][name].values()]
y_range = (min(values)/3.0, max(max(values), lsst)*12.0)
decades = 10.0**np.arange(np.floor(np.log10(y_range[0])), np.ceil(np.log10(y_range[1])) + 1)
y_ticks = decades[(decades > 1.6*y_range[0]) & (decades < y_range[1]/1.6)]

# one column of a two-column paper: wide, short panels with a thin gap
fig, axes = plt.subplots(len(layouts), 1, figsize=(3.5, 1.15*len(layouts)), sharex=True,
                         sharey=True, gridspec_kw={'hspace': 0.06})
for ax, lay in zip(axes, layouts):
    for k in ('baseline', 'pessimistic'):
        ax.axvspan(rungs.index(k) - 0.22, rungs.index(k) + 0.22, color='0.93', zorder=0)
    for name, st in STYLES.items():
        y = np.array([lay['fom'][name][k] for k in rungs])
        ax.plot(x, y, color=lay['color'], ls=st['ls'], lw=st['lw'], marker=st['marker'],
                ms=st['ms'])
    ax.axhline(lsst, color='0.3', ls='--', lw=1.0)
    ax.set_yscale('log')
    ax.set_ylim(*y_range)
    ax.set_yticks(y_ticks)
    ax.grid(False, axis='x')
    # the layout's label inside the panel, top right
    ax.text(0.97, 0.92, lay['label'], transform=ax.transAxes, ha='right', va='top',
            fontsize=10)
axes[-1].set_xlim(-0.4, x[-1] + 0.4)
axes[-1].set_xticks(x)
axes[-1].set_xticklabels([DISPLAY[r['key']] for r in data['rungs']], fontsize=9.5)
fig.supylabel(r'FoM$(w_0, w_a)$, DESI $C_\ell^{gg}$ alone', x=-0.03, fontsize=10)


def line_points(ax):
    '''Display coordinates of points densely sampled along every line of ax
    (in log y for a log axis, as drawn), the LSST Y1 line included.'''
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
            xs = xd[i] + t*(xd[i + 1] - xd[i])
            ys = np.exp(np.log(yd[i]) + t*(np.log(yd[i + 1]) - np.log(yd[i])))
            pts.append(ax.transData.transform(np.column_stack([xs, ys])))
    return np.vstack(pts)


# A label must never sit on the data: stop if one does.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for ax, lay in zip(axes, layouts):
    pts = line_points(ax)
    box = ax.texts[0].get_window_extent(renderer).expanded(1.05, 1.15)
    if np.any((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1)
              & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1)):
        raise RuntimeError(f'{lay["name"]}: the panel label touches a line')

handles = [Line2D([], [], color='0.25', ls=st['ls'], lw=st['lw'], marker=st['marker'],
                  ms=st['ms'], label=st['label']) for st in STYLES.values()]
handles.append(Line2D([], [], color='0.3', ls='--', lw=1.0, label=f'LSST Y1 3x2pt ({lsst:g})'))
axes[0].legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False, fontsize=9, handlelength=2.4, columnspacing=1.2,
               handletextpad=0.5)
fig.savefig(os.path.join(HERE, 'baseline_ladder.pdf'))
fig.savefig(os.path.join(HERE, 'baseline_ladder.png'), dpi=300)
