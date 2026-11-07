"""Redraw baseline_ladder.pdf from data.json: the FoM(w0, wa) of each DESI
layout down the ladder of linear-bias assumptions, from the tied bias of
every group to one free bias per bin, under model A (one-loop bias,
notes/higher_order_bias) with three higher-order variants, next to the v1
linear-bias values. One panel per layout. data.json is written by
EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb.
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
# the line style of each bias model; v1 is the linear-bias ladder of the
# first note
STYLES = {
    'v1 linear': {'ls': ':', 'lw': 1.4, 'marker': 's', 'ms': 4.5, 'mfc': 'white',
                  'label': 'v1: linear bias'},
    'A calibrated': {'ls': '-.', 'lw': 1.3, 'marker': 'v', 'ms': 5.0, 'mfc': 'white',
                     'label': 'model A, calibrations exact'},
    'A baseline': {'ls': '-', 'lw': 1.9, 'marker': 'o', 'ms': 6.0, 'mfc': None,
                   'label': 'model A baseline'},
    'A pessimistic': {'ls': '--', 'lw': 1.6, 'marker': 'D', 'ms': 5.0, 'mfc': None,
                      'label': 'model A pessimistic'},
}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
rungs = [r['key'] for r in data['rungs']]
x = np.arange(len(rungs))

fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), sharex=True,
                         gridspec_kw={'hspace': 0.16, 'wspace': 0.22})
for ax, lay in zip(axes.ravel(), data['layouts']):
    for k in ('baseline', 'pessimistic'):
        ax.axvspan(rungs.index(k) - 0.18, rungs.index(k) + 0.18, color='0.93', zorder=0)
    series = {'v1 linear': lay['v1_linear']}
    series.update({v: lay['fom'][v] for v in data['variants']})
    for name, fom in series.items():
        st = STYLES[name]
        y = np.array([fom[k] for k in rungs])
        ax.plot(x, y, color=lay['color'], ls=st['ls'], lw=st['lw'], marker=st['marker'],
                ms=st['ms'], mfc=st['mfc'] or lay['color'], mec=lay['color'])
    ax.axhline(data['lsst_y1_3x2pt_fom'], color='0.3', ls='--', lw=1.0)
    ax.set_yscale('log')
    ax.set_title(lay['label'], fontsize=11.5)
    ax.set_xlim(-0.4, x[-1] + 0.4)
    # room below the lowest point for the headline numbers
    lowest = min(min(s.values()) for s in series.values())
    ax.set_ylim(bottom=lowest/3.5)

# The headline numbers of each panel, model A baseline at the baseline rung
# and model A pessimistic at the pessimistic rung, as a two-line block that
# names them. Four series share each panel, so a label beside a point would
# either sit on a line or read as the value of a neighboring point; the
# block takes the first panel corner where its box crosses no drawn line.
CORNERS = [((0.03, 0.03), 'left', 'bottom'), ((0.97, 0.97), 'right', 'top'),
           ((0.97, 0.03), 'right', 'bottom'), ((0.03, 0.97), 'left', 'top')]


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


fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for ax, lay in zip(axes.ravel(), data['layouts']):
    pts = line_points(ax)
    text = (f'model A baseline, baseline rung: {lay["fom"]["A baseline"]["baseline"]:.3g}\n'
            f'model A pessimistic, pessimistic rung: '
            f'{lay["fom"]["A pessimistic"]["pessimistic"]:.3g}')
    for xy, ha, va in CORNERS:
        block = ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5,
                        color=lay['color'], linespacing=1.3)
        box = block.get_window_extent(renderer).expanded(1.04, 1.1)
        hits = np.sum((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1)
                      & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1))
        if hits == 0:
            break
        block.remove()
    else:
        raise RuntimeError(f'{lay["label"]}: no free corner for the headline numbers')
for ax in axes[:, 0]:
    ax.set_ylabel(r'FoM$(w_0, w_a)$, DESI $C_\ell^{gg}$ alone')
for ax in axes[1]:
    ax.set_xticks(x)
    ax.set_xticklabels([r['label'] for r in data['rungs']])
handles = [Line2D([], [], color='0.25', ls=st['ls'], lw=st['lw'], marker=st['marker'],
                  ms=st['ms'], mfc=st['mfc'] or '0.25', mec='0.25', label=st['label'])
           for st in STYLES.values()]
handles.append(Line2D([], [], color='0.3', ls='--', lw=1.0,
                      label=f'LSST Y1 3x2pt ({data["lsst_y1_3x2pt_fom"]:g})'))
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.94), ncol=3,
           frameon=False, fontsize=9.5, handlelength=2.4, columnspacing=1.4)
# k_max is the z = 0 value of the per-bin cut at k_max/D(z) (the caption
# explains the UV-fraction criterion and the shipped mask)
fig.suptitle(rf'Linear bias assumptions from rigid to free under model A '
             rf'($k_\mathrm{{max}} = {data["kmax_h_per_Mpc"]:g}\,h\,\mathrm{{Mpc}}^{{-1}}$ at $z = 0$)',
             fontsize=12.5, y=1.04)
fig.savefig(os.path.join(HERE, 'baseline_ladder.pdf'))
fig.savefig(os.path.join(HERE, 'baseline_ladder.png'), dpi=300)
