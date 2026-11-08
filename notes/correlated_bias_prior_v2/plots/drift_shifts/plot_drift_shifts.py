"""Redraw drift_shifts.pdf from data.json: the shift of a model A baseline
fit when the true linear biases follow a known drift away from the tied
model (b2 and bs2 following b1 through their calibrations), on the tied
and baseline rungs of the linear-bias ladder. Two drifts: a conserved
galaxy population in every group, and the r < 19.5 flux limit in BGS.
(w0, wa) is scored in units of the fit's ellipse, Omega_m and A_s in their
own marginalized sigma. One wide panel per scored parameter, stacked, one
column per bin layout. data.json, written by
EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb, also holds the pessimistic and
independent rungs and model A pessimistic, which the figure leaves out
(the caption quotes them).
Run from any folder: python plot_drift_shifts.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
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
VARIANT = 'A baseline'
PARAM_LABELS = {'w0wa': r'$(w_0, w_a)$', 'omegam': r'$\Omega_\mathrm{m}$',
                'As_1e9': r'$A_\mathrm{s}$'}
RUNG_MARKERS = {'tied': {'marker': 's', 'ms': 5.5, 'label': 'growth-tied'},
                'baseline': {'marker': 'o', 'ms': 5.5, 'label': 'baseline prior'}}
# each drift sits at its own offset inside the layout's column; the BGS
# flux limit is drawn with open markers
DRIFTS = {'conserved': {'dx': -0.15, 'open': False, 'label': 'conserved population'},
          'flux_limit': {'dx': 0.15, 'open': True, 'label': 'BGS flux limit'}}
TICK_LABELS = {'Delta d_com = 62 Mpc': '$\\Delta d_\\mathrm{com}$\n62 Mpc',
               'Delta z = 0.02': '$\\Delta z$\n0.02', 'Delta z = 0.05': '$\\Delta z$\n0.05',
               'Delta z = 0.2': '$\\Delta z$\n0.2'}
# Shifts below FLOOR sit on it: the exact zeros of the baseline rung on
# Delta z = 0.2 (its groups hold one or two bins, so the free amplitude and
# tilt absorb any drift) and the 2e-4 A_s shift of the 62 Mpc flux limit.
FLOOR = 1.0e-3

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
layouts = data['layouts']
params = data['params']
x = np.arange(len(layouts))
values = [lay['shift'][VARIANT][d][r][p] for lay in layouts for d in DRIFTS
          for r in RUNG_MARKERS for p in params]
# one y range for every panel
y_range = (FLOOR/2.0, max(values)*6.0)

fig, axes = plt.subplots(len(params), 1, figsize=(3.5, 1.15*len(params)), sharex=True,
                         sharey=True, gridspec_kw={'hspace': 0.06})
for ax, p in zip(axes, params):
    ax.axhline(data['shift_threshold'], color='firebrick', ls='--', lw=1.0)
    for i, lay in enumerate(layouts):
        for drift, dst in DRIFTS.items():
            shift = lay['shift'][VARIANT][drift]
            xi = i + dst['dx']
            y = {r: max(shift[r][p], FLOOR) for r in RUNG_MARKERS}
            ax.plot([xi, xi], [y['baseline'], y['tied']], color=lay['color'], lw=1.0,
                    alpha=0.6)
            for rung, st in RUNG_MARKERS.items():
                ax.plot(xi, y[rung], ls='none', marker=st['marker'], ms=st['ms'],
                        color=lay['color'], mfc='white' if dst['open'] else lay['color'],
                        mew=1.1)
    ax.set_yscale('log')
    ax.set_ylim(*y_range)
    # labeled decades away from the edges, where the panels sit close
    ax.set_yticks([1e-2, 1e0])
    ax.grid(False, axis='x')
    # the scored parameter on the right side, rotated like the y label, so
    # it can never sit on the data
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(PARAM_LABELS[p], rotation=90, fontsize=11, labelpad=5)
axes[-1].set_xlim(-0.5, len(layouts) - 0.5)
axes[-1].set_xticks(x)
axes[-1].set_xticklabels([TICK_LABELS.get(lay['name'], lay['name']) for lay in layouts],
                         fontsize=9)
fig.supylabel(r'shift [$\sigma$]', x=-0.04, fontsize=10)
rung_handles = [Line2D([], [], ls='none', marker=st['marker'], ms=st['ms'], color='0.3')
                for st in RUNG_MARKERS.values()]
# each drift's entry shows the pair of markers it is drawn with: filled for
# the conserved population, open for the BGS flux limit
drift_handles = [tuple(Line2D([], [], ls='none', marker=st['marker'], ms=st['ms'],
                              color='0.3', mfc='white' if dst['open'] else '0.3', mew=1.1)
                       for st in RUNG_MARKERS.values()) for dst in DRIFTS.values()]
threshold = Line2D([], [], color='firebrick', ls='--', lw=1.0)
# legend columns fill top to bottom: the first row reads the two priors and
# sigma/3, the second the two drifts
handles = [rung_handles[0], drift_handles[0], rung_handles[1], drift_handles[1], threshold]
labels = [RUNG_MARKERS['tied']['label'], DRIFTS['conserved']['label'],
          RUNG_MARKERS['baseline']['label'], DRIFTS['flux_limit']['label'], r'$\sigma/3$']
axes[0].legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, fontsize=8.5, handlelength=2.2, columnspacing=1.0,
               handletextpad=0.4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)})
fig.savefig(os.path.join(HERE, 'drift_shifts.pdf'))
fig.savefig(os.path.join(HERE, 'drift_shifts.png'), dpi=300)
