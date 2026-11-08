"""Redraw goodness_of_fit.pdf from data.json: the goodness of fit of the
parameter recovery under model A baseline, on the tied and baseline rungs
of the linear-bias ladder, when the truths carry the known bias drifts
plus a residual drawn from the pessimistic prior.

The statistic is sqrt(E[chi2]/n_dof), with chi2 the distance of the
recovered parameters from the truth in units of the fit's own covariance
((w0, wa): n_dof = 2; Omega_m or A_s: n_dof = 1) and E the average over
the random truths and the data noise. It is 1 when the error bars cover
the truth as often as they claim; above 1 it is the factor by which they
are too small. One wide panel per scored parameter, stacked, one column
per bin layout. data.json, written by EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb,
also holds the pessimistic and independent rungs and model A pessimistic,
which the figure leaves out (the caption quotes them).
Run from any folder: python plot_goodness_of_fit.py
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
VARIANT, TRUTH = 'A baseline', 'pessimistic'
PARAM_LABELS = {'w0wa': r'$(w_0, w_a)$', 'omegam': r'$\Omega_\mathrm{m}$',
                'As_1e9': r'$A_\mathrm{s}$'}
# the marker of each drawn rung (the color is the layout's)
RUNG_MARKERS = {'tied': {'marker': 's', 'ms': 6.0, 'label': 'growth-tied'},
                'baseline': {'marker': 'o', 'ms': 6.0, 'label': 'baseline prior'}}
TICK_LABELS = {'Delta d_com = 62 Mpc': '$\\Delta d_\\mathrm{com}$\n62 Mpc',
               'Delta z = 0.02': '$\\Delta z$\n0.02', 'Delta z = 0.05': '$\\Delta z$\n0.05',
               'Delta z = 0.2': '$\\Delta z$\n0.2'}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
layouts = data['layouts']
params = data['params']
x = np.arange(len(layouts))
values = [lay['gof'][VARIANT][TRUTH][r][p] for lay in layouts for r in RUNG_MARKERS
          for p in params]
# one y range for every panel
y_range = (0.7, max(values)*1.8)

# one column of a two-column article: wide, short panels with a thin gap
fig, axes = plt.subplots(len(params), 1, figsize=(3.5, 1.1*len(params)), sharex=True,
                         sharey=True, gridspec_kw={'hspace': 0.06})
for ax, p in zip(axes, params):
    ax.axhline(1.0, color='0.3', ls='--', lw=1.0)
    for i, lay in enumerate(layouts):
        fit = lay['gof'][VARIANT][TRUTH]
        # the drop from the tied to the baseline rung
        ax.plot([i, i], [fit['baseline'][p], fit['tied'][p]], color=lay['color'], lw=1.2,
                alpha=0.6)
        for rung, st in RUNG_MARKERS.items():
            ax.plot(i, fit[rung][p], ls='none', marker=st['marker'], ms=st['ms'],
                    color=lay['color'])
    ax.set_yscale('log')
    ax.set_ylim(*y_range)
    ax.set_yticks([1, 3, 10])
    ax.set_yticklabels(['1', '3', '10'])
    ax.grid(False, axis='x')
    # the scored parameter on the right side, rotated like the y label, so
    # it can never sit on the data
    ax.yaxis.set_label_position('right')
    ax.set_ylabel(PARAM_LABELS[p], rotation=90, fontsize=11, labelpad=5)
axes[-1].set_xlim(-0.5, len(layouts) - 0.5)
axes[-1].set_xticks(x)
axes[-1].set_xticklabels([TICK_LABELS.get(lay['name'], lay['name']) for lay in layouts],
                         fontsize=9)
fig.supylabel(r'goodness of fit $\sqrt{\langle\chi^2\rangle/n_\mathrm{dof}}$', x=-0.02,
              fontsize=10)
handles = [Line2D([], [], ls='none', marker=st['marker'], ms=st['ms'], color='0.3',
                  label=st['label']) for st in RUNG_MARKERS.values()]
handles.append(Line2D([], [], color='0.3', ls='--', lw=1.0, label='consistent (1)'))
axes[0].legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3,
               frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.0,
               handletextpad=0.4)
fig.savefig(os.path.join(HERE, 'goodness_of_fit.pdf'))
fig.savefig(os.path.join(HERE, 'goodness_of_fit.png'), dpi=300)
