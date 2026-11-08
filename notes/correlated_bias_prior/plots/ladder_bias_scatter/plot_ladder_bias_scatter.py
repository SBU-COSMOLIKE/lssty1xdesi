"""Redraw ladder_bias_scatter.pdf from data.json: the figure of bias of every
rung of the linear-bias ladder, averaged over random truths. Each truth is
the known drift (BGS flux limit in BGS, a conserved population elsewhere)
plus a residual drawn from the baseline prior (top) or the pessimistic prior
(bottom). Points: the mean shift of (w0, wa) in units of the fitted rung's
marginalized ellipse; bars: the 16-84 percentiles over the draws.
data.json is written by EXAMPLE_EVALUATE_FISHER.ipynb. Run from any folder:
python plot_ladder_bias_scatter.py
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
# shifts below this floor are drawn at the floor (the independent rung, and
# every rung below tied on Delta z = 0.2, absorb the truth exactly)
SHIFT_FLOOR = 1.0e-3
# horizontal offsets that keep the four layouts' error bars apart
OFFSETS = [-0.15, -0.05, 0.05, 0.15]

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
rungs = [r['key'] for r in data['rungs']]
x = np.arange(len(rungs))
# the names the figures print for the linear-bias priors (the data keys
# keep the notebook's names)
DISPLAY = {'tied': 'growth-tied', 'baseline': 'baseline',
           'pessimistic': 'pessimistic', 'independent': 'free'}

fig, axes = plt.subplots(2, 1, figsize=(5.4, 6.0), sharex=True, gridspec_kw={'hspace': 0.14})
for ax, truth in zip(axes, data['truths']):
    for k in ('baseline', 'pessimistic'):
        ax.axvspan(rungs.index(k) - 0.24, rungs.index(k) + 0.24, color='0.92', zorder=0)
    for lay, dx in zip(data['layouts'], OFFSETS):
        stats = [lay['truth'][truth][rung] for rung in rungs]
        mean = np.maximum([s['mean'] for s in stats], SHIFT_FLOOR)
        low = np.maximum([s['p16'] for s in stats], SHIFT_FLOOR)
        high = np.maximum([s['p84'] for s in stats], SHIFT_FLOOR)
        ax.errorbar(x + dx, mean, yerr=[mean - low, high - mean], color=lay['color'], lw=1.0,
                    marker='o', ms=5.5, capsize=2.5, elinewidth=1.2)
    ax.axhline(data['shift_threshold'], color='firebrick', lw=1.0, ls='--')
    ax.text(x[-1] + 0.3, 1.15*data['shift_threshold'], r'$\sigma/3$', va='bottom', fontsize=10,
            color='firebrick')
    ax.text(x[-1], 1.6*SHIFT_FLOOR, 'zero', ha='center', va='bottom', fontsize=9, color='0.3')
    # the rung whose prior is the truth's scatter: there the shift is the
    # prior's own share of the error ellipse, not a bias
    ax.text(rungs.index(truth), 25.0, 'fit prior =\ntruth scatter', ha='center', va='top',
            fontsize=8.5, color='0.35')
    ax.set_yscale('log')
    ax.set_ylim(0.6*SHIFT_FLOOR, 60.0)
    ax.set_ylabel(r'$(w_0, w_a)$ shift [$\sigma$]')
    ax.set_title(f'truth: known drifts + {truth} scatter', fontsize=11.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'fit:\n{DISPLAY[r["key"]]}' for r in data['rungs']])
axes[1].set_xlim(-0.45, x[-1] + 0.75)
handles = [Line2D([], [], color=lay['color'], lw=1.0, marker='o', ms=5.5, label=lay['label'])
           for lay in data['layouts']]
# the upper right of the top panel is empty: every curve falls below sigma/3
# before the independent rung
axes[0].legend(handles=handles, loc='upper right', frameon=False, fontsize=9.5,
               handlelength=1.6, labelspacing=0.5)
fig.savefig(os.path.join(HERE, 'ladder_bias_scatter.pdf'))
fig.savefig(os.path.join(HERE, 'ladder_bias_scatter.png'), dpi=200)
