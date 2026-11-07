"""Redraw ladder_bias_params.pdf from data.json: the accuracy test of the
correlated-bias-prior note (known drifts plus a baseline or a pessimistic
residual, 2000 random truths, fitted with every linear rung) scored for
(w0, wa) in its ellipse and for Omega_m and A_s in their own marginalized
sigma. Rows: the parameters; columns: the truth's residual scatter.
data.json is written by EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb.
Run from any folder: python plot_ladder_bias_params.py
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
SHIFT_FLOOR = 1.0e-3
OFFSETS = [-0.15, -0.05, 0.05, 0.15]
PARAM_LABELS = {'w0wa': r'$(w_0, w_a)$ shift [$\sigma$]',
                'omegam': r'$\Omega_\mathrm{m}$ shift [$\sigma$]',
                'As_1e9': r'$A_\mathrm{s}$ shift [$\sigma$]'}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
rungs = data['rungs']
x = np.arange(len(rungs))

fig, axes = plt.subplots(len(data['params']), len(data['truths']),
                         figsize=(4.6*len(data['truths']), 2.6*len(data['params'])),
                         sharex=True, sharey='row', gridspec_kw={'hspace': 0.12, 'wspace': 0.06})
for col, truth in enumerate(data['truths']):
    for row, what in enumerate(data['params']):
        ax = axes[row, col]
        for k in ('baseline', 'pessimistic'):
            ax.axvspan(rungs.index(k) - 0.24, rungs.index(k) + 0.24, color='0.93', zorder=0)
        for lay, dx in zip(data['layouts'], OFFSETS):
            stats = [lay['fits'][r][truth][what] for r in rungs]
            raw = np.array([s['mean'] for s in stats])
            mean = np.maximum(raw, SHIFT_FLOOR)
            low = np.maximum([s['p16'] for s in stats], SHIFT_FLOOR)
            high = np.maximum([s['p84'] for s in stats], SHIFT_FLOOR)
            # a fit with as much freedom as the truth recovers it exactly (zero
            # shift); those points sit on the floor as open markers
            fit = raw >= SHIFT_FLOOR
            ax.errorbar((x + dx)[fit], mean[fit], yerr=[(mean - low)[fit], (high - mean)[fit]],
                        color=lay['color'], ls='none', marker='o', ms=4.5, capsize=2.0,
                        elinewidth=1.0)
            ax.plot((x + dx)[~fit], mean[~fit], ls='none', marker='o', ms=4.5, mfc='white',
                    mec=lay['color'])
        ax.axhline(data['shift_threshold'], color='firebrick', lw=1.0, ls='--')
        ax.set_yscale('log')
        ax.set_ylim(0.6*SHIFT_FLOOR, 100.0)
        if row == 0:
            ax.set_title(f'truth: known drifts + {truth} scatter', fontsize=11)
        if col == 0:
            ax.set_ylabel(PARAM_LABELS[what])
        if row == len(data['params']) - 1:
            ax.set_xticks(x)
            ax.set_xticklabels([f'fit:\n{r}' for r in rungs])
handles = [Line2D([], [], color=lay['color'], ls='none', marker='o', ms=4.5, label=lay['label'])
           for lay in data['layouts']]
handles.append(Line2D([], [], color='firebrick', lw=1.0, ls='--', label=r'$\sigma/3$'))
handles.append(Line2D([], [], color='0.3', ls='none', marker='o', ms=4.5, mfc='white',
                      mec='0.3', label=r'exact recovery (at $10^{-3}$)'))
# just above the column titles (the axes end at 0.88 of the figure); two rows
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.905), ncol=3,
           frameon=False, fontsize=10)
fig.savefig(os.path.join(HERE, 'ladder_bias_params.pdf'))
fig.savefig(os.path.join(HERE, 'ladder_bias_params.png'), dpi=200)
