"""Redraw ladder_bias.pdf from data.json: on each rung of the linear-bias
ladder, the shift of (w0, wa) that a known bias drift causes, in units of
that rung's marginalized w0-wa error ellipse (a figure of bias,
sqrt(dp^T C_w^-1 dp), linear response). Circles: a conserved population in
every group; squares: the BGS flux limit. data.json is written by
EXAMPLE_EVALUATE_FISHER.ipynb. Run from any folder: python plot_ladder_bias.py
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
# shifts below this floor are drawn at the floor (the independent rung is
# exactly zero: free per-bin biases absorb any drift)
SHIFT_FLOOR = 1.0e-3

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
rungs = [r['key'] for r in data['rungs']]
# shifts below this floor are drawn at the floor (the independent rung is
# exactly zero: free per-bin biases absorb any drift)
SHIFT_FLOOR = 1.0e-3

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
rungs = [r['key'] for r in data['rungs']]
x = np.arange(len(rungs))

fig, ax = plt.subplots(figsize=(7.6, 4.6))
for k in ('baseline', 'pessimistic'):
    ax.axvspan(rungs.index(k) - 0.18, rungs.index(k) + 0.18, color='0.92', zorder=0)
for lay in data['layouts']:
    for key, marker, fill in (('shift_conserved', 'o', lay['color']),
                              ('shift_flux_limit', 's', 'white')):
        shift = np.maximum([lay[key][k] for k in rungs], SHIFT_FLOOR)
        ax.plot(x, shift, color=lay['color'], lw=1.2,
                ls='-' if key == 'shift_conserved' else ':',
                marker=marker, ms=6, mfc=fill, mec=lay['color'])
ax.axhline(data['shift_threshold'], color='firebrick', lw=1.0, ls='--')
ax.text(x[-1] + 0.2, 1.15*data['shift_threshold'], r'$\sigma/3$', va='bottom', fontsize=10,
        color='firebrick')
ax.text(x[-1], 1.6*SHIFT_FLOOR, 'zero', ha='center', va='bottom', fontsize=9, color='0.3')
ax.set_yscale('log')
ax.set_ylim(0.6*SHIFT_FLOOR, 60.0)
ax.set_ylabel(r'$(w_0, w_a)$ shift [$\sigma$]')
layout_handles = [Line2D([], [], color=lay['color'], lw=1.6, label=lay['label'])
                  for lay in data['layouts']]
drift_handles = [Line2D([], [], color='0.3', lw=1.2, marker='o', ms=6,
                        label='conserved population (every group)'),
                 Line2D([], [], color='0.3', lw=1.2, ls=':', marker='s', ms=6, mfc='white',
                        label=r'BGS flux limit $r < 19.5$')]
ax.legend(handles=drift_handles, loc='upper right', frameon=False, fontsize=10)
fig.legend(handles=layout_handles, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=4,
           frameon=False, fontsize=10, handlelength=1.6, columnspacing=1.2)
ax.set_xticks(x)
ax.set_xticklabels([r['label'] for r in data['rungs']])
ax.set_xlim(-0.4, x[-1] + 0.75)
fig.savefig(os.path.join(HERE, 'ladder_bias.pdf'))
fig.savefig(os.path.join(HERE, 'ladder_bias.png'), dpi=200)
