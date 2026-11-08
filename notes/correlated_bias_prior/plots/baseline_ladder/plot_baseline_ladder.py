"""Redraw baseline_ladder.pdf from data.json: the FoM(w0, wa) of each DESI
layout down the ladder of linear-bias assumptions, from the tied bias of
every group to one free bias per bin, with the baseline and pessimistic
assumptions shaded. data.json is written by EXAMPLE_EVALUATE_FISHER.ipynb.
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
# shifts below this floor are drawn at the floor (the independent rung is
# exactly zero: free per-bin biases absorb any drift)
SHIFT_FLOOR = 1.0e-3

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
rungs = [r['key'] for r in data['rungs']]

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
rungs = [r['key'] for r in data['rungs']]
x = np.arange(len(rungs))
# the names the figures print for the linear-bias priors (the data keys
# keep the notebook's names)
DISPLAY = {'tied': 'growth-tied', 'baseline': 'baseline',
           'pessimistic': 'pessimistic', 'independent': 'free'}

fig, ax = plt.subplots(figsize=(7.6, 5.0))
# the two stated assumptions of the note
for k in ('baseline', 'pessimistic'):
    ax.axvspan(rungs.index(k) - 0.18, rungs.index(k) + 0.18, color='0.92', zorder=0)
for lay in data['layouts']:
    fom = np.array([lay['fom'][k] for k in rungs])
    ax.plot(x, fom, color=lay['color'], lw=1.8, marker='o', ms=6, label=lay['label'])
    for k in ('baseline', 'pessimistic'):
        i = rungs.index(k)
        ax.annotate(f'{fom[i]:.3g}', (x[i], fom[i]), textcoords='offset points',
                    xytext=(6, 6), va='bottom', fontsize=9, color=lay['color'])
ax.axhline(data['lsst_y1_3x2pt_fom'], color='0.3', ls='--', lw=1.2)
ax.text(x[-1] + 0.65, 0.94*data['lsst_y1_3x2pt_fom'], 'LSST Y1 3x2pt', ha='right', va='top',
        fontsize=9, color='0.3')
ax.set_yscale('log')
ax.set_ylabel(r'FoM$(w_0, w_a)$, DESI $C_\ell^{gg}$ alone')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=10,
          handlelength=1.6, columnspacing=1.2)
fig.suptitle(rf'Linear bias assumptions from rigid to free ($k_\mathrm{{max}} = '
             rf'{data["kmax_h_per_Mpc"]:g}\,h\,\mathrm{{Mpc}}^{{-1}}$)', fontsize=12, y=1.02)
ax.set_xticks(x)
ax.set_xticklabels([DISPLAY[r['key']] for r in data['rungs']])
ax.set_xlim(-0.4, x[-1] + 0.75)
fig.savefig(os.path.join(HERE, 'baseline_ladder.pdf'))
fig.savefig(os.path.join(HERE, 'baseline_ladder.png'), dpi=300)
