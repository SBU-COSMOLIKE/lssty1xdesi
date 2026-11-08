"""Redraw ladder_bias_scatter.pdf from data.json: the (w0, wa) shift of
fits to random truths under model A (one-loop bias, notes/higher_order_bias),
in units of each fit's own ellipse. Top two panels: the v1 test (known drifts
plus a baseline or a pessimistic residual, the higher-order biases following
b1), fitted with every linear rung; bottom panel: a drift of the b2 and bs2
calibrations themselves. Filled markers: model A baseline (one offset per
chunk); open markers: model A pessimistic (offsets plus tilts).
data.json is written by EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb.
Run from any folder: python plot_ladder_bias_scatter.py
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
# shifts below this floor are drawn at the floor (the independent rung
# absorbs any b1 truth exactly)
SHIFT_FLOOR = 1.0e-3
# horizontal offsets: one per layout, and a smaller one per variant
OFFSETS = [-0.21, -0.07, 0.07, 0.21]
VARIANTS = {'A baseline': {'dx': -0.025, 'filled': True},
            'A pessimistic': {'dx': 0.025, 'filled': False}}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
rungs = [r['key'] for r in data['rungs']]
x = np.arange(len(rungs))
# the names the figures print for the linear-bias priors (the data keys
# keep the notebook's names)
DISPLAY = {'tied': 'growth-tied', 'baseline': 'baseline',
           'pessimistic': 'pessimistic', 'independent': 'free'}
panels = [(t, f'truth: known drifts + {t} scatter') for t in data['truths']]
panels.append(('calibration_drift',
               rf'truth: $b_2$ and $b_{{s^2}}$ offsets drifting by '
               rf'$\pm{data["calibration_drift"]["b2"]/2:g}$ and '
               rf'$\pm{data["calibration_drift"]["bs2"]/2:g}$ across each chunk'))

fig, axes = plt.subplots(len(panels), 1, figsize=(5.8, 8.4), sharex=True,
                         gridspec_kw={'hspace': 0.2})
for ax, (truth, title) in zip(axes, panels):
    for k in ('baseline', 'pessimistic'):
        ax.axvspan(rungs.index(k) - 0.28, rungs.index(k) + 0.28, color='0.93', zorder=0)
    for lay, dx in zip(data['layouts'], OFFSETS):
        for variant, st in VARIANTS.items():
            fits = [lay['fits'][variant][rung][truth] for rung in rungs]
            face = lay['color'] if st['filled'] else 'white'
            if truth == 'calibration_drift':
                y = np.maximum([s['shift'] for s in fits], SHIFT_FLOOR)
                ax.plot(x + dx + st['dx'], y, ls='none', marker='o', ms=5.0,
                        mfc=face, mec=lay['color'], mew=1.1)
                continue
            mean = np.maximum([s['mean'] for s in fits], SHIFT_FLOOR)
            low = np.maximum([s['p16'] for s in fits], SHIFT_FLOOR)
            high = np.maximum([s['p84'] for s in fits], SHIFT_FLOOR)
            ax.errorbar(x + dx + st['dx'], mean, yerr=[mean - low, high - mean], ls='none',
                        color=lay['color'], marker='o', ms=5.0, mfc=face, mec=lay['color'],
                        mew=1.1, capsize=2.0, elinewidth=1.0)
    ax.axhline(data['shift_threshold'], color='firebrick', lw=1.0, ls='--')
    ax.text(x[-1] + 0.36, 1.15*data['shift_threshold'], r'$\sigma/3$', va='bottom',
            fontsize=10, color='firebrick')
    ax.set_yscale('log')
    ax.set_ylim(0.6*SHIFT_FLOOR, 60.0)
    ax.set_ylabel(r'$(w_0, w_a)$ shift [$\sigma$]')
    ax.set_title(title, fontsize=10.5)
axes[-1].set_xticks(x)
axes[-1].set_xticklabels([f'fit:\n{DISPLAY[r["key"]]}' for r in data['rungs']])
axes[-1].set_xlim(-0.45, x[-1] + 0.8)
handles = [Line2D([], [], ls='none', marker='o', ms=5.0, mfc=lay['color'], mec=lay['color'],
                  label=lay['label']) for lay in data['layouts']]
handles += [Line2D([], [], ls='none', marker='o', ms=5.0, mfc='0.3', mec='0.3',
                   label='model A baseline'),
            Line2D([], [], ls='none', marker='o', ms=5.0, mfc='white', mec='0.3',
                   label='model A pessimistic')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.905), ncol=3,
           frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.2)
fig.savefig(os.path.join(HERE, 'ladder_bias_scatter.pdf'))
fig.savefig(os.path.join(HERE, 'ladder_bias_scatter.png'), dpi=200)
