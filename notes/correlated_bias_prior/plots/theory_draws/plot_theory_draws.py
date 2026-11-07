"""Redraw theory_draws.pdf from data.json: random draws of the BGS bin
deviations from the exponential-kernel prior (left) and from the memory
prior (right), against the BGS flux-limit drift. data.json comes from
../make_theory_data.py. Run from any folder: python plot_theory_draws.py
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
DRAW_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b']

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
z = np.array(data['z'])
kernel = data['kernel_prior']
memory = data['memory_prior']

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True, gridspec_kw={'wspace': 0.06})
for draw, color in zip(data['kernel_draws'], DRAW_COLORS):
    axes[0].plot(z, 100.0*np.array(draw), color=color, lw=1.4, marker='o', ms=3)
for draw, color in zip(data['memory_draws'], DRAW_COLORS):
    axes[1].plot(z, 100.0*np.array(draw), color=color, lw=1.4, marker='o', ms=3)
for ax in axes:
    ax.plot(z, 100.0*np.array(data['flux_limit']), color='black', lw=3.0, ls='-.')
    ax.axhline(0.0, color='0.2', lw=0.8)
    ax.set_xlabel(r'redshift $z$ (BGS, $\Delta z = 0.02$ bins)')
axes[0].set_ylabel(r'$\delta_i$ [%], mean removed')
axes[0].set_title(rf'kernel: $\sigma_b = {100*kernel["sigma_b"]:.0f}\%$, $L = {kernel["L"]:g}$',
                  fontsize=12)
axes[1].set_title(rf'memory: free tilt + $\sigma_b = {100*memory["sigma_b"]:.0f}\%$, '
                  rf'$L = {memory["L"]:g}$', fontsize=12)
handles = [Line2D([], [], color='black', lw=3.0, ls='-.', label=r'BGS flux limit $r < 19.5$'),
           Line2D([], [], color='0.4', lw=1.4, marker='o', ms=3, label='prior draws')]
axes[1].legend(handles=handles, loc='lower right', frameon=False, fontsize=11)
fig.savefig(os.path.join(HERE, 'theory_draws.pdf'))
