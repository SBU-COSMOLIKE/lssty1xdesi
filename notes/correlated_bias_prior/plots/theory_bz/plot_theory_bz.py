"""Redraw theory_bz.pdf from data.json: the linear bias b(z) of every DESI
group under the tied shape B_g/D(z) (solid), a conserved population
(dashed) and, for BGS, the r < 19.5 flux limit (dash-dot).
data.json comes from ../make_theory_data.py. Run from any folder:
python plot_theory_bz.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
GROUP_COLORS = {'BGS': '#1f77b4', 'LRG1': '#d62728', 'LRG2': '#9467bd',
                'LRG3': '#8c564b', 'ELG1': '#2ca02c', 'ELG2': '#ff7f0e'}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for group in data['groups']:
    color = GROUP_COLORS[group['name']]
    ax.plot(group['z'], group['b_tied'], color=color, lw=2.0)
    ax.plot(group['z'], group['b_conserved'], color=color, lw=2.0, ls='--')
    if 'b_flux_limit' in group:
        ax.plot(group['z'], group['b_flux_limit'], color=color, lw=2.0, ls='-.')
    # ELG2 starts where ELG1 ends: label ELG1 at its first redshift
    if group['name'] == 'ELG1':
        ax.text(group['z'][0] - 0.015, group['b_tied'][0], group['name'], color=color,
                fontsize=11, va='center', ha='right')
    else:
        ax.text(group['z'][-1] + 0.015, group['b_tied'][-1], group['name'], color=color,
                fontsize=11, va='center')
ax.set_xlabel(r'redshift $z$')
ax.set_ylabel(r'linear bias $b(z)$')
ax.set_xlim(0.05, 1.75)
handles = [
    Line2D([], [], color='0.25', lw=2.0, label=r'tied: $B_g/D(z)$'),
    Line2D([], [], color='0.25', lw=2.0, ls='--',
           label=r'conserved: $1 + (b_c - 1)\,D(z_c)/D(z)$'),
    Line2D([], [], color='0.25', lw=2.0, ls='-.', label=r'BGS flux limit $r < 19.5$'),
]
ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=11)
fig.savefig(os.path.join(HERE, 'theory_bz.pdf'))
