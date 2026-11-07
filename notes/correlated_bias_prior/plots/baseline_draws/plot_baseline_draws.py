"""Redraw baseline_draws.pdf from data.json: per DESI group, the known bias
drift against random histories of the baseline memory prior that share its
straight line. data.json comes from ../make_theory_data.py. Run from any
folder: python plot_baseline_draws.py
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
DRAW_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
LAW_STYLE = {'conserved': {'ls': '--', 'label': 'conserved population'},
             'flux_limit': {'ls': '-.', 'label': r'BGS flux limit $r < 19.5$'}}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
prior = data['prior']

fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), gridspec_kw={'hspace': 0.42, 'wspace': 0.22})
for ax, group in zip(axes.ravel(), data['groups']):
    z = np.array(group['z'])
    for draw, color in zip(group['draws'], DRAW_COLORS):
        ax.plot(z, 100.0*np.array(draw), color=color, lw=1.1, marker='o', ms=2.5, alpha=0.85)
    for law, values in group['laws'].items():
        ax.plot(z, 100.0*np.array(values), color='black', lw=2.6, ls=LAW_STYLE[law]['ls'])
    ax.axhline(0.0, color='0.2', lw=0.8)
    ax.set_title(rf'{group["name"]}: $\sigma_b = {100*prior["sigma_b"][group["name"]]:.0f}\%$',
                 fontsize=12)
    ax.set_xlabel(r'redshift $z$')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$\delta(z)$ [%]')
handles = [Line2D([], [], color='black', lw=2.6, ls=LAW_STYLE[k]['ls'], label=LAW_STYLE[k]['label'])
           for k in ('conserved', 'flux_limit')]
handles.append(Line2D([], [], color='0.45', lw=1.1, marker='o', ms=2.5,
                      label=rf'baseline prior draws, $L = {prior["L"]:g}$'))
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=11,
           bbox_to_anchor=(0.5, 1.02))
fig.savefig(os.path.join(HERE, 'baseline_draws.pdf'))
fig.savefig(os.path.join(HERE, 'baseline_draws.png'), dpi=300)
