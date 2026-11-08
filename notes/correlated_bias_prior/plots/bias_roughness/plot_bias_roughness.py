"""Redraw bias_roughness.pdf from data.json: for the exponential kernel (left)
and the memory model (right), the fraction of the tied FoM kept (top) and
the (w0, wa) shift caused by a conserved-population drift (bottom), against
the roughness x = sigma_b sqrt(0.1/L), one thin line per correlation length
L. data.json is written by EXAMPLE_EVALUATE_FISHER.ipynb. Run from any
folder: python plot_bias_roughness.py
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
L_STYLES = ['-', '--', '-.', ':', (0, (6, 1.5, 1, 1.5, 1, 1.5))]
SHIFT_FLOOR = 1.0e-3

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
sigma_b = np.array(data['sigma_b'])
Ls = data['L']
marks = data['markers']

fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), sharex=True, sharey='row',
                         gridspec_kw={'height_ratios': [1.4, 1.0], 'hspace': 0.08, 'wspace': 0.06})
for col, (model, title) in enumerate((('kernel', 'exponential kernel'),
                                      ('memory', 'memory model (free tilt per group)'))):
    top, bottom = axes[0, col], axes[1, col]
    for lay in data['layouts']:
        for L, ls in zip(Ls, L_STYLES):
            curve = lay[model][str(L)]
            x = sigma_b*np.sqrt(0.1/L)
            top.plot(100*x, np.array(curve['fom'])/lay['fom_tied'], color=lay['color'], lw=1.1, ls=ls)
            bottom.plot(100*x, np.maximum(curve['shift_conserved'], SHIFT_FLOOR),
                        color=lay['color'], lw=1.1, ls=ls)
        top.axhline(lay['fom_independent']/lay['fom_tied'], color=lay['color'], lw=0.8, ls=':', alpha=0.6)
    for ax in (top, bottom):
        for value, label in ((marks['baseline_other'], 'baseline, not BGS'),
                             (marks['baseline_BGS'], 'baseline, BGS'),
                             (marks['pessimistic'], 'pessimistic')):
            ax.axvline(100*value, color='0.35', lw=0.9, ls='--')
            if ax is top:
                ax.text(100*value/1.05, 0.03, label, fontsize=8.5, color='0.3', va='bottom',
                        ha='right', rotation=90)
    bottom.axhline(data['shift_threshold'], color='firebrick', lw=1.0, ls='--')
    top.set_title(title, fontsize=12)
    bottom.set_xscale('log')
    bottom.set_xlabel(r'roughness $x = \sigma_b\sqrt{0.1/L}$: rms change over $\Delta z = 0.05$ [%]')
axes[0, 0].set_ylim(0.0, 1.05)
axes[0, 0].set_ylabel('FoM / FoM(growth-tied)')
axes[1, 0].set_yscale('log')
axes[1, 0].set_ylim(0.6*SHIFT_FLOOR, 60.0)
axes[1, 0].set_ylabel(r'$(w_0, w_a)$ shift [$\sigma$], conserved drift')
axes[1, 1].text(0.98, 1.15*data['shift_threshold'], r'$\sigma/3$', transform=axes[1, 1].get_yaxis_transform(),
                ha='right', va='bottom', fontsize=10, color='firebrick')
layout_handles = [Line2D([], [], color=lay['color'], lw=1.6, label=lay['label']) for lay in data['layouts']]
L_handles = [Line2D([], [], color='0.3', lw=1.1, ls=ls, label=rf'$L = {L:g}$') for L, ls in zip(Ls, L_STYLES)]
axes[0, 0].legend(handles=layout_handles, loc='upper right', frameon=False, fontsize=9.5)
axes[0, 1].legend(handles=L_handles, loc='upper right', frameon=False, fontsize=9.5)
axes[1, 0].set_xlim(0.05, 50.0)
fig.savefig(os.path.join(HERE, 'bias_roughness.pdf'))
fig.savefig(os.path.join(HERE, 'bias_roughness.png'), dpi=200)
