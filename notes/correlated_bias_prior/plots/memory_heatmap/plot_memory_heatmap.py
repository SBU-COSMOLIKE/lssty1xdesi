"""Redraw memory_heatmap.pdf from data.json: the fraction of the tied FoM
kept, FoM/FoM_tied, on the (L, sigma_b) grid of every DESI layout under the
memory prior (a free tilt per group plus the exponential kernel for the
residual), with the FoM in parentheses, on the color scale of
fom_heatmap.pdf. data.json is written by the export cell of
EXAMPLE_EVALUATE_FISHER.ipynb. Run from any folder: python plot_memory_heatmap.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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
with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
sigmas = data['sigma_b']
lengths = data['L']

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), gridspec_kw={'hspace': 0.35})
for n, (ax, layout) in enumerate(zip(axes.ravel(), data['layouts'])):
    fom = np.array(layout['fom_memory'])
    frac = fom/layout['fom_tied']
    image = ax.imshow(frac, origin='lower', cmap='viridis', vmin=0.0, vmax=1.0, aspect='auto')
    for a in range(len(sigmas)):
        for c in range(len(lengths)):
            ax.text(c, a, f'{frac[a, c]:.2f}\n({fom[a, c]:.3g})', ha='center', va='center',
                    fontsize=8.5, color='white' if frac[a, c] < 0.6 else 'black')
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels([f'{L:g}' for L in lengths])
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels([f'{s:.0%}' for s in sigmas])
    ax.grid(False)
    if n >= 2:  # shared labels: x on the bottom row, y on the left column
        ax.set_xlabel(r'correlation length $L$ of the residual (in redshift)')
    if n % 2 == 0:
        ax.set_ylabel(r'residual scatter $\sigma_b$')
    ax.set_title(f'{layout["title"]}\n'
                 f'tied {layout["fom_tied"]:.3g}, tied + free tilt '
                 f'{layout["fom_tied_free_tilt"]:.3g}', fontsize=10)
fig.colorbar(image, ax=axes, label=r'FoM / FoM$_\mathrm{tied}$', shrink=0.8)
fig.suptitle('Memory model: fraction of the tied FoM kept with a free tilt per group '
             rf'($k_\mathrm{{max}} = {data["kmax_h_per_Mpc"]:g}\,h$ Mpc$^{{-1}}$)', fontsize=12)
fig.text(0.45, 0.955, 'PRELIMINARY: LINEAR BIAS ONLY', ha='center', va='top', fontsize=12,
         fontweight='bold', color='firebrick')
fig.savefig(os.path.join(HERE, 'memory_heatmap.pdf'))
