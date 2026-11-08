"""Redraw memory_lines.pdf from data.json: the fraction of the tied FoM kept
against the residual scatter sigma_b at the baseline L, for the
exponential-kernel prior (solid, circles) and the memory prior with a free
tilt per group (dashed, triangles); at sigma_b = 0 the kernel model is the
tied model and the memory model the tied model plus a free tilt; dotted
lines mark each layout's independent fraction. data.json is written by the
export cell of EXAMPLE_EVALUATE_FISHER.ipynb. Run from any folder:
python plot_memory_lines.py
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
with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
L_ref = data['baseline']['L']
c_ref = data['L'].index(L_ref)
sig = 100.0*np.array([0.0] + list(data['sigma_b']))

fig, ax = plt.subplots(figsize=(8.5, 5.5))
for layout in data['layouts']:
    color = layout['color']
    tied = layout['fom_tied']
    kernel = np.r_[1.0, np.array(layout['fom_kernel'])[:, c_ref]/tied]
    memory = np.r_[layout['fom_tied_free_tilt'], np.array(layout['fom_memory'])[:, c_ref]]/tied
    ax.plot(sig, kernel, color=color, marker='o', ms=5, lw=1.8)
    ax.plot(sig, memory, color=color, marker='^', ms=6, lw=1.8, ls='--')
    ax.axhline(layout['fom_independent']/tied, color=color, ls=':', lw=1.2)
ax.set_xlim(-0.3, 10.5)
ax.set_ylim(0.0, 1.05)
ax.set_xlabel(r'residual scatter $\sigma_b$ [%]')
ax.set_ylabel('FoM / FoM(growth-tied)')
ax.set_title(rf'Exponential kernel against the memory model ($L = {L_ref:g}$, '
             rf'$k_\mathrm{{max}} = {data["kmax_h_per_Mpc"]:g}\,h$ Mpc$^{{-1}}$)')
ax.text(0.5, 0.97, 'PRELIMINARY: LINEAR BIAS ONLY', transform=ax.transAxes, ha='center',
        va='top', fontsize=12, fontweight='bold', color='firebrick')
lines = [Line2D([], [], color=layout['color'], lw=2, label=layout['label'])
         for layout in data['layouts']]
styles = [
    Line2D([], [], color='0.3', marker='o', ms=5, lw=1.8, label='exponential kernel'),
    Line2D([], [], color='0.3', marker='^', ms=6, lw=1.8, ls='--',
           label='memory: free tilt + kernel'),
    Line2D([], [], color='0.3', ls=':', lw=1.2, label='free biases'),
]
styles += [Line2D([], [], color='none', label=' ')]*(len(lines) - len(styles))
ax.legend(handles=lines + styles, fontsize=8.5, ncol=2, loc='upper right',
          bbox_to_anchor=(1.0, 0.9), frameon=False)
fig.savefig(os.path.join(HERE, 'memory_lines.pdf'))
