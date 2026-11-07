"""Redraw fom_range.pdf from data.json: per DESI layout, the FoM(w0, wa) of
DESI C_gg alone with the bin biases tied per group (black cap), independent
(gray cap) and under the exponential-kernel prior (dots: every scanned
(L, sigma_b); band: the plausible region; diamond: the baseline), with the
LSST Y1 3x2pt reference (dashed). data.json is written by the export cell
of EXAMPLE_EVALUATE_FISHER.ipynb. Run from any folder: python plot_fom_range.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
plausible_L = data['plausible_L']
plausible_sigma = data['plausible_sigma_b']
baseline = data['baseline']
ia = [a for a, s in enumerate(sigmas) if plausible_sigma[0] <= s <= plausible_sigma[1]]
ic = [c for c, L in enumerate(lengths) if plausible_L[0] <= L <= plausible_L[1]]
npts = len(sigmas)*len(lengths)

fig, ax = plt.subplots(figsize=(9, 5.5))
for k, layout in enumerate(data['layouts']):
    color = layout['color']
    grid = np.array(layout['fom_kernel'])
    band = grid[np.ix_(ia, ic)]
    ax.plot([k, k], [layout['fom_independent'], layout['fom_tied']], color='0.75', lw=1.0,
            zorder=0)
    ax.fill_between([k - 0.22, k + 0.22], band.min(), band.max(), color=color, alpha=0.3, lw=0)
    ax.scatter(k + np.linspace(-0.18, 0.18, npts), grid.ravel(), s=12, color=color, alpha=0.5,
               lw=0)
    ax.plot([k - 0.28, k + 0.28], [layout['fom_tied']]*2, color='0.1', lw=2.5)
    ax.plot([k - 0.28, k + 0.28], [layout['fom_independent']]*2, color='0.55', lw=2.5)
    base = grid[sigmas.index(baseline['sigma_b']), lengths.index(baseline['L'])]
    ax.plot(k, base, marker='D', ms=9, color=color, mec='black', zorder=5)
    ax.text(k + 0.3, base, f'{base:.3g}', va='center', fontsize=9)
ax.axhline(data['lsst_y1_3x2pt_fom'], color='0.35', ls='--', lw=1.2)
ax.set_yscale('log')
ax.set_ylim(0.3, 60.0)
ax.set_xlim(-0.6, len(data['layouts']) - 0.4)
ax.set_xticks(range(len(data['layouts'])))
ax.set_xticklabels([layout['label'] for layout in data['layouts']])
ax.set_ylabel(r'FoM$(w_0, w_a)$')
ax.set_title(rf'DESI $C_\ell^{{gg}}$ alone (non-Limber), UV-fraction cut at '
             rf'$k_\mathrm{{max}} = {data["kmax_h_per_Mpc"]:g}\,h$ Mpc$^{{-1}}$: '
             'freedom of the bin biases')
ax.text(0.5, 0.97, 'PRELIMINARY: LINEAR BIAS ONLY', transform=ax.transAxes, ha='center',
        va='top', fontsize=13, fontweight='bold', color='firebrick')
handles = [
    Line2D([], [], color='0.1', lw=2.5, label='biases tied per group'),
    Line2D([], [], color='0.55', lw=2.5, label='independent biases, flat priors'),
    Patch(facecolor='0.6', alpha=0.4,
          label=rf'plausible: $L$ = {plausible_L[0]:g}–{plausible_L[1]:g}, '
                rf'$\sigma_b$ = {plausible_sigma[0]:.0%}–{plausible_sigma[1]:.0%}'),
    Line2D([], [], color='0.4', marker='D', ls='none', ms=8, mec='black',
           label=rf'baseline: $L$ = {baseline["L"]:g}, $\sigma_b$ = {baseline["sigma_b"]:.0%}'),
    Line2D([], [], color='0.4', marker='o', ls='none', ms=4, alpha=0.5,
           label=r'every scanned ($L$, $\sigma_b$)'),
    Line2D([], [], color='0.35', ls='--', lw=1.2,
           label=f'LSST Y1 3x2pt (SRD), {data["lsst_y1_3x2pt_fom"]:.1f}'),
]
ax.legend(handles=handles, fontsize=8.5, loc='lower left', frameon=False)
fig.savefig(os.path.join(HERE, 'fom_range.pdf'))
