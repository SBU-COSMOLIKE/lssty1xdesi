"""Redraw fom_collapse.pdf from data.json: the fraction of the tied FoM kept
under the exponential-kernel prior against the roughness combination
sqrt(2) kappa = sigma_b sqrt(2/L), one color per DESI layout and one marker
per L; markers are filled where (L, sigma_b) lies inside the plausible
region and open outside it; dotted lines mark each layout's independent
fraction. data.json is written by the export cell of
EXAMPLE_EVALUATE_FISHER.ipynb. Run from any folder: python plot_fom_collapse.py
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
L_MARKERS = {0.02: 'o', 0.05: 's', 0.1: '^', 0.2: 'D'}
with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
sigmas = data['sigma_b']
lengths = data['L']
plausible_L = data['plausible_L']
plausible_sigma = data['plausible_sigma_b']


def is_plausible(L, sigma_b):
    """True when (L, sigma_b) lies inside the plausible region."""
    return (plausible_L[0] <= L <= plausible_L[1]
            and plausible_sigma[0] <= sigma_b <= plausible_sigma[1])


fig, ax = plt.subplots(figsize=(8.5, 5.5))
for layout in data['layouts']:
    color = layout['color']
    fom = np.array(layout['fom_kernel'])
    kappas = []
    fracs = []
    for c, L in enumerate(lengths):
        for a, sigma_b in enumerate(sigmas):
            kappa2 = np.sqrt(2.0)*sigma_b/np.sqrt(L)
            frac = fom[a, c]/layout['fom_tied']
            kappas.append(kappa2)
            fracs.append(frac)
            ax.plot(kappa2, frac, marker=L_MARKERS[L], color=color, ls='none', ms=6.5,
                    mew=1.4, mfc=color if is_plausible(L, sigma_b) else 'none')
    order = np.argsort(kappas)
    ax.plot(np.array(kappas)[order], np.array(fracs)[order], color=color, lw=0.8, alpha=0.5)
    ax.axhline(layout['fom_independent']/layout['fom_tied'], color=color, ls=':', lw=1.2)
ax.set_xscale('log')
ax.set_ylim(0.0, 1.05)
ax.set_xlabel(r'$\sqrt{2}\,\kappa = \sigma_b\sqrt{2/L}$')
ax.set_ylabel(r'FoM / FoM$_\mathrm{tied}$')
ax.set_title('Collapse test')
ax.text(0.5, 0.97, 'PRELIMINARY: LINEAR BIAS ONLY', transform=ax.transAxes, ha='center',
        va='top', fontsize=12, fontweight='bold', color='firebrick')
ax.text(0.02, 0.03, r'$\mathrm{rms}(\delta_i - \delta_j) \approx \sqrt{2}\,\kappa\,\sqrt{\Delta z}$'
        r' for $\Delta z \ll L$', transform=ax.transAxes, ha='left', va='bottom', fontsize=10)
# legend: lines in the first column, marker types in the second (blank
# entries pad the shorter column)
lines = [Line2D([], [], color=layout['color'], lw=2, label=layout['label'])
         for layout in data['layouts']]
lines.append(Line2D([], [], color='0.4', ls=':', lw=1.2, label='independent biases'))
markers = [Line2D([], [], color='0.4', marker=mk, ls='none', ms=6.5, label=rf'$L = {L:g}$')
           for L, mk in L_MARKERS.items()]
markers.append(Line2D([], [], color='0.4', marker='o', ls='none', ms=6.5,
                      label=r'filled: plausible $(L, \sigma_b)$'))
markers.append(Line2D([], [], color='0.4', marker='o', ls='none', ms=6.5, mfc='none', mew=1.4,
                      label='open: outside'))
lines += [Line2D([], [], color='none', label=' ')]*(len(markers) - len(lines))
ax.legend(handles=lines + markers, fontsize=8.5, ncol=2, loc='upper right',
          bbox_to_anchor=(1.0, 0.9), frameon=False)
fig.savefig(os.path.join(HERE, 'fom_collapse.pdf'))
