"""Redraw fob_kmax.pdf from data.json: the shift of a linear-bias fit to a
one-loop (model A) truth against the UV-fraction cut k_max, per layout
(columns) and scored parameter (rows: (w0, wa) in its ellipse, Omega_m and
A_s in their own marginalized sigma). One line per linear-bias prior, for
the truth on the calibrations: each column takes its layout's color, the
growth-tied prior is black, and the priors differ by line width and marker.
The band, drawn for the pessimistic prior only (BAND_RUNG), spans its other
truths (b2 offsets of +-1, alpha_0 = +-12.5 h^-2 Mpc^2); data.json holds the
other priors' brackets too. Each row's y range is fitted to the drawn data,
in whole decades. Cuts up to k_max = 0.2 h/Mpc (KMAX_MAX) are drawn. data.json is written by
EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb.
Run from any folder: python plot_fob_kmax.py
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
SHIFT_FLOOR = 1.0e-3
# the largest cut drawn [h/Mpc at z = 0]; data.json may hold larger ones
KMAX_MAX = 0.2
# Each column is one layout and takes the layout's color (the notes' layout
# colors, Okabe-Ito without its green); the growth-tied prior is black
# (color set here), the other priors take the column's color (color None).
# The priors differ by line width and marker, the notes' prior markers
# (growth-tied star, baseline down-triangle, pessimistic X, free plus);
# circles, squares, diamonds and up-triangles are kept for the layouts. The
# star is drawn larger: at one size it looks smaller than the others.
RUNG_STYLE = {'tied': {'color': 'black', 'lw': 2.4, 'marker': '*', 'ms': 8.0,
                       'label': 'growth-tied'},
              'baseline': {'color': None, 'lw': 1.8, 'marker': 'v', 'ms': 5.5,
                           'label': 'baseline prior'},
              'pessimistic': {'color': None, 'lw': 1.3, 'marker': 'X', 'ms': 5.5,
                              'label': 'pessimistic prior'},
              'independent': {'color': None, 'lw': 0.9, 'marker': 'P', 'ms': 5.5,
                              'label': 'free'}}
# the prior whose band of other truths is drawn (one band keeps it legible),
# and the band's opacity
BAND_RUNG = 'pessimistic'
BAND_ALPHA = 0.2
# the legend's color for the priors drawn in their column's color
LEGEND_COLOR = '0.35'
PARAM_LABELS = {'w0wa': r'$(w_0, w_a)$ shift [$\sigma$]',
                'omegam': r'$\Omega_\mathrm{m}$ shift [$\sigma$]',
                'As_1e9': r'$A_\mathrm{s}$ shift [$\sigma$]'}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax_all = np.array(data['kmax_h_per_Mpc'])
# the cuts drawn: KMAX_MAX and below (1e-9 absorbs the float rounding)
DRAWN_CUTS = kmax_all <= KMAX_MAX + 1.0e-9
kmax = kmax_all[DRAWN_CUTS]
params = data['params']

fig, axes = plt.subplots(len(params), len(data['layouts']),
                         figsize=(3.1*len(data['layouts']), 2.5*len(params)),
                         sharex=True, sharey='row', gridspec_kw={'hspace': 0.12, 'wspace': 0.08})
# the drawn values of each row, for its y range
row_values = {row: [] for row in range(len(params))}
for col, lay in enumerate(data['layouts']):
    for row, what in enumerate(params):
        ax = axes[row, col]
        for rung in data['rungs']:
            st = RUNG_STYLE[rung]
            color = lay['color'] if st['color'] is None else st['color']
            curves = {}
            for truth in data['truths']:
                vals = [np.nan if r is None else r[what] for r in lay['shift'][rung][truth]]
                curves[truth] = np.maximum(np.array(vals, dtype=float)[DRAWN_CUTS],
                                           SHIFT_FLOOR)
            others = np.array([curves[t] for t in data['truths'] if t != 'calibrated'])
            # a cut that leaves no data for the cosmology has no fit (NaN for
            # every truth) and no band
            fit = np.isfinite(others).any(axis=0)
            if rung == BAND_RUNG:
                low = np.nanmin(others[:, fit], axis=0)
                high = np.nanmax(others[:, fit], axis=0)
                ax.fill_between(kmax[fit], low, high, color=color, alpha=BAND_ALPHA, lw=0)
                row_values[row] += [*low, *high]
            ax.plot(kmax, curves['calibrated'], color=color, lw=st['lw'],
                    marker=st['marker'], ms=st['ms'])
            calibrated = curves['calibrated']
            row_values[row] += list(calibrated[np.isfinite(calibrated)])
        ax.axhline(data['shift_threshold'], color='0.1', lw=1.0, ls='--')
        ax.set_yscale('log')
        if row == 0:
            ax.set_title(lay['label'], fontsize=11)
        if col == 0:
            ax.set_ylabel(PARAM_LABELS[what])
        if row == len(params) - 1:
            ax.set_xlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$')
# each row's y range: the whole decades around its drawn values and sigma/3
for row in range(len(params)):
    values = [*row_values[row], data['shift_threshold']]
    axes[row, 0].set_ylim(10.0**np.floor(np.log10(min(values))),
                          10.0**np.ceil(np.log10(max(values))))
handles = [Line2D([], [], color=LEGEND_COLOR if st['color'] is None else st['color'],
                  lw=st['lw'], marker=st['marker'], ms=st['ms'],
                  label=f'linear fit: {st["label"]}') for st in RUNG_STYLE.values()]
# the reference line (sigma/3) carries no legend entry: the caption names it
# just above the column titles (the axes end at 0.88 of the figure)
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.905), ncol=4,
           frameon=False, fontsize=10)
fig.savefig(os.path.join(HERE, 'fob_kmax.pdf'))
fig.savefig(os.path.join(HERE, 'fob_kmax.png'), dpi=200)
