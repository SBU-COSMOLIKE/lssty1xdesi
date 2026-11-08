"""Redraw fob_kmax.pdf from data.json: the shift of a linear-bias fit to a
one-loop (model A) truth against the UV-fraction cut k_max, per layout
(columns) and scored parameter (rows: (w0, wa) in its ellipse, Omega_m and
A_s in their own marginalized sigma). One line per linear rung, for the
truth on the calibrations; the band spans the other truths (b2 offsets of
+-1, alpha_0 = +-12.5 h^-2 Mpc^2). data.json is written by
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
RUNG_STYLE = {'tied': {'color': '0.15', 'label': 'growth-tied'},
              'baseline': {'color': 'C0', 'label': 'baseline prior'},
              'pessimistic': {'color': 'C3', 'label': 'pessimistic prior'},
              'independent': {'color': 'C2', 'label': 'free'}}
PARAM_LABELS = {'w0wa': r'$(w_0, w_a)$ shift [$\sigma$]',
                'omegam': r'$\Omega_\mathrm{m}$ shift [$\sigma$]',
                'As_1e9': r'$A_\mathrm{s}$ shift [$\sigma$]'}

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
kmax = np.array(data['kmax_h_per_Mpc'])
params = data['params']

fig, axes = plt.subplots(len(params), len(data['layouts']),
                         figsize=(3.1*len(data['layouts']), 2.5*len(params)),
                         sharex=True, sharey='row', gridspec_kw={'hspace': 0.12, 'wspace': 0.08})
for col, lay in enumerate(data['layouts']):
    for row, what in enumerate(params):
        ax = axes[row, col]
        for rung in data['rungs']:
            st = RUNG_STYLE[rung]
            curves = {}
            for truth in data['truths']:
                vals = [np.nan if r is None else r[what] for r in lay['shift'][rung][truth]]
                curves[truth] = np.maximum(np.array(vals, dtype=float), SHIFT_FLOOR)
            others = np.array([curves[t] for t in data['truths'] if t != 'calibrated'])
            # a cut that leaves no data for the cosmology has no fit (NaN for
            # every truth) and no band
            fit = np.isfinite(others).any(axis=0)
            ax.fill_between(kmax[fit], np.nanmin(others[:, fit], axis=0),
                            np.nanmax(others[:, fit], axis=0), color=st['color'], alpha=0.13,
                            lw=0)
            ax.plot(kmax, curves['calibrated'], color=st['color'], lw=1.6, marker='o', ms=3)
        ax.axhline(data['shift_threshold'], color='firebrick', lw=1.0, ls='--')
        ax.set_yscale('log')
        ax.set_ylim(0.5*SHIFT_FLOOR, 300.0)
        if row == 0:
            ax.set_title(lay['label'], fontsize=11)
        if col == 0:
            ax.set_ylabel(PARAM_LABELS[what])
        if row == len(params) - 1:
            ax.set_xlabel(r'$k_\mathrm{max}$ [$h$ Mpc$^{-1}$] at $z = 0$')
handles = [Line2D([], [], color=st['color'], lw=1.6, marker='o', ms=3,
                  label=f'linear fit: {st["label"]}') for st in RUNG_STYLE.values()]
handles.append(Line2D([], [], color='firebrick', lw=1.0, ls='--', label=r'$\sigma/3$'))
# just above the column titles (the axes end at 0.88 of the figure)
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.905), ncol=5,
           frameon=False, fontsize=10)
fig.savefig(os.path.join(HERE, 'fob_kmax.pdf'))
fig.savefig(os.path.join(HERE, 'fob_kmax.png'), dpi=200)
