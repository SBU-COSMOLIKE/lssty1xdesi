"""Redraw fog_smearing.pdf and .png from data.json: how the Fingers-of-God
(FoG) blur at the default velocity dispersions changes the overlap of
neighbouring thin bins.

What is drawn.

  Left column, top to bottom Delta z = 0.02, Delta d_com = 62 Mpc and
  Delta z = 0.05, on one narrow redshift window inside the LRG1 group:
  each bin's raw n(z) (dashed) and its FoG-convolved n(z) (solid), both
  divided by the bin's raw peak, so the two curves enclose the same
  area. The shaded lenses are the overlap of two neighbouring convolved
  bins, min(n_i, n_j). The bar marks plus and minus sigma_z, the width of
  the Gaussian blur, sigma_z = (1 + zbar) sigma_v / c.

  Right column, against redshift, for every layout:
    top    - sigma_z divided by the bin width (the redshift range the
             bin's galaxies were selected in): how large the blur is
             against the bin;
    bottom - the normalized overlap integral of every pair of
             neighbouring bins of one sample,
             O_ij = int n_i n_j dz / sqrt(int n_i^2 dz int n_j^2 dz),
             for the raw (dashed, open markers) and the convolved
             (solid, filled markers) n(z), on a logarithmic axis; black
             crosses mark the unmeasured LRG3 x ELG1 pairs of
             Delta d_com = 62 Mpc whose raw ranges miss by 3 to 5 DESI grid steps and
             whose convolved n(z) overlap (their raw overlap is zero).

Each layout has its house color, line width and marker
(.claude/skills/paper-figures/SKILL.md); raw against convolved is dashed
against solid everywhere, so the figure reads in black and white. The
texts inside the panels are checked in code against every drawn line
(line_points); the script stops with an error when one would touch the
data.

data.json is written by ../make_cross_correlations_data.py. Run from any
folder: python plot_fog_smearing.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.legend_handler import HandlerTuple
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

# The house line width of each layout; the marker and color come from
# data.json.
LAYOUT_LW = {
    'dr3': 2.4,
    'dr1_dz0p02': 1.8,
    'dr1_dz0p05': 1.3,
    'dr1_dz0p20': 0.9,
}
LABEL = {
    'dr1_dz0p20': r'$\Delta z = 0.2$',
    'dr1_dz0p05': r'$\Delta z = 0.05$',
    'dr1_dz0p02': r'$\Delta z = 0.02$',
    'dr3': r'$\Delta d_\mathrm{com} = 62$ Mpc',
}
# In the close-ups every panel is one layout, so its lines are thinner than
# the house widths (a 2.4 pt line would hide the blurred edge): the
# convolved curve takes 0.6 of the layout's width, the raw one a fixed
# thin dash.
ZOOM_LW_FACTOR = 0.6
RAW_ZOOM_LW = 0.8
# Dash pattern of the raw curves, in points: matplotlib scales a dash
# pattern by the line width, so dividing by the width keeps the dashes
# visible on the thinnest line.
DASH = (3.0, 2.0)
MARKER_SIZE = 3.0
FONT = 8


def dashed(lw):
    '''The raw-curve line style for a line of width lw.'''
    return (0, (DASH[0]/lw, DASH[1]/lw))


def line_points(ax):
    '''Display coordinates of points densely sampled along every line of
    ax (curves, bars, the overlap lenses' outlines are their curves).'''
    pts = []
    for line in ax.get_lines():
        xd, yd = (np.asarray(v, dtype=float) for v in line.get_data())
        keep = np.isfinite(xd) & np.isfinite(yd)
        xd = xd[keep]
        yd = yd[keep]
        if xd.size == 1:
            pts.append(ax.transData.transform(np.column_stack([xd, yd])))
            continue
        for i in range(len(xd) - 1):
            t = np.linspace(0.0, 1.0, 8)
            pts.append(ax.transData.transform(np.column_stack(
                [xd[i] + t*(xd[i + 1] - xd[i]), yd[i] + t*(yd[i + 1] - yd[i])])))
    return np.vstack(pts)


def assert_clear(fig, ax, artists, name):
    '''Raise when a text of ax touches a drawn line or marker.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = line_points(ax)
    for artist in artists:
        box = artist.get_window_extent(renderer).expanded(1.04, 1.15)
        inside = ((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1)
                  & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1))
        if np.any(inside):
            raise RuntimeError(f'{name}: the text {artist.get_text()!r} touches '
                               'the data')


with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)

fig = plt.figure(figsize=(6.4, 3.35))
outer = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.24,
                 left=0.06, right=0.995, top=0.86, bottom=0.12)
left = GridSpecFromSubplotSpec(len(data['zoom']), 1, subplot_spec=outer[0],
                               hspace=0.06)
right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.06)

# ---- the close-ups ----
window = data['zoom_window']
zoom_axes = []
for k, lay in enumerate(data['zoom']):
    ax = fig.add_subplot(left[k], sharex=zoom_axes[0] if zoom_axes else None)
    zoom_axes.append(ax)
    ax.grid(False)
    color = lay['color']
    lw = ZOOM_LW_FACTOR*LAYOUT_LW[lay['key']]
    z_raw = np.asarray(lay['z_raw'])
    z_fog = np.asarray(lay['z_fog'])
    curves = [np.asarray(b['n_fog']) for b in lay['bins']]
    # the overlap of each pair of neighbours (consecutive bins of the
    # window, all of one sample here): the smaller of the two curves
    for a, b in zip(curves[:-1], curves[1:]):
        ax.fill_between(z_fog, 0.0, np.minimum(a, b), color=color, alpha=0.55,
                        lw=0, zorder=1)
    for b, n_fog in zip(lay['bins'], curves):
        ax.plot(z_raw, b['n_raw'], color=color, lw=RAW_ZOOM_LW,
                ls=dashed(RAW_ZOOM_LW), zorder=2)
        ax.plot(z_fog, n_fog, color=color, lw=lw, zorder=3)
    ax.set_xlim(*window)
    ax.set_ylim(0.0, 1.62)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis='y', labelsize=FONT)
    if k < len(data['zoom']) - 1:
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel('redshift $z$', fontsize=9)
        ax.tick_params(axis='x', labelsize=FONT)
    # the blur width: a bar of plus and minus sigma_z centred on the
    # bin edge nearest the window centre, above the curves
    centre = 0.5*(window[0] + window[1])
    edges = []
    for b in lay['bins']:
        n_raw = np.asarray(b['n_raw'])
        live = np.nonzero(n_raw > 0)[0]
        if live.size:
            edges.append(z_raw[live[-1]])
    edge = min(edges, key=lambda e: abs(e - centre))
    sigma_z = [b['sigma_z'] for b in lay['bins']][0]
    y_bar = 1.32
    ax.plot([edge - sigma_z, edge + sigma_z], [y_bar, y_bar], color='0.1',
            lw=0.9, solid_capstyle='butt', zorder=4)
    ax.plot([edge - sigma_z, edge - sigma_z], [y_bar - 0.07, y_bar + 0.07],
            color='0.1', lw=0.9, zorder=4)
    ax.plot([edge + sigma_z, edge + sigma_z], [y_bar - 0.07, y_bar + 0.07],
            color='0.1', lw=0.9, zorder=4)
    bar_text = ax.text(edge + sigma_z + 0.0015, y_bar, r'$\pm\sigma_z$',
                       ha='left', va='center', fontsize=FONT)
    name = ax.text(window[0] + 0.002, 1.33, LABEL[lay['key']], ha='left',
                   va='center', fontsize=FONT + 0.5)
    assert_clear(fig, ax, [bar_text, name], lay['key'])
fig.text(0.005, 0.49, r'$n(z)$ / raw peak', rotation=90, ha='left',
         va='center', fontsize=9)

# ---- sigma_z against the bin width ----
ax_ratio = fig.add_subplot(right[0])
ax_overlap = fig.add_subplot(right[1], sharex=ax_ratio)
for lay in data['layouts']:
    color = lay['color']
    lw = LAYOUT_LW[lay['key']]
    style = {'color': color, 'lw': lw, 'marker': lay['marker'],
             'ms': MARKER_SIZE, 'mec': color}
    samples = np.asarray(lay['samples'])
    zbar = np.asarray(lay['zbar'])
    ratio = np.asarray(lay['sigma_z_over_width'])
    # one line per sample: LRG and ELG bins share redshifts, and a line
    # through both would zig-zag between their dispersions
    for sample in ('bgs', 'lrg', 'elg'):
        pick = samples == sample
        ax_ratio.plot(zbar[pick], ratio[pick], mfc=color, **style)
    nb = lay['neighbours']
    nb_samples = np.asarray(nb['samples'])
    zmid = np.asarray(nb['zmid'])
    raw = np.asarray(nb['raw'])
    fog = np.asarray(nb['fog'])
    for sample in ('bgs', 'lrg', 'elg'):
        pick = nb_samples == sample
        ax_overlap.plot(zmid[pick], fog[pick], mfc=color, **style)
        ax_overlap.plot(zmid[pick], raw[pick], mfc='white', ls=dashed(lw), **style)
    # unmeasured pairs that overlap only after the blur (their raw overlap
    # is zero, which a log axis cannot show): black crosses
    near = lay['unmeasured_near']
    if near['zmid']:
        ax_overlap.plot(near['zmid'], near['fog'], ls='none', marker='x', ms=4.0,
                        mew=1.0, color='black', zorder=5)
ax_ratio.set_ylim(0.0, 0.142)
ax_ratio.set_yticks([0.0, 0.04, 0.08, 0.12])
ax_ratio.set_ylabel(r'$\sigma_z$ / bin width', fontsize=9)
ax_ratio.tick_params(labelsize=FONT)
plt.setp(ax_ratio.get_xticklabels(), visible=False)
ax_overlap.set_yscale('log')
ax_overlap.set_ylim(1.5e-4, 0.6)
ax_overlap.set_ylabel(r'overlap $O_{ij}$', fontsize=9)
ax_overlap.set_xlabel(r'mean redshift $\bar z$', fontsize=9)
ax_overlap.tick_params(labelsize=FONT)
ax_overlap.set_xlim(0.05, 1.65)
for ax in (ax_ratio, ax_overlap):
    ax.grid(False)
# the crosses' meaning, in the empty top of the overlap panel; checked
# against every line and marker
cross_legend = ax_overlap.legend(
    handles=[Line2D([], [], ls='none', marker='x', ms=4.0, mew=1.0, color='black')],
    labels=[r'unmeasured LRG3$\times$ELG1 pairs, $\Delta d_\mathrm{com} = 62$ Mpc'],
    loc='upper left', frameon=False, fontsize=FONT - 0.5, handletextpad=0.3,
    borderaxespad=0.2, borderpad=0.1)
fig.canvas.draw()
legend_box = cross_legend.get_window_extent(fig.canvas.get_renderer())
pts = line_points(ax_overlap)
if np.any((pts[:, 0] > legend_box.x0) & (pts[:, 0] < legend_box.x1)
          & (pts[:, 1] > legend_box.y0) & (pts[:, 1] < legend_box.y1)):
    raise RuntimeError('fog_smearing: the cross legend touches the data')

# ---- legends above the figure: the two variants, then the layouts ----
order = ['dr3', 'dr1_dz0p02', 'dr1_dz0p05', 'dr1_dz0p20']
by_key = {lay['key']: lay for lay in data['layouts']}


def variant_handle(key, raw):
    '''A legend line as drawn on the right: dashed and open for raw n(z),
    solid and filled for the convolved n(z).'''
    lay = by_key[key]
    lw = LAYOUT_LW[key]
    return Line2D([], [], color=lay['color'], lw=lw, marker=lay['marker'],
                  ms=MARKER_SIZE, mec=lay['color'],
                  mfc='white' if raw else lay['color'],
                  ls=dashed(lw) if raw else '-')


variant_handles = [tuple(variant_handle(key, True) for key in order),
                   tuple(variant_handle(key, False) for key in order),
                   Patch(facecolor='0.45', alpha=0.55, lw=0)]
variant_labels = ['raw $n(z)$', r'FoG-convolved $n(z)$, default $\sigma_v$',
                  'overlap of neighbours']
fig.legend(handles=variant_handles, labels=variant_labels,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
           loc='lower center', bbox_to_anchor=(0.53, 0.935), ncol=3,
           frameon=False, fontsize=FONT, handlelength=5.0, columnspacing=1.6,
           handletextpad=0.5, borderaxespad=0.0)
layout_handles = [variant_handle(key, False) for key in order]
fig.legend(handles=layout_handles, labels=[LABEL[key] for key in order],
           loc='lower center', bbox_to_anchor=(0.53, 0.875), ncol=4,
           frameon=False, fontsize=FONT, handlelength=2.4, columnspacing=1.6,
           handletextpad=0.5, borderaxespad=0.0)

fig.savefig(os.path.join(HERE, 'fog_smearing.pdf'))
fig.savefig(os.path.join(HERE, 'fog_smearing.png'), dpi=300)
plt.close(fig)
