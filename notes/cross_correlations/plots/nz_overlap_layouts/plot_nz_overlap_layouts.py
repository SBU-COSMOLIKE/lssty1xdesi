"""Redraw nz_overlap_layouts.pdf and .png from data.json: every thin bin of
the four DESI layouts and the cross pairs measured between them.

What is drawn. One block per layout, top to bottom Delta z = 0.2, 0.05,
0.02 and Delta d_com = 62 Mpc, on one shared redshift axis:

  - the upper panel of a block draws each bin's raw redshift distribution
    n(z), the histogram of its galaxies' measured redshifts, scaled to
    unit peak. The panel has two lanes: BGS and LRG bins on top (the two
    samples never share a redshift), ELG bins below. The six groups
    (BGS, LRG1-3, ELG1-2) differ by fill shade, light and dark in turn
    along each lane, and by the line style of the bin outlines;
  - the lower panel is the pair graph: one dot per bin at its mean
    redshift zbar, on the row of its lane, and one segment for every
    cross spectrum C_ij in the data vector. Segments inside a row join
    neighbouring bins of one sample; segments between the rows join an
    LRG bin to an ELG bin at about the same redshift.

A gray band marks the redshifts where LRG3 and ELG1, two different
galaxy samples, observe the same volume. A dashed segment is a pair
whose raw ranges neither overlap nor touch (only the BGS-LRG1 boundary
pair of Delta d_com = 62 Mpc, which the DESI forecast's pair list
includes).

A header above the blocks names the groups with a swatch of their fill
and outline. The texts inside the panels are checked in code against
every drawn line (line_points); the script stops with an error when one
would touch the data.

data.json is written by ../make_cross_correlations_data.py. Run from any
folder: python plot_nz_overlap_layouts.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

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

# The six groups. Each lane alternates a light and a dark fill, so two
# neighbouring groups always differ in shade, and every group has its own
# outline style (dash patterns in points), so the groups stay apart in
# black and white. 'lane' 1 is the upper lane (BGS and LRG), 0 the lower
# (ELG).
GROUPS = {
    'BGS': {'lane': 1, 'alpha': 0.22, 'ls': '-'},
    'LRG1': {'lane': 1, 'alpha': 0.62, 'ls': (0, (3.0, 1.2))},
    'LRG2': {'lane': 1, 'alpha': 0.22, 'ls': (0, (0.9, 0.9))},
    'LRG3': {'lane': 1, 'alpha': 0.62, 'ls': (0, (4.0, 1.0, 1.0, 1.0))},
    'ELG1': {'lane': 0, 'alpha': 0.22, 'ls': (0, (2.0, 1.0))},
    'ELG2': {'lane': 0, 'alpha': 0.62, 'ls': (0, (4.0, 1.0, 1.0, 1.0, 1.0, 1.0))},
}
# bottom of each lane in the n(z) panels; each bin rises one unit from it
LANE_BASE = {1: 1.2, 0: 0.0}
# the row of each lane in the pair graph
GRAPH_ROW = {1: 1.0, 0: 0.0}
# the outline of every bin: thin and dark, so the shared edge of two
# touching bins shows as one line
OUTLINE = {'color': '0.15', 'lw': 0.55}
# the pair segments and the bin dots of the pair graph
EDGE = {'color': '0.1', 'lw': 0.6}
NODE_SIZE = 2.6
# the gray band where LRG3 and ELG1 share a volume
BAND_COLOR = '0.86'
# the shared redshift axis
Z_RANGE = (0.075, 1.625)
# the layout whose group ranges the header shows: its bins tile each
# group's nominal range exactly
HEADER_LAYOUT = 'dr1_dz0p02'
# display labels of the layouts (data.json carries plain-text names)
LABEL = {
    'dr1_dz0p20': r'$\Delta z = 0.2$',
    'dr1_dz0p05': r'$\Delta z = 0.05$',
    'dr1_dz0p02': r'$\Delta z = 0.02$',
    'dr3': r'$\Delta d_\mathrm{com} = 62$ Mpc',
}
FONT = 7.5

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)


def bin_curve(bin_entry, dz_node):
    '''The redshift nodes and unit-peak n(z) of one bin.'''
    n = np.asarray(bin_entry['n'], dtype=float)
    z = bin_entry['z_first'] + dz_node*np.arange(n.size)
    return z, n


def line_points(ax):
    '''Display coordinates of points densely sampled along every line of
    ax: the bin outlines, the pair segments and the swatch edges.'''
    pts = []
    for line in ax.get_lines():
        xd, yd = (np.asarray(v, dtype=float) for v in line.get_data())
        if xd.size == 1:
            pts.append(ax.transData.transform(np.column_stack([xd, yd])))
            continue
        for i in range(len(xd) - 1):
            t = np.linspace(0.0, 1.0, 12)
            pts.append(ax.transData.transform(np.column_stack(
                [xd[i] + t*(xd[i + 1] - xd[i]), yd[i] + t*(yd[i + 1] - yd[i])])))
    for patch in ax.patches:
        if patch.get_label() == 'band':
            continue
        corners = patch.get_patch_transform().transform(
            patch.get_path().vertices)
        pts.append(ax.transData.transform(corners))
    return np.vstack(pts)


def assert_clear(fig, ax, artists, name):
    '''Raise when a text of ax touches a drawn line, outline or swatch.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = line_points(ax)
    for artist in artists:
        box = artist.get_window_extent(renderer).expanded(1.04, 1.12)
        inside = ((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1)
                  & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1))
        if np.any(inside):
            raise RuntimeError(f'{name}: the text {artist.get_text()!r} touches '
                               'the data')


layouts = data['layouts']
nblocks = len(layouts)
# rows: header, spacer, then per block an n(z) panel, a thin gap and the
# pair graph, with a spacer between blocks (heights in inches)
heights = [0.50, 0.07]
for k in range(nblocks):
    heights += [0.62, 0.025, 0.30]
    if k < nblocks - 1:
        heights.append(0.11)
fig = plt.figure(figsize=(6.4, sum(heights) + 0.05))
grid = GridSpec(len(heights), 1, figure=fig, height_ratios=heights, hspace=0.0,
                left=0.09, right=0.995, top=0.995, bottom=0.07)

# ---- header: the groups, their swatches and the band ----
header = fig.add_subplot(grid[0])
header_layout = [lay for lay in layouts if lay['key'] == HEADER_LAYOUT][0]
first_band = header_layout['lrg3_elg1_range']
header.axvspan(first_band[0], first_band[1], color=BAND_COLOR, lw=0, zorder=0,
               label='band')
header_texts = []
for name, style in GROUPS.items():
    members = [b for b in header_layout['bins'] if b['group'] == name]
    z_low = bin_curve(members[0], header_layout['dz_node'])[0][1]
    z_high = bin_curve(members[-1], header_layout['dz_node'])[0][-2]
    row = 1.05 if style['lane'] == 1 else 0.05
    swatch = Rectangle((z_low, row), z_high - z_low, 0.32, facecolor='0.35',
                       alpha=style['alpha'], lw=0)
    header.add_patch(swatch)
    header.plot([z_low, z_high, z_high, z_low, z_low],
                [row, row, row + 0.32, row + 0.32, row], ls=style['ls'], **OUTLINE)
    header_texts.append(header.text(0.5*(z_low + z_high), row + 0.40, name,
                                    ha='center', va='bottom', fontsize=FONT))
# the band's meaning, in the empty upper row right of LRG3 (BGS and LRG
# end at z = 1.1)
header_texts.append(header.text(
    Z_RANGE[1] - 0.01, 1.42, 'gray band: LRG3 and ELG1,\ntwo samples in one volume',
    ha='right', va='center', fontsize=FONT, linespacing=1.0))
header.set_xlim(*Z_RANGE)
header.set_ylim(-0.02, 1.98)
header.axis('off')
assert_clear(fig, header, header_texts, 'header')

# ---- one block per layout ----
row = 2
axes_nz = []
for k, lay in enumerate(layouts):
    ax = fig.add_subplot(grid[row])
    graph = fig.add_subplot(grid[row + 2], sharex=ax)
    row += 4
    axes_nz.append(ax)
    band = lay['lrg3_elg1_range']
    for target in (ax, graph):
        target.axvspan(band[0], band[1], color=BAND_COLOR, lw=0, zorder=0,
                       label='band')
        target.grid(False)

    # the n(z) panel: one filled outline per bin, in its lane
    for b in lay['bins']:
        style = GROUPS[b['group']]
        base = LANE_BASE[style['lane']]
        z, n = bin_curve(b, lay['dz_node'])
        ax.fill_between(z, base, base + n, color=lay['color'], alpha=style['alpha'],
                        lw=0, zorder=1)
        ax.plot(z, base + n, ls=style['ls'], zorder=2, **OUTLINE)
    ax.set_ylim(-0.12, 2.36)
    ax.set_yticks([0.5, 1.7])
    ax.set_yticklabels(['ELG', 'BGS\nLRG'], fontsize=FONT, linespacing=0.95)
    ax.tick_params(axis='y', length=0, pad=2)
    plt.setp(ax.get_xticklabels(), visible=False)
    summary = lay['summary']
    label = ax.text(Z_RANGE[1] - 0.012, 1.72,
                    f"{LABEL[lay['key']]}: {summary['nbins']} bins, "
                    f"{summary['npairs']} cross pairs",
                    ha='right', va='center', fontsize=FONT + 0.5)
    assert_clear(fig, ax, [label], lay['key'])

    # the pair graph: dots at the bin means, one segment per pair
    zbar = np.array([b['zbar'] for b in lay['bins']])
    lane = np.array([GROUPS[b['group']]['lane'] for b in lay['bins']])
    yrow = np.array([GRAPH_ROW[v] for v in lane])
    for (i, j), touching in zip(lay['pairs'], lay['touching']):
        graph.plot([zbar[i], zbar[j]], [yrow[i], yrow[j]],
                   ls='-' if touching else (0, (2.0, 1.5)), zorder=2, **EDGE)
    graph.plot(zbar, yrow, ls='none', marker=lay['marker'], ms=NODE_SIZE,
               mfc=lay['color'], mec='0.1', mew=0.35, zorder=3)
    graph.set_ylim(-0.5, 1.5)
    graph.set_yticks([0.0, 1.0])
    graph.set_yticklabels(['ELG', 'BGS, LRG'], fontsize=FONT - 1)
    graph.tick_params(axis='y', length=0, pad=2)
    if k < nblocks - 1:
        plt.setp(graph.get_xticklabels(), visible=False)
    else:
        graph.set_xlabel('redshift $z$', fontsize=9)
        graph.tick_params(axis='x', labelsize=8)
    graph.set_xlim(*Z_RANGE)

fig.text(0.0, 0.5, r'raw $n(z)$ at unit peak; pairs', rotation=90,
         ha='left', va='center', fontsize=9)
fig.savefig(os.path.join(HERE, 'nz_overlap_layouts.pdf'))
fig.savefig(os.path.join(HERE, 'nz_overlap_layouts.png'), dpi=300)
plt.close(fig)
