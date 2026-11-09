"""Redraw nz_overlap_zoom.pdf and .png from data.json: the corner where the
LRG2 group ends and the LRG3 and ELG1 groups begin, on the two thin
layouts, with every bin numbered.

What is drawn. The same drawing as nz_overlap_layouts, magnified: for
Delta z = 0.02 (top block) and Delta d_com = 62 Mpc (bottom block),

  - the upper panel of a block draws each bin's raw n(z) at unit peak,
    LRG bins in the upper lane and ELG bins in the lower one, with the
    group fills and outline styles of nz_overlap_layouts;
  - the lower panel is the pair graph: a dot per bin at its mean redshift
    zbar, labelled with the bin's number in the data vector (counted from
    1, group by group), and a segment per measured cross spectrum: solid
    when the two raw ranges overlap or touch, dash-dotted vermillion for
    the pairs that scripts/make_desi_pairs.py added because their
    Fingers-of-God-blurred n(z) overlap (on Delta d_com = 62 Mpc only:
    the LRG edges sit 0.001-0.002 above the ELG edges, so an LRG3 bin
    starts just above the end of an ELG1 bin). A legend under the figure
    names the two styles.

The numbers show why the LRG-ELG pairs look far apart in the bin
numbering and are not: the numbering runs through all LRG bins before the
first ELG bin, so an LRG3 bin and the ELG1 bin at the same redshift
differ by 11 to 16 in number. A segment that leaves the panel joins a
bin outside the window.

data.json is written by ../make_cross_correlations_data.py. Run from any
folder: python plot_nz_overlap_zoom.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
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

# The group styles of nz_overlap_layouts (fill shade alternating along
# each lane, one outline style per group); only the groups of the window
# appear. 'lane' 1 is the upper lane (LRG), 0 the lower (ELG).
GROUPS = {
    'LRG2': {'lane': 1, 'alpha': 0.22, 'ls': (0, (0.9, 0.9))},
    'LRG3': {'lane': 1, 'alpha': 0.62, 'ls': (0, (4.0, 1.0, 1.0, 1.0))},
    'ELG1': {'lane': 0, 'alpha': 0.22, 'ls': (0, (2.0, 1.0))},
}
LANE_BASE = {1: 1.2, 0: 0.0}
GRAPH_ROW = {1: 1.0, 0: 0.0}
OUTLINE = {'color': '0.15', 'lw': 0.6}
# The pair segments: raw ranges that overlap or touch (solid), and pairs
# added because the blurred n(z) overlap (vermillion and dash-dotted, so
# they read apart in black and white too), as in nz_overlap_layouts
PAIR_STYLE = {
    'touching': {'color': '0.1', 'lw': 0.7, 'ls': '-'},
    'added': {'color': '#D55E00', 'lw': 1.0, 'ls': (0, (3.0, 1.0, 1.0, 1.0))},
}
PAIR_LABEL = {
    'touching': 'raw ranges overlap or touch',
    'added': 'added: blurred kernels overlap',
}
NODE_SIZE = 3.4
BAND_COLOR = '0.86'
LABEL = {
    'dr1_dz0p02': r'$\Delta z = 0.02$',
    'dr3': r'$\Delta d_\mathrm{com} = 62$ Mpc',
}
FONT = 7.5

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
window = data['window']


def bin_curve(bin_entry, dz_node):
    '''The redshift nodes and unit-peak n(z) of one bin.'''
    n = np.asarray(bin_entry['n'], dtype=float)
    z = bin_entry['z_first'] + dz_node*np.arange(n.size)
    return z, n


def line_points(ax):
    '''Display coordinates of points densely sampled along every line of
    ax, and the corners of every swatch.'''
    pts = []
    for line in ax.get_lines():
        xd, yd = (np.asarray(v, dtype=float) for v in line.get_data())
        if xd.size == 1:
            pts.append(ax.transData.transform(np.column_stack([xd, yd])))
            continue
        for i in range(len(xd) - 1):
            t = np.linspace(0.0, 1.0, 16)
            pts.append(ax.transData.transform(np.column_stack(
                [xd[i] + t*(xd[i + 1] - xd[i]), yd[i] + t*(yd[i + 1] - yd[i])])))
    for patch in ax.patches:
        if patch.get_label() == 'band':
            continue
        corners = patch.get_patch_transform().transform(patch.get_path().vertices)
        pts.append(ax.transData.transform(corners))
    return np.vstack(pts)


def assert_clear(fig, ax, artists, name):
    '''Raise when a text of ax touches a drawn line, dot or swatch, or
    another text of the list.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = line_points(ax)
    boxes = []
    for artist in artists:
        box = artist.get_window_extent(renderer).expanded(1.04, 1.10)
        inside = ((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1)
                  & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1))
        if np.any(inside):
            raise RuntimeError(f'{name}: the text {artist.get_text()!r} touches '
                               'the data')
        for other in boxes:
            if box.overlaps(other):
                raise RuntimeError(f'{name}: the text {artist.get_text()!r} '
                                   'touches another text')
        boxes.append(box)


layouts = data['layouts']
heights = [0.48, 0.05]
for k in range(len(layouts)):
    heights += [0.62, 0.025, 0.55]
    if k < len(layouts) - 1:
        heights.append(0.10)
fig = plt.figure(figsize=(3.5, sum(heights) + 0.05))
grid = GridSpec(len(heights), 1, figure=fig, height_ratios=heights, hspace=0.0,
                left=0.13, right=0.99, top=0.995, bottom=0.11)

# ---- header: the three groups of the window ----
header = fig.add_subplot(grid[0])
first = layouts[0]
header.axvspan(*first['lrg3_elg1_range'], color=BAND_COLOR, lw=0, zorder=0,
               label='band')
texts = []
for name, style in GROUPS.items():
    members = [b for b in first['bins'] if b['group'] == name]
    z_low = max(bin_curve(members[0], first['dz_node'])[0][1], window[0])
    z_high = min(bin_curve(members[-1], first['dz_node'])[0][-2], window[1])
    row = 1.05 if style['lane'] == 1 else 0.05
    header.add_patch(Rectangle((z_low, row), z_high - z_low, 0.32,
                               facecolor='0.35', alpha=style['alpha'], lw=0))
    header.plot([z_low, z_high, z_high, z_low, z_low],
                [row, row, row + 0.32, row + 0.32, row], ls=style['ls'], **OUTLINE)
    texts.append(header.text(0.5*(z_low + z_high), row + 0.40, name, ha='center',
                             va='bottom', fontsize=FONT))
header.set_xlim(*window)
header.set_ylim(-0.02, 1.98)
header.axis('off')
assert_clear(fig, header, texts, 'header')

row = 2
for k, lay in enumerate(layouts):
    ax = fig.add_subplot(grid[row])
    graph = fig.add_subplot(grid[row + 2], sharex=ax)
    row += 4
    for target in (ax, graph):
        target.axvspan(*lay['lrg3_elg1_range'], color=BAND_COLOR, lw=0, zorder=0,
                       label='band')
        target.grid(False)
    for b in lay['bins']:
        style = GROUPS[b['group']]
        base = LANE_BASE[style['lane']]
        z, n = bin_curve(b, lay['dz_node'])
        ax.fill_between(z, base, base + n, color=lay['color'], alpha=style['alpha'],
                        lw=0, zorder=1)
        ax.plot(z, base + n, ls=style['ls'], zorder=2, **OUTLINE)
    ax.set_xlim(*window)
    ax.set_ylim(-0.12, 2.36)
    ax.set_yticks([0.5, 1.7])
    ax.set_yticklabels(['ELG', 'LRG'], fontsize=FONT)
    ax.tick_params(axis='y', length=0, pad=2)
    plt.setp(ax.get_xticklabels(), visible=False)
    # the layout's name in the empty corner of the ELG lane (ELG starts
    # at z = 0.8)
    name = ax.text(window[0] + 0.002, 0.5, LABEL[lay['key']], ha='left',
                   va='center', fontsize=FONT)
    assert_clear(fig, ax, [name], lay['key'])

    zbar = np.array([b['zbar'] for b in lay['bins']])
    lane = np.array([GROUPS[b['group']]['lane'] for b in lay['bins']])
    yrow = np.array([GRAPH_ROW[v] for v in lane])
    for (i, j), touching, added in zip(lay['pairs'], lay['touching'], lay['added']):
        if not touching and not added:
            raise ValueError(f'pair {i}, {j}: raw ranges apart and not added; '
                             'this figure has no style for it')
        kind = 'added' if added else 'touching'
        graph.plot([zbar[i], zbar[j]], [yrow[i], yrow[j]], zorder=2,
                   **PAIR_STYLE[kind])
    graph.plot(zbar, yrow, ls='none', marker=lay['marker'], ms=NODE_SIZE,
               mfc=lay['color'], mec='0.1', mew=0.4, zorder=3)
    # bin numbers: above the LRG row, below the ELG row
    numbers = []
    for b, z_node, y_node in zip(lay['bins'], zbar, yrow):
        if not window[0] < z_node < window[1]:
            continue
        offset = 0.40 if y_node > 0.5 else -0.40
        numbers.append(graph.text(z_node, y_node + offset, str(b['number']),
                                  ha='center', va='center', fontsize=FONT - 0.5))
    graph.set_ylim(-0.8, 1.8)
    graph.set_yticks([0.0, 1.0])
    graph.set_yticklabels(['ELG', 'LRG'], fontsize=FONT - 0.5)
    graph.tick_params(axis='y', length=0, pad=2)
    graph.set_xlim(*window)
    assert_clear(fig, graph, numbers, lay['key'] + ' numbers')
    if k < len(layouts) - 1:
        plt.setp(graph.get_xticklabels(), visible=False)
    else:
        graph.set_xlabel('redshift $z$', fontsize=9)
        graph.tick_params(axis='x', labelsize=8)

# the two kinds of pair segment, under the x label
fig.legend(handles=[Line2D([], [], **PAIR_STYLE[kind]) for kind in PAIR_STYLE],
           labels=[PAIR_LABEL[kind] for kind in PAIR_STYLE],
           loc='upper center', bbox_to_anchor=(0.56, 0.0), ncol=1,
           frameon=False, fontsize=7.5, handlelength=3.2, handletextpad=0.5,
           labelspacing=0.3)
fig.savefig(os.path.join(HERE, 'nz_overlap_zoom.pdf'))
fig.savefig(os.path.join(HERE, 'nz_overlap_zoom.png'), dpi=300)
plt.close(fig)
