"""Redraw pair_graph_bandwidth2.pdf and .png from data.json: the pair graph
of the four DESI layouts when every pair of bins two apart in one sample,
C_{i,i+2}, is added to today's cross pairs, with the bins that have the
most partners marked.

Words used below. Two bins are partners when a measured cross spectrum
C_ij contains both. The degree of a bin is its number of partners. The
derivative columns are batched by a greedy coloring: each bin, in bin
(data-vector) order, takes the lowest batch that none of its earlier
partners holds. A bin's batch is therefore at most 1 + (the number of its
partners that come before it in bin order), its 'earlier' count, which
can be much smaller than its degree.

What is drawn. The upper part has one block per layout, top to bottom
Delta z = 0.2, 0.05, 0.02 and Delta d_com = 62 Mpc, on one redshift
axis:

  - the upper panel of a block draws each bin's raw n(z) at unit peak, as
    in notes/cross_correlations/plots/nz_overlap_layouts: BGS and LRG bins
    in the upper lane, ELG bins in the lower one, the six groups told
    apart by fill shade and outline style;
  - the lower panel is the pair graph: a dot per bin at its mean redshift
    zbar, on the row of its lane. Today's pairs are thin dark segments.
    The added pairs C_{i,i+2} join two bins of one row, so they are drawn
    as vermillion dash-dotted arcs, above the upper row and below the
    lower one, to keep them off today's segments. The bins of largest
    degree are larger black dots; a text beside each row of them gives
    their degree and their earlier count.

The lower part magnifies the corner where LRG3 and ELG1 share a volume,
one pair graph per layout, with each bin's greedy batch written inside
its dot (white on black for the bins of largest degree). A segment or
arc that leaves a panel joins a bin outside the window.

A gray band marks the redshifts where LRG3 and ELG1, two different
galaxy samples, observe the same volume. Every text inside the panels is
checked in code against every drawn line and dot (line_points); the
script stops with an error when one would touch the data.

data.json is written by ../make_pair_graph_bandwidth2_data.py. Run from
any folder: python plot_pair_graph_bandwidth2.py
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

# The six groups, as in nz_overlap_layouts. Each lane alternates a light
# and a dark fill, and every group has its own outline style (dash
# patterns in points), so the groups stay apart in black and white.
# 'lane' 1 is the upper lane (BGS and LRG), 0 the lower (ELG).
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
# the row of each lane in the pair graphs
GRAPH_ROW = {1: 1.0, 0: 0.0}
OUTLINE = {'color': '0.15', 'lw': 0.55}
# today's pairs: thin and dark; the added pairs C_{i,i+2}: vermillion and
# dash-dotted, so they read apart from today's in black and white too
PAIR_STYLE = {
    'today': {'color': '0.1', 'lw': 0.6, 'ls': '-'},
    'added': {'color': '#D55E00', 'lw': 0.9, 'ls': (0, (3.0, 1.0, 1.0, 1.0))},
}
PAIR_LABEL = {'today': "today's pairs", 'added': 'added: bins two apart'}
# the bins of largest degree: black, larger than the other dots
TOP_COLOR = 'black'
NODE_SIZE = 2.6
TOP_SIZE = 4.4
# the zoom: dots big enough to carry the batch number
ZOOM_NODE_SIZE = 8.6
ZOOM_NUMBER_FONT = 5.6
# height of the C_{i,i+2} arcs above or below their row, in row units
ARC_HEIGHT = {'full': 0.48, 'zoom': 0.62}
BAND_COLOR = '0.86'
Z_RANGE = (0.075, 1.625)
# the zoom window: the LRG3/ELG1 volume (0.8 < z < 1.1) with room on the
# left, where the ELG row is empty, for the panel texts
Z_ZOOM = (0.655, 1.15)
HEADER_LAYOUT = 'dr1_dz0p02'
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


def arc(z0, z1, row, height, sign):
    '''A parabola from (z0, row) to (z1, row) bulging by height, upward for
    sign = +1 and downward for sign = -1.'''
    t = np.linspace(0.0, 1.0, 31)
    return z0 + t*(z1 - z0), row + sign*height*4.0*t*(1.0 - t)


def line_points(ax):
    '''Display coordinates of points densely sampled along every line of
    ax, plus the footprint of every marker: outlines, segments, arcs, dots
    and swatch edges. The gray band is left out (texts may sit on it).'''
    fig = ax.figure
    pts = []
    for line in ax.get_lines():
        xd, yd = (np.asarray(v, dtype=float) for v in line.get_data())
        xy = ax.transData.transform(np.column_stack([xd, yd]))
        if line.get_linestyle() not in ('None', 'none', '', ' '):
            for i in range(len(xd) - 1):
                t = np.linspace(0.0, 1.0, 12)[:, None]
                pts.append(xy[i] + t*(xy[i + 1] - xy[i]))
        if line.get_marker() not in (None, 'None', 'none', '', ' '):
            # a marker of size ms points covers a square of half side ms/2
            half = 0.5*line.get_markersize()*fig.dpi/72.0
            for dx in (-half, 0.0, half):
                for dy in (-half, 0.0, half):
                    pts.append(xy + np.array([dx, dy]))
        else:
            pts.append(xy)
    for patch in ax.patches:
        if patch.get_label() == 'band':
            continue
        corners = patch.get_patch_transform().transform(patch.get_path().vertices)
        pts.append(ax.transData.transform(corners))
    pts = np.vstack(pts)
    # only what the panel shows can touch a text
    box = ax.get_window_extent()
    keep = ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
            & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))
    return pts[keep]


def touches(fig, ax, artist):
    '''True when the text artist touches a drawn line or dot of ax, or
    leaves the panel.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = artist.get_window_extent(renderer).expanded(1.04, 1.15)
    frame = ax.get_window_extent(renderer)
    if (box.x0 < frame.x0 or box.x1 > frame.x1 or box.y0 < frame.y0
            or box.y1 > frame.y1):
        return True
    pts = line_points(ax)
    inside = ((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1)
              & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1))
    return bool(np.any(inside))


def place_text(fig, ax, candidates, text, name, **kwargs):
    '''Put text at the first candidate (x, y, ha) that touches nothing;
    raise when none clears.'''
    for x, y, ha in candidates:
        artist = ax.text(x, y, text, ha=ha, **kwargs)
        if not touches(fig, ax, artist):
            return artist
        artist.remove()
    raise RuntimeError(f'{name}: no clear place for the text {text!r}')


def header_axes(ax, layouts, z_range, groups, band_note):
    '''The group swatches of the header layout, clipped to z_range.'''
    lay = [x for x in layouts if x['key'] == HEADER_LAYOUT][0]
    band = lay['lrg3_elg1_range']
    ax.axvspan(band[0], band[1], color=BAND_COLOR, lw=0, zorder=0, label='band')
    texts = []
    for name in groups:
        style = GROUPS[name]
        members = [b for b in lay['bins'] if b['group'] == name]
        z_low = max(bin_curve(members[0], lay['dz_node'])[0][1], z_range[0])
        z_high = min(bin_curve(members[-1], lay['dz_node'])[0][-2], z_range[1])
        row = 1.05 if style['lane'] == 1 else 0.05
        ax.add_patch(Rectangle((z_low, row), z_high - z_low, 0.32, facecolor='0.35',
                               alpha=style['alpha'], lw=0))
        ax.plot([z_low, z_high, z_high, z_low, z_low],
                [row, row, row + 0.32, row + 0.32, row], ls=style['ls'], **OUTLINE)
        texts.append(ax.text(0.5*(z_low + z_high), row + 0.40, name, ha='center',
                             va='bottom', fontsize=FONT))
    if band_note:
        texts.append(ax.text(z_range[1] - 0.01, 1.42,
                             'gray band: LRG3 and ELG1,\ntwo samples in one volume',
                             ha='right', va='center', fontsize=FONT, linespacing=1.0))
    ax.set_xlim(*z_range)
    ax.set_ylim(-0.02, 1.98)
    ax.axis('off')
    for t in texts:
        if touches(ax.figure, ax, t):
            raise RuntimeError(f'header: the text {t.get_text()!r} touches the data')


def draw_pairs(graph, lay, zbar, yrow, height):
    '''Today's pairs as segments, the added pairs C_{i,i+2} as arcs.'''
    for i, j in lay['pairs_today']:
        graph.plot([zbar[i], zbar[j]], [yrow[i], yrow[j]], zorder=2, **PAIR_STYLE['today'])
    for i, j in lay['pairs_added']:
        assert yrow[i] == yrow[j]
        sign = 1.0 if yrow[i] == GRAPH_ROW[1] else -1.0
        za, ya = arc(zbar[i], zbar[j], yrow[i], height, sign)
        graph.plot(za, ya, zorder=2, **PAIR_STYLE['added'])


def check_counts(lay):
    '''The degrees, earlier counts and batches of data.json agree with the
    pair lists (a cheap guard against an inconsistent data.json).'''
    nb = len(lay['bins'])
    degree = [0]*nb
    earlier = [0]*nb
    for a, b in lay['pairs_today'] + lay['pairs_added']:
        degree[a] += 1
        degree[b] += 1
        earlier[max(a, b)] += 1
    for k, b in enumerate(lay['bins']):
        assert b['degree'] == degree[k] and b['earlier'] == earlier[k], (lay['key'], k)
        assert 1 <= b['batch'] <= b['earlier'] + 1, (lay['key'], k)


def top_text(bins, members):
    '''The text of one row of largest-degree bins: their groups, degree and
    earlier count (a range when the earlier counts differ).'''
    groups = sorted({bins[i]['group'] for i in members})
    degree = bins[members[0]]['degree']
    early = sorted({bins[i]['earlier'] for i in members})
    early_txt = str(early[0]) if len(early) == 1 else f'{early[0]}-{early[-1]}'
    return f"{', '.join(groups)}: {degree} partners, {early_txt} earlier in bin order"


layouts = data['layouts']
nblocks = len(layouts)
# rows (heights in inches): header and spacer; per layout an n(z) panel, a
# thin gap and the pair graph, with a spacer between layouts; room for the
# x axis; the zoom header and spacer; the four zoom pair graphs
heights = [0.50, 0.07]
for k in range(nblocks):
    heights += [0.46, 0.025, 0.44]
    if k < nblocks - 1:
        heights.append(0.10)
heights += [0.50, 0.40, 0.04]
for k in range(nblocks):
    heights.append(0.62)
    if k < nblocks - 1:
        heights.append(0.06)
bottom_room = 0.40
fig_h = (sum(heights) + bottom_room)/0.995
fig = plt.figure(figsize=(6.4, fig_h))
grid = GridSpec(len(heights), 1, figure=fig, height_ratios=heights, hspace=0.0,
                left=0.09, right=0.995, top=0.995, bottom=bottom_room/fig_h)

# ---- the full-range part ----
header_axes(fig.add_subplot(grid[0]), layouts, Z_RANGE, list(GROUPS), band_note=True)
row = 2
graphs = []
first_nz = None
for k, lay in enumerate(layouts):
    check_counts(lay)
    ax = fig.add_subplot(grid[row])
    graph = fig.add_subplot(grid[row + 2], sharex=ax)
    row += 4
    first_nz = first_nz or ax
    graphs.append(graph)
    band = lay['lrg3_elg1_range']
    for target in (ax, graph):
        target.axvspan(band[0], band[1], color=BAND_COLOR, lw=0, zorder=0, label='band')
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
    ax.set_xlim(*Z_RANGE)
    s = lay['summary']
    place_text(fig, ax, [(Z_RANGE[1] - 0.012, y, 'right') for y in (1.72, 1.85, 1.6)],
               f"{LABEL[lay['key']]}: {s['nbins']} bins, {s['npairs_today']} + "
               f"{s['npairs_added']} pairs", lay['key'], va='center', fontsize=FONT + 0.5)

    # the pair graph: dots at the bin means, today's segments, added arcs
    bins = lay['bins']
    zbar = np.array([b['zbar'] for b in bins])
    yrow = np.array([GRAPH_ROW[GROUPS[b['group']]['lane']] for b in bins])
    draw_pairs(graph, lay, zbar, yrow, ARC_HEIGHT['full'])
    dmax = s['max_degree']
    top = np.array([b['degree'] == dmax for b in bins])
    graph.plot(zbar[~top], yrow[~top], ls='none', marker=lay['marker'], ms=NODE_SIZE,
               mfc=lay['color'], mec='0.1', mew=0.35, zorder=3)
    graph.plot(zbar[top], yrow[top], ls='none', marker='o', ms=TOP_SIZE, mfc=TOP_COLOR,
               mec=TOP_COLOR, mew=0.3, zorder=4)
    graph.set_ylim(-0.78, 1.78)
    graph.set_yticks([0.0, 1.0])
    graph.set_yticklabels(['ELG', 'BGS, LRG'], fontsize=FONT - 1)
    graph.tick_params(axis='y', length=0, pad=2)
    graph.set_xlim(*Z_RANGE)
    if k < nblocks - 1:
        plt.setp(graph.get_xticklabels(), visible=False)
    else:
        graph.set_xlabel('redshift $z$', fontsize=9)
        graph.tick_params(axis='x', labelsize=8)
    # the degree texts: the upper row's beside the band on the right (the
    # BGS/LRG row ends at z = 1.1), the lower row's beside it on the left
    # (the ELG row starts at z = 0.8)
    for lane_row, side in ((GRAPH_ROW[1], 'right'), (GRAPH_ROW[0], 'left')):
        members = [i for i in range(len(bins)) if top[i] and yrow[i] == lane_row]
        if not members:
            continue
        if side == 'right':
            cands = [(band[1] + dz, lane_row + dy, 'left') for dy in (0.0, 0.3, -0.3)
                     for dz in (0.015, 0.025, 0.05)]
        else:
            cands = [(band[0] - dz, lane_row + dy, 'right') for dy in (0.0, -0.3, 0.3)
                     for dz in (0.025, 0.05, 0.08)]
        place_text(fig, graph, cands, top_text(bins, members), lay['key'] + ' graph',
                   va='center', fontsize=FONT - 0.5)

# ---- the zoom on the LRG3/ELG1 volume ----
zoom_header = fig.add_subplot(grid[row])
header_axes(zoom_header, layouts, Z_ZOOM, ['LRG2', 'LRG3', 'ELG1', 'ELG2'],
            band_note=False)
row += 2
zooms = []
for k, lay in enumerate(layouts):
    graph = fig.add_subplot(grid[row], sharex=zooms[0] if zooms else None)
    row += 2
    zooms.append(graph)
    band = lay['lrg3_elg1_range']
    graph.axvspan(band[0], band[1], color=BAND_COLOR, lw=0, zorder=0, label='band')
    graph.grid(False)
    bins = lay['bins']
    zbar = np.array([b['zbar'] for b in bins])
    yrow = np.array([GRAPH_ROW[GROUPS[b['group']]['lane']] for b in bins])
    draw_pairs(graph, lay, zbar, yrow, ARC_HEIGHT['zoom'])
    graph.set_xlim(*Z_ZOOM)
    graph.set_ylim(-0.98, 1.98)
    # the dots whose disc lies inside the window: white with the layout's
    # color as edge, black for the bins of largest degree; each carries its
    # greedy batch
    margin = 0.006
    shown = (zbar > Z_ZOOM[0] + margin) & (zbar < Z_ZOOM[1] - margin)
    dmax = lay['summary']['max_degree']
    top = np.array([b['degree'] == dmax for b in bins])
    for sel, face, edge, ink in ((shown & ~top, 'white', lay['color'], 'black'),
                                 (shown & top, TOP_COLOR, TOP_COLOR, 'white')):
        graph.plot(zbar[sel], yrow[sel], ls='none', marker='o', ms=ZOOM_NODE_SIZE,
                   mfc=face, mec=edge, mew=1.0, zorder=3)
        for i in np.flatnonzero(sel):
            graph.text(zbar[i], yrow[i], str(bins[i]['batch']), ha='center',
                       va='center_baseline', fontsize=ZOOM_NUMBER_FONT, color=ink,
                       zorder=5)
    graph.set_yticks([0.0, 1.0])
    graph.set_yticklabels(['ELG', 'LRG'], fontsize=FONT - 1)
    graph.tick_params(axis='y', length=0, pad=2)
    if k < nblocks - 1:
        plt.setp(graph.get_xticklabels(), visible=False)
    else:
        graph.set_xlabel('redshift $z$', fontsize=9)
        graph.tick_params(axis='x', labelsize=8)
    s = lay['summary']
    # the layout and its batch counts, in the empty ELG row left of the band
    cands = [(Z_ZOOM[0] + 0.006, y, 'left') for y in (0.0, -0.25, 0.25, 0.5)]
    place_text(fig, graph, cands,
               f"{LABEL[lay['key']]}\ngreedy: {s['greedy_batches']} batches, "
               f"fewest: {s['fewest_batches']}", lay['key'] + ' zoom', va='center',
               fontsize=FONT - 0.5, linespacing=1.05)

# the shared y labels of the two parts
y_full = 0.5*(first_nz.get_position().y1 + graphs[-1].get_position().y0)
y_zoom = 0.5*(zooms[0].get_position().y1 + zooms[-1].get_position().y0)
fig.text(0.0, y_full, r'raw $n(z)$ at unit peak; pairs', rotation=90, ha='left',
         va='center', fontsize=9)
fig.text(0.0, y_zoom, 'pairs; greedy batch in each dot', rotation=90, ha='left',
         va='center', fontsize=9)
# one legend under the zoom: the two kinds of pair and the largest degree
handles = [Line2D([], [], **PAIR_STYLE['today']), Line2D([], [], **PAIR_STYLE['added']),
           Line2D([], [], ls='none', marker='o', ms=TOP_SIZE, mfc=TOP_COLOR,
                  mec=TOP_COLOR, mew=0.3)]
fig.legend(handles=handles,
           labels=[PAIR_LABEL['today'], PAIR_LABEL['added'], 'largest degree'],
           loc='upper center', bbox_to_anchor=(0.54, 0.0), ncol=3, frameon=False,
           fontsize=8, handlelength=3.2, columnspacing=2.0, handletextpad=0.5)
fig.savefig(os.path.join(HERE, 'pair_graph_bandwidth2.pdf'))
fig.savefig(os.path.join(HERE, 'pair_graph_bandwidth2.png'), dpi=300)
plt.close(fig)
