"""Redraw batching.pdf from data.json: how the notebooks compute the per-bin
bias derivatives in batches, on the real bins of the Delta d_com = 62 Mpc
layout.

The DESI data vector m holds, for every thin bin i, its auto spectrum C_ii
in 53 multipole bands, then the cross spectrum C_ij of every pair of bins
the dataset lists (bins whose redshift distributions overlap or touch). A
derivative such as dm/db_{2,i} (b_{2,i} the quadratic bias of bin i) is
nonzero only in the spectra that contain bin i. The notebooks therefore
shift b_{2,i} in every bin i of a "batch" at once, where no two bins of a batch
share a measured cross spectrum, and evaluate the vector at +-h and +-2h
(a five-point stencil, four evaluations). Every spectrum then holds at most
one shifted bin, so the stencil difference on bin i's spectra is bin i's
column, and the column is exactly zero elsewhere. The batches come from a
greedy pass (pair_coloring of the notebooks): each bin, in index order,
takes the lowest batch that none of its lower-index partners holds.

Four panels, the full text width of the note (6.3 x 6.3 in):

  (a) the 76 bins at their mean redshift zbar_i, on two rows (BGS and LRG
      above, ELG below, since LRG3 and ELG1 overlap in redshift), each
      measured cross spectrum an arc or a line between two dots, the six
      groups bracketed, each dot's batch shown by marker and shade;
  (b) a zoom on eight bins where all four batches meet (bins 37 to 41 and
      50 to 52 of the note), with bin numbers, and how the greedy pass
      gives bin 51 batch 4;
  (c) the b_{2,i} derivative block of those eight bins: rows are their spectra
      (autos, then crosses), columns dm/db_{2,i}; a filled cell takes the
      shade and the marker of the batch that fills it, a white cell is
      exactly zero, and the cells of batch 3 are outlined;
  (d) evaluations per bias family against the number of bins N, on log
      axes: 4N bin by bin (dashed, open markers), 4 per batch in batches
      (solid, filled markers; 16 on every layout, since every layout has
      four batches), with the four layouts marked and the saving of 62 Mpc
      drawn as an arrow.

Bins are numbered from 1 on the figure, as in the note: data.json, the
data files and the code count from 0, so the drawn label of bin i is
i + 1 (shown()).

Each batch has a marker and a shade of the cividis map (dark to light, so
the batches stay apart in grayscale and for colorblind readers; no red,
no green). The layouts of panel (d) take the project's colors and markers.
Every text is checked in code against every drawn line, marker and cell,
and against every other text; the script stops with an error when one
touches.

data.json is written by make_batching_data.py (this folder).
Run from any folder: python plot_batching.py
With BATCHING_DEBUG=1 in the environment, a text that finds no clear
position also saves the figure as it stood, debug_batching.png, before
the error.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

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

# Each batch has a marker and a shade. The shades are the cividis map at 0,
# 1/3, 2/3 and 1: its lightness grows evenly, so the four stay apart in
# grayscale and for every common color-vision deficiency. 'ms' evens out
# the visual size of the markers (a diamond and a triangle look smaller
# than a circle of the same nominal size).
BATCH_STYLE = {
    1: {'marker': 'o', 'color': '#00224e', 'ms': 1.00},
    2: {'marker': 's', 'color': '#575d6d', 'ms': 0.88},
    3: {'marker': 'D', 'color': '#a59c74', 'ms': 0.82},
    4: {'marker': '^', 'color': '#fee838', 'ms': 1.08},
}
# every marker gets a thin dark edge, so the light batch-4 marker reads on white
MARKER_EDGE = '0.1'
# the cross pairs: one mid-gray for every arc and line, under the markers
PAIR_COLOR = '0.45'
# The layouts of panel (d): the project's color and marker of each
# (paper-figures skill of lssty1xdesi), and a short name for the in-panel
# labels.
LAYOUT_STYLE = {
    'Delta d_com = 62 Mpc': {'color': '#0072B2', 'marker': 'o',
                             'short': r'$\Delta d_\mathrm{com} = 62$ Mpc'},
    'Delta z = 0.02': {'color': '#E69F00', 'marker': 's', 'short': r'$\Delta z = 0.02$'},
    'Delta z = 0.05': {'color': '#CC79A7', 'marker': 'D', 'short': r'$\Delta z = 0.05$'},
    'Delta z = 0.2': {'color': '#56B4E9', 'marker': '^', 'short': r'$\Delta z = 0.2$'},
}
# The two rows of panels (a) and (b): BGS and LRG bins above, ELG bins
# below. LRG3 and ELG1 cover the same redshifts, so on one row their dots
# would sit on top of each other.
ROW_OF_GROUP = {'BGS': 1.0, 'LRG1': 1.0, 'LRG2': 1.0, 'LRG3': 1.0,
                'ELG1': 0.0, 'ELG2': 0.0}
# The height of the arc between two bins of one row, in units of the row
# spacing: the arcs bulge away from the other row, so the lines between
# the rows (the LRG3 x ELG1 pairs) stay apart from them.
ARC_HEIGHT = {'overview': 0.24, 'zoom': 0.30}
# the redshift window of the zoom, panel (b): it holds the 0-based bins 36
# to 40 and 49 to 51 (zbar = 0.768 to 0.859) with a margin on each side,
# and stops short of bins 41 and 52 (zbar = 0.883); the arcs and lines to
# bins 35, 41 and 52 leave the window, a reminder that the graph goes on
ZOOM_WINDOW = (0.757, 0.871)
# font sizes [pt]: tick labels, axis labels, in-panel teaching notes
TICK_FS = 7.5
LABEL_FS = 8.5
NOTE_FS = 7.3
# the margin [pt] a text keeps from any line, marker, cell or other text
CLEAR_PT = 1.5
# the figure width [in]: the text width of the note (letter paper, 1.15 in
# margins)
FIG_WIDTH = 6.3
# the worked examples of panels (b) and (c), as 0-based indices of
# data.json: the bin whose batch the greedy pass explains (index 50, bin 51
# of the note, takes batch 4, the only bin of the stretch that does), the
# batch whose cells are outlined, and the cross spectrum that is filled
# twice (indices 38 and 49: an LRG3 bin of batch 1 and an ELG1 bin of
# batch 3, C_{39,50} of the note)
GREEDY_EXAMPLE = 50
OUTLINED = 3
TWICE = (38, 49)


def shown(i):
    '''Return the drawn number of bin i: data.json counts bins from 0, the
    note and the figure from 1.'''
    return i + 1

with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
focus = data['focus']
bins = focus['bins']
pairs = [tuple(p) for p in focus['pairs']]
zbar = np.array([b['zbar'] for b in bins])
batch_of = [b['batch'] for b in bins]
row_of = np.array([ROW_OF_GROUP[b['group']] for b in bins])
stretch = list(focus['stretch'])
layouts = data['layouts']
evals_per_batch = data['evaluations_per_batch']


# ---------------------------------------------------------------------------
# The clearance check: texts never touch drawn data or each other.
# ---------------------------------------------------------------------------

def drawn_points(ax):
    '''Return display points sampled along every line and marker of ax.

    Every Line2D of the axes (the arcs, the pair lines, the markers, the
    brackets and the curves of panel (d)) is sampled in display
    coordinates: each vertex, and 40 points along each segment of a drawn
    line. Each point carries a radius, half the marker size for a vertex
    with a marker and half the line width for a segment point, so a
    marker counts with its full area. Points outside the axes box are
    dropped: the lines are clipped there and invisible.

    Arguments:
      ax = the axes to sample.

    Returns:
      (points, radii): an (n, 2) array of display coordinates [px] and an
      (n,) array of radii [px].
    '''
    px_per_pt = ax.figure.dpi/72.0
    box = ax.get_window_extent()
    points = [np.zeros((0, 2))]
    radii = [np.zeros(0)]
    t = np.linspace(0.0, 1.0, 40)[:, None]
    for line in ax.get_lines():
        if not line.get_visible():
            continue
        xy = np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()])
        display_xy = line.get_transform().transform(xy)
        if line.get_marker() not in ('None', 'none', '', ' ', None):
            points.append(display_xy)
            radii.append(np.full(len(display_xy), 0.5*line.get_markersize()*px_per_pt))
        if line.get_linestyle() in ('None', 'none', ' ', ''):
            continue
        for i in range(len(display_xy) - 1):
            # the segment from vertex i to vertex i + 1, as 40 points
            segment = display_xy[i] + t*(display_xy[i + 1] - display_xy[i])
            points.append(segment)
            radii.append(np.full(len(segment), 0.5*line.get_linewidth()*px_per_pt))
    points = np.vstack(points)
    radii = np.concatenate(radii)
    inside = ((points[:, 0] >= box.x0) & (points[:, 0] <= box.x1)
              & (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1))
    return points[inside], radii[inside]


def cell_boxes(ax):
    '''Return the display boxes of every heat-map cell of ax.

    The cells are the Rectangle patches tagged 'cell' in panel (c);
    a text must not overlap any of them, white (zero) cells included,
    since a zero cell is data too.

    Arguments:
      ax = the axes whose cells to collect.

    Returns:
      list of matplotlib Bbox in display coordinates.
    '''
    boxes = []
    for patch in ax.patches:
        if patch.get_gid() == 'cell':
            boxes.append(patch.get_window_extent())
    return boxes


def text_box(artist, renderer):
    '''Return the display box of a text alone, without its arrow.

    For an annotation, get_window_extent also covers the arrow, which is
    meant to touch the data it points to; Text.get_window_extent gives the
    text alone. A legend returns its own box.

    Arguments:
      artist   = a Text, an Annotation or a Legend.
      renderer = the figure's renderer.

    Returns:
      matplotlib Bbox in display coordinates.
    '''
    if isinstance(artist, Text):
        return Text.get_window_extent(artist, renderer)
    return artist.get_window_extent(renderer)


def scene(axes_list):
    '''Collect every drawn line, marker and cell of the figure once.

    Sampling the lines is the slow part of a check, and texts never move
    a line, so place() samples them once and then tries every candidate
    position against the same arrays.

    Arguments:
      axes_list = every axes of the figure.

    Returns:
      list of (panel label, points, radii, cells), one per axes, with the
      arrays of drawn_points and the boxes of cell_boxes.
    '''
    out = []
    for ax in axes_list:
        points, radii = drawn_points(ax)
        out.append((ax.get_label(), points, radii, cell_boxes(ax)))
    return out


def touching(box, drawn, pad):
    '''Return a description of the first drawn element a box touches, or None.

    Arguments:
      box   = a display Bbox (a text's box).
      drawn = what scene() returned: the lines, markers and cells of every
              panel (a text near a panel edge can reach into its neighbour).
      pad   = the clearance a text keeps [px].

    Returns:
      str naming the panel and the element, or None when the box is clear.
    '''
    for label, points, radii, cells in drawn:
        # a point of radius r touches the box when it lies within r + pad
        # of it
        reach = radii + pad
        hit = ((points[:, 0] > box.x0 - reach) & (points[:, 0] < box.x1 + reach)
               & (points[:, 1] > box.y0 - reach) & (points[:, 1] < box.y1 + reach))
        if np.any(hit):
            return f'a line or marker of panel {label}'
        for cell in cells:
            if boxes_meet(box, cell, pad):
                return f'a cell of panel {label}'
    return None


def boxes_meet(a, b, pad):
    '''Return True when two display boxes come closer than pad [px].'''
    return (a.x0 - pad < b.x1 and a.x1 + pad > b.x0
            and a.y0 - pad < b.y1 and a.y1 + pad > b.y0)


def visible_boxes(texts, renderer):
    '''Return (name, box) for every visible, nonempty text of a list.

    Arguments:
      texts    = list of (name, artist) pairs.
      renderer = the figure's renderer.

    Returns:
      list of (name, display Bbox); hidden and empty texts (unused tick
      labels) are left out.
    '''
    boxes = []
    for name, artist in texts:
        if not artist.get_visible():
            continue
        if isinstance(artist, Text) and artist.get_text() == '':
            continue
        box = text_box(artist, renderer)
        if box.width == 0 or box.height == 0:
            continue
        boxes.append((name, box))
    return boxes


def both_ticks(name_a, name_b):
    '''Return True for two tick-like labels, which check_clear does not
    compare with each other (see its docstring).'''
    return name_a.startswith('tick') and name_b.startswith('tick')


def check_clear(fig, axes_list, texts):
    '''Raise when a text touches a drawn element or two texts overlap.

    Every text is checked against every line, marker and cell of every
    panel, and against every other text, with a margin of CLEAR_PT. One
    exception: two tick-like labels (names starting with 'tick': the axis
    tick labels and the row names of panel (c), one per row) are not
    checked against each other. Their positions are fixed one per tick or
    row, and a mathtext box such as that of C_{36,37} is about 9 pt tall
    whatever its font size, taller than the ink and than a row of
    panel (c).

    Arguments:
      fig       = the figure.
      axes_list = every axes of the figure.
      texts     = list of (name, artist) pairs: every in-panel note, label,
                  legend, and the tick labels.

    Raises:
      RuntimeError naming the text and what it touches.
    '''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = CLEAR_PT*fig.dpi/72.0
    drawn = scene(axes_list)
    boxes = visible_boxes(texts, renderer)
    for name, box in boxes:
        hit = touching(box, drawn, pad)
        if hit is not None:
            raise RuntimeError(f'batching: the text "{name}" touches {hit}; move it')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if both_ticks(boxes[i][0], boxes[j][0]):
                continue
            if boxes_meet(boxes[i][1], boxes[j][1], pad):
                raise RuntimeError(f'batching: "{boxes[i][0]}" overlaps "{boxes[j][0]}"')


def place(fig, axes_list, placed, name, artist, candidates):
    '''Move a text to the first candidate position where it is clear.

    Each candidate is a position in the text's own coordinates (data
    coordinates for a text drawn with ax.text, the text position of an
    annotation). The first one at which the text touches no line, marker,
    cell or already placed text is kept. The lines and the placed texts
    are measured once; only the moving text is measured at each candidate.

    Arguments:
      fig        = the figure.
      axes_list  = every axes of the figure.
      placed     = list of (name, artist) already placed; the text is
                   appended to it.
      name       = the text's name for error messages.
      artist     = the Text or Annotation to move.
      candidates = list of (x, y) positions, best first.

    Raises:
      RuntimeError when no candidate clears.
    '''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = CLEAR_PT*fig.dpi/72.0
    drawn = scene(axes_list)
    others = visible_boxes(placed, renderer)
    # why each candidate failed, so the error names what is in the way
    reasons = []
    for xy in candidates:
        artist.set_position(xy)
        box = text_box(artist, renderer)
        reason = touching(box, drawn, pad)
        if reason is None:
            for other_name, other_box in others:
                if boxes_meet(box, other_box, pad) and not both_ticks(name, other_name):
                    reason = f'the text "{other_name}"'
                    break
        if reason is not None:
            reasons.append(f'  at {xy}: touches {reason}')
            continue
        placed.append((name, artist))
        return
    if os.environ.get('BATCHING_DEBUG'):
        # the figure as it stood, to see what blocks the text
        fig.savefig(os.path.join(HERE, 'debug_batching.png'), dpi=150)
    raise RuntimeError(f'batching: no candidate position clears the text "{name}":\n'
                       + '\n'.join(reasons))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def log_grid(best, xs, ys):
    '''Return the grid points (x, y), sorted by log distance from best.

    The candidate positions of a text on the log-log panel (d): every
    pair of xs and ys, the one nearest the preferred position first, so
    place() keeps the nearest position that clears.

    Arguments:
      best = the preferred (x, y), in data units.
      xs   = the candidate x values, positive.
      ys   = the candidate y values, positive.

    Returns:
      list of (x, y) tuples.
    '''
    points = []
    for x in xs:
        for y in ys:
            distance = np.hypot(np.log10(x/best[0]), np.log10(y/best[1]))
            points.append((distance, (float(x), float(y))))
    points.sort(key=first_entry)
    return [p[1] for p in points]


def first_entry(item):
    '''Return item[0], the sort key of log_grid's (distance, point) pairs.'''
    return item[0]


def batch_marker(ax, x, y, batch, size, zorder=4):
    '''Draw one batch marker (its shape and shade) at (x, y), size in pt.'''
    st = BATCH_STYLE[batch]
    ax.plot([x], [y], ls='none', marker=st['marker'], ms=size*st['ms'],
            mfc=st['color'], mec=MARKER_EDGE, mew=0.5, zorder=zorder)


def arc(z0, z1, y0, height):
    '''Return the points of a half ellipse from (z0, y0) to (z1, y0).

    Arguments:
      z0, z1 = the two ends on the redshift axis.
      y0     = the row of both ends.
      height = the bulge, positive upwards, in row units.

    Returns:
      (z, y) arrays of 40 points.
    '''
    t = np.linspace(0.0, np.pi, 40)
    center = 0.5*(z0 + z1)
    half = 0.5*(z1 - z0)
    return center - half*np.cos(t), y0 + height*np.sin(t)


def draw_pairs(ax, height, lw):
    '''Draw every measured cross pair of the 62 Mpc layout on ax.

    Two bins of one row are joined by an arc that bulges away from the
    other row; a bin of the upper row and one of the lower row (an LRG3
    and an ELG1 bin at the same redshift) are joined by a straight line.

    Arguments:
      ax     = the axes (panel (a) or (b)).
      height = the arc height in row units.
      lw     = the line width [pt].
    '''
    for a, b in pairs:
        if row_of[a] == row_of[b]:
            sign = 1.0 if row_of[a] > 0.5 else -1.0
            z, y = arc(zbar[a], zbar[b], row_of[a], sign*height)
        else:
            z, y = [zbar[a], zbar[b]], [row_of[a], row_of[b]]
        ax.plot(z, y, color=PAIR_COLOR, lw=lw, zorder=2, solid_capstyle='round')


def draw_dots(ax, size):
    '''Draw every bin as its batch marker at (zbar_i, row of i), size in pt.'''
    for i in range(len(bins)):
        batch_marker(ax, zbar[i], row_of[i], batch_of[i], size)


def draw_groups(ax, upper_y, lower_y, tick, fontsize, placed, fig, axes_list):
    '''Bracket the six groups along their rows and name them.

    The upper-row groups are bracketed above the arcs, the lower-row groups
    below them; each name sits beyond its bracket, centered on its span.

    Arguments:
      ax        = the axes of panel (a).
      upper_y   = the height of the upper brackets.
      lower_y   = the height of the lower brackets.
      tick      = the length of the bracket ends, toward the row.
      fontsize  = the size of the names [pt].
      placed, fig, axes_list = as in place().
    '''
    for group in focus['groups']:
        z0, z1 = group['zbar_range']
        # half a bin spacing beyond the first and the last bin
        margin = 0.006
        upper = ROW_OF_GROUP[group['name']] > 0.5
        y = upper_y if upper else lower_y
        end = y - tick if upper else y + tick
        ax.plot([z0 - margin, z0 - margin, z1 + margin, z1 + margin], [end, y, y, end],
                color='0.15', lw=0.6, zorder=2)
        offset = 0.08 if upper else -0.08
        label = ax.text(0.5*(z0 + z1), y + offset, group['name'], fontsize=fontsize,
                        ha='center', va='bottom' if upper else 'top')
        place(fig, axes_list, placed, f'group {group["name"]}', label,
              [(0.5*(z0 + z1), y + offset)])


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

def plain_number(value, position):
    '''Return a tick label as a plain number (16, not 1.6 x 10^1).'''
    return f'{value:g}'


def inches_to_rect(fig, left, bottom, width, height):
    '''Return an axes rectangle in figure fractions from one in inches.'''
    W, H = fig.get_size_inches()
    return [left/W, bottom/H, width/W, height/H]


# The layout [in], bottom to top: the bottom strip holds panel (c) on the
# left and, on the right, the notes of (c) above panel (d); then (b), then
# (a). Panel (c)'s axes span the whole strip without a frame, so its notes
# share its data coordinates: one unit is one cell, CELL_W wide and CELL_H
# tall.
CELL_W = 0.22
CELL_H = 0.118
# panel (c) holds one row per auto and per cross spectrum of the stretch,
# a gap of GAP_ROWS rows between them, HEADER_ROWS rows of column header
# above and FOOTER_ROWS of margin below; its strip is at least H_BLOCK_MIN
# tall, the room panel (d) and the notes of (c) need
GAP_ROWS = 0.6
HEADER_ROWS = 4.9
FOOTER_ROWS = 0.7
H_BLOCK_MIN = 3.10
inside_stretch = set(stretch)
n_cross_rows = sum(1 for p in pairs
                   if p[0] in inside_stretch and p[1] in inside_stretch)
H_BLOCK = max(H_BLOCK_MIN, CELL_H*(HEADER_ROWS + len(stretch) + GAP_ROWS
                                   + n_cross_rows + FOOTER_ROWS))
H_ZOOM = 1.25
H_OVER = 1.15
GAP_LOW = 0.40
GAP_HIGH = 0.36
FIG_HEIGHT = H_BLOCK + GAP_LOW + H_ZOOM + GAP_HIGH + H_OVER + 0.05
# panel (d): its box inside the bottom strip [in]
SAVE_LEFT = 3.50
SAVE_BOTTOM = 0.33
SAVE_WIDTH = FIG_WIDTH - SAVE_LEFT - 0.05
SAVE_HEIGHT = 1.55
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
y_zoom = H_BLOCK + GAP_LOW
y_over = y_zoom + H_ZOOM + GAP_HIGH
ax_over = fig.add_axes(inches_to_rect(fig, 0.62, y_over, FIG_WIDTH - 0.66, H_OVER),
                       label='(a)')
ax_zoom = fig.add_axes(inches_to_rect(fig, 0.62, y_zoom, FIG_WIDTH - 0.66, H_ZOOM),
                       label='(b)')
ax_block = fig.add_axes(inches_to_rect(fig, 0.0, 0.0, FIG_WIDTH, H_BLOCK), label='(c)')
ax_save = fig.add_axes(inches_to_rect(fig, SAVE_LEFT, SAVE_BOTTOM, SAVE_WIDTH,
                                      SAVE_HEIGHT), label='(d)')
axes_list = [ax_over, ax_zoom, ax_block, ax_save]
placed = []

# --- (a) the 76 bins and their cross pairs ---------------------------------
ax = ax_over
ax.grid(False)
ax.axvspan(*ZOOM_WINDOW, color='0.92', lw=0, zorder=0)
draw_pairs(ax, ARC_HEIGHT['overview'], lw=0.6)
draw_dots(ax, size=3.6)
ax.set_xlim(0.065, 1.59)
ax.set_ylim(-1.30, 2.12)
ax.set_yticks([0.0, 1.0])
ax.set_yticklabels(['ELG', 'BGS, LRG'], fontsize=TICK_FS)
ax.tick_params(axis='y', length=0)
ax.tick_params(axis='x', labelsize=TICK_FS, length=2.5, pad=1.5)
ax.set_xlabel(r'mean redshift $\bar{z}_i$ of bin $i$', fontsize=LABEL_FS, labelpad=1.5)
draw_groups(ax, upper_y=1.52, lower_y=-0.52, tick=0.14, fontsize=NOTE_FS + 0.3,
            placed=placed, fig=fig, axes_list=axes_list)
note = ax.text(0, 0, '(a)', fontsize=LABEL_FS, ha='left', va='top', fontweight='bold')
place(fig, axes_list, placed, 'panel (a)', note, [(0.075, 2.05)])
note = ax.text(0, 0, 'zoom: (b)', fontsize=NOTE_FS - 0.5, ha='center', va='top',
               color='0.3')
place(fig, axes_list, placed, 'zoom note', note,
      [(0.5*sum(ZOOM_WINDOW), 2.05), (0.5*sum(ZOOM_WINDOW), -0.95)])
# the batch legend and the pair line, in the empty lower left (no ELG bin
# lies below z = 0.81)
handles = []
for k in sorted(BATCH_STYLE):
    st = BATCH_STYLE[k]
    handles.append(Line2D([], [], ls='none', marker=st['marker'], ms=4.6*st['ms'],
                          mfc=st['color'], mec=MARKER_EDGE, mew=0.5, label=f'batch {k}'))
legend = ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 0.0),
                   ncol=4, frameon=False, fontsize=NOTE_FS, handlelength=1.0,
                   handletextpad=0.35, columnspacing=1.1, borderaxespad=0.3)
ax.add_artist(legend)
placed.append(('batch legend', legend))
# the pair line, in a second legend whose bottom sits CLEAR_PT + 1 pt above
# the top of the batch legend (in axes fractions)
fig.canvas.draw()
batch_top = legend.get_window_extent(fig.canvas.get_renderer()).y1
batch_top += (CLEAR_PT + 1.0)*fig.dpi/72.0
anchor_y = ax.transAxes.inverted().transform((0.0, batch_top))[1]
pair_handle = Line2D([], [], color=PAIR_COLOR, lw=1.0,
                     label=r'a measured cross spectrum $C_{ij}$ of bins $i$ and $j$')
# the same pads as the batch legend, so the two start at the same x
legend = ax.legend(handles=[pair_handle], loc='lower left',
                   bbox_to_anchor=(0.0, anchor_y), frameon=False, fontsize=NOTE_FS,
                   handlelength=1.6, handletextpad=0.35, borderaxespad=0.3)
placed.append(('pair legend', legend))
note = ax.text(0, 0, 'batches 3 and 4 hold only ELG1 bins,\n'
               'where ELG1 and LRG3 overlap in redshift', fontsize=NOTE_FS,
               ha='left', va='top', linespacing=1.15)
place(fig, axes_list, placed, 'batches 3 and 4 note', note,
      [(1.105, 1.30), (1.105, 1.55), (1.105, 1.80)])

# --- (b) the zoom on eight bins --------------------------------------------
ax = ax_zoom
ax.grid(False)
ax.set_facecolor('#fbfbfb')
draw_pairs(ax, ARC_HEIGHT['zoom'], lw=1.1)
draw_dots(ax, size=10.0)
ax.set_xlim(*ZOOM_WINDOW)
ax.set_ylim(-1.18, 1.98)
ax.set_yticks([0.0, 1.0])
ax.set_yticklabels(['ELG1', 'LRG2, LRG3'], fontsize=TICK_FS)
ax.tick_params(axis='y', length=0)
ax.tick_params(axis='x', labelsize=TICK_FS, length=2.5, pad=1.5)
ax.set_xlabel(r'$\bar{z}_i$', fontsize=LABEL_FS, labelpad=0.5)
# the bin numbers, beyond the arcs of each row
for i in stretch:
    upper = row_of[i] > 0.5
    y = row_of[i] + (0.40 if upper else -0.40)
    label = ax.text(zbar[i], y, f'{shown(i)}', fontsize=NOTE_FS + 0.4, ha='center',
                    va='bottom' if upper else 'top')
    place(fig, axes_list, placed, f'bin label {i}', label,
          [(zbar[i], y), (zbar[i], y + (0.12 if upper else -0.12))])
note = ax.text(0, 0, '(b)', fontsize=LABEL_FS, ha='left', va='top', fontweight='bold')
place(fig, axes_list, placed, 'panel (b)', note, [(0.7585, 1.93)])
note = ax.text(0, 0, 'no arc or line joins\ntwo dots of the same batch',
               fontsize=NOTE_FS, ha='left', va='center', linespacing=1.15)
place(fig, axes_list, placed, 'no arc note', note,
      [(0.7585, 0.30), (0.7585, 0.10), (0.7585, -0.35)])
# How the greedy pass gives bin GREEDY_EXAMPLE its batch: its lower-index
# partners (39, 40 and 50 for bin 51, numbered from 1) already hold the
# lower batches.
g = GREEDY_EXAMPLE
lower = []
for p in bins[g]['partners']:
    if p < g:
        lower.append(p)
held = [batch_of[p] for p in lower]
lower_text = ', '.join(str(shown(p)) for p in lower)
held_text = ', '.join(str(h) for h in held)
note = ax.annotate(
    f'greedy, in bin order: the partners {lower_text} of bin {shown(g)} already '
    f'hold batches {held_text}, so {shown(g)} takes batch {batch_of[g]}',
    xy=(zbar[g] + 0.0012, row_of[g] - 0.17), xytext=(0, 0),
    textcoords='data', fontsize=NOTE_FS, ha='left', va='center',
    arrowprops={'arrowstyle': '-|>', 'color': '0.25', 'lw': 0.6,
                'shrinkA': 1.5, 'shrinkB': 4.0, 'mutation_scale': 6})
place(fig, axes_list, placed, 'greedy note', note,
      [(0.7585, -0.93), (0.7585, -1.0), (0.7585, -0.85)])


# --- (c) the b_{2,i} derivative block of the eight bins --------------------
ax = ax_block
ax.set_axis_off()
columns = list(stretch)
inside = set(stretch)
autos = [(i, i) for i in columns]
# the cross spectra of two bins of the stretch, in pair-file order (the
# order of the data vector)
crosses = []
for p in pairs:
    if p[0] in inside and p[1] in inside:
        crosses.append(p)
# row positions: the autos first, then a gap of GAP_ROWS, then the crosses
spectra = []
for k, s in enumerate(autos):
    spectra.append((s, float(k)))
for k, s in enumerate(crosses):
    spectra.append((s, len(autos) + GAP_ROWS + k))
n_cols = len(columns)
# x = 0 is the first column and y = 0 the first row; the axes start 4.15
# cells left of the first column (room for the row names) and 4.9 rows
# above the first row (room for the column header)
x_left = -4.15
x_right = x_left + FIG_WIDTH/CELL_W
y_top = -HEADER_ROWS
y_bottom = y_top + H_BLOCK/CELL_H
ax.set_xlim(x_left, x_right)
ax.set_ylim(y_bottom, y_top)
for (a, b), y in spectra:
    for c, i in enumerate(columns):
        # Column i is nonzero exactly on the spectra that contain bin i
        # (rows[i] of spectrum_rows); there it takes the stencil difference
        # of the batch of bin i. Every other cell is exactly zero.
        filled = i in (a, b)
        face = 'white'
        if filled:
            face = BATCH_STYLE[batch_of[i]]['color']
        outlined = filled and batch_of[i] == OUTLINED
        cell = Rectangle((c - 0.5, y - 0.5), 1.0, 1.0, facecolor=face,
                         edgecolor='black' if outlined else '0.78',
                         lw=1.1 if outlined else 0.35, zorder=3 if outlined else 2,
                         gid='cell')
        ax.add_patch(cell)
        if filled:
            # the batch's marker inside the cell as well as its shade, so a
            # cell names its batch in grayscale too: white on the two dark
            # shades, black on the two light ones
            st = BATCH_STYLE[batch_of[i]]
            glyph = 'white' if batch_of[i] <= 2 else 'black'
            ax.plot([c], [y], ls='none', marker=st['marker'], ms=3.4*st['ms'],
                    mfc=glyph, mec='none', zorder=4)
# the line between the autos and the crosses
y_split = len(autos) - 0.5 + 0.5*GAP_ROWS
ax.plot([-0.5, n_cols - 0.5], [y_split, y_split], color='0.2', lw=0.6, zorder=1)
# the row names, and above each column its bin number and batch marker
for (a, b), y in spectra:
    label = ax.text(-0.75, y, rf'$C_{{{shown(a)},{shown(b)}}}$', fontsize=TICK_FS - 0.6,
                    ha='right',
                    va='center')
    placed.append((f'tick row {a},{b}', label))
for c, i in enumerate(columns):
    label = ax.text(c, -0.85, f'{shown(i)}', fontsize=TICK_FS - 0.3, ha='center',
                    va='bottom')
    placed.append((f'column {i}', label))
    batch_marker(ax, c, -2.75, batch_of[i], 6.0)
title = ax.text(0.5*(n_cols - 1), -3.75, r'column $i$: $\partial m/\partial b_{2,i}$',
                fontsize=LABEL_FS, ha='center', va='bottom')
placed.append(('block title', title))
# the two kinds of rows, bracketed along the left edge
for text, first, last in [('auto spectra', 0, len(autos) - 1),
                          ('cross spectra', len(autos), len(spectra) - 1)]:
    y0 = spectra[first][1] - 0.4
    y1 = spectra[last][1] + 0.4
    ax.plot([-3.15, -3.0, -3.0, -3.15], [y0, y0, y1, y1], color='0.15', lw=0.6)
    label = ax.text(-3.4, 0.5*(y0 + y1), text, fontsize=NOTE_FS, rotation=90,
                    ha='center', va='center')
    placed.append((text, label))
note = ax.text(x_left + 0.1, y_top + 0.1, '(c)', fontsize=LABEL_FS, ha='left', va='top',
               fontweight='bold')
placed.append(('panel (c)', note))


def row_y(a, b):
    '''Return the y position of the row of spectrum C_ab in panel (c).'''
    for s, y in spectra:
        if s == (a, b):
            return y
    raise KeyError(f'C_{a},{b} is not a row of panel (c)')


# The notes of (c), right of the block and above panel (d), top to bottom
# in the order of the rows they point to, so no two arrows cross. Each
# arrow leaves its note at the middle of the note's left edge (relpos), so
# it runs left to the block and never crosses panel (d).
ARROW = {'arrowstyle': '-|>', 'color': '0.25', 'lw': 0.6, 'shrinkA': 1.5,
         'shrinkB': 0.5, 'mutation_scale': 6, 'relpos': (0.0, 0.5)}
x_note = n_cols + 0.6
block = focus['block']
zero_fraction = 1.0 - block['filled_spectrum_cells']/block['spectrum_cells']
note = ax.annotate(
    'white: exactly zero, the spectrum does not contain bin $i$.\n'
    f'The full block, {block["spectra"]} spectra by {block["columns"]} columns, is '
    f'{100*zero_fraction:.0f} percent zeros',
    xy=(n_cols - 0.5, row_y(columns[0], columns[0])), xytext=(0, 0), textcoords='data',
    fontsize=NOTE_FS, ha='left', va='center', linespacing=1.15, arrowprops=ARROW)
place(fig, axes_list, placed, 'zero note', note,
      [(x_note, -3.9), (x_note, -3.6), (x_note, -3.3)])
members = [i for i in columns if batch_of[i] == OUTLINED]
target = members[-1]
note = ax.annotate(
    f'outlined: what the {evals_per_batch} evaluations of batch {OUTLINED} fill, '
    'at once.\n'
    rf'$b_{{2,i}}$ shifts in bins $i$ = {shown(members[0])}, {shown(members[1])} '
    f'and every other batch-{OUTLINED} bin;\n'
    'no spectrum holds two of them, so each changed row goes to one column',
    xy=(n_cols - 1.5, row_y(target, target)), xytext=(0, 0), textcoords='data',
    fontsize=NOTE_FS, ha='left', va='center', linespacing=1.15, arrowprops=ARROW)
place(fig, axes_list, placed, 'batch 3 note', note,
      [(x_note, -0.5), (x_note, -0.2), (x_note, 0.1)])
twice = TWICE
note = ax.annotate(
    rf'$C_{{{shown(twice[0])},{shown(twice[1])}}}$ is filled twice: in column '
    f'{shown(twice[0])} by batch {batch_of[twice[0]]},\nin column {shown(twice[1])} '
    f'by batch {batch_of[twice[1]]}',
    xy=(n_cols - 0.5, row_y(*twice)), xytext=(0, 0), textcoords='data',
    fontsize=NOTE_FS, ha='left', va='center', linespacing=1.15, arrowprops=ARROW)
place(fig, axes_list, placed, 'filled twice note', note,
      [(x_note, 2.9), (x_note, 3.2), (x_note, 2.6)])

# --- (d) what batching buys --------------------------------------------------
ax = ax_save
nbatches = sorted({lay['batches'] for lay in layouts})
if len(nbatches) != 1:
    raise RuntimeError(f'batching: the layouts use {nbatches} batches; panel (d) draws '
                       'one batched line and needs one count')
N = np.geomspace(1.0, 100.0, 300)
per_bin = evals_per_batch*N
# a layout of N < 4 bins cannot need more than N batches
batched = evals_per_batch*np.minimum(N, nbatches[0])
ax.plot(N, per_bin, color='0.35', lw=1.2, ls=(0, (5.0, 2.5)), zorder=2)
ax.plot(N, batched, color='black', lw=1.8, zorder=2)
layout_handles = []
for lay in layouts:
    st = LAYOUT_STYLE[lay['name']]
    open_marker = {'ls': 'none', 'marker': st['marker'], 'ms': 5.0, 'mfc': 'white',
                   'mec': st['color'], 'mew': 1.3}
    full_marker = {'ls': 'none', 'marker': st['marker'], 'ms': 5.0, 'mfc': st['color'],
                   'mec': st['color'], 'mew': 1.3}
    ax.plot([lay['nbins']], [lay['evaluations_per_family_per_bin']], zorder=4,
            **open_marker)
    ax.plot([lay['nbins']], [lay['evaluations_per_family_batched']], zorder=4,
            **full_marker)
    layout_handles.append((Line2D([], [], **open_marker), Line2D([], [], **full_marker)))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.85, 130.0)
ax.set_ylim(3.0, 2000.0)
ax.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
ax.yaxis.set_major_locator(FixedLocator([4, 16, 100, 1000]))
ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(plain_number))
ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
ax.tick_params(labelsize=TICK_FS, length=2.5, pad=1.5)
ax.set_xlabel(r'number of bins $N$', fontsize=LABEL_FS, labelpad=1.0)
ax.set_ylabel('evaluations per family', fontsize=LABEL_FS, labelpad=2.0)
# the saving of 62 Mpc, as a two-headed arrow between its two markers
focus_layout = [lay for lay in layouts if lay['name'] == data['focus_layout']][0]
n_focus = focus_layout['nbins']
low = focus_layout['evaluations_per_family_batched']
high = focus_layout['evaluations_per_family_per_bin']
color = LAYOUT_STYLE[focus_layout['name']]['color']
ax.plot([n_focus, n_focus], [low*1.35, high/1.35], color=color, lw=0.8, zorder=3)
ax.plot([n_focus], [high/1.35], ls='none', marker='^', ms=3.2, color=color, zorder=3)
ax.plot([n_focus], [low*1.35], ls='none', marker='v', ms=3.2, color=color, zorder=3)
legend = ax.legend(handles=layout_handles, labels=[lay['label'] for lay in layouts],
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
                   loc='upper left', ncol=2, frameon=False, fontsize=NOTE_FS - 0.3,
                   handlelength=2.0, handletextpad=0.4, columnspacing=0.8,
                   labelspacing=0.3, borderaxespad=0.3)
placed.append(('layout legend', legend))
# The saving of 62 Mpc goes first, under the legend, in the empty upper
# left; then the two line names, between and under the lines: the per-bin
# line in the wedge between the two lines, left of the 62 Mpc arrow, and
# the batched line under its flat part.
families = len(data['families'])
note = ax.text(0, 0, (f"{LAYOUT_STYLE[focus_layout['name']]['short']}:\n"
                      f"{low} against {high} per family,\n"
                      f"{families*low} against {families*high:,} "
                      f"for {families} families"),
               fontsize=NOTE_FS, ha='left', va='top', linespacing=1.15)
place(fig, axes_list, placed, 'saving note', note,
      log_grid((1.0, 900.0), [0.95, 1.0, 1.1, 1.25], np.geomspace(250.0, 1900.0, 16)))
# plain text, no mathtext: a mathtext box is about 9 pt tall whatever its
# font size, too tall for the space beside the lines
label = ax.text(0, 0, f'bin by bin: {evals_per_batch} per bin', fontsize=NOTE_FS,
                ha='left', va='center', color='0.25')
place(fig, axes_list, placed, 'per-bin line label', label,
      log_grid((12.0, 30.0), np.geomspace(11.0, 20.0, 8), np.geomspace(24.0, 45.0, 8)))
label = ax.text(0, 0, f'in batches: {evals_per_batch} per batch', fontsize=NOTE_FS,
                ha='left', va='top')
place(fig, axes_list, placed, 'batched line label', label,
      log_grid((5.0, 12.5), np.geomspace(4.5, 30.0, 10), np.geomspace(8.0, 13.5, 6)))
note = ax.text(0, 0, '(d)', fontsize=LABEL_FS, ha='right', va='bottom', fontweight='bold')
place(fig, axes_list, placed, 'panel (d)', note, [(120.0, 3.4), (120.0, 4.0)])

# every tick label and axis label joins the overlap check (panel (c) has
# no axis)
for ax in [ax_over, ax_zoom, ax_save]:
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        placed.append((f'tick {ax.get_label()} {t.get_text()}', t))
    placed.append((f'tick xlabel {ax.get_label()}', ax.xaxis.label))
    placed.append((f'tick ylabel {ax.get_label()}', ax.yaxis.label))
check_clear(fig, axes_list, placed)

fig.savefig(os.path.join(HERE, 'batching.pdf'))
fig.savefig(os.path.join(HERE, 'batching.png'), dpi=300)
plt.close(fig)
