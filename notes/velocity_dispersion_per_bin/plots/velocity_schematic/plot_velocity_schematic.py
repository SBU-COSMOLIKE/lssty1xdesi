"""Redraw velocity_schematic.pdf from data.json: the didactic schematic
of the note's two central ideas.

Top panel (no axes): a halo, drawn as a dashed circle, drifts as a whole
with the large-scale flow (blue arrow, v_halo) while its central galaxy
sits near the center (black dot) and its satellites orbit inside it
(orange dots with orange arrows, v_int).  Blue is the linear-flow color
and orange the halo-part color of the note's other figures.

Bottom panel: the line of sight in comoving distance, with three 62 Mpc
bins marked by dashed edges.  A galaxy's velocity shifts its apparent
position by (1 + z) v / H(z): the example galaxy (orange filled dot,
true position) is displaced across the bin edge (open dot, apparent
position).  The blue Gaussian is the blur of width sigma_chi for the
whole bin, drawn to scale against the bin width at z = 0.5.
Run from any folder: python plot_velocity_schematic.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

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

BLUE = '#0072B2'    # the linear flow, as in lin_decomposition
ORANGE = '#E69F00'  # the halo part, as in lin_decomposition

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
width = data['bin_width_Mpc']
sigma_chi = data['sigma_chi_Mpc']
shift = data['example_shift_Mpc']


def check_clear(fig, texts, lines_xy):
    '''Raise when a text box touches one of the drawn lines (given as
    display-coordinate point arrays) or another text box.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    for box in boxes:
        padded = box.expanded(1.03, 1.08)
        for pts in lines_xy:
            if np.any((pts[:, 0] > padded.x0) & (pts[:, 0] < padded.x1)
                      & (pts[:, 1] > padded.y0) & (pts[:, 1] < padded.y1)):
                raise RuntimeError('velocity_schematic: a label touches a line')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                raise RuntimeError(
                    f'velocity_schematic: labels {i} and {j} overlap')


def sampled(ax, x, y, n=400):
    '''Display coordinates densely sampled along a polyline.'''
    x, y = np.asarray(x, float), np.asarray(y, float)
    t = np.linspace(0.0, 1.0, n)
    xs = np.interp(t*(len(x) - 1), np.arange(len(x)), x)
    ys = np.interp(t*(len(y) - 1), np.arange(len(y)), y)
    return ax.transData.transform(np.column_stack([xs, ys]))


fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(3.5, 3.5), gridspec_kw={'height_ratios': [1.0, 1.0]})
fig.subplots_adjust(hspace=0.12)
ax1.set_anchor('S')  # keep the equal-aspect sketch close to the lower panel
ax2.set_anchor('N')

# --- top: a halo and its galaxies -------------------------------------
ax1.set_xlim(-2.9, 2.9)
ax1.set_ylim(-1.65, 1.55)
ax1.set_aspect('equal')
ax1.axis('off')
halo = Circle((-0.6, 0.0), 1.08, fill=False, ls=(0, (4, 2)),
              lw=1.2, ec='black')
ax1.add_patch(halo)
# central galaxy
ax1.plot([-0.6], [0.0], marker='o', ms=5, color='black', ls='none')
# satellites with internal-velocity arrows
sats = [(-1.15, 0.45, -0.28, -0.30), (-0.05, 0.55, 0.26, 0.24),
        (-0.25, -0.62, 0.30, -0.18), (-1.25, -0.35, -0.10, 0.36)]
for x, y, dx, dy in sats:
    ax1.plot([x], [y], marker='o', ms=3.4, color=ORANGE, ls='none')
    ax1.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy),
                                  arrowstyle='-|>', mutation_scale=7,
                                  lw=1.1, color=ORANGE))
# the halo's bulk motion
ax1.add_patch(FancyArrowPatch((0.62, 0.0), (2.05, 0.0),
                              arrowstyle='-|>', mutation_scale=12,
                              lw=2.4, color=BLUE))
t_halo = ax1.text(1.33, 0.22, r'$v_{\rm halo}$', fontsize=9,
                  color=BLUE, ha='center', va='bottom')
t_flow = ax1.text(1.33, -0.26, 'large-scale flow', fontsize=7,
                  color=BLUE, ha='center', va='top')
t_cen = ax1.text(-0.6, -0.22, 'central', fontsize=7, ha='center',
                 va='top', color='black')
t_sat = ax1.text(0.45, 0.95, r'satellites ($v_{\rm int}$)', fontsize=7,
                 color=ORANGE, ha='left', va='center')
t_haloname = ax1.text(-2.05, 1.0, 'halo', fontsize=7, ha='center',
                      va='center', color='black')

# --- bottom: the blur across a thin bin's edge ------------------------
ax2.set_xlim(0.0, 3.0*width)
ax2.set_ylim(0.0, 1.0)
ax2.set_yticks([])
ax2.grid(False)
ax2.set_xlabel(r'comoving distance along the line of sight [Mpc]',
               fontsize=8)
ax2.tick_params(labelsize=7)
edge_pts = []
for k in range(4):
    ax2.axvline(k*width, color='black', ls=(0, (4, 2)), lw=1.0)
    edge_pts.append(sampled(ax2, [k*width, k*width], [0.0, 1.0]))
bin_labels = []
for k, name in enumerate([r'bin $i-1$', r'bin $i$', r'bin $i+1$']):
    bin_labels.append(ax2.text((k + 0.5)*width, 0.9, name, fontsize=7,
                               ha='center', va='center'))
# the blur Gaussian of the whole bin, width sigma_chi, drawn to scale
xg = np.linspace(0.0, 3.0*width, 600)
mu = 1.5*width
yg = 0.52*np.exp(-0.5*((xg - mu)/sigma_chi)**2)
ax2.plot(xg, yg, color=BLUE, lw=1.6)
gauss_pts = sampled(ax2, xg, yg)
t_sig = ax2.text(mu + 1.7*sigma_chi, 0.44, r'$\sigma_\chi$',
                 fontsize=8, color=BLUE, ha='left', va='center')
# one galaxy displaced across the edge
x_true = 2.0*width - 4.0
x_app = x_true + shift
ax2.plot([x_true], [0.2], marker='o', ms=4.5, color=ORANGE, ls='none')
ax2.plot([x_app], [0.2], marker='o', ms=4.5, mfc='none', mec=ORANGE,
         ls='none')
ax2.add_patch(FancyArrowPatch((x_true, 0.2), (x_app - 1.2, 0.2),
                              arrowstyle='-|>', mutation_scale=8,
                              lw=1.1, color=ORANGE))
t_shift = ax2.text(x_app + 3.0, 0.2, r'$(1+z)\,v/H$',
                   fontsize=7, color=ORANGE, ha='left', va='center')

texts = [t_halo, t_flow, t_cen, t_sat, t_haloname, t_sig, t_shift,
         *bin_labels]
check_clear(fig, texts, edge_pts + [gauss_pts])
fig.savefig(os.path.join(HERE, 'velocity_schematic.pdf'))
fig.savefig(os.path.join(HERE, 'velocity_schematic.png'), dpi=300)
plt.close(fig)
print('wrote velocity_schematic.pdf/.png')
