"""Redraw fom_convergence.pdf from data.json: how the figures of merit of
the paper model converge with the chi grid of the non-Limber integrals,
and whether they then depend on the stencil step of the derivatives.

The paper model is model B + tilts without alpha_2, every sigma_v prior
width 6 times the exploration ones, on the pessimistic linear-bias prior;
the Fisher systems are scored on the UV-fraction cuts of the paper
fiducial.
Each FoM is divided by its converged value, the five-point rule at step h
(h = 0.02, relative) on the finest grid of its layout. The x axis is the
combined chi-grid factor accuracyboost x nonlimber_accuracyboost (the
grid has 511 times that many intervals); 32 is the MCMC grid of
Delta z = 0.05.

Four stacked panels, one per FoM and cut, on one shared y range. On
Delta z = 0.05 the line joins the five-point FoM at step h, and the
markers beside each grid point give the steps h/2, 2h and 3h; the other
controls stay at the reference's values. Two more points use other
settings: the five-point rule with every control at its MCMC value (grid
32) and DerivKit at the MCMC settings (grid 32) and at the final settings
(grid 256). The spot check of Delta d_com = 62 Mpc, at the final
settings, is drawn the same way in its own color. A gray band marks 1
+- 1%. data.json holds the study's results (tests/studies/run_fisher_derivatives.py)
rescored for the paper model on the paper-fiducial cuts with the arithmetic
of tests/studies/generate_fisher_derivatives_figures.py
(rescore_convergence.py of the notebook-builder backup folder).
Run from any folder: python plot_fom_convergence.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
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
# the layouts: color, line width and the marker of step h from the
# project's figure rules (paper-figures skill)
# The spot check of 62 Mpc sits at grid 256 only, on top of Delta z = 0.05's
# points there: it is drawn as open rings (no face), so the filled markers
# under it stay visible.
LAYOUT_STYLE = {'Delta z = 0.05': {'color': '#CC79A7', 'lw': 1.3, 'marker': 'D',
                                   'label': r'$\Delta z = 0.05$', 'open': False},
                'Delta d_com = 62 Mpc': {'color': '#0072B2', 'lw': 2.4, 'marker': 'o',
                                         'label': r'$\Delta d_\mathrm{com} = 62$ Mpc',
                                         'open': True}}
OPEN_SIZE = 5.6
OPEN_EDGE = 1.0
# the steps other than h: marker and horizontal offset from the grid point,
# in octaves of the chi grid (the points of one grid sit side by side)
STEP_STYLE = {'0.5': {'marker': 'v', 'offset': -0.13, 'label': r'$h/2$'},
              '2': {'marker': '^', 'offset': 0.13, 'label': r'$2h$'},
              '3': {'marker': 's', 'offset': 0.26, 'label': r'$3h$'}}
# the points at other settings (Delta z = 0.05): the five-point rule at
# the MCMC settings, and DerivKit, left of their grid point
POINT_STYLE = {'mcmc': {'marker': 'P', 'offset': -0.15, 'ms': 5.0},
               'derivkit_mcmc': {'marker': '*', 'offset': -0.32, 'ms': 6.5},
               'derivkit_final': {'marker': '*', 'offset': -0.30, 'ms': 6.5}}
MARKER_SIZE = 3.6
PANEL_LABELS = {'fom_w0wa': r'$(w_0, w_a)$', 'fom_om_as': r'$(\Omega_\mathrm{m}, A_\mathrm{s})$'}
BAND = 0.01  # the 1% band around 1 (no legend entry: the caption names it)
# candidate y ticks, labelled only away from the panel edges
TICK_CANDIDATES = [0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
# the y headroom above the highest point, grown until every label clears
HEADROOM = [0.06, 0.10, 0.16, 0.24]
LEGEND_GAP = 0.02


def as_array(values):
    '''A list with None (not computed) as a float array with NaN.'''
    out = []
    for v in values:
        out.append(np.nan if v is None else float(v))
    return np.array(out)


def shifted(grid, offset):
    '''The x position of a point offset by some octaves from its grid.'''
    return np.asarray(grid, dtype=float)*2.0**offset


def line_points(ax):
    '''Display coordinates of points densely sampled along every drawn line
    and marker of ax, inside the axes only. Segments are sampled in display
    space, where a segment of a log axis is straight.'''
    box = ax.get_window_extent()
    pts = [np.zeros((0, 2))]
    t = np.linspace(0.0, 1.0, 60)[:, None]
    for line in ax.get_lines():
        xy = np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()])
        display_xy = line.get_transform().transform(xy)
        finite = np.all(np.isfinite(display_xy), axis=1)
        pts.append(display_xy[finite])
        if line.get_linestyle() in ('None', 'none', ' ', ''):
            continue
        for i in range(len(display_xy) - 1):
            if finite[i] and finite[i + 1]:
                pts.append(display_xy[i] + t*(display_xy[i + 1] - display_xy[i]))
    pts = np.vstack(pts)
    inside = ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
              & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))
    return pts[inside]


def touches(box, axes):
    '''True when a display box, with a margin of 3 points, holds a line point.'''
    pad = 3.0*axes[0].figure.dpi/72.0
    for ax in axes:
        pts = line_points(ax)
        hit = ((pts[:, 0] > box.x0 - pad) & (pts[:, 0] < box.x1 + pad)
               & (pts[:, 1] > box.y0 - pad) & (pts[:, 1] < box.y1 + pad))
        if np.any(hit):
            return True
    return False


def check_clear(fig, axes, artists):
    '''Raise when a legend, a label or a tick label touches a drawn line or
    marker, or when two of them overlap.'''
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for name, artist in artists:
        box = artist.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        if touches(box, axes):
            raise RuntimeError(f'fom_convergence: {name} touches a line')
        boxes.append((name, box))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                raise RuntimeError(f'fom_convergence: {boxes[i][0]} overlaps {boxes[j][0]}')


def plain_number(value, position):
    '''A tick label as a plain number: 1, 1.2, 32 (position is unused).'''
    return f'{value:g}'


def marker_style(st):
    '''The marker keywords of one layout: filled, or an open ring on top.'''
    if st['open']:
        return {'ms': OPEN_SIZE, 'mfc': 'none', 'mec': st['color'], 'mew': OPEN_EDGE,
                'zorder': 5}
    return {'ms': MARKER_SIZE, 'mfc': st['color'], 'mec': st['color'], 'zorder': 3}


def draw_layout(ax, lay, fom, cut):
    '''One layout's line at step h (no line through a single grid) and its
    markers at the other steps.'''
    st = LAYOUT_STYLE[lay['name']]
    ratio = lay['ratio'][fom][cut]
    y = as_array(ratio['1'])
    line = '-' if np.sum(np.isfinite(y)) > 1 else 'none'
    ax.plot(lay['grid'], y, color=st['color'], lw=st['lw'], ls=line, marker=st['marker'],
            **marker_style(st))
    for step, sst in STEP_STYLE.items():
        y = as_array(ratio[step])
        if np.all(np.isnan(y)):
            continue
        ax.plot(shifted(lay['grid'], sst['offset']), y, ls='none', marker=sst['marker'],
                color=st['color'], **marker_style(st))


with open(os.path.join(HERE, 'data.json')) as f:
    data = json.load(f)
foms = data['foms']
cuts = [f'{k:g}' for k in data['kmax_h_per_Mpc']]
layouts = data['layouts']
# the points drawn (data.json also holds the final settings' five-point FoM,
# which the caption quotes: at grid 256 it sits on the line)
points = {}
for name, point in data['points'].items():
    if name not in POINT_STYLE:
        continue
    if any(v is not None for fom in point['ratio'].values() for v in fom.values()):
        points[name] = point
values = [1.0 - BAND, 1.0 + BAND]
grids = set()
for lay in layouts:
    grids.update(lay['grid'])
    for fom in foms:
        for cut in cuts:
            for step in lay['ratio'][fom][cut].values():
                values += [float(v) for v in step if v is not None]
for point in points.values():
    for fom in foms:
        for cut in cuts:
            if point['ratio'][fom][cut] is not None:
                values.append(float(point['ratio'][fom][cut]))
grids = sorted(grids)
low = min(values)
high = max(values)
span = high - low

for headroom in HEADROOM:
    # one column of a two-column article: four wide, short panels with a
    # thin gap and one shared y range
    panels = [(fom, cut) for fom in foms for cut in cuts]
    fig, axes = plt.subplots(len(panels), 1, figsize=(3.5, 1.0*len(panels)), sharex=True,
                             sharey=True, gridspec_kw={'hspace': 0.06})
    for ax, (fom, cut) in zip(axes, panels):
        # the reference band 1 +- 1% carries no legend entry: the caption names it
        ax.axhspan(1.0 - BAND, 1.0 + BAND, color='0.85', lw=0, zorder=1)
        # the spot check under the main layout, so the thinner line stays visible
        for lay in reversed(layouts):
            draw_layout(ax, lay, fom, cut)
        main = LAYOUT_STYLE['Delta z = 0.05']
        for name, point in points.items():
            pst = POINT_STYLE[name]
            value = point['ratio'][fom][cut]
            if value is None:
                continue
            filled = name != 'derivkit_mcmc'
            ax.plot([shifted(point['grid'], pst['offset'])], [value], ls='none',
                    marker=pst['marker'], ms=pst['ms'], color=main['color'],
                    mfc=main['color'] if filled else 'white', mec=main['color'], mew=0.9,
                    zorder=4)
        ax.set_xscale('log', base=2)
        bottom = low - 0.10*span
        ax.set_ylim(bottom, high + headroom*span)
        ticks = []
        for tick in TICK_CANDIDATES:
            if bottom + 0.04*span < tick < high + headroom*span - 0.04*span:
                ticks.append(tick)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(False, axis='x')
        # the FoM and the cut on the right side, rotated like the y label
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(PANEL_LABELS[fom] + '\n' + rf'$k_\mathrm{{max}} = {cut}$', rotation=90,
                      fontsize=8, labelpad=6)
    axes[-1].set_xlim(grids[0]*2.0**-0.55, grids[-1]*2.0**0.45)
    axes[-1].xaxis.set_major_locator(FixedLocator(grids))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(plain_number))
    axes[-1].xaxis.set_minor_locator(NullLocator())
    axes[-1].set_xlabel(r'$\chi$-grid factor', fontsize=10)
    fig.supylabel('FoM / converged FoM', x=-0.02, fontsize=10)

    # legends above the top panel, stacked close together: the layouts and
    # steps first, the other settings above them
    layout_handles = []
    for lay in layouts:
        st = LAYOUT_STYLE[lay['name']]
        drawn_line = any(np.sum(np.isfinite(as_array(lay['ratio'][fom][cut]['1']))) > 1
                         for fom in foms for cut in cuts)
        layout_handles.append(Line2D([], [], color=st['color'], lw=st['lw'],
                                     ls='-' if drawn_line else 'none', marker=st['marker'],
                                     label=st['label'] + r', step $h$', **marker_style(st)))
    step_handles = []
    step_labels = []
    for step, sst in STEP_STYLE.items():
        drawn = []
        for lay in layouts:
            if any(v is not None for fom in foms for cut in cuts
                   for v in lay['ratio'][fom][cut][step]):
                lst = LAYOUT_STYLE[lay['name']]
                drawn.append(Line2D([], [], ls='none', marker=sst['marker'], color=lst['color'],
                                    **marker_style(lst)))
        if drawn:
            step_handles.append(tuple(drawn))
            step_labels.append(sst['label'])
    legend_layouts = axes[0].legend(handles=layout_handles, loc='lower center',
                                    bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                                    fontsize=7, handlelength=2.2, columnspacing=1.0,
                                    handletextpad=0.4, borderaxespad=0.2, borderpad=0.1)
    axes[0].add_artist(legend_layouts)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top = axes[0].transAxes.inverted().transform(
        (0.0, legend_layouts.get_window_extent(renderer).y1))[1]
    legend_steps = axes[0].legend(handles=step_handles, labels=step_labels, loc='lower center',
                                  bbox_to_anchor=(0.5, top + LEGEND_GAP), ncol=len(step_handles),
                                  frameon=False, fontsize=7, columnspacing=1.4,
                                  handletextpad=0.3, borderaxespad=0.0, borderpad=0.1,
                                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
    axes[0].add_artist(legend_steps)
    legends = [('layout legend', legend_layouts), ('step legend', legend_steps)]
    point_handles = []
    main = LAYOUT_STYLE['Delta z = 0.05']
    if 'mcmc' in points:
        point_handles.append(Line2D([], [], ls='none', marker='P', ms=5.0, color=main['color'],
                                    label='five-point, MCMC settings'))
    dkit = []
    for name in ('derivkit_mcmc', 'derivkit_final'):
        if name in points:
            filled = name != 'derivkit_mcmc'
            dkit.append(Line2D([], [], ls='none', marker='*', ms=6.5, color=main['color'],
                               mfc=main['color'] if filled else 'white', mec=main['color'],
                               mew=0.9))
    point_labels = [h.get_label() for h in point_handles]
    if dkit:
        point_handles.append(tuple(dkit))
        point_labels.append('DerivKit, MCMC / final settings' if len(dkit) == 2 else 'DerivKit')
    if point_handles:
        fig.canvas.draw()
        top = axes[0].transAxes.inverted().transform(
            (0.0, legend_steps.get_window_extent(fig.canvas.get_renderer()).y1))[1]
        legend_points = axes[0].legend(handles=point_handles, labels=point_labels,
                                       loc='lower center', bbox_to_anchor=(0.5, top + LEGEND_GAP),
                                       ncol=2, frameon=False, fontsize=7, columnspacing=1.2,
                                       handletextpad=0.3, borderaxespad=0.0, borderpad=0.1,
                                       handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})
        legends.append(('settings legend', legend_points))
    artists = list(legends) + [('x label', axes[-1].xaxis.label)]
    for k, ax in enumerate(axes):
        artists.append((f'panel {k} label', ax.yaxis.label))
        for t, label in enumerate(ax.get_yticklabels()):
            if label.get_visible() and label.get_text():
                artists.append((f'panel {k} y tick {t}', label))
    for t, label in enumerate(axes[-1].get_xticklabels()):
        artists.append((f'x tick {t}', label))
    try:
        check_clear(fig, list(axes), artists)
    except RuntimeError as error:
        plt.close(fig)
        last_error = error
        continue
    break
else:
    raise RuntimeError(f'fom_convergence: no headroom in {HEADROOM} clears the labels '
                       f'({last_error})')
fig.savefig(os.path.join(HERE, 'fom_convergence.pdf'))
fig.savefig(os.path.join(HERE, 'fom_convergence.png'), dpi=300)
plt.close(fig)
