"""Write plots/pair_axes_attribution/data.json from the attribution fits of
the pair-axes study (one JSON per layout, written by its bias_attribution.py).

The fits repeat the gap axis of plots/pair_axes_gap with blocks of nuisance
parameters held at their fiducial values. Holding a block fixed deletes its
rows and columns from the full Fisher matrix (data plus every prior,
including the correlated prior of the 90 per-bin linear biases) before the
inversion, which for a Gaussian likelihood is exact conditioning. Blocks:

- b1 per bin: the linear bias of every bin. Per sample group the bin biases are
  a free level and a free slope in redshift plus residuals with a correlated
  Gaussian prior; the two split blocks hold only the residuals, or only the
  level and slope, fixed (b = U a + E e per group, U = (1, z - zbar), E the
  residual directions, U^T Q e = 0 with Q the prior's correlated precision);
- b2, bs2: the second-order and tidal biases of each sample and their
  redshift tilts;
- a0: the counterterm of each sample and its tilt;
- shot noise: the two shot-noise parameters of each sample;
- all nuisance: every block above at once (cosmology and sigma_v stay free).

    python make_pair_axes_attribution_data.py <attribution_dcom62.json> \
        <attribution_dz0p02.json> <attribution_dz0p05.json>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {  # layout key -> (name, label, color)
    'Delta d_com = 62 Mpc': ('Delta d_com = 62 Mpc', r'$\Delta d_\mathrm{com} = 62$ Mpc', '#0072B2'),
    'Delta z = 0.02': ('Delta z = 0.02', r'$\Delta z = 0.02$', '#E69F00'),
    'Delta z = 0.05': ('Delta z = 0.05', r'$\Delta z = 0.05$', '#CC79A7'),
}
BLOCKS = ['all free', 'b1 per bin fixed', 'b1 residuals fixed (level, slope free)',
          'b1 level and slope fixed (residuals free)', 'b2 bs2 fixed', 'a0 fixed',
          'shot noise fixed', 'all nuisance fixed']


def main(paths):
    layouts = []
    cuts = None
    for path in paths:
        with open(path) as f:
            fit = json.load(f)
        name, label, color = LABELS[fit['layout']]
        cuts = list(fit['cuts'])
        points = {}
        for cut, entry in fit['cuts'].items():
            today = entry['today']
            for point, blocks in entry.items():
                if point == 'gap<=0':  # the same pair list as 'today'
                    continue
                g = 0 if point == 'today' else int(point.split('<=')[1])
                p = points.setdefault(g, {'g': g})
                p[cut] = {
                    block: {
                        sv: {
                            'fom_ratio': blocks[block][sv]['fom']/today[block][sv]['fom'],
                            'fom_om_as_ratio':
                                blocks[block][sv]['fom_om_as']/today[block][sv]['fom_om_as'],
                        } for sv in ('sigma_v free', 'sigma_v fixed')
                    } for block in BLOCKS
                }
                if point == 'today':
                    p[cut]['cost_of_fitting_sigma_v'] = {
                        block: today[block]['sigma_v fixed']['fom']/today[block]['sigma_v free']['fom']
                        for block in BLOCKS}
        layouts.append({'name': name, 'label': label, 'color': color,
                        'points': [points[g] for g in sorted(points)]})
    data = {
        'about': (
            'FoM(w0, wa) and FoM(Omega_m, 10^9 A_s) of the paper model (alpha_2 fixed '
            'at 0, sigma_v prior widths x6, pessimistic linear-bias prior) along the '
            'gap axis, over the same fit on the current pair list, with blocks of '
            'nuisance parameters held at the fiducial: ' + ', '.join(BLOCKS) + '. '
            'Holding a block fixed deletes its rows and columns from the full Fisher '
            'matrix (data plus every prior) before the inversion. Row selection from '
            'each layout\'s largest derivative base (the pair-axes study); the "all '
            'free" ratios equal those of plots/pair_axes_gap. Written by '
            'notes/cross_correlations/plots/make_pair_axes_attribution_data.py from '
            'the study\'s attribution_<layout>.json files. k_max is the z = 0 value of '
            'a per-bin cut at k_max/D(zbar_i).'),
        'kmax_h_per_Mpc': cuts,
        'blocks': BLOCKS,
        'layouts': layouts,
    }
    folder = os.path.join(HERE, 'pair_axes_attribution')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.join(folder, 'data.json'))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
