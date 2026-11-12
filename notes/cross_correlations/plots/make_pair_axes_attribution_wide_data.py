"""Write plots/pair_axes_attribution_wide/data.json: the attribution fits of
plots/pair_axes_attribution, measured against wide bins.

For each layout, gap-axis point g (every pair whose raw n(z) ranges are at
most g bin widths apart; g = 0 is the current list) and block of nuisance
parameters held at the fiducial (deleted from the Fisher matrix before the
inversion), the figure of merit of the fiducial model with sigma_v fitted
is divided by the same figure of merit of the Delta z = 0.2 layout on its
current list, with the same block held fixed and at the same k_max:

    R_block(layout, g) = FoM_block(layout, g) / FoM_block(Delta z = 0.2, current).

The ratio says how much more the thin bins measure than the wide bins under
each assumption about the nuisance parameters. k_max is the z = 0 value of
a per-bin cut at k_max/D(zbar_i).

    python make_pair_axes_attribution_wide_data.py <pair_axes_fits_dcom62.json> \
        <pair_axes_fits_dz0p02.json> <pair_axes_fits_dz0p05.json> \
        <pair_axes_fits_dz0p20.json>
(the per-layout files of scripts/compute_pair_axes_fits.py, in any order).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUTS = [  # results key, name, label
    ('dcom62', 'Delta d_com = 62 Mpc', r'$\Delta d_\mathrm{com} = 62$ Mpc'),
    ('dz0p02', 'Delta z = 0.02', r'$\Delta z = 0.02$'),
    ('dz0p05', 'Delta z = 0.05', r'$\Delta z = 0.05$'),
]
REFERENCE = 'dz0p20'
SV = 'sigma_v free'
BLOCKS = ['all free', 'b1 residuals fixed (level, slope free)', 'b2 bs2 fixed',
          'b1 level and slope fixed (residuals free)', 'all nuisance fixed']


def main(paths):
    fits = {}
    for path in paths:
        with open(path) as f:
            fit = json.load(f)
        fits[fit['tag']] = fit
    missing = [tag for tag in [t for t, *_ in LAYOUTS] + [REFERENCE] if tag not in fits]
    if missing:
        raise SystemExit(f'missing per-layout fits: {missing}')
    ref = fits[REFERENCE]['attribution']['cuts']
    cuts = list(ref)
    layouts = []
    for tag, name, label in LAYOUTS:
        att = fits[tag]['attribution']['cuts']
        points = []
        for point in att[cuts[0]]:
            if point == 'gap<=0':  # the same pair list as 'today'
                continue
            entry = {'g': 0 if point == 'today' else int(point.split('<=')[1])}
            for cut in cuts:
                entry[cut] = {}
                for block in BLOCKS:
                    num = att[cut][point][block][SV]
                    den = ref[cut]['today'][block][SV]
                    entry[cut][block] = {
                        'fom_ratio_to_wide': num['fom']/den['fom'],
                        'fom_om_as_ratio_to_wide': num['fom_om_as']/den['fom_om_as']}
            points.append(entry)
        points.sort(key=lambda p: p['g'])
        layouts.append({'name': name, 'label': label, 'points': points})
    data = {
        'about': (
            'FoM(w0, wa) and FoM(Omega_m, 10^9 A_s) of the fiducial model (the paper model, '
            'pessimistic linear-bias prior, sigma_v fitted) along the gap axis with blocks of '
            'nuisance parameters held at the fiducial (deleted from the Fisher matrix before '
            'the inversion), over the same figure of merit of the Delta z = 0.2 layout on its '
            'current pair list with the same block held fixed, at the same k_max. Written by '
            'notes/cross_correlations/plots/make_pair_axes_attribution_wide_data.py from the '
            'per-layout fits of scripts/compute_pair_axes_fits.py. k_max is the z = 0 value '
            'of a per-bin cut at k_max/D(zbar_i).'),
        'kmax_h_per_Mpc': cuts,
        'blocks': BLOCKS,
        'layouts': layouts,
    }
    folder = os.path.join(HERE, 'pair_axes_attribution_wide')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.join(folder, 'data.json'))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
