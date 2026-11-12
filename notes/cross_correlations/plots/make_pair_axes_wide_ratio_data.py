"""Write plots/pair_axes_wide_ratio/data.json: what thin bins gain over wide
bins, and how that gain grows as cross pairs beyond the current list are
measured.

For each layout and each point of the gap axis (every pair whose raw n(z)
ranges are at most g bin widths apart; g = 0 is the current list), the
figure of merit of the fiducial model (the paper model, sigma_v fitted) is
divided by the same figure of merit of the Delta z = 0.2 layout on its
current list, at the same k_max:

    R(layout, g) = FoM(layout, g) / FoM(Delta z = 0.2, current list).

Delta z = 0.2, five bins of width 0.2 per sample group, is the wide-bin
reference. The FoMs are FoM(w0, wa) and FoM(Omega_m, 10^9 A_s); k_max is
the z = 0 value of a per-bin cut at k_max/D(zbar_i).

    python make_pair_axes_wide_ratio_data.py <pair_axes_fits_dcom62.json> \
        <pair_axes_fits_dz0p02.json> <pair_axes_fits_dz0p05.json> \
        <pair_axes_fits_dz0p20.json>
(the per-layout files of scripts/compute_pair_axes_fits.py, in any order).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUTS = [  # results key, name, label, color (the project palette)
    ('dcom62', 'Delta d_com = 62 Mpc', r'$\Delta d_\mathrm{com} = 62$ Mpc', '#0072B2'),
    ('dz0p02', 'Delta z = 0.02', r'$\Delta z = 0.02$', '#E69F00'),
    ('dz0p05', 'Delta z = 0.05', r'$\Delta z = 0.05$', '#CC79A7'),
    ('dz0p20', 'Delta z = 0.2', r'$\Delta z = 0.2$', '#56B4E9'),
]
REFERENCE = 'dz0p20'
MODEL = 'paper'
SV = 'sigma_v free'
KEYS = {'fom': 'fom_ratio_to_wide', 'fom_om_as': 'fom_om_as_ratio_to_wide'}


def main(paths):
    fits = {}
    for path in paths:
        with open(path) as f:
            fit = json.load(f)
        fits[fit['tag']] = fit
    missing = [tag for tag, *_ in LAYOUTS if tag not in fits]
    if missing:
        raise SystemExit(f'missing per-layout fits: {missing}')
    cuts = list(fits[REFERENCE]['panel']['cuts'])
    wide = {cut: fits[REFERENCE]['panel']['cuts'][cut]['today'][MODEL][SV] for cut in cuts}
    layouts = []
    for tag, name, label, color in LAYOUTS:
        panel = fits[tag]['panel']['cuts']
        points = []
        for point in panel[cuts[0]]:
            if point != 'today' and not point.startswith('gap<='):
                continue
            if point == 'gap<=0':  # the same pair list as 'today'
                continue
            entry = {'g': 0 if point == 'today' else int(point.split('<=')[1])}
            for cut in cuts:
                fom = panel[cut][point][MODEL][SV]
                entry[cut] = {
                    'fom': fom['fom'], 'fom_om_as': fom['fom_om_as'],
                    **{out: fom[key]/wide[cut][key] for key, out in KEYS.items()}}
            points.append(entry)
        points.sort(key=lambda p: p['g'])
        layouts.append({'name': name, 'label': label, 'color': color, 'points': points})
    data = {
        'about': (
            'FoM(w0, wa) and FoM(Omega_m, 10^9 A_s) of the fiducial model (the paper model: '
            + fits[REFERENCE]['panel']['models'][MODEL] + ', pessimistic linear-bias prior, '
            'sigma_v fitted) along the gap axis of each layout, over the same figure of merit '
            'of the Delta z = 0.2 layout on its current pair list at the same k_max (the '
            'wide-bin reference). Written by notes/cross_correlations/plots/'
            'make_pair_axes_wide_ratio_data.py from the per-layout fits of '
            'scripts/compute_pair_axes_fits.py. k_max is the z = 0 value of a per-bin cut at '
            'k_max/D(zbar_i).'),
        'kmax_h_per_Mpc': cuts,
        'reference_fom': {cut: {'fom': wide[cut]['fom'], 'fom_om_as': wide[cut]['fom_om_as']}
                          for cut in cuts},
        'layouts': layouts,
    }
    folder = os.path.join(HERE, 'pair_axes_wide_ratio')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.join(folder, 'data.json'))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
