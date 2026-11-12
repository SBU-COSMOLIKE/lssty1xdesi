"""Write plots/pair_axes_priors/data.json: how much the added cross pairs of
the gap axis raise the figures of merit of the paper model under each
linear-bias prior that passes the accuracy test of
notes/correlated_bias_prior_v2 (the growth-tied prior fails it and gets no
figure of merit).

The priors act on the per-bin amplitudes beta_i = b_1,i D(zbar_i):
- baseline: per sample group a free level and a free slope in redshift, plus
  per-bin residuals with a Gaussian prior of width 1% (BGS 3%) correlated as
  exp(-|z_i - z_j|/L), L = 0.2;
- pessimistic (the paper prior): the same with 3% in every group and L = 0.05;
- free: every beta_i free, no prior.
Every point is a row selection from the layout's largest derivative base (the
pair-axes study), sigma_v fitted and fixed, k_max = 0.075, 0.1, 0.2 h/Mpc
(z = 0 value of a per-bin cut at k_max/D(zbar_i)).

    python make_pair_axes_priors_data.py <prior_gain_dcom62.json> \
        <prior_gain_dz0p02.json> <prior_gain_dz0p05.json>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    'Delta d_com = 62 Mpc': (r'$\Delta d_\mathrm{com} = 62$ Mpc', '#0072B2'),
    'Delta z = 0.02': (r'$\Delta z = 0.02$', '#E69F00'),
    'Delta z = 0.05': (r'$\Delta z = 0.05$', '#CC79A7'),
}
PRIORS = ['baseline', 'pessimistic', 'free']


def main(paths):
    layouts = []
    cuts = None
    for path in paths:
        with open(path) as f:
            fit = json.load(f)
        label, color = LABELS[fit['layout']]
        cuts = list(fit['cuts'])
        points = {}
        for cut, entry in fit['cuts'].items():
            today = entry['today']
            for point, priors in entry.items():
                if point != 'today' and not point.startswith('gap<='):
                    continue
                if point == 'gap<=0':
                    continue
                g = 0 if point == 'today' else int(point.split('<=')[1])
                p = points.setdefault(g, {'g': g})
                p[cut] = {
                    prior: {sv: {'fom': priors[prior][sv]['fom'],
                                 'fom_om_as': priors[prior][sv]['fom_om_as'],
                                 'fom_ratio': priors[prior][sv]['fom']/today[prior][sv]['fom'],
                                 'fom_om_as_ratio':
                                     priors[prior][sv]['fom_om_as']/today[prior][sv]['fom_om_as']}
                            for sv in ('sigma_v free', 'sigma_v fixed')}
                    for prior in PRIORS}
        layouts.append({'name': fit['layout'], 'label': label, 'color': color,
                        'points': [points[g] for g in sorted(points)]})
    data = {
        'about': (
            'FoM(w0, wa) and FoM(Omega_m, 10^9 A_s) of the paper model along the gap '
            'axis under the baseline, pessimistic and free linear-bias priors, absolute '
            'and over the same prior\'s fit on the current pair list. The growth-tied '
            'prior fails the accuracy test of notes/correlated_bias_prior_v2 and is not '
            'included. Row selection from each layout\'s largest derivative base (the '
            'pair-axes study); the pessimistic ratios equal those of plots/pair_axes_gap. '
            'Written by notes/cross_correlations/plots/make_pair_axes_priors_data.py from '
            'the study\'s prior_gain_<layout>.json files. k_max is the z = 0 value of a '
            'per-bin cut at k_max/D(zbar_i).'),
        'kmax_h_per_Mpc': cuts,
        'priors': PRIORS,
        'layouts': layouts,
    }
    folder = os.path.join(HERE, 'pair_axes_priors')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.join(folder, 'data.json'))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
