"""Write the data.json of the two pair-axes figures (pair_axes_gap and
pair_axes_sigv) from the results.json of the pair-axes study.

The study adds cross pairs to the current data vector along two axes, one
layout at a time, and repeats the Fisher forecast for each new pair list:

- gap: every pair whose raw n(z) ranges are at most g bin widths apart
  (g = 0 is the current list);
- sigma_v: every pair whose Fingers-of-God-blurred kernels overlap by
  O_ij >= 1e-6 when every sigma_v is set to s km/s. s only selects the
  pairs: the model keeps the fiducial sigma_v.

Each point is a row selection from one derivative base per layout, computed
on that layout's largest pair list. The study's panel scan reports the
figures of merit over those of the current list, for the paper model and the
reference model, with sigma_v fitted and with sigma_v held at the fiducial.

    python make_pair_axes_data.py <path to the study's results.json>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUTS = [  # results.json key, name, label, color (the project palette)
    ('dcom62', 'Delta d_com = 62 Mpc', r'$\Delta d_\mathrm{com} = 62$ Mpc', '#0072B2'),
    ('dz0p02', 'Delta z = 0.02', r'$\Delta z = 0.02$', '#E69F00'),
    ('dz0p05', 'Delta z = 0.05', r'$\Delta z = 0.05$', '#CC79A7'),
    ('dz0p20', 'Delta z = 0.2', r'$\Delta z = 0.2$', '#56B4E9'),
]
KMAX = ['0.075', '0.1', '0.2']
MODELS = ['paper', 'reference']
# where the fiducial point of the sigma_v axis sits: the fiducial dispersions
# are 289-371 km/s, one per sample group
X_FIDUCIAL_SIGV = 330.0

AXES = {
    'gap': {
        'results_key': 'axis_A_gap',
        'about_key': 'axis_A',
        'xlabel': r'$g$: pairs measured up to $g$ bin widths apart',
        'x_of': lambda key: 0.0 if key == 'today' else float(key.split('<=')[1]),
        'tick_label': lambda key: '0' if key == 'today' else key.split('<=')[1],
    },
    'sigv': {
        'results_key': 'axis_B_sigma_v',
        'about_key': 'axis_B',
        'xlabel': r'$\sigma_v$ that selects the pairs [km/s]',
        'x_of': lambda key: X_FIDUCIAL_SIGV if key == 'today' else float(key.split('=')[1]),
        'tick_label': lambda key: 'fid.' if key == 'today' else key.split('=')[1],
    },
}


def main(results_path):
    with open(results_path) as f:
        results = json.load(f)
    for axis, spec in AXES.items():
        layouts = []
        ticks = {}
        for key, name, label, color in LAYOUTS:
            lay = results['layouts'][key]
            if lay['status'] != 'computed':
                raise RuntimeError(f'{key}: {lay["status"]}')
            points = []
            for pkey, point in lay[spec['results_key']].items():
                if pkey == 'gap<=0':  # the same pair list as 'today'
                    continue
                x = spec['x_of'](pkey)
                ticks[x] = spec['tick_label'](pkey)
                entry = {'x': x, 'key': pkey, 'n_pairs': point['n_pairs']}
                for cut in KMAX:
                    entry[cut] = {
                        model: {
                            variant: {
                                'fom_ratio': point[cut][model][f'sigma_v {variant}']['fom_ratio'],
                                'fom_om_as_ratio':
                                    point[cut][model][f'sigma_v {variant}']['fom_om_as_ratio'],
                            } for variant in ('free', 'fixed')
                        } for model in MODELS
                    }
                points.append(entry)
            points.sort(key=lambda p: p['x'])
            layouts.append({'name': name, 'label': label, 'color': color,
                            'n_pairs_current': lay[spec['results_key']]['today']['n_pairs'],
                            'points': points})
        xs = sorted(ticks)
        data = {
            'about': (
                'Figures of merit over those of the current data vector when cross '
                'pairs are added along one axis (' + results[spec['about_key']] + '). '
                'Extracted by notes/cross_correlations/plots/make_pair_axes_data.py '
                'from the results.json of the pair-axes study (written '
                + results['written'] + '): one Fisher derivative base per layout on '
                'its largest pair list (final derivative tier), every point a row '
                'selection from it, checked bit for bit against the current caches '
                'on the shared rows. Models: '
                + '; '.join(f'{m}: {results["models"][m]}' for m in MODELS)
                + '. Variants: free = sigma_v fitted; fixed = '
                + results['models']['sigma_v fixed'] + '. k_max is the z = 0 value '
                'of a per-bin cut at k_max/D(zbar_i).'),
            'axis': axis,
            'xlabel': spec['xlabel'],
            'x_ticks': xs,
            'x_ticklabels': [ticks[x] for x in xs],
            'kmax_h_per_Mpc': KMAX,
            'models': MODELS,
            'layouts': layouts,
        }
        folder = os.path.join(HERE, f'pair_axes_{axis}')
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, 'data.json'), 'w') as f:
            json.dump(data, f, indent=1)
        print('wrote', os.path.join(folder, 'data.json'))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
