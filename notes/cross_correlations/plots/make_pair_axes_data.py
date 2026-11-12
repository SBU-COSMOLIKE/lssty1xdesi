"""Write the data.json of the two pair-axes figures (pair_axes_gap and
pair_axes_sigv) from the per-layout fits of scripts/compute_pair_axes_fits.py.

The study adds cross pairs to the current data vector along two axes, one
layout at a time, and repeats the Fisher forecast for each new pair list:

- gap: every pair whose raw n(z) ranges are at most g bin widths apart
  (g = 0 is the current list);
- sigma_v: every pair whose Fingers-of-God-blurred kernels overlap by
  O_ij >= 1e-6 when every sigma_v is set to s km/s. s only selects the
  pairs: the model keeps the fiducial sigma_v.

Each point is a row selection from one derivative base per layout, computed
on that layout's largest pair list. The fits give the figures of merit of
every point for the paper model and the reference model, with sigma_v fitted
and with sigma_v held at the fiducial; this script divides each by the same
fit on the current list.

    python make_pair_axes_data.py <pair_axes_fits_dcom62.json> \
        <pair_axes_fits_dz0p02.json> <pair_axes_fits_dz0p05.json> \
        <pair_axes_fits_dz0p20.json>

The four files are the chains/pair_axes_fits_<layout>.json that
scripts/compute_pair_axes_fits.py writes, in any order.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUTS = [  # the layout tag of the fits file, name, label, color (the project palette)
    ('dcom62', 'Delta d_com = 62 Mpc', r'$\Delta d_\mathrm{com} = 62$ Mpc', '#0072B2'),
    ('dz0p02', 'Delta z = 0.02', r'$\Delta z = 0.02$', '#E69F00'),
    ('dz0p05', 'Delta z = 0.05', r'$\Delta z = 0.05$', '#CC79A7'),
    ('dz0p20', 'Delta z = 0.2', r'$\Delta z = 0.2$', '#56B4E9'),
]
KMAX = ['0.075', '0.1', '0.2']
MODELS = ['paper', 'reference']
# the notebook variant of each model, which the about text describes; the
# fits file must name the same ones
MODEL_VARIANTS = {'paper': 'B tilts noA2 sv x6', 'reference': 'B tilts'}
MODEL_ABOUT = {
    'paper': 'B tilts noA2 sv x6 on the pessimistic prior (alpha_2 fixed at 0, '
             'sigma_v widths x6)',
    'reference': 'B tilts on the pessimistic prior (alpha_2 free, exploration sigma_v '
                 'widths): the model of the pair_threshold figure',
}
# where the fiducial point of the sigma_v axis sits: the fiducial dispersions
# are 289-371 km/s, one per sample group
X_FIDUCIAL_SIGV = 330.0

AXES = {
    'gap': {
        'prefix': 'gap<=',
        'about': "gap<=g: the current pairs + every pair whose raw n(z) ranges are at "
                 "most g bin widths apart",
        'xlabel': r'$g$: pairs measured up to $g$ bin widths apart',
        'x_of': lambda key: 0.0 if key == 'today' else float(key.split('<=')[1]),
        'tick_label': lambda key: '0' if key == 'today' else key.split('<=')[1],
    },
    'sigv': {
        'prefix': 'sigv=',
        'about': "sigv=s: the current pairs + every pair with blurred-kernel overlap "
                 "O_ij >= 1e-6 when every sigma_v = s km/s (selection only; the model "
                 "stays at the fiducial sigma_v)",
        'xlabel': r'$\sigma_v$ that selects the pairs [km/s]',
        'x_of': lambda key: X_FIDUCIAL_SIGV if key == 'today' else float(key.split('=')[1]),
        'tick_label': lambda key: 'fid.' if key == 'today' else key.split('=')[1],
    },
}


def ratio(value, reference):
    """Return value/reference, or None when either FoM is None (no fit)."""
    if value is None or reference is None:
        return None
    return value/reference


def read_fits(paths):
    """Read the four fits files and return them by layout tag.

    Arguments:
      paths = the paths of the four chains/pair_axes_fits_<layout>.json files.

    Returns:
      dict {tag: the file's content}.

    Raises:
      RuntimeError when a layout is missing or given twice, or when a file
      names other model variants than MODEL_VARIANTS.
    """
    fits = {}
    for path in paths:
        with open(path) as f:
            content = json.load(f)
        tag = content['tag']
        if tag in fits:
            raise RuntimeError(f'{path}: the layout {tag} is given twice')
        if content['panel']['models'] != MODEL_VARIANTS:
            raise RuntimeError(f'{path} fits the models {content["panel"]["models"]}; '
                               f'the about text describes {MODEL_VARIANTS}')
        fits[tag] = content
    for tag, name, label, color in LAYOUTS:
        if tag not in fits:
            raise RuntimeError(f'no fits file of the layout {tag} ({name}) among {paths}')
    return fits


def main(paths):
    """Write pair_axes_gap/data.json and pair_axes_sigv/data.json.

    Arguments:
      paths = the four fits files (read_fits).

    Side effects: writes the two data.json files next to this script.
    """
    fits = read_fits(paths)
    for axis, spec in AXES.items():
        layouts = []
        ticks = {}
        for key, name, label, color in LAYOUTS:
            lay = fits[key]
            cuts = lay['panel']['cuts']
            points = []
            for pkey, point in lay['points'].items():
                if pkey != 'today' and not pkey.startswith(spec['prefix']):
                    continue
                if pkey == 'gap<=0':  # the same pair list as 'today'
                    continue
                x = spec['x_of'](pkey)
                ticks[x] = spec['tick_label'](pkey)
                entry = {'x': x, 'key': pkey, 'n_pairs': point['n_pairs']}
                for cut in KMAX:
                    entry[cut] = {}
                    for model in MODELS:
                        entry[cut][model] = {}
                        for variant in ('free', 'fixed'):
                            fit = cuts[cut][pkey][model][f'sigma_v {variant}']
                            today = cuts[cut]['today'][model][f'sigma_v {variant}']
                            entry[cut][model][variant] = {
                                'fom_ratio': ratio(fit['fom'], today['fom']),
                                'fom_om_as_ratio': ratio(fit['fom_om_as'],
                                                         today['fom_om_as']),
                            }
                points.append(entry)
            points.sort(key=lambda p: p['x'])
            layouts.append({'name': name, 'label': label, 'color': color,
                            'n_pairs_current': lay['points']['today']['n_pairs'],
                            'points': points})
        xs = sorted(ticks)
        data = {
            'about': (
                'Figures of merit over those of the current data vector when cross '
                'pairs are added along one axis (' + spec['about'] + '). '
                'Extracted by notes/cross_correlations/plots/make_pair_axes_data.py '
                'from the per-layout fits of scripts/compute_pair_axes_fits.py: one '
                'Fisher derivative base per layout on its largest pair list (the '
                'archived final-tier caches model_*_<layout>_pairmax_final), every '
                'point a row selection from it. Models: '
                + '; '.join(f'{m}: {MODEL_ABOUT[m]}' for m in MODELS)
                + '. Variants: free = sigma_v fitted; fixed = the same model with '
                'every sigma_v column removed (held at the fiducial). k_max is the '
                'z = 0 value of a per-bin cut at k_max/D(zbar_i).'),
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
    if len(sys.argv) != 1 + len(LAYOUTS):
        raise SystemExit(__doc__)
    main(sys.argv[1:])
