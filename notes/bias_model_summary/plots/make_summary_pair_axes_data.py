"""Write the data.json of the summary's two pair-axes figures from the
cross-correlations note's figure data (all in this repository):

- plots/summary_pair_axes/data.json: the paper model's FoM(w0, wa) and
  FoM(Omega_m, 10^9 A_s) over those of the current pair list, sigma_v
  fitted, at k_max = 0.075 and 0.2 h/Mpc, along the gap axis (pairs whose
  raw n(z) ranges are at most g bin widths apart) and the sigma_v axis
  (pairs whose blurred kernels overlap by O_ij >= 1e-6 with every
  dispersion set to s; s selects the pairs only);
- plots/summary_pair_attribution/data.json: the gap axis of the three
  layouts that gain, FoM(w0, wa) with sigma_v fitted, with every parameter
  free, with each group's bias line (the free level and slope of
  beta_i = b_1,i D(zbar_i)) held fixed, and with every nuisance parameter
  held fixed, at k_max = 0.075 and 0.2.

Inputs: ../../cross_correlations/plots/pair_axes_gap/data.json,
pair_axes_sigv/data.json and pair_axes_attribution/data.json.
    python make_summary_pair_axes_data.py      (from any folder)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, '..', '..', 'cross_correlations', 'plots')
CUTS = ['0.075', '0.2']
MODEL = 'paper'
SV = 'sigma_v free'
BLOCKS = {'all free': 'every parameter free',
          'b1 level and slope fixed (residuals free)': 'bias line fixed',
          'all nuisance fixed': 'every nuisance parameter fixed'}


def load(name):
    with open(os.path.join(SOURCE, name, 'data.json')) as f:
        return json.load(f)


def main():
    axes = {}
    for axis in ('gap', 'sigv'):
        src = load(f'pair_axes_{axis}')
        axes[axis] = {
            'xlabel': src['xlabel'], 'x_ticks': src['x_ticks'],
            'x_ticklabels': src['x_ticklabels'],
            'layouts': [{
                'name': lay['name'], 'label': lay['label'], 'color': lay['color'],
                'x': [p['x'] for p in lay['points']],
                **{cut: {key: [p[cut][MODEL]['free'][key] for p in lay['points']]
                         for key in ('fom_ratio', 'fom_om_as_ratio')} for cut in CUTS},
            } for lay in src['layouts']],
        }
    folder = os.path.join(HERE, 'summary_pair_axes')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump({
            'about': ('The paper model\'s FoM(w0, wa) and FoM(Omega_m, 10^9 A_s) with sigma_v '
                      'fitted, over the same figures of merit of the current pair list, along '
                      'the gap axis and the sigma_v axis of the pair-axes study, at k_max = '
                      '0.075 and 0.2 h/Mpc (z = 0 value of a per-bin cut at k_max/D(zbar_i)). '
                      'Copied by notes/bias_model_summary/plots/make_summary_pair_axes_data.py '
                      'from notes/cross_correlations/plots/pair_axes_gap and pair_axes_sigv.'),
            'kmax_h_per_Mpc': CUTS, 'axes': axes}, f, indent=1)

    src = load('pair_axes_attribution')
    folder = os.path.join(HERE, 'summary_pair_attribution')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump({
            'about': ('The paper model\'s FoM(w0, wa) with sigma_v fitted along the gap axis, '
                      'over the same fit on the current pair list, with every parameter free, '
                      'with each group\'s bias line (the free level and slope of beta_i = '
                      'b_1,i D(zbar_i)) held fixed and with every nuisance parameter held fixed '
                      '(their rows and columns deleted from the Fisher matrix before the '
                      'inversion), at k_max = 0.075 and 0.2 h/Mpc (z = 0 value of a per-bin cut '
                      'at k_max/D(zbar_i)). Copied by notes/bias_model_summary/plots/'
                      'make_summary_pair_axes_data.py from notes/cross_correlations/plots/'
                      'pair_axes_attribution.'),
            'kmax_h_per_Mpc': CUTS,
            'blocks': BLOCKS,
            'layouts': [{
                'name': lay['name'], 'label': lay['label'],
                'g': [p['g'] for p in lay['points']],
                **{cut: {block: [p[cut][block][SV]['fom_ratio'] for p in lay['points']]
                         for block in BLOCKS} for cut in CUTS},
            } for lay in src['layouts']]}, f, indent=1)
    print('wrote summary_pair_axes/data.json and summary_pair_attribution/data.json')


if __name__ == '__main__':
    main()
