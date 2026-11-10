"""Write the data.json of every figure of notes/scale_cut_fiducial_dependence.

The scale-cut fiducial study computes the non-Limber UV-fraction scale cut
(arXiv:2603.13031) of every bin layout twice, with the call of run_cuts in
tests/studies/run_fisher_derivatives.py at its reference settings: on the
linear-bias example yaml (exploration/EXAMPLE_EVALUATE_LINEAR<n>.yaml, the
fiducial every Fisher result's cut is built on) and on the paper yaml
(EXAMPLE_EVALUATE<n>.yaml, the paper model at its fiducial point). It then
rescores the paper model with the model B notebook builder's own functions
(fit_system, paper_truths, truth_scores, honesty, the break rule) on the
linear notebook's cut cache and on the paper-fiducial cuts. Its scripts
(cuts_fid.py, compare_masks.py, score_fid.py) write three JSON files into
one results folder, which this script reads:

  compare_pkl_linear_vs_paper.json   kept band powers per layout and cut
  score_pkl_linear.json              the paper model on the linear-bias cuts
  score_paper.json                   the paper model on the paper-fiducial cuts

Usage: python make_scale_cut_fiducial_data.py <results folder>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUTS = [('dcom62', 'Delta d_com = 62 Mpc', r'$\Delta d_\mathrm{com} = 62$ Mpc', '#0072B2'),
           ('dz0p02', 'Delta z = 0.02', r'$\Delta z = 0.02$', '#E69F00'),
           ('dz0p05', 'Delta z = 0.05', r'$\Delta z = 0.05$', '#CC79A7'),
           ('dz0p20', 'Delta z = 0.2', r'$\Delta z = 0.2$', '#56B4E9')]
VARIANT = 'B tilts noA2 sv x6'
KMAX = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
PRODUCER = ('make_scale_cut_fiducial_data.py in notes/scale_cut_fiducial_dependence/plots/, '
            'from the scale-cut fiducial study (cuts_fid.py: run_cuts of '
            'tests/studies/run_fisher_derivatives.py at its reference settings on the paper '
            'yamls EXAMPLE_EVALUATE1-4.yaml; score_fid.py: the model B notebook builder\'s '
            'scoring functions)')


def write(name, data):
    folder = os.path.join(HERE, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.join(folder, 'data.json'))


def main(results):
    with open(os.path.join(results, 'compare_pkl_linear_vs_paper.json')) as f:
        compare = json.load(f)
    with open(os.path.join(results, 'score_pkl_linear.json')) as f:
        linear = json.load(f)['variants']
    with open(os.path.join(results, 'score_paper.json')) as f:
        paper = json.load(f)['variants']

    missing = [short for short, *_ in LAYOUTS if short not in compare]
    if missing:
        print('warning: no paper-fiducial cut yet for', missing, '(left out)')
    present = [entry for entry in LAYOUTS if entry[0] in compare]
    layouts = []
    for short, name, label, color in present:
        per = compare[short]['per_kmax']
        entry = {'name': name, 'label': label, 'color': color}
        for kind in ('autos', 'pairs'):
            entry[f'kept_{kind}_linear'] = [per[f'{k:g}'][f'kept_{kind}'][0] for k in KMAX]
            entry[f'kept_{kind}_paper'] = [per[f'{k:g}'][f'kept_{kind}'][1] for k in KMAX]
        layouts.append(entry)
    write('cut_kept_kmax', {
        'about': ('Band powers kept by the UV-fraction scale cut (arXiv:2603.13031; a band is '
                  'kept while |C_UV|/(|C_UV| + |C_IR|) < 0.05, C_IR from P(k) times a step at '
                  'k_max/D(zbar_i)), intersected with the shipped mask, split into clustering '
                  'autos and cross-bin pairs, with the cut computed on the linear-bias fiducial '
                  '(kept_*_linear: the linear notebook\'s cache, chains/EXAMPLE_EVALUATE_FISHER_'
                  'SCALE_CUTS.pkl, which run_cuts reproduces at its reference settings) and on '
                  'the paper model\'s fiducial (kept_*_paper: b2 and bs2 on their calibrations, '
                  'every other higher-order term zero). k_max [h/Mpc] is the z = 0 value of the '
                  'per-bin cut. Written by ' + PRODUCER + '.'),
        'kmax_h_per_Mpc': KMAX, 'layouts': layouts})

    layouts = []
    for short, name, label, color in present:
        entry = {'name': name, 'label': label, 'color': color}
        lin = linear[VARIANT][name]['per_cut']
        pap = paper[VARIANT][name]['per_cut']
        for field in ('fom', 'fom_om_as'):
            a = [None if lin[f'{k:g}'] is None else lin[f'{k:g}'][field] for k in KMAX]
            b = [None if pap[f'{k:g}'] is None else pap[f'{k:g}'][field] for k in KMAX]
            entry[f'{field}_linear_cut'] = a
            entry[f'{field}_paper_cut'] = b
            entry[f'{field}_ratio'] = [None if x is None or y is None else y/x
                                       for x, y in zip(a, b)]
        layouts.append(entry)
    write('fom_ratio_kmax', {
        'about': ('FoM(w0, wa) and FoM(Omega_m, 10^9 A_s) of the paper model (model B + tilts '
                  'without alpha_2, the seven sigma_v prior widths 6x the exploration ones: the '
                  'model B builder variant "' + VARIANT + '"), pessimistic linear-bias prior, DESI C_gg '
                  'alone, with the UV-fraction cut computed on the linear-bias fiducial '
                  '(*_linear_cut) and on the paper fiducial (*_paper_cut), and their ratio '
                  '(paper / linear). The Fisher matrices come from the model A derivative caches '
                  'and the model B templates (final derivative tier); only the kept rows differ. '
                  'k_max [h/Mpc] is the z = 0 value of the per-bin cut at k_max/D(zbar_i), '
                  'intersected with the shipped mask; None: no fit at that cut. Written by '
                  + PRODUCER + '.'),
        'kmax_h_per_Mpc': KMAX, 'variant': VARIANT, 'prior': 'pessimistic',
        'layouts': layouts})


if __name__ == '__main__':
    main(sys.argv[1])
