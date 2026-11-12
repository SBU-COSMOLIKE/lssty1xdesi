"""Write plots/limber_real/data.json: the correlation coefficients of one
Delta z = 0.02 bin with its neighbour, the bin two apart and the bin three
apart, from the forecast's exact (non-Limber) spectra and from the Limber
approximation of the same model.

The bins are 56, 57, 58 and 59 (ELG1, zbar = 0.93 to 0.99). The exact
spectra are the fiducial vector of the Delta z = 0.02 pair-axes derivative
base (fisher_caches/model_derivatives_dz0p02_pairmax_final.npz, copied to
chains/ with fm.load_fisher_from_cache), which measures every pair up to
two bin widths apart. The Limber spectra are one evaluation of the same
model (EXAMPLE_EVALUATE2_PAIRMAX.yaml at the final derivative settings)
with adopt_limber_gg: 1, which computes every clustering spectrum, auto and
cross, in the Limber approximation. The correlation coefficient of bins i
and j at band ell is r_ell = C_ell^ij / sqrt(C_ell^ii C_ell^jj) of the
signal (no shot noise). The script also stores the blurred n(z) of the four
bins and the bands the UV-fraction cut keeps at k_max = 0.1 and 0.2 h/Mpc
(z = 0 values of the per-bin cut at k_max/D(zbar_i)).

Run from Cocoa/ with the cocoa environment (it evaluates CosmoLike once,
a few minutes):  python -I make_limber_real_data.py
"""
import json
import os
import sys

import numpy as np

ROOTDIR = os.environ['ROOTDIR']
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOTDIR + '/external_modules/code/cosmolike_core')
sys.path.insert(0, ROOTDIR + '/projects/lssty1xdesi/interface')
import cosmolike_lssty1xdesi_fisher as fm  # noqa: E402

PROJECT = 'projects/lssty1xdesi/'
DATA = ROOTDIR + '/external_modules/data/lssty1xdesi/'
STEM = 'lssty1xdesi_desi_dr1_dz0p02_pairmax'
BASE = 'model_derivatives_dz0p02_pairmax_final'
CUTS = 'model_scale_cuts_dz0p02_pairmax_final'
BIN = 56
OTHERS = [57, 58, 59]
NBINS = 90
COSMO = ['As_1e9', 'ns', 'H0', 'omegab', 'omegam', 'w0', 'wa']


def main():
    base = fm.read_fisher_cache(BASE, tier='final')
    m_exact = np.asarray(base['m0'], dtype=float)
    acc = fm.derivative_settings(STEM, 'final')
    overrides = {**acc['likelihood_overrides'], 'adopt_limber_gg': 1}
    model = fm.ModelFisher(PROJECT + 'EXAMPLE_EVALUATE2_PAIRMAX.yaml', COSMO,
                           likelihood_overrides=overrides, invcov_scale=1.0,
                           accuracy_boost=acc['accuracy_boost'],
                           camb_overrides=acc['camb_overrides'])
    m_limber = np.asarray(model.dv(), dtype=float)
    del model
    if m_limber.shape != m_exact.shape:
        raise RuntimeError(f'Limber vector {m_limber.shape} against exact {m_exact.shape}')
    pairs = [tuple(p) for p in np.loadtxt(DATA + STEM + '_pairs.txt', dtype=int)]
    spectra = [(b, b) for b in range(NBINS)] + pairs
    ell = np.loadtxt(DATA + 'lssty1xdesi_desi_dr3_ells.txt')
    nell = ell.size
    n_ee = m_exact.size - len(spectra)*nell

    def block(m, i, j):
        s = spectra.index((i, j))
        return m[n_ee + s*nell:n_ee + (s + 1)*nell]

    cuts = fm.read_fisher_cache(CUTS, tier='final')['Delta z = 0.02']
    kmax = np.asarray(cuts['kmax'], dtype=float).ravel()
    mask = np.asarray(cuts['mask'])
    curves = []
    for j in OTHERS:
        entry = {'bins': [BIN, j], 'apart': j - BIN}
        for label, m in (('exact', m_exact), ('limber', m_limber)):
            cij, cii, cjj = block(m, BIN, j), block(m, BIN, BIN), block(m, j, j)
            entry[f'r_{label}'] = (cij/np.sqrt(cii*cjj)).tolist()
        s = spectra.index((BIN, j))
        entry['kept'] = {f'{k:g}': [bool(v) for v in mask[s, :, int(np.argmin(np.abs(kmax - k)))]]
                         for k in (0.1, 0.2)}
        curves.append(entry)
    table = [line.split() for line in open(DATA + 'lssty1xdesi_desi_dr1_dz0p02_bins.txt')
             if not line.startswith('#')]
    zbar = {int(r[0]): float(r[4]) for r in table}
    data = {
        'about': (
            'Correlation coefficients r_ell = C^ij/sqrt(C^ii C^jj) of the Delta z = 0.02 bin '
            f'{BIN} (ELG1) with bins {OTHERS}: exact non-Limber spectra (the fiducial vector of '
            f'fisher_caches/{BASE}.npz) against one Limber evaluation of the same model '
            '(EXAMPLE_EVALUATE2_PAIRMAX.yaml, final derivative settings, adopt_limber_gg: 1); '
            'kept bands of the UV-fraction cut at k_max = 0.1 and 0.2 h/Mpc (z = 0 values of '
            'the per-bin cut at k_max/D(zbar_i)). Written by notes/cross_correlations/plots/'
            'make_limber_real_data.py.'),
        'ell': ell.tolist(), 'bin': BIN, 'zbar': {str(b): zbar[b] for b in [BIN] + OTHERS},
        'curves': curves,
    }
    folder = os.path.join(HERE, 'limber_real')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.join(folder, 'data.json'))
    for c in curves:
        print(c['bins'], 'exact', np.round(c['r_exact'][:8], 4), 'limber', np.round(c['r_limber'][:8], 4))


if __name__ == '__main__':
    main()
