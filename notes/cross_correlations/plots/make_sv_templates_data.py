"""Write plots/sv_templates/data.json: how three Delta z = 0.02 spectra of
one ELG bin respond to the velocity dispersion sigma_v, to the linear bias
b_1 and to w0, band by band, in units of each band's error.

For a parameter p, its template on a band is sigma_ell^-1 dC_ell/d p, with
sigma_ell = sqrt(Cov_ll) the band's error from the diagonal of the Gaussian
covariance. sigma_v and b_1 are taken logarithmically (d/d ln p = p d/dp,
every dispersion and every sample's b_1 moved together, which for spectra of
one sample is that sample's own parameter); w0 is taken as is.

The templates of all 90 auto spectra on the bands the scale cut keeps at
k_max = 0.1 h/Mpc define how alike two parameters act there: the cosine
between the sigma_v template and the b_1 (or w0) template, and the
coefficients c of the best match of the sigma_v template by
c_b1 * (b_1 template) + c_w0 * (w0 template), fitted over those bands. The
figure applies these auto-spectrum coefficients to the three spectra of one
bin: its auto spectrum, its cross spectrum with the next bin (in the current
list) and with the bin two apart (added at gap g = 1).

Inputs (all written by the pair-axes study): the Delta z = 0.02 derivative
base on the largest pair list (chains/model_derivatives_dz0p02_pairmax_final.npz),
its scale cuts (chains/model_scale_cuts_dz0p02_pairmax_final.npz), and the
dataset files (mask, pair list, bin table, covariance).

    python make_sv_templates_data.py      (from any folder)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRJ = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA = os.path.abspath(os.path.join(PRJ, '..', '..', 'external_modules', 'data', 'lssty1xdesi'))
STEM = 'lssty1xdesi_desi_dr1_dz0p02_pairmax'
BIN = 56          # an ELG1 bin whose three spectra all keep bands at the cut
KMAX = 0.1        # h/Mpc, z = 0 value of the per-bin cut at k_max/D(zbar_i)
NBINS = 90


def load_npz(name):
    z = np.load(os.path.join(PRJ, 'chains', name), allow_pickle=False)
    return z, json.loads(str(z['__header__']))


def main():
    base, header = load_npz('model_derivatives_dz0p02_pairmax_final.npz')
    names = header['parameter_names']['result/names']
    fid = base['result/fiducial']
    deriv = base['result/D']
    cuts, _ = load_npz('model_scale_cuts_dz0p02_pairmax_final.npz')
    key = 'result/Delta z = 0.02/'
    kmax = cuts[key + 'kmax'][:, 0]
    ell = cuts[key + 'ell'][:, 0]
    m = int(np.argmin(np.abs(kmax - KMAX)))
    uv_mask = cuts[key + 'mask'][:, :, m] > 0.5
    nell = ell.size

    pairs = [tuple(p) for p in np.loadtxt(os.path.join(DATA, STEM + '_pairs.txt'), dtype=int)]
    spectra = [(b, b) for b in range(NBINS)] + pairs
    table = [line.split() for line in open(os.path.join(DATA, 'lssty1xdesi_desi_dr1_dz0p02_bins.txt'))
             if not line.startswith('#')]
    chunk = [row[3] for row in table]
    zbar = [float(row[4]) for row in table]
    n_ee = deriv.shape[0] - len(spectra)*nell
    shipped = np.loadtxt(os.path.join(DATA, STEM + '.mask'))[:, 1] > 0.5
    var = np.zeros(deriv.shape[0])
    with open(os.path.join(DATA, STEM + '_cov')) as f:
        for line in f:
            i, j, v = line.split()
            if i == j:
                var[int(i)] = float(v)

    def rows(s):
        r = n_ee + s*nell + np.arange(nell)
        return r, uv_mask[s] & shipped[r]

    sv = [k for k, n in enumerate(names) if n.startswith('DESI_SIGV')]
    b1 = [k for k, n in enumerate(names) if n.startswith('DESI_B1')]
    sd = np.sqrt(var)
    templates = {
        'sigma_v': (deriv[:, sv]*fid[sv]).sum(axis=1)/np.where(var > 0, sd, np.inf),
        'b1': (deriv[:, b1]*fid[b1]).sum(axis=1)/np.where(var > 0, sd, np.inf),
        'w0': deriv[:, names.index('w0')]/np.where(var > 0, sd, np.inf),
    }

    # the autos' kept bands define the cosines and the matching coefficients
    auto_rows = np.concatenate([rows(s)[0][rows(s)[1]] for s in range(NBINS)])
    t = {p: v[auto_rows] for p, v in templates.items()}

    def cos(a, b):
        return float(a @ b/np.sqrt((a @ a)*(b @ b)))
    single = {p: float(t['sigma_v'] @ t[p]/(t[p] @ t[p])) for p in ('b1', 'w0')}
    basis = np.column_stack([t['b1'], t['w0']])
    joint, *_ = np.linalg.lstsq(basis, t['sigma_v'], rcond=None)

    shown = []
    for label, pair, role in [('auto', (BIN, BIN), 'auto spectrum'),
                              ('neighbours', (BIN, BIN + 1), 'cross with the next bin (current list)'),
                              ('two apart', (BIN, BIN + 2), 'cross with the bin two apart (added at g = 1)')]:
        s = spectra.index(pair)
        r, keep = rows(s)
        shown.append({
            'name': label, 'role': role, 'bins': list(pair),
            'sample': chunk[pair[0]], 'zbar': [zbar[pair[0]], zbar[pair[1]]],
            'ell': ell[keep].tolist(),
            **{f't_{p}': v[r[keep]].tolist() for p, v in templates.items()},
        })

    data = {
        'about': (
            'Templates sigma_ell^-1 dC_ell/dp of three Delta z = 0.02 spectra of bin '
            f'{BIN} ({chunk[BIN]}): its auto spectrum, its cross with bin {BIN + 1} and '
            f'with bin {BIN + 2}, on the bands kept at k_max = {kmax[m]} h/Mpc (z = 0 '
            'value of the per-bin cut at k_max/D(zbar_i), UV-fraction cut intersected '
            'with the shipped mask). sigma_ell from the diagonal of the Gaussian '
            'covariance; p = ln sigma_v, ln b_1 (every dispersion / every sample bias '
            'together) and w0. Cosines and matching coefficients fitted over the kept '
            'bands of all 90 auto spectra. Written by '
            'notes/cross_correlations/plots/make_sv_templates_data.py from the pair-axes '
            'derivative base chains/model_derivatives_dz0p02_pairmax_final.npz, its cuts '
            'chains/model_scale_cuts_dz0p02_pairmax_final.npz and the dataset files of '
            + STEM + '.'),
        'kmax_h_per_Mpc': float(kmax[m]),
        'auto_bands': int(auto_rows.size),
        'cosine_autos': {'b1': cos(t['sigma_v'], t['b1']), 'w0': cos(t['sigma_v'], t['w0'])},
        'match_single': single,
        'match_joint': {'b1': float(joint[0]), 'w0': float(joint[1])},
        'spectra': shown,
    }
    folder = os.path.join(HERE, 'sv_templates')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.join(folder, 'data.json'))
    print('auto bands', auto_rows.size, 'cosines', data['cosine_autos'],
          'single', single, 'joint', data['match_joint'])
    for sp in shown:
        print(sp['name'], len(sp['ell']), 'bands')


if __name__ == '__main__':
    main()
