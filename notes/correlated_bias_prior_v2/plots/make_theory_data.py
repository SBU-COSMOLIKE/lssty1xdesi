"""Compute the data of the note's three theory figures and write each into
its plot folder as data.json:

  theory_bz/data.json     b(z) of every DESI group under the tied shape
                          B/D(z), a conserved population (the bias
                          continuity equation) and, for BGS, the r < 19.5
                          flux limit.
  theory_delta/data.json  the same laws as fractional deviations from B/D at
                          the Delta z = 0.05 bin centers, with the best
                          straight line through each and its residual.
  theory_draws/data.json  random draws of the BGS deviations (Delta z = 0.02
                          bins) from the exponential-kernel prior and from
                          the memory prior, and the flux-limit drift.
  prior_draws/data.json   per group, the known drifts and random histories
                          of the note's baseline and pessimistic memory
                          priors that share the drift's straight line.

Flat LCDM with Omega_m = 0.3; the fiducial amplitudes of the project's
params files. Each folder's plot_<name>.py redraws its figure from data.json
alone. Run from any folder: python make_theory_data.py
"""
import json
import os

import numpy as np
from scipy.integrate import quad

OMEGA_MATTER = 0.3
# the six DESI groups: (first redshift, last redshift, fiducial B_g)
GROUPS = {
    'BGS': (0.1, 0.4, 1.34),
    'LRG1': (0.4, 0.6, 1.70),
    'LRG2': (0.6, 0.8, 1.70),
    'LRG3': (0.8, 1.1, 1.70),
    'ELG1': (0.8, 1.1, 0.84),
    'ELG2': (1.1, 1.6, 0.84),
}
BIN_WIDTH = 0.05        # bin centers of the line fit (the Delta z = 0.05 layout)
DRAW_BIN_WIDTH = 0.02   # bin centers of the prior draws (the Delta z = 0.02 layout)
BGS_FLUX_LIMIT = 19.5   # r-band apparent magnitude limit of BGS Bright
ZEHAVI_MSTAR = -20.44   # M* - 5 log10 h of the Zehavi et al. 2011 fit
KERNEL_PRIOR = {'sigma_b': 0.10, 'L': 0.05}
MEMORY_PRIOR = {'sigma_b': 0.02, 'L': 0.05, 'slope_width': 0.6}
NUMBER_OF_DRAWS = 5
RANDOM_SEED = 20261107
# The note's two stated assumptions on the residual around each group's free
# line (the memory model): one width sigma_b per group and one correlation
# length L. Keep them equal to BIAS_ASSUMPTIONS of EXAMPLE_EVALUATE_FISHER.ipynb.
STATED_PRIORS = {
    'baseline': {'L': 0.2, 'sigma_b': {'BGS': 0.03, 'LRG1': 0.01, 'LRG2': 0.01,
                                       'LRG3': 0.01, 'ELG1': 0.01, 'ELG2': 0.01}},
    'pessimistic': {'L': 0.05, 'sigma_b': {'BGS': 0.03, 'LRG1': 0.03, 'LRG2': 0.03,
                                          'LRG3': 0.03, 'ELG1': 0.03, 'ELG2': 0.03}},
}
STATED_PRIOR_DRAWS = 4
PLOTS = os.path.dirname(os.path.abspath(__file__))


def hubble_rate(z):
    '''H(z)/H0 of flat LCDM.'''
    return np.sqrt(OMEGA_MATTER*(1.0 + z)**3 + 1.0 - OMEGA_MATTER)


def growth(z):
    '''Linear growth factor D(z) of flat LCDM, normalized to D(0) = 1.

    Arguments:
      z = redshift (scalar or array).

    Returns:
      D(z) from D ~ H(a) int_0^a da'/(a' H(a'))^3, exact for LCDM.
    '''
    def unnormalized(redshift):
        a = 1.0/(1.0 + redshift)
        integral = quad(lambda x: 1.0/(x*hubble_rate(1.0/x - 1.0))**3, 0.0, a)[0]
        return hubble_rate(redshift)*integral
    values = np.array([unnormalized(x) for x in np.atleast_1d(z)])
    return values/unnormalized(0.0)


def comoving_distance(z):
    '''Comoving distance in Mpc/h of flat LCDM.'''
    return 2997.92458*quad(lambda x: 1.0/hubble_rate(x), 0.0, z)[0]


def conserved_bias(z, z_center, b_center):
    '''Bias of a conserved population, from the bias continuity equation.

    Arguments:
      z        = redshifts.
      z_center = the anchor redshift.
      b_center = the bias at the anchor.

    Returns:
      b(z) = 1 + (b_center - 1) D(z_center)/D(z).
    '''
    return 1.0 + (b_center - 1.0)*growth(z_center)/growth(z)


def bgs_flux_limit_bias(z):
    '''Bias of a sample cut at r < 19.5, through the threshold-bias fit of
    Zehavi et al. 2011, b(>L) = 1.06 + 0.21 (L/L*)^1.12, with no
    K-correction and no evolution of the fit (an order-of-magnitude model).

    Arguments:
      z = redshifts.

    Returns:
      b(z) of the luminosity threshold M_lim(z) = r_lim - 25 - 5 log10 d_L.
    '''
    z = np.atleast_1d(z)
    distance = np.array([(1.0 + x)*comoving_distance(x) for x in z])
    magnitude = BGS_FLUX_LIMIT - 25.0 - 5.0*np.log10(distance)
    return 1.06 + 0.21*(10.0**(-0.4*(magnitude - ZEHAVI_MSTAR)))**1.12


def deviation_from_tied(z, bias):
    '''Fractional deviation of b(z) from B/D(z), with B set so that the
    mean deviation over z is zero (B_g is free).'''
    amplitude = bias*growth(z)
    return amplitude/amplitude.mean() - 1.0


def line_fit(z, delta):
    '''Best straight line through delta(z) with equal weights.

    Returns:
      (line values at z, residual delta - line).
    '''
    design = np.column_stack([np.ones_like(z), z - z.mean()])
    coefficients = np.linalg.lstsq(design, delta, rcond=None)[0]
    line = design @ coefficients
    return line, delta - line


def write(name, data):
    '''Write one figure's data.json into its plot folder.'''
    folder = os.path.join(PLOTS, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.join(name, 'data.json'))


def theory_bz():
    groups = []
    for name, (z0, z1, amplitude) in GROUPS.items():
        z = np.linspace(z0, z1, 61)
        z_center = 0.5*(z0 + z1)
        b_center = amplitude/growth(z_center)[0]
        entry = {'name': name, 'B_g': amplitude, 'z_center': z_center,
                 'z': z.tolist(),
                 'b_tied': (amplitude/growth(z)).tolist(),
                 'b_conserved': conserved_bias(z, z_center, b_center).tolist()}
        if name == 'BGS':
            flux = bgs_flux_limit_bias(z)
            entry['b_flux_limit'] = (flux*b_center/bgs_flux_limit_bias(z_center)[0]).tolist()
        groups.append(entry)
    write('theory_bz', {
        'about': ('linear bias b(z) of every DESI group: the tied shape B_g/D(z), a '
                  'conserved population 1 + (b_c - 1) D(z_c)/D(z) anchored at the group '
                  'midpoint z_c, and for BGS the r < 19.5 flux-limit bias (Zehavi et al. '
                  '2011 threshold fit, no K-correction) scaled to b_c at z_c; flat LCDM '
                  'with Omega_m = 0.3'),
        'omega_matter': OMEGA_MATTER, 'groups': groups})


def theory_delta():
    curves = []
    for name, (z0, z1, amplitude) in GROUPS.items():
        z = np.arange(z0 + BIN_WIDTH/2.0, z1, BIN_WIDTH)
        z_center = 0.5*(z0 + z1)
        b_center = amplitude/growth(z_center)[0]
        laws = [('conserved', conserved_bias(z, z_center, b_center))]
        if name == 'BGS':
            laws.append(('flux_limit', bgs_flux_limit_bias(z)))
        for law, bias in laws:
            delta = deviation_from_tied(z, bias)
            line, residual = line_fit(z, delta)
            curves.append({'group': name, 'law': law, 'z': z.tolist(),
                           'delta': delta.tolist(), 'line': line.tolist(),
                           'residual': residual.tolist()})
    write('theory_delta', {
        'about': ('fractional deviation delta(z) = b D/B_g - 1 of each law from the tied '
                  'shape at the Delta z = 0.05 bin centers, with B_g set so the mean '
                  'deviation is zero; line = the best straight line (equal weights), '
                  'residual = delta - line; law is conserved (bias continuity equation) '
                  'or flux_limit (BGS r < 19.5); flat LCDM with Omega_m = 0.3'),
        'omega_matter': OMEGA_MATTER, 'bin_width': BIN_WIDTH, 'curves': curves})


def theory_draws():
    z0, z1, _ = GROUPS['BGS']
    z = np.arange(z0 + DRAW_BIN_WIDTH/2.0, z1, DRAW_BIN_WIDTH)
    lag = np.abs(z[:, None] - z[None, :])
    rng = np.random.default_rng(RANDOM_SEED)
    kernel_cov = KERNEL_PRIOR['sigma_b']**2*np.exp(-lag/KERNEL_PRIOR['L'])
    memory_cov = MEMORY_PRIOR['sigma_b']**2*np.exp(-lag/MEMORY_PRIOR['L'])
    kernel_draws = []
    memory_draws = []
    for _ in range(NUMBER_OF_DRAWS):
        draw = rng.multivariate_normal(np.zeros(len(z)), kernel_cov)
        kernel_draws.append((draw - draw.mean()).tolist())
        slope = rng.normal(0.0, MEMORY_PRIOR['slope_width'])
        draw = slope*(z - z.mean()) + rng.multivariate_normal(np.zeros(len(z)), memory_cov)
        memory_draws.append((draw - draw.mean()).tolist())
    write('theory_draws', {
        'about': ('random draws of the fractional deviations delta_i of the BGS bins '
                  '(Delta z = 0.02 centers) from the exponential-kernel prior and from the '
                  'memory prior (a linear tilt plus the kernel), each with its mean removed '
                  '(B_g is free); the tilt is free in the model and drawn here from a '
                  'Gaussian of width slope_width per unit z for display only; flux_limit '
                  '= the BGS r < 19.5 drift'),
        'z': z.tolist(), 'kernel_prior': KERNEL_PRIOR, 'memory_prior': MEMORY_PRIOR,
        'random_seed': RANDOM_SEED,
        'flux_limit': deviation_from_tied(z, bgs_flux_limit_bias(z)).tolist(),
        'kernel_draws': kernel_draws, 'memory_draws': memory_draws})


def stated_prior_draws(case):
    '''Return, per group at Delta z = 0.02 centers: the known drift laws, the best
    straight line through the reference law (the BGS flux limit for BGS, the
    conserved population otherwise), and random histories of one stated
    prior: that line plus a residual drawn from the exponential kernel at the
    group's sigma_b and L, with the residual's own best line removed (the
    amplitude and tilt are free in the model, so only the residual carries
    the prior).

    Arguments:
      case = a key of STATED_PRIORS.

    Returns:
      list of one dict per group; each prior has its own generator started
      at RANDOM_SEED, so a prior's draws do not depend on the other's.
    '''
    prior = STATED_PRIORS[case]
    rng = np.random.default_rng(RANDOM_SEED)
    groups = []
    for name, (z0, z1, amplitude) in GROUPS.items():
        z = np.arange(z0 + DRAW_BIN_WIDTH/2.0, z1, DRAW_BIN_WIDTH)
        z_center = 0.5*(z0 + z1)
        b_center = amplitude/growth(z_center)[0]
        laws = {'conserved': deviation_from_tied(z, conserved_bias(z, z_center, b_center))}
        if name == 'BGS':
            laws['flux_limit'] = deviation_from_tied(z, bgs_flux_limit_bias(z))
        reference = 'flux_limit' if name == 'BGS' else 'conserved'
        line, residual = line_fit(z, laws[reference])
        sigma = prior['sigma_b'][name]
        cov = sigma**2*np.exp(-np.abs(z[:, None] - z[None, :])/prior['L'])
        draws = []
        for _ in range(STATED_PRIOR_DRAWS):
            _, draw_residual = line_fit(z, rng.multivariate_normal(np.zeros(len(z)), cov))
            draws.append((line + draw_residual).tolist())
        groups.append({'name': name, 'z': z.tolist(),
                       'laws': {k: v.tolist() for k, v in laws.items()},
                       'reference_law': reference, 'line': line.tolist(),
                       'residual_rms': float(np.std(residual)),
                       'neighbor_rms_prior': float(sigma*np.sqrt(
                           2.0*(1.0 - np.exp(-DRAW_BIN_WIDTH/prior['L'])))),
                       'draws': draws})
    return groups


def prior_draws():
    '''Write prior_draws/data.json: per group, the known drift laws and the
    random histories of both stated priors (baseline and pessimistic).'''
    by_case = {case: stated_prior_draws(case) for case in ('baseline', 'pessimistic')}
    groups = []
    for i, group in enumerate(by_case['baseline']):
        merged = {k: v for k, v in group.items() if k not in ('draws', 'neighbor_rms_prior')}
        merged['draws'] = {case: by_case[case][i]['draws'] for case in by_case}
        merged['neighbor_rms_prior'] = {case: by_case[case][i]['neighbor_rms_prior']
                                        for case in by_case}
        groups.append(merged)
    write('prior_draws', {
        'about': ('per DESI group at Delta z = 0.02 bin centers: the known drift laws '
                  '(fractional deviation from B_g/D with zero mean), the best straight line '
                  'through the reference law, and random histories of the baseline and '
                  'pessimistic memory priors: that line plus a residual drawn from the '
                  'exponential kernel at the group width sigma_b and L, with its own best '
                  'line removed; neighbor_rms_prior = the prior rms change between '
                  'adjacent bins'),
        'priors': {case: STATED_PRIORS[case] for case in by_case},
        'random_seed': RANDOM_SEED, 'bin_width': DRAW_BIN_WIDTH, 'groups': groups})


def main():
    theory_bz()
    theory_delta()
    theory_draws()
    prior_draws()


if __name__ == '__main__':
    main()
