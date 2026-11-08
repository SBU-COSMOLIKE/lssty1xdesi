"""Write the data.json of the five figures of notes/higher_order_bias.

The figures carry the numbers of the note's one-loop study. Each folder next
to this file holds data.json (written here), plot_<name>.py (it redraws the
figure from data.json alone) and the figure, <name>.pdf and <name>.png:

  term_sizes          - each one-loop term of P_gg in percent of b_1^2 P_mm,
                        against k, for four DESI-like tracers
  poisson_power       - the Poisson power 1/nbar in percent of b_1^2 P_mm
  coevolution_biases  - b_1 = B_g/D(z) and the calibrated b_2, b_{s^2} and
                        b_3nl across the six DESI chunks
  draft_bias_check    - the draft paper's linear-bias column against B_g/D(z)
  k_neg               - the wavenumber where the subtracted one-loop P_gg first
                        turns negative, against the offset Delta_2 of b_2

Symbols (the note's notation, in the CCL / FAST-PT normalization):
  b_1, b_2, b_{s^2}, b_3nl - the Eulerian linear, quadratic, tidal and
                        third-order galaxy biases;
  B_g                 - a chunk's bias amplitude: b_1(z) = B_g/D(z), with D
                        the linear growth factor, D(0) = 1;
  P_lin, P_mm         - the linear and the halofit matter power spectra;
  P_d1d2, ..., P_s2s2 - the five quadratic one-loop templates of FAST-PT at
                        z = 0, scaled by D^4(z) at redshift z;
  sig3nl              - FAST-PT's third-order template, sigma_3^2(k) P_lin(k);
  sigma_4             - the k -> 0 limit of P_d2d2 / 2, a white-noise constant;
  nbar                - the tracer number density in (h/Mpc)^3, and 1/nbar
                        its Poisson power in (Mpc/h)^3;
  Delta_2             - the offset of b_2 from the Lazeyras et al. fit
                        (arXiv:1511.01096),
                        b_2 = 0.412 - 2.143 b_1 + 0.929 b_1^2 + 0.008 b_1^3;
  co-evolution        - b_{s^2} = -(4/7)(b_1 - 1), b_3nl = (32/315)(b_1 - 1).

Two numerical set-ups, both stated in the note's appendix on the numerical
set-up of the figures:

  set-up A (every figure except k_neg): the project fiducial of the draft
    paper (Omega_m = 0.3, Omega_b = 0.04, h = 0.6732, 10^9 A_s = 2.1,
    n_s = 0.966, massless neutrinos), with A_s the primordial scalar
    amplitude and n_s its spectral index; CAMB linear and halofit
    (Takahashi, the version the likelihood uses) spectra on 1200
    logarithmic points between 1e-4 and 30 h/Mpc; FAST-PT padded with 600
    points; growth from the evolution of the k = 0.05/Mpc mode. The loop
    templates keep their low-k constants (CCL's default, sub_lowk off).
  set-up B (k_neg): the likelihood's evaluation fiducial (one massive
    neutrino of 0.06 eV, n_s = 0.96605), CAMB linear spectra on 2000
    logarithmic points between 1e-4 and 40 h/Mpc, FAST-PT padded with 1000
    points, growth from the linear spectra at k = 1e-4 h/Mpc. The templates
    are renormalized as CosmoLike does: the three low-k constants are
    subtracted.

Both run FAST-PT with CCL's EulerianPTCalculator extrapolation and window
(low_extrap = -5, high_extrap = 3, C_window = 0.75).

Run in the group's ccl conda environment, which carries CAMB 1.6.5 and
FAST-PT 4.0 (the cocoapy311 environment carries neither FAST-PT nor pyccl):

  source /Users/vivianmiranda/miniforge/etc/profile.d/conda.sh
  conda activate ccl
  python make_figure_data.py [--data-dir <projects/lssty1xdesi/data>]

The script overwrites the data.json of the five folders next to it. The
figures are then redrawn with each folder's plot_<name>.py.
"""
import argparse
import json
import os

import camb
import fastpt
import numpy as np
from scipy.integrate import simpson

HERE = os.path.dirname(os.path.abspath(__file__))
# notes/higher_order_bias/plots -> projects/lssty1xdesi/data
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'data'))
# the versions the note's appendix states; other versions change the numbers
REQUIRED_VERSIONS = {'camb': '1.6.5', 'fastpt': '4.0.0'}

# Set-up A: the project fiducial of the note's appendix.
SETUP_A = {
    'H0_km_s_mpc': 67.32,
    'omega_m': 0.30,
    'omega_b': 0.04,
    'A_s': 2.1e-9,
    'n_s': 0.966,
    'mnu_ev': 0.0,
    # CAMB's name of the halofit version the likelihood uses (CAMB's default
    # nonlinear model is HMcode-2020, 'mead2020')
    'halofit_version': 'takahashi',
    'k_min_h_per_mpc': 1.0e-4,
    'k_max_h_per_mpc': 30.0,
    'n_k': 1200,
    'fastpt_n_pad': 600,
    'growth': 'evolution of the k = 0.05/Mpc mode (CAMB delta_tot)',
    'loop_constants': 'kept (CCL default, sub_lowk off)',
}
# Set-up B: the likelihood's evaluation fiducial (EXAMPLE_EVALUATE1.yaml).
SETUP_B = {
    'H0_km_s_mpc': 67.32,
    'omega_m': 0.30,
    'omega_b': 0.04,
    'A_s': 2.1e-9,
    'n_s': 0.96605,
    'mnu_ev': 0.06,
    'k_min_h_per_mpc': 1.0e-4,
    'k_max_h_per_mpc': 40.0,
    'n_k': 2000,
    'fastpt_n_pad': 1000,
    'growth': 'sqrt of the linear-spectrum ratio at k = 1e-4 h/Mpc',
    'loop_constants': 'subtracted (CosmoLike convention, sub_lowk on)',
}
# FAST-PT settings of CCL's EulerianPTCalculator: power-law extrapolation
# over 5 decades below and 3 above the grid, and the FFTLog window on 75% of
# the Fourier coefficients
FASTPT_EXTRAP = {'low_extrap': -5, 'high_extrap': 3}
FASTPT_C_WINDOW = 0.75
# CAMB's Boltzmann k range in 1/Mpc, above the largest k the grids request
CAMB_KMAX_PER_MPC = 50.0
# the DESI DR1 one-sigma width of the counterterm prior, alpha_0 in (Mpc/h)^2
ALPHA0_MPC2_PER_H2 = 12.5
# comoving width of each thin bin of the Delta d_com = 62 Mpc layout, in Mpc
BIN_WIDTH_MPC = 62.28
ARCMIN2_PER_SR = (180.0*60.0/np.pi)**2
# the k at which the growth factor of set-up A is taken, in 1/Mpc
GROWTH_K_PER_MPC = 0.05

# The four tracers of the term-size figures: the chunk amplitude B_g of the
# project fiducial and a redshift inside the chunk.
SIZE_TRACERS = [
    {'name': 'BGS-like', 'sample': 'bgs', 'z': 0.3, 'B_g': 1.34},
    {'name': 'LRG-like', 'sample': 'lrg', 'z': 0.5, 'B_g': 1.70},
    {'name': 'LRG-like', 'sample': 'lrg', 'z': 0.8, 'B_g': 1.70},
    {'name': 'ELG-like', 'sample': 'elg', 'z': 1.3, 'B_g': 0.84},
]
# the wavenumbers the note's text reads, in h/Mpc
TEXT_K_H_PER_MPC = [0.05, 0.075, 0.1, 0.15, 0.2]
# the curves of the term-size figures, in h/Mpc
SIZE_K_GRID = np.linspace(0.02, 0.20, 181)

# The six DESI chunks: amplitude, redshift range and the DESI DR1 effective
# redshift (ELG1, excluded from the DESI DR1 full-shape fits, near the
# center of its range).
CHUNKS = [
    {'name': 'BGS', 'sample': 'bgs', 'B_g': 1.34,
     'z_lo': 0.1, 'z_hi': 0.4, 'z_eff': 0.295},
    {'name': 'LRG1', 'sample': 'lrg', 'B_g': 1.70,
     'z_lo': 0.4, 'z_hi': 0.6, 'z_eff': 0.510},
    {'name': 'LRG2', 'sample': 'lrg', 'B_g': 1.70,
     'z_lo': 0.6, 'z_hi': 0.8, 'z_eff': 0.706},
    {'name': 'LRG3', 'sample': 'lrg', 'B_g': 1.70,
     'z_lo': 0.8, 'z_hi': 1.1, 'z_eff': 0.919},
    {'name': 'ELG1', 'sample': 'elg', 'B_g': 0.84,
     'z_lo': 0.8, 'z_hi': 1.1, 'z_eff': 0.955},
    {'name': 'ELG2', 'sample': 'elg', 'B_g': 0.84,
     'z_lo': 1.1, 'z_hi': 1.6, 'z_eff': 1.317},
]
# points per chunk of the drawn bias curves
CHUNK_CURVE_POINTS = 41
# The draft paper's linear-bias column ("b_1^E"), as the note transcribes it;
# the draft has no ELG1 row.
DRAFT_LINEAR_COLUMN = {'BGS': 1.375, 'LRG1': 1.448, 'LRG2': 1.403, 'LRG3': 1.388,
                       'ELG2': 0.704}

# The three bins of the k_neg figure: two BGS redshifts and one ELG redshift.
KNEG_BINS = [
    {'name': 'BGS', 'z': 0.1, 'B_g': 1.34},
    {'name': 'BGS', 'z': 0.3, 'B_g': 1.34},
    {'name': 'ELG', 'z': 1.1, 'B_g': 0.84},
]
# Delta_2 from -1 to +1 (model A's one-sigma prior range) in steps of 0.005,
# so the offsets the text quotes (multiples of 0.5) are grid points
DELTA2_GRID = np.linspace(-1.0, 1.0, 401)
TEXT_DELTA2 = [-1.0, -0.5, 0.0, 0.5, 1.0]
# P_gg is searched for a sign change only above this k, in h/Mpc: below it
# the loops vanish and b_1^2 P_lin dominates
KNEG_SEARCH_FROM_H_PER_MPC = 0.01


def check_versions():
    '''Stop unless CAMB and FAST-PT have the versions the note states.

    Raises:
      RuntimeError when either package has another version.
    '''
    found = {'camb': camb.__version__, 'fastpt': fastpt.__version__}
    for package, required in REQUIRED_VERSIONS.items():
        if found[package] != required:
            raise RuntimeError(
                f'{package} is version {found[package]}, but the note states {required}: '
                f'activate the ccl conda environment, which carries {package} {required}')


class FASTPTTrapezoid(fastpt.FASTPT):
    '''FAST-PT 4.0 with its sigma_4 integral written with np.trapezoid.

    FAST-PT 4.0 integrates sigma_4 = int dq q^2 P_lin(q)^2 / (2 pi^2) with
    np.trapz, a name numpy 2.4 (the ccl environment) no longer provides. This
    subclass repeats the same trapezoid rule on the same extrapolated grid
    under numpy's current name; every template is FAST-PT's own code.
    '''

    def _get_sig4(self, P, P_window=None, C_window=None):
        '''Return sigma_4 in (Mpc/h)^3 for the linear spectrum P [n_k].'''
        P_extrapolated, _ = self.J_k_scalar(P, self.X_spt, -2, P_window=P_window,
                                            C_window=C_window)
        integrand = self.k_extrap**3*P_extrapolated**2
        return np.trapezoid(integrand, x=np.log(self.k_extrap))/(2.0*np.pi**2)


def camb_parameters(setup):
    '''Return CAMB parameters for one of the two set-ups.

    The matter density Omega_m includes the massive neutrino of set-up B:
    its density, mnu/(93.14 eV) h^-2, comes out of the cold dark matter, so
    Omega_m stays 0.3.

    Arguments:
      setup = SETUP_A or SETUP_B
    Returns:
      a camb.CAMBparams instance
    '''
    h = setup['H0_km_s_mpc']/100.0
    ombh2 = setup['omega_b']*h**2
    neutrino_omh2 = setup['mnu_ev']/93.14
    omch2 = (setup['omega_m'] - setup['omega_b'])*h**2 - neutrino_omh2
    if setup['mnu_ev'] == 0.0:
        return camb.set_params(H0=setup['H0_km_s_mpc'], ombh2=ombh2, omch2=omch2,
                               As=setup['A_s'], ns=setup['n_s'], mnu=0.0,
                               num_massive_neutrinos=0, nnu=3.044, WantTransfer=True,
                               kmax=CAMB_KMAX_PER_MPC)
    return camb.set_params(H0=setup['H0_km_s_mpc'], ombh2=ombh2, omch2=omch2,
                           As=setup['A_s'], ns=setup['n_s'], mnu=setup['mnu_ev'])


def matter_spectra(setup, redshifts, halofit_version):
    '''Run CAMB and return the linear (and optionally halofit) spectra.

    Arguments:
      setup           = SETUP_A or SETUP_B
      redshifts       = list of the redshifts needed
      halofit_version = CAMB's name of the nonlinear model ('takahashi',
                        'mead2020', ...), or None for linear spectra only
    Returns:
      (results, k, P_lin, P_mm): the CAMB results of the linear run; k [n_k]
      in h/Mpc; dicts redshift -> spectrum [n_k] in (Mpc/h)^3 (P_mm is empty
      when halofit_version is None)
    '''
    pars = camb_parameters(setup)
    pars.set_matter_power(redshifts=list(redshifts), kmax=CAMB_KMAX_PER_MPC,
                          nonlinear=False)
    results = camb.get_results(pars)
    k, z_out, plin_grid = results.get_matter_power_spectrum(
        minkh=setup['k_min_h_per_mpc'], maxkh=setup['k_max_h_per_mpc'],
        npoints=setup['n_k'])
    plin = {}
    for z in redshifts:
        plin[z] = plin_grid[row_of_redshift(z_out, z)]
    pmm = {}
    if halofit_version is not None:
        pars.NonLinear = camb.model.NonLinear_pk
        pars.NonLinearModel.set_params(halofit_version=halofit_version)
        results_nl = camb.get_results(pars)
        k_nl, z_nl, pmm_grid = results_nl.get_matter_power_spectrum(
            minkh=setup['k_min_h_per_mpc'], maxkh=setup['k_max_h_per_mpc'],
            npoints=setup['n_k'])
        if not np.allclose(k_nl, k, rtol=1e-12):
            raise RuntimeError('the halofit and linear k grids differ: both must come '
                               'from the same minkh, maxkh and npoints')
        for z in redshifts:
            pmm[z] = pmm_grid[row_of_redshift(z_nl, z)]
    return results, k, plin, pmm


def row_of_redshift(z_out, z):
    '''Return the row of CAMB's output redshifts z_out that holds redshift z.'''
    distance = np.abs(np.asarray(z_out) - z)
    row = int(np.argmin(distance))
    if distance[row] > 1e-9:
        raise RuntimeError(f'CAMB returned no spectrum at z = {z}')
    return row


def redshift_key(z):
    '''Return the key of redshift z in the growth dictionaries: z to 6 decimals.'''
    return round(float(z), 6)


def growth_from_mode(results, redshifts):
    '''Return D(z) / D(0) from the evolution of the k = 0.05/Mpc mode.

    Arguments:
      results   = CAMB results of set-up A
      redshifts = list of redshifts
    Returns:
      dict redshift_key(z) -> D, with D(0) = 1
    '''
    if 0.0 not in redshifts:
        raise RuntimeError('growth_from_mode needs z = 0 in its redshift list, '
                           'for D(0) = 1')
    # One CAMB call evolves the mode once and samples it at every redshift.
    # Its integrator needs strictly increasing output times, so the redshifts
    # go from high to low, and redshifts equal to 6 decimals (the tracer
    # redshift 0.5 and the chunk-curve point 0.4 + 20 x 0.005, say) are merged
    # under one key: two output times a rounding error apart stop CAMB's
    # integrator.
    unique_keys = set()
    for z in redshifts:
        unique_keys.add(redshift_key(z))
    z_descending = sorted(unique_keys, reverse=True)
    evolution = results.get_redshift_evolution(GROWTH_K_PER_MPC, z_descending,
                                               ['delta_tot'])
    delta = np.abs(np.ravel(evolution))
    if delta.size != len(z_descending):
        raise RuntimeError(f'CAMB returned {delta.size} values of delta_tot for '
                           f'{len(z_descending)} redshifts')
    delta_today = delta[z_descending.index(0.0)]
    growth = {}
    for i, z in enumerate(z_descending):
        growth[z] = float(delta[i]/delta_today)
    return growth


def sigma8_tophat(k, plin):
    '''Return sigma_8, the rms linear overdensity in 8 Mpc/h spheres.

    Arguments:
      k    = wavenumbers [n_k] in h/Mpc
      plin = linear spectrum [n_k] in (Mpc/h)^3
    '''
    x = 8.0*k
    window = 3.0*(np.sin(x) - x*np.cos(x))/x**3
    integrand = k**3*plin*window**2/(2.0*np.pi**2)
    return float(np.sqrt(simpson(integrand, x=np.log(k))))


def loop_templates(k, plin_z0, n_pad):
    '''Return the FAST-PT one-loop bias templates at z = 0.

    Arguments:
      k       = wavenumbers [n_k] in h/Mpc, logarithmically spaced
      plin_z0 = linear spectrum at z = 0 [n_k] in (Mpc/h)^3
      n_pad   = number of zero-padding points of FAST-PT
    Returns:
      dict with 'P_d1d2', 'P_d2d2', 'P_d1s2', 'P_d2s2', 'P_s2s2' and 'sig3nl'
      [n_k] in (Mpc/h)^3, and 'sigma_4' (a float, (Mpc/h)^3)
    '''
    calculator = FASTPTTrapezoid(k, low_extrap=FASTPT_EXTRAP['low_extrap'],
                                 high_extrap=FASTPT_EXTRAP['high_extrap'], n_pad=n_pad)
    out = calculator.one_loop_dd_bias_b3nl(plin_z0, C_window=FASTPT_C_WINDOW)
    # FAST-PT returns the tuple
    # (P_1loop, P_s, P_d1d2, P_d2d2, P_d1s2, P_d2s2, P_s2s2, sigma_4, sig3nl)
    templates = {
        'P_d1d2': np.asarray(out[2]),
        'P_d2d2': np.asarray(out[3]),
        'P_d1s2': np.asarray(out[4]),
        'P_d2s2': np.asarray(out[5]),
        'P_s2s2': np.asarray(out[6]),
        'sigma_4': float(out[7]),
        'sig3nl': np.asarray(out[8]),
    }
    return templates


def at_k(k_grid, values, k_eval):
    '''Interpolate values [n_k] on k_grid to k_eval, linearly in ln k.'''
    return np.interp(np.log(k_eval), np.log(k_grid), values)


def lazeyras_b2(b1):
    '''Return the Lazeyras et al. (arXiv:1511.01096) quadratic bias b_2(b_1).'''
    return 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3


def coevolution_bs2(b1):
    '''Return the co-evolution tidal bias b_{s^2} = -(4/7)(b_1 - 1).'''
    return -(4.0/7.0)*(b1 - 1.0)


def coevolution_b3nl(b1):
    '''Return the co-evolution third-order bias b_3nl = (32/315)(b_1 - 1).'''
    return (32.0/315.0)*(b1 - 1.0)


def term_percentages(k_eval, biases, growth, k, templates, plin_z, pmm_z):
    '''Return each one-loop term of P_gg in percent of b_1^2 P_mm.

    The terms are those of the note's one-loop P_gg in the CCL convention,
    with the templates scaled by D^4 and the low-k constants kept.

    Arguments:
      k_eval    = wavenumbers [n_eval] in h/Mpc
      biases    = dict with 'b1', 'b2', 'bs2', 'b3nl'
      growth    = D(z) of the tracer redshift
      k         = the template grid [n_k] in h/Mpc
      templates = output of loop_templates
      plin_z    = linear spectrum at the tracer redshift [n_k], (Mpc/h)^3
      pmm_z     = halofit spectrum at the tracer redshift [n_k], (Mpc/h)^3
    Returns:
      dict term key -> percentages [n_eval]: 'b1b2', 'b2b2', 'b1bs2', 'b2bs2',
      'bs2bs2', 'b1b3nl', 'tidal' (the three b_{s^2} terms), 'loop_sum' (all six
      loop terms) and 'counterterm' (alpha_0 k^2 P_lin at the DESI width)
    '''
    b1 = biases['b1']
    b2 = biases['b2']
    bs2 = biases['bs2']
    b3nl = biases['b3nl']
    growth4 = growth**4
    reference = b1**2*at_k(k, pmm_z, k_eval)
    terms = {
        'b1b2': b1*b2*growth4*at_k(k, templates['P_d1d2'], k_eval),
        'b2b2': 0.25*b2*b2*growth4*at_k(k, templates['P_d2d2'], k_eval),
        'b1bs2': b1*bs2*growth4*at_k(k, templates['P_d1s2'], k_eval),
        'b2bs2': 0.5*b2*bs2*growth4*at_k(k, templates['P_d2s2'], k_eval),
        'bs2bs2': 0.25*bs2*bs2*growth4*at_k(k, templates['P_s2s2'], k_eval),
        'b1b3nl': b1*b3nl*growth4*at_k(k, templates['sig3nl'], k_eval),
    }
    tidal = terms['b1bs2'] + terms['b2bs2'] + terms['bs2bs2']
    loop_sum = tidal + terms['b1b2'] + terms['b2b2'] + terms['b1b3nl']
    terms['tidal'] = tidal
    terms['loop_sum'] = loop_sum
    terms['counterterm'] = ALPHA0_MPC2_PER_H2*k_eval**2*at_k(k, plin_z, k_eval)
    percentages = {}
    for key, value in terms.items():
        percentages[key] = 100.0*value/reference
    return percentages


def read_bin_densities(data_dir, results, h):
    '''Return the redshift, sample and 3D density of each thin bin.

    nbar = n_2D/(chi^2 Delta chi), with n_2D the surface density per steradian,
    chi the comoving distance to the bin mean redshift and Delta chi the
    comoving bin width.

    Arguments:
      data_dir = projects/lssty1xdesi/data
      results  = CAMB results of set-up A (for chi)
      h        = H0/100
    Returns:
      (z_bar [n_bin], sample [n_bin] of 'bgs'/'lrg'/'elg', nbar [n_bin] in (h/Mpc)^3)
    '''
    ngal_path = os.path.join(data_dir, 'lssty1xdesi_desi_dr3_ngal.txt')
    bins_path = os.path.join(data_dir, 'lssty1xdesi_desi_dr3_bins.txt')
    for path in (ngal_path, bins_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{path} does not exist: pass --data-dir with the '
                                    f'projects/lssty1xdesi/data folder')
    ngal_per_arcmin2 = np.loadtxt(ngal_path, usecols=1)
    # the bins file mixes text and numbers: columns bin, sample, chunk_index,
    # chunk_name, mean_redshift, after '#' comment lines
    samples = []
    mean_redshifts = []
    with open(bins_path) as f:
        for line in f:
            fields = line.split()
            if len(fields) == 0 or fields[0].startswith('#'):
                continue
            samples.append(fields[1])
            mean_redshifts.append(float(fields[4]))
    bin_sample = np.array(samples)
    z_bar = np.array(mean_redshifts)
    if ngal_per_arcmin2.size != z_bar.size:
        raise RuntimeError(f'{ngal_path} has {ngal_per_arcmin2.size} bins but '
                           f'{bins_path} has {z_bar.size}: the two files must describe '
                           f'the same bins')
    chi_mpc = results.comoving_radial_distance(z_bar)
    nbar_per_mpc3 = ngal_per_arcmin2*ARCMIN2_PER_SR/(chi_mpc**2*BIN_WIDTH_MPC)
    # a density per Mpc^3 is h^3 times smaller per (Mpc/h)^3
    nbar_h3_per_mpc3 = nbar_per_mpc3/h**3
    return z_bar, bin_sample, nbar_h3_per_mpc3


def nearest_bin(z_bar, bin_sample, sample, z):
    '''Return the index of the bin of one sample whose mean redshift is nearest z.'''
    candidates = np.flatnonzero(bin_sample == sample)
    if candidates.size == 0:
        raise RuntimeError(f'no bin of sample {sample} in the bins file')
    distance = np.abs(z_bar[candidates] - z)
    return int(candidates[np.argmin(distance)])


def first_negative_k(k, p_gg):
    '''Return the k where p_gg first turns negative above the search start.

    The crossing is interpolated linearly in p_gg and ln k between the last
    positive and the first negative grid point.

    Arguments:
      k    = wavenumbers [n_k] in h/Mpc
      p_gg = galaxy power spectrum [n_k]
    Returns:
      the crossing in h/Mpc, or None when p_gg stays positive on the grid
    '''
    searched = np.flatnonzero(k > KNEG_SEARCH_FROM_H_PER_MPC)
    negative = searched[p_gg[searched] < 0.0]
    if negative.size == 0:
        return None
    i = int(negative[0])
    if p_gg[i - 1] < 0.0:
        raise RuntimeError(f'P_gg is already negative at the search start '
                           f'k = {k[i - 1]:.4g} h/Mpc: lower KNEG_SEARCH_FROM_H_PER_MPC')
    fraction = p_gg[i - 1]/(p_gg[i - 1] - p_gg[i])
    ln_k = np.log(k[i - 1]) + fraction*(np.log(k[i]) - np.log(k[i - 1]))
    return float(np.exp(ln_k))


def subtracted_p_gg(b1, b2, bs2, growth, plin_z0, templates):
    '''Return the one-loop P_gg in CosmoLike's subtracted convention.

    P_gg = b_1^2 D^2 P_lin + D^4 [b_1 b_2 P_d1d2 + b_2^2/4 (P_d2d2 - 2 sigma_4)
           + b_1 b_{s^2} P_d1s2 + b_2 b_{s^2}/2 (P_d2s2 - 4/3 sigma_4)
           + b_{s^2}^2/4 (P_s2s2 - 8/9 sigma_4)],
    with b_3nl and the counterterm zero; the leading term is linear.

    Returns:
      P_gg [n_k] in (Mpc/h)^3
    '''
    sigma_4 = templates['sigma_4']
    loops = (b1*b2*templates['P_d1d2']
             + 0.25*b2**2*(templates['P_d2d2'] - 2.0*sigma_4)
             + b1*bs2*templates['P_d1s2']
             + 0.5*b2*bs2*(templates['P_d2s2'] - (4.0/3.0)*sigma_4)
             + 0.25*bs2**2*(templates['P_s2s2'] - (8.0/9.0)*sigma_4))
    return b1**2*growth**2*plin_z0 + growth**4*loops


def rounded(values, digits=6):
    '''Return values (a float or an array) rounded to significant digits, for JSON.'''
    if values is None:
        return None
    if np.isscalar(values):
        return float(f'{float(values):.{digits}g}')
    result = []
    for value in np.asarray(values, dtype=float).ravel():
        result.append(float(f'{value:.{digits}g}'))
    return result


def fastpt_description():
    '''Return the FAST-PT version and settings as one string, for data.json.'''
    return (f'FAST-PT {fastpt.__version__}, low_extrap = {FASTPT_EXTRAP["low_extrap"]}, '
            f'high_extrap = {FASTPT_EXTRAP["high_extrap"]}, C_window = {FASTPT_C_WINDOW}')


def write_json(name, payload):
    '''Write payload to <name>/data.json next to this script.'''
    folder = os.path.join(HERE, name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'{folder} does not exist: the figure folders sit next '
                                f'to make_figure_data.py')
    path = os.path.join(folder, 'data.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=1)
    print(f'wrote {path}')


def setup_a_inputs(halofit_version):
    '''Run set-up A and return everything its four figures need.

    Arguments:
      halofit_version = CAMB's name of the nonlinear model of P_mm
    Returns:
      dict with 'results', 'k', 'plin', 'pmm', 'growth', 'sigma8' and 'templates'
    '''
    redshifts = [0.0]
    for tracer in SIZE_TRACERS:
        redshifts.append(tracer['z'])
    for chunk in CHUNKS:
        redshifts.append(chunk['z_eff'])
    redshifts = sorted(set(redshifts))
    results, k, plin, pmm = matter_spectra(SETUP_A, redshifts, halofit_version)
    chunk_redshifts = []
    for chunk in CHUNKS:
        z_curve = np.linspace(chunk['z_lo'], chunk['z_hi'], CHUNK_CURVE_POINTS)
        for z in z_curve:
            chunk_redshifts.append(float(z))
    growth = growth_from_mode(results, redshifts + chunk_redshifts)
    sigma8 = {}
    for z in redshifts:
        sigma8[z] = sigma8_tophat(k, plin[z])
    templates = loop_templates(k, plin[0.0], SETUP_A['fastpt_n_pad'])
    return {'results': results, 'k': k, 'plin': plin, 'pmm': pmm, 'growth': growth,
            'sigma8': sigma8, 'templates': templates}


def setup_b_inputs():
    '''Run set-up B and return its k grid, z = 0 spectrum, growth and templates.'''
    redshifts = [0.0]
    for item in KNEG_BINS:
        redshifts.append(item['z'])
    redshifts = sorted(set(redshifts))
    _, k, plin, _ = matter_spectra(SETUP_B, redshifts, halofit_version=None)
    growth = {}
    for z in redshifts:
        growth[z] = float(np.sqrt(plin[z][0]/plin[0.0][0]))
    templates = loop_templates(k, plin[0.0], SETUP_B['fastpt_n_pad'])
    return {'k': k, 'plin_z0': plin[0.0], 'growth': growth, 'templates': templates}


def size_payloads(inputs, data_dir):
    '''Return the data.json contents of term_sizes and poisson_power.'''
    h = SETUP_A['H0_km_s_mpc']/100.0
    z_bar, bin_sample, nbar = read_bin_densities(data_dir, inputs['results'], h)
    k_text = np.array(TEXT_K_H_PER_MPC)
    size_tracers = []
    poisson_tracers = []
    for tracer in SIZE_TRACERS:
        z = tracer['z']
        growth = inputs['growth'][redshift_key(z)]
        b1 = tracer['B_g']/growth
        biases = {'b1': b1, 'b2': lazeyras_b2(b1), 'bs2': coevolution_bs2(b1),
                  'b3nl': coevolution_b3nl(b1)}
        plin_z = inputs['plin'][z]
        pmm_z = inputs['pmm'][z]
        curves = term_percentages(SIZE_K_GRID, biases, growth, inputs['k'],
                                  inputs['templates'], plin_z, pmm_z)
        points = term_percentages(k_text, biases, growth, inputs['k'],
                                  inputs['templates'], plin_z, pmm_z)
        i_bin = nearest_bin(z_bar, bin_sample, tracer['sample'], z)
        poisson_power = 1.0/nbar[i_bin]
        reference_curve = biases['b1']**2*at_k(inputs['k'], inputs['pmm'][z], SIZE_K_GRID)
        reference_text = biases['b1']**2*at_k(inputs['k'], inputs['pmm'][z], k_text)
        curves_out = {}
        points_out = {}
        for key in curves:
            curves_out[key] = rounded(curves[key])
            points_out[key] = rounded(points[key])
        size_tracers.append({
            'name': tracer['name'],
            'z': z,
            'B_g': tracer['B_g'],
            'growth_D': rounded(growth),
            'biases': {
                'b1': rounded(biases['b1']),
                'b2': rounded(biases['b2']),
                'bs2': rounded(biases['bs2']),
                'b3nl': rounded(biases['b3nl']),
            },
            'percent_of_b1sq_pmm': curves_out,
            'at_text_k': points_out,
        })
        poisson_tracers.append({
            'name': tracer['name'],
            'z': z,
            'b1': rounded(biases['b1']),
            'density_bin': i_bin,
            'density_bin_mean_z': rounded(z_bar[i_bin]),
            'nbar_h3_per_mpc3': rounded(nbar[i_bin]),
            'poisson_power_mpc3_per_h3': rounded(poisson_power),
            'percent_of_b1sq_pmm': rounded(100.0*poisson_power/reference_curve),
            'at_text_k': rounded(100.0*poisson_power/reference_text),
        })
    setup = dict(SETUP_A)
    setup['fastpt'] = fastpt_description()
    setup['camb'] = 'CAMB ' + camb.__version__
    setup['sigma_4_mpc3_per_h3'] = rounded(inputs['templates']['sigma_4'])
    setup['sigma8_z0'] = rounded(inputs['sigma8'][0.0])
    sizes = {
        'about': (
            'Each one-loop term of the galaxy power spectrum P_gg, in percent of '
            'the linear-bias model b_1^2 P_mm (P_mm halofit), for four DESI-like '
            'tracers at the co-evolution biases: b_1 = B_g/D(z), b_2 from the '
            'Lazeyras et al. fit, b_{s^2} = -(4/7)(b_1 - 1), b_3nl = (32/315)(b_1 - 1). '
            'Terms in the CCL convention: b1b2 = b_1 b_2 P_d1d2, b2b2 = b_2^2/4 '
            'P_d2d2, b1bs2 = b_1 b_{s^2} P_d1s2, b2bs2 = b_2 b_{s^2}/2 P_d2s2, '
            'bs2bs2 = b_{s^2}^2/4 P_s2s2, b1b3nl = b_1 b_3nl sigma_3^2 P_lin, '
            'tidal = b1bs2 + b2bs2 + bs2bs2, loop_sum = all six; counterterm = '
            'alpha_0 k^2 P_lin(k, z) at the DESI DR1 one-sigma width alpha_0 = 12.5 '
            '(Mpc/h)^2. FAST-PT templates at z = 0 scaled by D^4, low-k constants '
            'kept (sub_lowk off). k in h/Mpc at the tracer redshift; at_text_k holds '
            'the values at the wavenumbers of text_k_h_per_mpc. Written by '
            '../make_figure_data.py.'),
        'setup': setup,
        'k_h_per_mpc': rounded(SIZE_K_GRID),
        'text_k_h_per_mpc': TEXT_K_H_PER_MPC,
        'tracers': size_tracers,
    }
    poisson = {
        'about': (
            'The Poisson power 1/nbar of the project thin-bin densities, in percent '
            'of the linear-bias model b_1^2 P_mm (P_mm halofit) of the four '
            'term_sizes tracers. nbar = n_2D/(chi^2 Delta chi) at the bin nearest '
            'the tracer redshift, with n_2D from data/lssty1xdesi_desi_dr3_ngal.txt, '
            'the bin mean redshift from data/lssty1xdesi_desi_dr3_bins.txt and '
            'Delta chi = 62.28 Mpc. The covariance carries this power; model A gives '
            'SN_0 a prior width of twice it. k in h/Mpc; at_text_k holds the values '
            'at text_k_h_per_mpc. Written by ../make_figure_data.py.'),
        'setup': setup,
        'k_h_per_mpc': rounded(SIZE_K_GRID),
        'text_k_h_per_mpc': TEXT_K_H_PER_MPC,
        'tracers': poisson_tracers,
    }
    return sizes, poisson


def chunk_payloads(inputs):
    '''Return the data.json contents of coevolution_biases and draft_bias_check.'''
    chunks_out = []
    draft_out = []
    for chunk in CHUNKS:
        z_curve = np.linspace(chunk['z_lo'], chunk['z_hi'], CHUNK_CURVE_POINTS)
        b1_curve = []
        for z in z_curve:
            b1_curve.append(chunk['B_g']/inputs['growth'][redshift_key(z)])
        b1_curve = np.array(b1_curve)
        z_eff = chunk['z_eff']
        b1_eff = chunk['B_g']/inputs['growth'][redshift_key(z_eff)]
        b1_edges = np.array([b1_curve[0], b1_curve[-1]])
        chunks_out.append({
            'name': chunk['name'],
            'sample': chunk['sample'],
            'B_g': chunk['B_g'],
            'z_lo': chunk['z_lo'],
            'z_hi': chunk['z_hi'],
            'z_eff': z_eff,
            'z': rounded(z_curve),
            'b1': rounded(b1_curve),
            'b2': rounded(lazeyras_b2(b1_curve)),
            'bs2': rounded(coevolution_bs2(b1_curve)),
            'b3nl': rounded(coevolution_b3nl(b1_curve)),
            'at_z_eff': {
                'b1': rounded(b1_eff),
                'b2': rounded(lazeyras_b2(b1_eff)),
                'bs2': rounded(coevolution_bs2(b1_eff)),
                'b3nl': rounded(coevolution_b3nl(b1_eff)),
            },
            'bs2_at_edges': rounded(coevolution_bs2(b1_edges)),
        })
        if chunk['name'] not in DRAFT_LINEAR_COLUMN:
            continue
        printed = DRAFT_LINEAR_COLUMN[chunk['name']]
        sigma8_eff = inputs['sigma8'][z_eff]
        implied_b1 = printed/sigma8_eff
        draft_out.append({
            'name': chunk['name'],
            'z_lo': chunk['z_lo'],
            'z_hi': chunk['z_hi'],
            'z_eff': z_eff,
            'z': rounded(z_curve),
            'project_b1_curve': rounded(b1_curve),
            'draft_value_printed': printed,
            'sigma8_z_eff': rounded(sigma8_eff),
            'draft_value_over_sigma8': rounded(implied_b1),
            'project_b1_at_z_eff': rounded(b1_eff),
            'ratio_implied_to_project': rounded(implied_b1/b1_eff),
        })
    setup = {
        'cosmology': ('set-up A: Omega_m = 0.3, Omega_b = 0.04, h = 0.6732, '
                      '10^9 A_s = 2.1, n_s = 0.966, massless neutrinos'),
        'growth': SETUP_A['growth'],
        'sigma8': 'top-hat of radius 8 Mpc/h on the CAMB linear spectrum at z_eff',
        'camb': 'CAMB ' + camb.__version__,
        'sigma8_z0': rounded(inputs['sigma8'][0.0]),
    }
    coevolution = {
        'about': (
            'The fiducial biases across the six DESI chunks in the CCL convention: '
            'b_1 = B_g/D(z) with the project amplitudes B_g, b_2 from the Lazeyras '
            'et al. fit b_2 = 0.412 - 2.143 b_1 + 0.929 b_1^2 + 0.008 b_1^3, and the '
            'co-evolution b_{s^2} = -(4/7)(b_1 - 1) and b_3nl = (32/315)(b_1 - 1). '
            'Each chunk is drawn over its redshift range [z_lo, z_hi]; at_z_eff holds '
            'the values at the DESI DR1 effective redshift (ELG1, excluded from the '
            'DR1 full-shape fits, at z = 0.955, near the center of its range); '
            'bs2_at_edges holds b_{s^2} at z_lo and z_hi. Written by '
            '../make_figure_data.py.'),
        'setup': setup,
        'chunks': chunks_out,
    }
    draft = {
        'about': (
            'The draft paper\'s linear-bias column ("b_1^E", draft_value_printed) '
            'read as the DESI DR1 physical combination (1 + b_1^L) sigma_8(z_eff) = '
            'b_1 sigma_8(z_eff): divided by sigma_8 at the DESI DR1 effective '
            'redshift it gives draft_value_over_sigma8, compared with the project '
            'fiducial b_1 = B_g/D(z) (project_b1_curve over each chunk, '
            'project_b1_at_z_eff at z_eff). The draft has no ELG1 row. Written by '
            '../make_figure_data.py.'),
        'setup': setup,
        'chunks': draft_out,
    }
    return coevolution, draft


def kneg_payload(inputs):
    '''Return the data.json contents of k_neg.'''
    curves = []
    for item in KNEG_BINS:
        z = item['z']
        growth = inputs['growth'][z]
        b1 = item['B_g']/growth
        bs2 = coevolution_bs2(b1)
        k_neg = []
        for delta2 in DELTA2_GRID:
            b2 = lazeyras_b2(b1) + delta2
            p_gg = subtracted_p_gg(b1, b2, bs2, growth, inputs['plin_z0'],
                                   inputs['templates'])
            k_neg.append(first_negative_k(inputs['k'], p_gg))
        at_text = []
        for delta2 in TEXT_DELTA2:
            i_grid = int(np.argmin(np.abs(DELTA2_GRID - delta2)))
            at_text.append(rounded(k_neg[i_grid]))
        # the offset at which b_2 + (2/3) b_{s^2}, and with it the high-k constant
        # P_inf = -(sigma_4/2) D^4 (b_2 + 2 b_{s^2}/3)^2, vanishes
        delta2_zero = -(2.0/3.0)*bs2 - lazeyras_b2(b1)
        curve_values = []
        for value in k_neg:
            curve_values.append(rounded(value))
        curves.append({
            'name': item['name'],
            'z': z,
            'B_g': item['B_g'],
            'growth_D': rounded(growth),
            'b1': rounded(b1),
            'b2_lazeyras': rounded(lazeyras_b2(b1)),
            'bs2': rounded(bs2),
            'delta2_where_constant_vanishes': rounded(delta2_zero),
            'k_neg_h_per_mpc': curve_values,
            'k_neg_at_text_delta2': at_text,
        })
    setup = dict(SETUP_B)
    setup['fastpt'] = fastpt_description()
    setup['camb'] = 'CAMB ' + camb.__version__
    setup['sigma_4_mpc3_per_h3'] = rounded(inputs['templates']['sigma_4'])
    payload = {
        'about': (
            'k_neg, the wavenumber [h/Mpc] at which the one-loop P_gg in '
            'CosmoLike\'s subtracted convention first turns negative (above '
            '0.01 h/Mpc), against the offset Delta_2 of b_2 from the Lazeyras et al. '
            'fit; b_{s^2} on co-evolution, b_1 = B_g/D(z), b_3nl and the counterterm '
            'zero, the leading term the linear b_1^2 D^2 P_lin. null where P_gg stays '
            'positive up to the grid end, 40 h/Mpc. delta2_where_constant_vanishes is '
            'the offset at which b_2 + (2/3) b_{s^2} = 0, where the high-k constant '
            'vanishes. k_neg_at_text_delta2 holds the values at text_delta2. Written '
            'by ../make_figure_data.py.'),
        'setup': setup,
        'delta2': rounded(DELTA2_GRID),
        'text_delta2': TEXT_DELTA2,
        'bins': curves,
    }
    return payload


def main():
    '''Compute the five figures' data and write their data.json files.'''
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help='the projects/lssty1xdesi/data folder (default: found '
                             'relative to this script)')
    args = parser.parse_args()
    check_versions()

    inputs_a = setup_a_inputs(halofit_version=SETUP_A['halofit_version'])
    sizes, poisson = size_payloads(inputs_a, args.data_dir)
    coevolution, draft = chunk_payloads(inputs_a)
    inputs_b = setup_b_inputs()
    kneg = kneg_payload(inputs_b)

    write_json('term_sizes', sizes)
    write_json('poisson_power', poisson)
    write_json('coevolution_biases', coevolution)
    write_json('draft_bias_check', draft)
    write_json('k_neg', kneg)


if __name__ == '__main__':
    main()
