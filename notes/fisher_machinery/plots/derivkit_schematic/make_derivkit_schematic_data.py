"""Write data.json for derivkit_schematic: a toy that shows how DerivKit's fit
lowers a derivative's sensitivity to numerical noise, next to the five-point
stencil and a two-point central difference.

The toy is m(theta) = 1 + s0 t + q t^2 + c t^3, t = theta - theta0, plus
independent Gaussian noise of width SIGMA at every evaluation (the stand-in
for the numerical noise of a data vector). The three estimators are called
the way the Fisher-derivative study calls them
(tests/studies/run_fisher_derivatives.py):

- two-point central difference at +-H (H = H_MAIN = 0.02 of theta0);
- five-point stencil at +-H, +-2H with the weights (-1, 8, -8, 1)/12
  (cosmolike_notebook_utils.fisher.get_ddv, the weights of
  cosmolike_lssty1xdesi_fisher.FIVE_POINT_WEIGHTS);
- DerivKit itself (cosmolike_notebook_utils.fisher.get_ddv_dkit, min_samples
  7, fallback_mode "poly_at_floor"), which samples theta0 and +-1, 1.5, 2.25,
  3.375 and 5% of theta0 and fits a weighted straight line.

Panel (a) records one realization (every sampled point and each estimator's
slope); panel (b) the distribution of each estimator over N_REAL
realizations, and the error each makes on a deterministic, smooth error
term added to m with no noise (the same in every realization).

Run with the cocoa environment (start_cocoa.sh sets ROOTDIR):
    python make_derivkit_schematic_data.py
"""
import json
import os
import sys
import warnings

import numpy as np

ROOTDIR = os.environ['ROOTDIR']
sys.path.insert(0, os.path.join(ROOTDIR, 'external_modules', 'code', 'cosmolike_core'))
from cosmolike_notebook_utils import fisher as cnu   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# the toy: m(theta) = 1 + S0 t + Q t^2 + C t^3 with t = theta - THETA0
THETA0 = 1.0
S0 = 1.0
Q = 0.8
C = 5.6
SIGMA = 1.0e-3
# the deterministic smooth error added in the second test: A sin(2 pi t / L),
# a slow ripple across the sampled range (its slope at theta0 is 2 pi A / L)
DET_A = 1.0e-3
DET_L = 0.2
# the study's settings
H_MAIN = 0.02
DKIT_MIN_SAMPLES = 7
DKIT_FALLBACK = 'poly_at_floor'
N_REAL = 2000
SEED = 20261109


def truth(theta):
    t = theta - THETA0
    return 1.0 + S0*t + Q*t**2 + C*t**3


def deterministic(theta):
    return DET_A*np.sin(2.0*np.pi*(theta - THETA0)/DET_L)


def make_dv(rng, extra=None, noise=True, log=None):
    '''The data-vector function of the study's calling convention: a
    one-entry vector at param[0], with fresh noise at every evaluation.'''
    def dv(param, AccuracyBoost=1.0):
        theta = float(param[0])
        value = truth(theta) + (extra(theta) if extra else 0.0)
        if noise:
            value += SIGMA*rng.standard_normal()
        if log is not None:
            log.append((theta, value))
        return np.array([value])
    return dv


def central(dv):
    '''Two-point central difference at +-H_MAIN theta0.'''
    h = H_MAIN*THETA0
    return float((dv(np.array([THETA0 + h]))[0] - dv(np.array([THETA0 - h]))[0])/(2.0*h))


def five_point(dv):
    return float(cnu.get_ddv(dv, index=0, h=H_MAIN, CV=np.array([THETA0]))[0])


def derivkit(dv):
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)   # a fallback would raise
        return float(np.atleast_1d(cnu.get_ddv_dkit(dv, index=0, CV=np.array([THETA0]),
                                                    min_samples=DKIT_MIN_SAMPLES,
                                                    fallback_mode=DKIT_FALLBACK))[0])


ESTIMATORS = {'central': central, 'five_point': five_point, 'derivkit': derivkit}

# the analytic noise spreads: weights on independent evaluations
h = H_MAIN*THETA0
dkit_steps = np.array([0.01, 0.015, 0.0225, 0.03375, 0.05])*THETA0
analytic_sd = {
    'central': SIGMA/(np.sqrt(2.0)*h),
    'five_point': SIGMA*np.sqrt((1.0 + 64.0 + 64.0 + 1.0)/144.0)/h,
    # the line through the center: the mean of the five central differences
    'derivkit': SIGMA*np.sqrt(np.sum(1.0/(2.0*dkit_steps**2)))/len(dkit_steps),
}

# (a) a typical realization, every sampled point logged: the first one (of
# its own seeded stream) in which every estimator lands within one of its
# own standard deviations of the true slope
rng_example = np.random.default_rng(SEED + 1)
for attempt in range(1, 1001):
    example = {}
    for name, estimator in ESTIMATORS.items():
        log = []
        value = estimator(make_dv(rng_example, log=log))
        example[name] = {'slope': value, 'theta': [p[0] for p in log],
                         'm': [p[1] for p in log]}
    if all(abs(example[n]['slope'] - S0) <= analytic_sd[n] for n in ESTIMATORS):
        break
EXAMPLE_ATTEMPT = attempt
rng = np.random.default_rng(SEED)
# (b) many realizations
draws = {name: [] for name in ESTIMATORS}
for _ in range(N_REAL):
    for name, estimator in ESTIMATORS.items():
        draws[name].append(estimator(make_dv(rng)))
# noise-free: the curvature (cubic) bias alone, and the deterministic error alone
curvature_only = {name: estimator(make_dv(rng, noise=False)) for name, estimator in ESTIMATORS.items()}
det_only = {name: estimator(make_dv(rng, extra=deterministic, noise=False)) - S0
            for name, estimator in ESTIMATORS.items()}

summary = {}
for name in ESTIMATORS:
    x = (np.array(draws[name]) - S0)/S0
    summary[name] = {
        'mean': float(np.mean(x)), 'sd': float(np.std(x, ddof=1)),
        'percentiles': {str(p): float(np.percentile(x, p)) for p in (2.5, 16, 50, 84, 97.5)},
        'analytic_sd': float(analytic_sd[name]/S0),
        'curvature_bias': float(curvature_only[name]/S0 - 1.0),
        'deterministic_error': float(det_only[name]/S0)}

t_line = np.linspace(-0.055, 0.055, 221)*THETA0
data = {
    'about': ('A toy, not a measurement: m(theta) = 1 + s0 t + q t^2 + c t^3 (t = theta - '
              'theta0) plus independent Gaussian noise sigma at every evaluation, '
              'differentiated at theta0 by a two-point central difference at +-h, the five-point '
              'stencil at +-h, +-2h (weights (-1, 8, -8, 1)/12; h = 0.02 theta0, the study\'s '
              'H_MAIN) and DerivKit (cosmolike_notebook_utils.fisher.get_ddv_dkit, min_samples '
              '7, fallback_mode poly_at_floor: theta0 and +-1, 1.5, 2.25, 3.375, 5% of theta0, '
              'a weighted straight line), as in tests/studies/run_fisher_derivatives.py. '
              'example: one typical realization (the first of its own seeded stream in which '
              'every estimator lands within one of its own analytic standard deviations of the '
              'true slope; example_attempt counts the draws), every evaluation (theta, m) and '
              'each slope. '
              'distribution: (estimate - s0)/s0 over n_real realizations (mean, sd, '
              'percentiles), the analytic sd from the estimator\'s weights, the bias of the '
              'noise-free toy (the cubic term), and deterministic_error, the error on the '
              'noise-free deterministic ripple A sin(2 pi t/L) alone (the same in every '
              'realization). Written by make_derivkit_schematic_data.py in this folder.'),
    'toy': {'theta0': THETA0, 's0': S0, 'q': Q, 'c': C, 'sigma': SIGMA, 'h': h,
            'deterministic_A': DET_A, 'deterministic_L': DET_L, 'n_real': N_REAL, 'seed': SEED,
            'dkit_steps': dkit_steps.tolist(), 'example_attempt': EXAMPLE_ATTEMPT},
    'curve': {'t': t_line.tolist(),
              'm_minus_tangent': (truth(THETA0 + t_line) - 1.0 - S0*t_line).tolist()},
    'example': example,
    'distribution': summary,
    'draws': {name: ((np.array(v) - S0)/S0).tolist() for name, v in draws.items()},
}
with open(os.path.join(HERE, 'data.json'), 'w') as f:
    json.dump(data, f, indent=1)
for name, s in summary.items():
    print(f"{name:10s} mean {100*s['mean']:+.3f}%  sd {100*s['sd']:.3f}% (analytic "
          f"{100*s['analytic_sd']:.3f}%)  curvature bias {100*s['curvature_bias']:+.4f}%  "
          f"deterministic {100*s['deterministic_error']:+.3f}%  evaluations "
          f"{len(example[name]['theta'])}")
print(f"derivkit/five_point sd ratio {summary['derivkit']['sd']/summary['five_point']['sd']:.3f}, "
      f"derivkit/central {summary['derivkit']['sd']/summary['central']['sd']:.3f}")
