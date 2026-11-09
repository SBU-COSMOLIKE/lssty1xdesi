"""Convergence of the Fisher derivatives of the paper model.

A Fisher forecast is built from derivatives of the data vector m (the
DESI band powers) with respect to every parameter. A derivative is a
difference of nearby vectors divided by a small step, so it amplifies
every numerical error of the vector that is not smooth in the
parameter: a chi grid that resamples thin kernels, tables that move
their nodes, a Boltzmann code at its default accuracy. Settings that
converge one chi2 to better than 0.2 (the MCMC settings of the example
yamls) need not converge a derivative. This study finds the accuracy
settings at which the derivatives stop changing what they feed: the
marginalized errors, the figures of merit and the linear-response
shifts of the paper model.

The paper model is model B + tilts on the pessimistic linear-bias prior
(EXAMPLE_EVALUATE<n>.yaml), scored at k_max = 0.075 and 0.2 h/Mpc at
z = 0 with the non-Limber UV-fraction scale cut (arXiv:2603.13031)
intersected with the shipped mask. Model A on the baseline prior at
0.075 is scored next to it. Both Fisher matrices are assembled exactly
as the model B notebook does (EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb,
section 5), from three kinds of columns of the model A likelihood at its
fiducial point, where models A and B coincide:

  - five-point stencil columns of the seven cosmological parameters and
    the seven velocity-dispersion groups (sigma_v);
  - per-bin columns of the bias operators, by five-point stencils on
    sets of bins that share no measured cross pair (one color of a
    greedy coloring of the pair graph, so every spectrum involves at
    most one stepped bin): the bin amplitude DESI_B1_<n> (with b2 and
    bs2 following b1 through their calibrations, the model A chain
    rule), the offsets DESI_DB2_<n> and DESI_DBS_<n> of b2 and bs2, and
    the counterterm DESI_A0_<n> = alpha_0;
  - exact templates of the terms that enter the vector linearly: the
    stochastic amplitudes SN0 and SN2 (stochastic_templates) and the
    radial counterterm alpha_2 (DESI_A2_<n>, a central difference,
    which is exact for a linear term at any step).

The per-bin parameters are tied to group parameters in the shipped
parameter files; here each becomes a sampled parameter at its fiducial
value (ModelFisher's sampled_overrides), so every step goes through the
likelihood exactly as in a chain.

The truths of the break test (notebook section 7) are applied at each
cut: alpha_2 at one prior width (model B) or +-12.5 h^-2 Mpc^2 (model
A), SN2 = 5, the offsets at one prior width, and the calibration drift
(Delta_2 and Delta_s linear across each group). A score is the shift of
(w0, wa) in units of its marginalized ellipse, or of Omega_m or A_s in
units of its marginalized sigma.

The ladder. One configuration is the reference, every control at a
high value; every other configuration lowers one control. The controls:

  stencil         three-point against five-point, from the same
                  evaluations (the reference evaluates every column at
                  +-h/2, +-h, +-2h and +-4h);
  step            h/2, h and 2h, every step scaled together;
  cosmolike       accuracyboost (its tables: the a and z grids, the
                  P(k) nodes, the ell sampling, the C-FAST-PT grids),
                  integration_accuracy (the Gauss-Legendre orders of
                  the Limber sums, which carry the one-loop cross
                  pairs), the combined chi-grid factor accuracyboost x
                  nonlimber_accuracyboost (NL_Nchi - 1 = 511 times it),
                  the low-rank engine's rank and multipole cap;
  CAMB            AccuracyBoost, k_per_logint (the transfer function's
                  output sampling) and the transfer kmax, which is the
                  likelihood's request kmax_boltzmann x accuracyboost
                  (the extra argument kmax of the yamls, 7.5, is below
                  it and does nothing).

Each control is varied with the others fixed, so a lower accuracyboost
keeps the chi grid and the CAMB kmax of the reference (nonlimber_
accuracyboost and kmax_boltzmann compensate). The C-FAST-PT
internal_accuracyboost stays at 1: the internal grid is that value
times accuracyboost, which the ladder varies, and 4 breaks the bias
tables.

The spot check. The full ladder runs on Delta z = 0.05. The thinnest
layout, Delta d_com = 62 Mpc, where the chi grid matters most, evaluates
only the candidate's stencil block at the step ladder
(candidate_ladder), to show that its FoM does not depend on the step;
if it did, the same at twice the chi grid (candidate_chi_double_ladder)
would decide the thin layouts' grid. Its per-bin columns are read from
the model A notebook's cache (compare --perbin-from chains/).

Stages (run from cocoa/Cocoa/ with the cocoa environment active, one
heavy stage at a time):

  python projects/lssty1xdesi/tests/studies/run_fisher_derivatives.py plan
  python ... run_fisher_derivatives.py cuts --layout dz0p05
  python ... run_fisher_derivatives.py derive --layout dz0p05 --config reference
  python ... run_fisher_derivatives.py compare --layout dz0p05
  python ... run_fisher_derivatives.py derive --layout dcom62 --config candidate_ladder
  python ... run_fisher_derivatives.py compare --layout dcom62 \\
      --base candidate_ladder --perbin-from projects/lssty1xdesi/chains

derive and cuts write their arrays to --work (default
projects/lssty1xdesi/chains/fisher_derivative_study/, gitignored: tens of
MB per configuration); compare writes a JSON summary to --results
(default tests/studies/results/).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# One BLAS thread: the parallelism belongs to cosmolike's OpenMP loops.
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
# The OpenMP runtime reads OMP_NUM_THREADS when the compiled interface
# loads, so the thread count is set here, before any import of it.
# start_cocoa.sh exports 1 (right for MPI chains); this study runs one
# process at a time and takes COCOA_STUDY_THREADS (default 8) instead.
os.environ["OMP_NUM_THREADS"] = os.environ.get("COCOA_STUDY_THREADS", "8")

ROOTDIR = os.environ.get("ROOTDIR")
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(STUDY_DIR))
PROJECT = "projects/lssty1xdesi/"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DEFAULT_WORK = os.path.join(PROJECT_DIR, "chains", "fisher_derivative_study")
DEFAULT_RESULTS = os.path.join(STUDY_DIR, "results")

COSMO = ["As_1e9", "ns", "H0", "omegab", "omegam", "w0", "wa"]
# one footprint: the DESI covariance files are built at 14,000 deg^2 and
# the forecasts rescale the inverse to 12,300 deg^2
INVCOV_SCALE = 12300.0/14000.0
# the cuts the paper model is scored at [h/Mpc at z = 0], and the list the
# UV-fraction cut is computed on (the linear notebook's KMAX_CUTS: where
# the UV fraction is not monotonic in k_max, the cut of a k_max is
# shortened to that of the next larger one, so the list matters)
SCORED_KMAX = (0.075, 0.2)
CUT_KMAX = tuple(float(k) for k in np.round(np.arange(0.05, 0.2501, 0.025), 3))
ALPHA_UV = 0.05

# The layouts: the model A example yaml (the likelihood whose derivatives
# every notebook takes), the linear one (the scale cut is built on the
# linear-bias spectra), the dataset stem, the linear parameter file (its
# DESI_B1 ties name the six bias groups) and the MCMC combined chi-grid
# factor of the paper yaml (accuracyboost 2 x nonlimber_accuracyboost).
LAYOUTS = {
    "dcom62": {
        "label": "Delta d_com = 62 Mpc",
        "model_a_yaml": PROJECT + "exploration/EXAMPLE_EVALUATE_MODEL_A1.yaml",
        "linear_yaml": PROJECT + "exploration/EXAMPLE_EVALUATE_LINEAR1.yaml",
        "dataset": "lssty1xdesi_desi_dr3",
        "bias_params": "likelihood/params_desi.yaml",
        "mcmc_chi_grid": 64.0,
    },
    "dz0p02": {
        "label": "Delta z = 0.02",
        "model_a_yaml": PROJECT + "exploration/EXAMPLE_EVALUATE_MODEL_A2.yaml",
        "linear_yaml": PROJECT + "exploration/EXAMPLE_EVALUATE_LINEAR2.yaml",
        "dataset": "lssty1xdesi_desi_dr1_dz0p02",
        "bias_params": "likelihood/params_desi_dr1_dz0p02.yaml",
        "mcmc_chi_grid": 64.0,
    },
    "dz0p05": {
        "label": "Delta z = 0.05",
        "model_a_yaml": PROJECT + "exploration/EXAMPLE_EVALUATE_MODEL_A3.yaml",
        "linear_yaml": PROJECT + "exploration/EXAMPLE_EVALUATE_LINEAR3.yaml",
        "dataset": "lssty1xdesi_desi_dr1_dz0p05",
        "bias_params": "likelihood/params_desi_dr1_dz0p05.yaml",
        "mcmc_chi_grid": 32.0,
    },
    "dz0p20": {
        "label": "Delta z = 0.2",
        "model_a_yaml": PROJECT + "exploration/EXAMPLE_EVALUATE_MODEL_A4.yaml",
        "linear_yaml": PROJECT + "exploration/EXAMPLE_EVALUATE_LINEAR4.yaml",
        "dataset": "lssty1xdesi_desi_dr1_dz0p20",
        "bias_params": "likelihood/params_desi_dr1_dz0p20.yaml",
        "mcmc_chi_grid": 8.0,
    },
}

# The settings every configuration resolves, at the MCMC values of the
# paper yamls (chi_grid per layout, above). camb_kmax [1/Mpc] is CAMB's
# transfer kmax, the likelihood's kmax_boltzmann x accuracyboost.
MCMC_SETTINGS = {
    "accuracyboost": 2.0,
    "integration_accuracy": 0,
    "chi_grid": None,
    "rank": 5,
    "lowrank_lmax": 1000,
    "camb_kmax": 40.0,
    "camb_accuracy": 1.05,
    "k_per_logint": 15,
}
# The reference: every control above its MCMC value. The chi grid is a
# multiple of the MCMC one (REFERENCE_CHI_GRID_FACTOR).
REFERENCE_SETTINGS = {
    "accuracyboost": 4.0,
    "integration_accuracy": 2,
    "chi_grid": None,
    "rank": 8,
    "lowrank_lmax": 2000,
    "camb_kmax": 80.0,
    "camb_accuracy": 2.0,
    "k_per_logint": 30,
}
REFERENCE_CHI_GRID_FACTOR = 4.0

# The steps at multiplier 1: the relative stencil step h of the cosmology
# and the velocity dispersions (cnu.get_ddv's), and the per-bin operator
# steps of the model A notebook (DESI_B1 relative, the b2 and bs2 offsets
# absolute, alpha_0 in h^-2 Mpc^2). alpha_2 is linear, so its central
# step is exact at any size.
H_MAIN = 0.02
OPERATOR_STEPS = {"b1": 0.01, "b2": 0.02, "bs2": 0.02, "a0": 5.0}
OPERATOR_PREFIX = {"b1": "DESI_B1", "b2": "DESI_DB2", "bs2": "DESI_DBS",
                   "a0": "DESI_A0"}
A2_STEP = 5.0
# The stencil offsets, in units of the step. "ladder" gives the
# three-point rule at d/2, d, 2d and 4d and the five-point rule at d/2, d
# and 2d from one set of evaluations; "five" only the five-point rule at d.
OFFSETS = {
    "ladder": (-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0),
    # the ladder plus +-3d and +-6d: the five-point rule also at 3d, the
    # h = 0.06 of the roman_real Fisher notebook
    "ladder_wide": (-6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0),
    # the five-point rule at d, 2d and 3d
    "steps": (-6.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 6.0),
    "five": (-2.0, -1.0, 1.0, 2.0),
}

# The configurations: settings changed from the reference, the stencil
# offsets, and the blocks evaluated ("all", or "stencil": only the
# cosmology and sigma_v columns, the per-bin columns then read from the
# reference). A setting that does not move with the bias - CAMB's, the
# engine's rank and multipole cap - reaches a per-bin column only through
# the fiducial power spectrum that every step of that column shares, so it
# cannot be amplified there: those rungs evaluate the stencil block only.
# The block swaps of the README show the same for the chi grid, which
# reaches the per-bin columns by less than 0.1% of the FoM. The chi grid
# rungs above the reference evaluate the stencil block at the ladder
# offsets, so they also give the step dependence at finer grids.
CONFIGS = {
    "reference": {"changes": {}, "offsets": "ladder", "blocks": "all"},
    # the reference at the five-point offsets only: the spot check of a
    # costly layout, where the step ladder is not repeated
    "reference_five": {"changes": {}, "offsets": "five", "blocks": "all"},
    "mcmc": {"changes": "mcmc", "offsets": "ladder", "blocks": "all"},
    "chi_grid_mcmc": {"changes": {"chi_grid_factor": 1.0}, "offsets": "five",
                      "blocks": "all"},
    "chi_grid_half": {"changes": {"chi_grid_factor": 2.0}, "offsets": "five",
                      "blocks": "all"},
    "chi_grid_double": {"changes": {"chi_grid_factor": 8.0}, "offsets": "ladder_wide",
                        "blocks": "stencil"},
    "chi_grid_quad": {"changes": {"chi_grid_factor": 16.0}, "offsets": "five",
                      "blocks": "stencil"},
    "accuracyboost_2": {"changes": {"accuracyboost": 2.0}, "offsets": "five",
                        "blocks": "all"},
    "accuracyboost_3": {"changes": {"accuracyboost": 3.0}, "offsets": "five",
                        "blocks": "all"},
    "accuracyboost_6": {"changes": {"accuracyboost": 6.0}, "offsets": "five",
                        "blocks": "all"},
    "integration_accuracy_0": {"changes": {"integration_accuracy": 0},
                               "offsets": "five", "blocks": "all"},
    "rank_5": {"changes": {"rank": 5}, "offsets": "five", "blocks": "stencil"},
    "lowrank_lmax_1000": {"changes": {"lowrank_lmax": 1000}, "offsets": "five",
                          "blocks": "stencil"},
    "camb_accuracy_1p05": {"changes": {"camb_accuracy": 1.05}, "offsets": "five",
                           "blocks": "stencil"},
    "camb_accuracy_1p5": {"changes": {"camb_accuracy": 1.5}, "offsets": "five",
                          "blocks": "stencil"},
    "camb_accuracy_3": {"changes": {"camb_accuracy": 3.0}, "offsets": "five",
                        "blocks": "stencil"},
    "camb_accuracy_4": {"changes": {"camb_accuracy": 4.0}, "offsets": "five",
                        "blocks": "stencil"},
    "k_per_logint_15": {"changes": {"k_per_logint": 15}, "offsets": "five",
                        "blocks": "stencil"},
    "camb_kmax_40": {"changes": {"camb_kmax": 40.0}, "offsets": "five",
                     "blocks": "stencil"},
    "candidate": {"changes": "candidate", "offsets": "five", "blocks": "all"},
    # the spot check of a costly layout (the per-bin columns from the
    # notebooks' caches, compare --perbin-from): the candidate's stencil
    # block at the five-point offsets and at the step ladder, and the same
    # at twice the candidate's chi grid
    "candidate_stencil": {"changes": "candidate", "offsets": "five", "blocks": "stencil"},
    "candidate_ladder": {"changes": "candidate", "offsets": "ladder", "blocks": "stencil"},
    "candidate_chi_double": {"changes": {"base": "candidate", "chi_grid": 512.0},
                             "offsets": "five", "blocks": "stencil"},
    "candidate_chi_double_ladder": {"changes": {"base": "candidate", "chi_grid": 512.0},
                                    "offsets": "ladder", "blocks": "stencil"},
}
# The converged settings this study proposes (the README's table). Every
# value is at or above its MCMC value. The chi grid is one combined factor,
# 256, on every layout: the coarsest grid at which the figures of merit do
# not depend on the stencil step (at 128 they move by 18% when the step is
# halved). That is 8 times the MCMC grid of Delta z = 0.05, 4 times the
# thin layouts' and 32 times Delta z = 0.2's.
CANDIDATE_SETTINGS = {
    "accuracyboost": 3.0,
    "integration_accuracy": 1,
    "chi_grid": 256.0,
    "rank": 5,
    "lowrank_lmax": 1000,
    "camb_kmax": 60.0,
    "camb_accuracy": 2.0,
    "k_per_logint": 30,
}

# The paper model and its scores (the model B notebook's settings):
# per-group families and their Gaussian widths (alpha_0 and alpha_2 at
# COUNTERTERM_FRACTION/k_max^2), the tilted families, the linear-bias
# priors, the truths and the inversion thresholds.
SYSTEMS = {
    "model B + tilts, pessimistic": {"model": "B", "tilts": True,
                                      "prior": "pessimistic",
                                      "kmax": SCORED_KMAX},
    "model A, baseline": {"model": "A", "tilts": False, "prior": "baseline",
                          "kmax": (0.075,)},
}
FIXED_WIDTHS = {
    "A": {"DB2": 1.0, "DBS": 0.5, "SN0": 2.0, "SN2": 5.0},
    "B": {"DB2": 2.0, "DBS": 1.0, "SN0": 2.0, "SN2": 5.0},
}
COUNTERTERM_FRACTION = 0.5
MODEL_FAMILIES = {"A": ["DB2", "DBS", "A0", "SN0", "SN2"],
                  "B": ["DB2", "DBS", "A0", "A2", "SN0", "SN2"]}
FAMILY_BLOCK = {"DB2": "b2", "DBS": "bs2", "A0": "a0", "A2": "a2",
                "SN0": "T0", "SN2": "T2"}
TILTED = ("DB2", "DBS", "A0", "A2")
BIAS_GROUP_NAMES = ["BGS", "LRG1", "LRG2", "LRG3", "ELG1", "ELG2"]
BIAS_PRIORS = {
    "baseline": {"L": 0.2, "sigma_b": {"BGS": 0.03, "LRG1": 0.01, "LRG2": 0.01,
                                       "LRG3": 0.01, "ELG1": 0.01, "ELG2": 0.01}},
    "pessimistic": {"L": 0.05, "sigma_b": {"BGS": 0.03, "LRG1": 0.03, "LRG2": 0.03,
                                           "LRG3": 0.03, "ELG1": 0.03, "ELG2": 0.03}},
}
ALPHA2_TRUTH_MODEL_A = 12.5
SN2_TRUTH = 5.0
CAL_DRIFT = {"b2": 1.0, "bs2": 0.5}
SINGLE_PARAMS = ["omegam", "As_1e9"]
PINV_RTOL = 1.0e-12
NO_DATA_RTOL = 1.0e-10
NULL_OVERLAP_MAX = 1.0e-6

# The tolerance: a configuration is converged when, against the
# reference, every marginalized sigma of (w0, wa, Omega_m, A_s) moves by
# less than SIGMA_TOL, each FoM by less than FOM_TOL (relative), and every
# score of every truth by less than SHIFT_TOL (in units of sigma). A change
# of 1-2% in a FoM does not alter any conclusion of the notes, and the
# noise floor of one evaluation moves the FoM at k_max = 0.075 by about
# that much between steps, so the FoM and sigma tolerances are 2%. The
# scores decide where a model breaks (sigma/3), so SHIFT_TOL is 10% of
# that threshold.
SIGMA_TOL = 0.02
FOM_TOL = 0.02
SHIFT_TOL = 0.03


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def resolve_settings(layout, config):
    """The settings of one configuration on one layout.

    Arguments:
      layout = a LAYOUTS key.
      config = a CONFIGS key.

    Returns:
      dict with the keys of MCMC_SETTINGS (chi_grid resolved), plus
      "offsets" and "blocks".

    Raises:
      KeyError naming an unknown layout, configuration or control.
    """
    if layout not in LAYOUTS:
        raise KeyError(f"layout {layout!r}: the layouts are {list(LAYOUTS)}")
    if config not in CONFIGS:
        raise KeyError(f"configuration {config!r}: the configurations are "
                       f"{list(CONFIGS)}")
    spec = CONFIGS[config]
    mcmc_grid = LAYOUTS[layout]["mcmc_chi_grid"]
    if spec["changes"] == "mcmc":
        settings = dict(MCMC_SETTINGS)
        settings["chi_grid"] = mcmc_grid
    else:
        settings = dict(REFERENCE_SETTINGS)
        settings["chi_grid"] = REFERENCE_CHI_GRID_FACTOR*mcmc_grid
        # the changes apply to the reference, or, with "base": "candidate", on
        # top of the candidate's settings
        if spec["changes"] == "candidate":
            change_list = [CANDIDATE_SETTINGS]
        elif spec["changes"].get("base") == "candidate":
            extra = {}
            for control, value in spec["changes"].items():
                if control != "base":
                    extra[control] = value
            change_list = [CANDIDATE_SETTINGS, extra]
        else:
            change_list = [spec["changes"]]
        for changes in change_list:
            for control, value in changes.items():
                if control == "chi_grid_factor":
                    settings["chi_grid"] = float(value)*mcmc_grid
                elif control in settings:
                    settings[control] = value
                else:
                    raise KeyError(f"configuration {config!r} changes {control!r}, "
                                   f"which is not a control ({list(settings)})")
    settings["offsets"] = spec["offsets"]
    settings["blocks"] = spec["blocks"]
    # every configuration runs on the layout's paper dataset, as the
    # notebooks do (fm.derivative_settings)
    settings["data_file"] = LAYOUTS[layout]["dataset"] + ".dataset"
    return settings


def likelihood_overrides(settings):
    """The likelihood options of a resolved configuration.

    The chi grid and CAMB's kmax are products with accuracyboost in the
    likelihood (the combined factor accuracyboost x nonlimber_accuracyboost,
    the request kmax_boltzmann x accuracyboost), so the two factors are
    divided by it: each control stays where the configuration sets it
    whatever accuracyboost is. The chi grid's factor is rounded up to a
    power of two in the C code (NL_Nchi - 1 = 511 m, m = 2^ceil(log2(f))),
    so nonlimber_accuracyboost is the grid divided by accuracyboost
    rounded up to a power of two: the product then lies in (grid/2, grid]
    and rounds to the grid exactly (accuracyboost 3: grid/4, not the
    grid/3 whose product could round to twice the grid).
    """
    boost = float(settings["accuracyboost"])
    boost_power_of_two = 2.0**np.ceil(np.log2(boost))
    return {
        "accuracyboost": boost,
        "integration_accuracy": int(settings["integration_accuracy"]),
        "nonlimber_accuracyboost": float(settings["chi_grid"])/boost_power_of_two,
        "nonlimber_lowrank_rank": int(settings["rank"]),
        "nonlimber_lowrank_lmax": int(settings["lowrank_lmax"]),
        "kmax_boltzmann": float(settings["camb_kmax"])/boost,
        "data_file": settings["data_file"],
    }


def camb_overrides(settings):
    """The CAMB extra arguments of a resolved configuration."""
    return {"AccuracyBoost": float(settings["camb_accuracy"]),
            "k_per_logint": int(settings["k_per_logint"])}


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------

def read_rows(path):
    """The data rows of a text file, split on blanks (# lines skipped)."""
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                rows.append(line.split())
    return rows


def dataset_entry(stem, key):
    """The value of one "key = value" line of a .dataset file."""
    with open(os.path.join(DATA_DIR, stem + ".dataset")) as handle:
        for row in handle:
            if row.split("=")[0].strip() == key:
                return row.split("=", 1)[1].strip()
    raise KeyError(f"{key} is not in {stem}.dataset")


def layout_tables(layout):
    """The bin table, cross pairs, band centers and data-vector layout.

    Returns:
      dict: 'zbar' (mean redshift per bin), 'pairs' (list of (a, b) bin
      indices, 0-based), 'nell', 'n' (the full vector length), 'n_ee' (the
      length of the empty shear blocks in front of the clustering
      spectra), 'mask' (bool, n: the shipped mask) and 'rows' (rows[i] =
      the vector entries of every spectrum that involves bin i).
    """
    stem = LAYOUTS[layout]["dataset"]
    bins = read_rows(os.path.join(DATA_DIR, stem + "_bins.txt"))
    pairs = []
    for row in read_rows(os.path.join(DATA_DIR, stem + "_pairs.txt")):
        pairs.append((int(row[0]), int(row[1])))
    ells = np.loadtxt(os.path.join(DATA_DIR, dataset_entry(stem, "ell_file"))).ravel()
    mask_rows = read_rows(os.path.join(DATA_DIR, dataset_entry(stem, "mask_file")))
    n = len(mask_rows)
    mask = np.zeros(n, dtype=bool)
    for row in mask_rows:
        mask[int(float(row[0]))] = float(row[1]) > 0
    nbins = len(bins)
    nell = len(ells)
    n_ee = n - (nbins + len(pairs))*nell
    rows = []
    for i in range(nbins):
        entries = list(range(n_ee + i*nell, n_ee + (i + 1)*nell))
        for p, (a, b) in enumerate(pairs):
            if i in (a, b):
                start = n_ee + (nbins + p)*nell
                entries += list(range(start, start + nell))
        rows.append(np.array(entries))
    zbar = np.array([float(row[4]) for row in bins])
    return {"zbar": zbar, "pairs": pairs, "nell": nell, "n": n, "n_ee": n_ee,
            "mask": mask, "rows": rows}


def layout_covariance(layout, n):
    """The covariance of one layout as a dense n x n matrix (its cov_file)."""
    stem = LAYOUTS[layout]["dataset"]
    path = os.path.join(DATA_DIR, dataset_entry(stem, "cov_file"))
    table = np.loadtxt(path, usecols=(0, 1, 2))
    C = np.zeros((n, n))
    i = table[:, 0].astype(int)
    j = table[:, 1].astype(int)
    C[i, j] = table[:, 2]
    C[j, i] = table[:, 2]
    return C


def bias_leaders(layout):
    """leader[i] = the DESI_B1_<n> that bin i + 1 is tied to (its group)."""
    import re
    from cobaya.yaml import yaml_load_file
    params = yaml_load_file(os.path.join(PROJECT_DIR, LAYOUTS[layout]["bias_params"]))
    params = params.get("params", params)
    leader = {}
    for name, spec in params.items():
        if not name.startswith("DESI_B1_"):
            continue
        value = ""
        if isinstance(spec, dict):
            value = str(spec.get("value", ""))
        found = re.search(r"lambda\s+(DESI_B1_\d+)\s*:", value)
        if found:
            leader[name] = found.group(1)
        else:
            leader[name] = name
    out = []
    for i in range(len(leader)):
        out.append(leader[f"DESI_B1_{i + 1}"])
    return out


def bias_groups(leaders):
    """(group name, leader, bin indices) of the six bias groups, in order."""
    groups = []
    ordered = list(dict.fromkeys(leaders))
    if len(ordered) != len(BIAS_GROUP_NAMES):
        raise ValueError(f"{len(ordered)} bias groups ({ordered}); the study "
                         f"expects {len(BIAS_GROUP_NAMES)}")
    for name, leader in zip(BIAS_GROUP_NAMES, ordered):
        members = []
        for i in range(len(leaders)):
            if leaders[i] == leader:
                members.append(i)
        groups.append((name, leader, np.array(members)))
    return groups


def pair_coloring(nbins, pairs):
    """A greedy coloring of the pair graph: no pair joins two bins of one color."""
    neighbors = []
    for _ in range(nbins):
        neighbors.append(set())
    for a, b in pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)
    color = []
    for i in range(nbins):
        used = set()
        for j in neighbors[i]:
            if j < i:
                used.add(color[j])
        c = 0
        while c in used:
            c += 1
        color.append(c)
    return np.array(color)


def work_file(work, layout, config):
    """The derive stage's output file of one layout and configuration."""
    return os.path.join(work, f"derivatives_{layout}_{config}.npz")


def cuts_file(work, layout):
    """The cuts stage's output file of one layout."""
    return os.path.join(work, f"cuts_{layout}.npz")


# ---------------------------------------------------------------------------
# Stage: cuts
# ---------------------------------------------------------------------------

def run_cuts(layout, work):
    """Compute the UV-fraction cuts of one layout on its linear-bias model.

    The cut is a data selection, the same for every configuration, so it
    is computed once, at the reference settings, on the linear example
    yaml (the notebooks build the cut on the linear-bias spectra).
    """
    import cosmolike_lssty1xdesi_fisher as fm
    settings = resolve_settings(layout, "reference")
    start = time.time()
    model = fm.ModelFisher(LAYOUTS[layout]["linear_yaml"], COSMO,
                           likelihood_overrides=likelihood_overrides(settings),
                           camb_overrides=camb_overrides(settings))
    model.dv()
    cuts = fm.ci.gg_scale_cuts_tomo(np.array(CUT_KMAX, dtype=float), ALPHA_UV)
    out = {"kmax": np.array(cuts["kmax"]).ravel(), "mask": np.array(cuts["mask"]),
           "nautos": int(cuts["nautos"]), "seconds": time.time() - start}
    os.makedirs(work, exist_ok=True)
    np.savez(cuts_file(work, layout), **out)
    print(f"{layout}: cuts at {list(CUT_KMAX)} in {out['seconds']:.0f} s",
          flush=True)


# ---------------------------------------------------------------------------
# Stage: derive
# ---------------------------------------------------------------------------

def run_derive(layout, config, work):
    """Evaluate every vector one configuration's derivatives need, and save them.

    The vectors at the stencil offsets are saved, not the derivatives: the
    compare stage forms the three- and five-point rules at every step the
    offsets allow.

    Saved arrays (n the vector length, K the number of offsets):
      m0, m_after       the fiducial vector before and after every step
                        (equal bit for bit when no step survives);
      T0, T2            the SN0 and SN2 templates, (n, nbins);
      offsets           the K offsets in units of the step;
      stencil_names     COSMO and the sigma_v leaders;
      stencil_steps     their steps at offset 1;
      stencil_vectors   (n_stencil, K, n);
      op_vectors        (4, n_colors, K, n), operators in OPERATOR_STEPS order;
      op_steps          the operator steps at offset 1 (b1 relative);
      a2_vectors        (n_colors, 2, n): alpha_2 = +A2_STEP, -A2_STEP;
      colors, b1_fiducial (DESI_B1_<n> per bin), prior_names, prior_sigmas;
      settings          the resolved settings, as JSON text;
      seconds, evaluations.
    """
    import cosmolike_lssty1xdesi_fisher as fm
    # the study measures the final tier, whose settings assume CAMB with the
    # halofit tolerance patch (fm.check_halofit_patch stops here without it)
    fm.check_halofit_patch()
    settings = resolve_settings(layout, config)
    if config in ("candidate", "candidate_stencil", "candidate_ladder"):
        # the candidate is what the notebooks run: the Fisher module's
        # derivative settings, with the yamls' rank and multipole cap
        adopted = fm.derivative_settings(LAYOUTS[layout]["dataset"], "final")
        adopted = dict(adopted["likelihood_overrides"])
        adopted["nonlimber_lowrank_rank"] = MCMC_SETTINGS["rank"]
        adopted["nonlimber_lowrank_lmax"] = MCMC_SETTINGS["lowrank_lmax"]
        if likelihood_overrides(settings) != adopted or \
                camb_overrides(settings) != dict(fm.DERIVATIVE_CAMB):
            raise ValueError(
                f"the candidate settings {likelihood_overrides(settings)} "
                f"{camb_overrides(settings)} differ from the Fisher module's "
                f"{adopted} {fm.DERIVATIVE_CAMB}: make CANDIDATE_SETTINGS and "
                "fm.DERIVATIVE_OVERRIDES / fm.DERIVATIVE_CAMB agree")
    if settings["blocks"] not in ("all", "stencil"):
        raise ValueError(f"blocks = {settings['blocks']!r}: 'all' or 'stencil'")
    tables = layout_tables(layout)
    nbins = len(tables["zbar"])
    colors = pair_coloring(nbins, tables["pairs"])
    ncolors = int(colors.max()) + 1
    leaders = bias_leaders(layout)
    offsets = OFFSETS[settings["offsets"]]

    # every per-bin parameter of the operators becomes a sampled parameter
    # at its fiducial (the bin amplitude at its group's value, the rest 0)
    from cobaya.yaml import yaml_load_file
    yaml_point = yaml_load_file(LAYOUTS[layout]["model_a_yaml"])["sampler"]["evaluate"]["override"]
    sampled = {}
    per_bin_names = []
    for i in range(nbins):
        sampled[f"DESI_B1_{i + 1}"] = float(yaml_point[leaders[i]])
        for prefix in ("DESI_DB2", "DESI_DBS", "DESI_A0", "DESI_A2"):
            sampled[f"{prefix}_{i + 1}"] = 0.0
    for name in sampled:
        per_bin_names.append(name)

    start = time.time()
    model = fm.ModelFisher(LAYOUTS[layout]["model_a_yaml"], COSMO + per_bin_names,
                           extra_prefixes=("DESI_SIGV_",),
                           likelihood_overrides=likelihood_overrides(settings),
                           invcov_scale=INVCOV_SCALE,
                           camb_overrides=camb_overrides(settings),
                           sampled_overrides=sampled)
    like = model.model.likelihood[model.likelihood_name]
    index = {}
    for k, name in enumerate(model.names):
        index[name] = k
    fiducial = np.array(model.fiducial, dtype=float)
    stencil_names = list(COSMO)
    for name in model.names:
        if name.startswith("DESI_SIGV_"):
            stencil_names.append(name)
    sigmas = model.prior_sigmas()
    prior_names = []
    prior_values = []
    for k, sigma in sigmas.items():
        prior_names.append(model.names[k])
        prior_values.append(sigma)
    count = [0]

    def evaluate(param):
        count[0] += 1
        return model.dv(param=param)

    m0 = evaluate(fiducial)
    # the templates read the kernels of the last evaluation: the fiducial
    T0 = like.stochastic_templates(order=0)
    T2 = like.stochastic_templates(order=2)
    n = len(m0)
    if n != layout_tables(layout)["n"]:
        raise ValueError(f"{layout}: the vector has {n} entries, the paper dataset "
                         f"{LAYOUTS[layout]['dataset']} lays out {layout_tables(layout)['n']}")

    # the stencil columns: relative steps h p0, absolute h s for a zero
    # fiducial (s = 1 for the flat wa; the sigma_v leaders are nonzero)
    stencil_steps = []
    stencil_vectors = np.zeros((len(stencil_names), len(offsets), n))
    for j, name in enumerate(stencil_names):
        k = index[name]
        p0 = fiducial[k]
        step = H_MAIN*abs(p0) if p0 != 0.0 else H_MAIN*model.zero_scale[k]
        stencil_steps.append(step)
        for o, offset in enumerate(offsets):
            param = fiducial.copy()
            param[k] = p0 + offset*step
            stencil_vectors[j, o] = evaluate(param)
        print(f"  {name}: {time.time() - start:.0f} s", flush=True)

    op_vectors = np.zeros((len(OPERATOR_STEPS), ncolors, len(offsets), n))
    a2_vectors = np.zeros((ncolors, 2, n))
    if settings["blocks"] == "all":
        for q, (op, op_step) in enumerate(OPERATOR_STEPS.items()):
            for c in range(ncolors):
                members = np.where(colors == c)[0]
                for o, offset in enumerate(offsets):
                    param = fiducial.copy()
                    for i in members:
                        k = index[f"{OPERATOR_PREFIX[op]}_{i + 1}"]
                        if op == "b1":
                            param[k] = fiducial[k]*(1.0 + offset*op_step)
                        else:
                            param[k] = fiducial[k] + offset*op_step
                    op_vectors[q, c, o] = evaluate(param)
            print(f"  operator {op}: {time.time() - start:.0f} s", flush=True)
        # alpha_2 enters the vector linearly: the central difference is exact
        for c in range(ncolors):
            members = np.where(colors == c)[0]
            for s, sign in enumerate((1.0, -1.0)):
                param = fiducial.copy()
                for i in members:
                    param[index[f"DESI_A2_{i + 1}"]] = sign*A2_STEP
                a2_vectors[c, s] = evaluate(param)
        print(f"  alpha_2: {time.time() - start:.0f} s", flush=True)
    m_after = evaluate(fiducial)
    seconds = time.time() - start

    b1_fiducial = np.zeros(nbins)
    for i in range(nbins):
        b1_fiducial[i] = fiducial[index[f"DESI_B1_{i + 1}"]]
    os.makedirs(work, exist_ok=True)
    np.savez(work_file(work, layout, config),
             m0=m0, m_after=m_after, T0=T0, T2=T2,
             offsets=np.array(offsets),
             stencil_names=np.array(stencil_names),
             stencil_steps=np.array(stencil_steps),
             stencil_vectors=stencil_vectors,
             op_names=np.array(list(OPERATOR_STEPS)),
             op_steps=np.array(list(OPERATOR_STEPS.values())),
             op_vectors=op_vectors, a2_vectors=a2_vectors,
             a2_step=A2_STEP, colors=colors, b1_fiducial=b1_fiducial,
             prior_names=np.array(prior_names), prior_sigmas=np.array(prior_values),
             settings=json.dumps(settings), seconds=seconds,
             evaluations=count[0])
    reset = float(np.max(np.abs(m_after - m0)))
    print(f"{layout} {config}: {count[0]} evaluations in {seconds:.0f} s "
          f"({seconds/count[0]:.1f} s each); max |m after - m0| = {reset:.1e}",
          flush=True)


# ---------------------------------------------------------------------------
# Stage: dkit
# ---------------------------------------------------------------------------

# The DerivKit columns: the stencil block and two per-bin operator families
# (the bin amplitude DESI_B1 and the b2 offset DESI_DB2), the latter on the
# same coloring of the pair graph as the five-point columns.
DKIT_FAMILIES = ("b1", "b2")
DKIT_MIN_SAMPLES = 7
DKIT_FALLBACK = "poly_at_floor"


def dkit_file(work, layout, config):
    """The dkit stage's output file of one layout and configuration."""
    return os.path.join(work, f"derivkit_{layout}_{config}.npz")


def run_dkit(layout, config, work):
    """DerivKit derivatives of one configuration, for the comparison stage.

    DerivKit (external_modules/code/derivkit, installed by Cocoa) samples
    the vector at the center and at +-1, 1.5, 2.25, 3.4 and 5% of the
    central value, fits a straight line weighted by the inverse distance to
    the center, prunes the points of largest relative residual down to
    min_samples, and returns the slope; the calls and defaults are those of
    cosmolike_notebook_utils.get_ddv_dkit (min_samples 7, fallback_mode
    "poly_at_floor"), the ones the roman_real Fisher notebook uses. A
    parameter whose fiducial is zero (wa, the b2 offsets) would get
    absolute steps of 1e-6 to 4e-5 there, far below the noise of the
    vector, so each is sampled through a shifted variable u = 1 + (p -
    p0)/s with s = 1: the steps are 0.01 to 0.05 in p, the range of the
    five-point steps. A per-bin family moves every bin of one color by the
    same amount (the bin amplitudes by one relative factor).

    Saved arrays: 'stencil_names', 'stencil' (n, n_stencil) and, per
    family, (n, nbins) columns, plus the evaluations and seconds of every
    column and the resolved settings.
    """
    import cosmolike_lssty1xdesi_fisher as fm
    from cosmolike_notebook_utils import fisher as cnu
    settings = resolve_settings(layout, config)
    tables = layout_tables(layout)
    nbins = len(tables["zbar"])
    colors = pair_coloring(nbins, tables["pairs"])
    ncolors = int(colors.max()) + 1
    leaders = bias_leaders(layout)
    from cobaya.yaml import yaml_load_file
    yaml_point = yaml_load_file(LAYOUTS[layout]["model_a_yaml"])["sampler"]["evaluate"]["override"]
    sampled = {}
    for i in range(nbins):
        sampled[f"DESI_B1_{i + 1}"] = float(yaml_point[leaders[i]])
        for prefix in ("DESI_DB2", "DESI_DBS", "DESI_A0", "DESI_A2"):
            sampled[f"{prefix}_{i + 1}"] = 0.0
    model = fm.ModelFisher(LAYOUTS[layout]["model_a_yaml"], COSMO + list(sampled),
                           extra_prefixes=("DESI_SIGV_",),
                           likelihood_overrides=likelihood_overrides(settings),
                           invcov_scale=INVCOV_SCALE,
                           camb_overrides=camb_overrides(settings),
                           sampled_overrides=sampled)
    index = {}
    for k, name in enumerate(model.names):
        index[name] = k
    fiducial = np.array(model.fiducial, dtype=float)
    stencil_names = list(COSMO)
    for name in model.names:
        if name.startswith("DESI_SIGV_"):
            stencil_names.append(name)
    count = [0]

    def dv_at(param, AccuracyBoost=1.0):
        # cnu.get_ddv_dkit's calling convention; the model's accuracy is set
        # by its settings, so AccuracyBoost is accepted and ignored
        count[0] += 1
        return model.dv(param=param)

    m0 = dv_at(fiducial)
    n = len(m0)
    if n != layout_tables(layout)["n"]:
        raise ValueError(f"{layout}: the vector has {n} entries, the paper dataset "
                         f"{LAYOUTS[layout]['dataset']} lays out {layout_tables(layout)['n']}")
    out = {"stencil_names": np.array(stencil_names), "m0": m0}
    evaluations = {}
    seconds = {}
    stencil = np.zeros((n, len(stencil_names)))
    for j, name in enumerate(stencil_names):
        k = index[name]
        start = time.time()
        before = count[0]
        if fiducial[k] != 0.0:
            stencil[:, j] = cnu.get_ddv_dkit(dv_at, index=k, CV=fiducial,
                                             min_samples=DKIT_MIN_SAMPLES,
                                             fallback_mode=DKIT_FALLBACK)
        else:
            # the shifted variable u = 1 + p (s = 1): central value 1
            def shifted(param, AccuracyBoost=1.0, k=k):
                p = np.array(param, dtype=float, copy=True)
                p[k] = p[k] - 1.0
                return dv_at(p)
            moved = fiducial.copy()
            moved[k] = 1.0
            stencil[:, j] = cnu.get_ddv_dkit(shifted, index=k, CV=moved,
                                             min_samples=DKIT_MIN_SAMPLES,
                                             fallback_mode=DKIT_FALLBACK)
        evaluations[name] = count[0] - before
        seconds[name] = time.time() - start
        print(f"  {name}: {evaluations[name]} evaluations, {seconds[name]:.0f} s", flush=True)
    out["stencil"] = stencil
    for family in DKIT_FAMILIES:
        block = np.zeros((n, nbins))
        for c in range(ncolors):
            members = np.where(colors == c)[0]
            keys = [index[f"{OPERATOR_PREFIX[family]}_{i + 1}"] for i in members]
            start = time.time()
            before = count[0]

            def color_dv(param, AccuracyBoost=1.0, keys=keys, family=family):
                # param[0] is the one-dimensional DerivKit variable: the
                # relative factor of the bin amplitudes, or 1 + the b2 offset
                p = fiducial.copy()
                for k in keys:
                    if family == "b1":
                        p[k] = fiducial[k]*param[0]
                    else:
                        p[k] = param[0] - 1.0
                return dv_at(p)

            d = cnu.get_ddv_dkit(color_dv, index=0, CV=np.array([1.0]),
                                 min_samples=DKIT_MIN_SAMPLES, fallback_mode=DKIT_FALLBACK)
            for i in members:
                rows = tables["rows"][i]
                scale = fiducial[index[f"{OPERATOR_PREFIX[family]}_{i + 1}"]] \
                    if family == "b1" else 1.0
                block[rows, i] = d[rows]/scale
            label = f"{family} color {c}"
            evaluations[label] = count[0] - before
            seconds[label] = time.time() - start
            print(f"  {label}: {evaluations[label]} evaluations, {seconds[label]:.0f} s",
                  flush=True)
        out[family] = block
    out["evaluations"] = json.dumps(evaluations)
    out["seconds"] = json.dumps(seconds)
    out["settings"] = json.dumps(settings)
    os.makedirs(work, exist_ok=True)
    np.savez(dkit_file(work, layout, config), **out)
    total = sum(evaluations.values())
    print(f"{layout} {config} DerivKit: {total} evaluations in "
          f"{sum(seconds.values()):.0f} s", flush=True)


# ---------------------------------------------------------------------------
# Stage: compare
# ---------------------------------------------------------------------------

def stencil_rule(vectors, offsets, rule, scale):
    """One derivative from stored stencil vectors.

    Arguments:
      vectors = (K, n) array at the offsets (in units of the step).
      offsets = the K offsets.
      rule    = "five" or "three".
      scale   = the multiple of the step the rule uses (0.5, 1, 2 or 4).

    Returns:
      (n,) array: the derivative per unit offset; divide by the step.

    Raises:
      KeyError when the offsets lack a point the rule needs.
    """
    where = {}
    for o, offset in enumerate(offsets):
        where[float(offset)] = o
    if rule == "five":
        points = ((2.0*scale, -1.0/12.0), (scale, 8.0/12.0),
                  (-scale, -8.0/12.0), (-2.0*scale, 1.0/12.0))
    elif rule == "three":
        points = ((scale, 0.5), (-scale, -0.5))
    else:
        raise KeyError(f"rule {rule!r}: 'five' or 'three'")
    total = np.zeros(vectors.shape[1])
    for offset, weight in points:
        if offset not in where:
            raise KeyError(f"the {rule}-point rule at {scale} steps needs the "
                           f"offset {offset}; stored: {list(offsets)}")
        total += weight*vectors[where[offset]]
    return total/scale


def base_columns(data, tables, rule="five", scale=1.0, fallback=None):
    """The base columns of one configuration under one stencil rule and step.

    Returns:
      dict of arrays: 'stencil' (n, n_stencil), 'b1', 'b2', 'bs2', 'a0',
      'a2', 'T0', 'T2' (each (n, nbins)): dm/dp per unit of the parameter.
      A configuration whose blocks are "stencil" takes the per-bin blocks
      from fallback (the reference's columns).

    Raises:
      ValueError when the stored vectors were made on another layout of the
      dataset (a pair list changed since: the arrays are stale).
    """
    if len(data["m0"]) != tables["n"]:
        raise ValueError(f"stored vectors of {len(data['m0'])} entries, but the dataset now "
                         f"lays out {tables['n']}: these arrays were made on an older "
                         "layout (stale); rerun the cuts and derive stages")
    offsets = list(data["offsets"])
    columns = {}
    vectors = data["stencil_vectors"]
    steps = data["stencil_steps"]
    stencil = np.zeros((vectors.shape[2], vectors.shape[0]))
    for j in range(vectors.shape[0]):
        stencil[:, j] = stencil_rule(vectors[j], offsets, rule, scale)/steps[j]
    columns["stencil"] = stencil
    settings = json.loads(str(data["settings"]))
    if settings["blocks"] == "stencil":
        if fallback is None:
            raise ValueError("a stencil-only configuration needs the "
                             "reference's per-bin columns")
        for block in ("b1", "b2", "bs2", "a0", "a2", "T0", "T2"):
            columns[block] = fallback[block]
        return columns
    colors = data["colors"]
    nbins = len(colors)
    n = vectors.shape[2]
    b1_fiducial = data["b1_fiducial"]
    for q, op in enumerate(data["op_names"]):
        op = str(op)
        block = np.zeros((n, nbins))
        for c in range(int(colors.max()) + 1):
            d = stencil_rule(data["op_vectors"][q, c], offsets, rule, scale)
            for i in np.where(colors == c)[0]:
                rows = tables["rows"][i]
                step = data["op_steps"][q]
                if op == "b1":
                    step = step*b1_fiducial[i]
                block[rows, i] = d[rows]/step
        columns[op] = block
    a2 = np.zeros((n, nbins))
    for c in range(int(colors.max()) + 1):
        d = (data["a2_vectors"][c, 0] - data["a2_vectors"][c, 1])/(2.0*float(data["a2_step"]))
        for i in np.where(colors == c)[0]:
            rows = tables["rows"][i]
            a2[rows, i] = d[rows]
    columns["a2"] = a2
    columns["T0"] = np.asarray(data["T0"])
    columns["T2"] = np.asarray(data["T2"])
    return columns


def stencil_noise(data, whiten):
    """The numerical noise of each stencil parameter's vectors, per point.

    Through the five equally spaced points p0 + (-2, -1, 0, 1, 2) d, the
    fourth difference f(-2) - 4 f(-1) + 6 f(0) - 4 f(1) + f(2) of a smooth
    vector is of order d^4 times its fourth derivative, negligible at
    these steps; noise that is independent from point to point gives it a
    variance 1 + 16 + 36 + 16 + 1 = 70 times that of one point. So
    |whitened fourth difference|/sqrt(70) estimates the noise of one
    evaluation in sqrt(chi^2) units (whitened by the shipped-mask
    covariance at the survey area), the jitter a derivative divides by
    its step.

    Arguments:
      data   = a derive stage's arrays.
      whiten = function mapping a full vector to its whitened kept entries.

    Returns:
      {stencil parameter: noise per point}.
    """
    where = {}
    for o, offset in enumerate(data["offsets"]):
        where[float(offset)] = o
    weights = ((-2.0, 1.0), (-1.0, -4.0), (1.0, -4.0), (2.0, 1.0))
    out = {}
    for j, name in enumerate(data["stencil_names"]):
        total = 6.0*np.asarray(data["m0"], dtype=float)
        for offset, weight in weights:
            total = total + weight*data["stencil_vectors"][j, where[offset]]
        out[str(name)] = float(np.linalg.norm(whiten(total))/np.sqrt(70.0))
    return out


def prior_widths(model, kmax):
    """The Gaussian widths of one model's per-group families at a cut."""
    widths = dict(FIXED_WIDTHS[model])
    widths["A0"] = COUNTERTERM_FRACTION/kmax**2
    if model == "B":
        widths["A2"] = COUNTERTERM_FRACTION/kmax**2
    return widths


def group_bias_precision(zbar, groups, fiducial_amplitude, L, sigma_by_group):
    """The memory-model prior precision on the bin amplitudes.

    Per group, the projected precision P = Q - Q U (U^T Q U)^-1 U^T Q, Q
    the inverse of the kernel width^2 exp(-|z_i - z_j|/L), U = [1, z -
    zbar_g]: it charges only the residual around the group's line (the
    model B notebook's group_bias_precision with a free tilt).
    """
    P = np.zeros((len(zbar), len(zbar)))
    for name, leader, idx in groups:
        z = zbar[idx]
        U = np.column_stack([np.ones(len(z)), z - z.mean()])
        if len(idx) <= U.shape[1]:
            continue
        width = sigma_by_group[name]*fiducial_amplitude[leader]
        Q = np.linalg.inv(width**2*np.exp(-np.abs(z[:, None] - z[None, :])/L))
        QU = Q @ U
        P[np.ix_(idx, idx)] = Q - QU @ np.linalg.inv(U.T @ QU) @ QU.T
    return P


def stable_inverse(F):
    """The pseudo-inverse of a Fisher matrix scaled to unit diagonal.

    Directions below PINV_RTOL of the largest eigenvalue carry no
    information and are dropped (the notebooks' stable_inverse).

    Returns:
      (F_inv, null): null holds the dropped directions (scaled basis).
    """
    s = 1.0/np.sqrt(np.diag(F))
    w, V = np.linalg.eigh(F*s[:, None]*s[None, :])
    keep = w > PINV_RTOL*w.max()
    F_inv = (s[:, None]*V[:, keep]) @ np.diag(1.0/w[keep]) @ (V[:, keep].T*s[None, :])
    return F_inv, V[:, ~keep]


def system_design(columns, system, groups, zbar, stencil_names, prior_sigma):
    """The derivatives D, names and prior tags of one system.

    The order and the columns are the model B notebook's variant_columns:
    the cosmology and the sigma_v leaders, the per-group families (and
    their tilts with the tilted variant), then one amplitude per bin.

    Returns:
      (D, names, tags): D = (n, n_par); tags[j] = ('flat',), ('gauss',
      sigma) or ('family', family).
    """
    model = system["model"]
    D = []
    names = []
    tags = []
    for j, name in enumerate(stencil_names):
        D.append(columns["stencil"][:, j])
        names.append(name)
        if name in prior_sigma:
            tags.append(("gauss", prior_sigma[name]))
        else:
            tags.append(("flat",))
    for family in MODEL_FAMILIES[model]:
        block = columns[FAMILY_BLOCK[family]]
        for name, leader, idx in groups:
            D.append(block[:, idx].sum(axis=1))
            names.append(f"DESI_{family}_{name}")
            tags.append(("family", family))
            if system["tilts"] and family in TILTED and len(idx) > 1:
                weights = zbar[idx] - zbar[idx].mean()
                D.append(block[:, idx] @ weights)
                names.append(f"DESI_T{family}_{name}")
                tags.append(("flat",))
    for i in range(columns["b1"].shape[1]):
        D.append(columns["b1"][:, i])
        names.append(f"DESI_B1_{i + 1}")
        tags.append(("flat",))
    return np.column_stack(D), names, tags


def system_truths(columns, system, groups, zbar, kmax):
    """{truth: delta m} of the break test at one cut (notebook section 7)."""
    model = system["model"]
    widths = FIXED_WIDTHS[model]
    if model == "A":
        alpha2 = ALPHA2_TRUTH_MODEL_A
    else:
        alpha2 = prior_widths("B", kmax)["A2"]
    shape = np.zeros(len(zbar))
    for name, leader, idx in groups:
        if len(idx) > 1:
            z = zbar[idx]
            shape[idx] = (z - z.mean())/(z.max() - z.min())
    truths = {
        "alpha2": alpha2*columns["a2"].sum(axis=1),
        "SN2": SN2_TRUTH*columns["T2"].sum(axis=1),
        "offsets": (widths["DB2"]*columns["b2"].sum(axis=1)
                    + widths["DBS"]*columns["bs2"].sum(axis=1)),
        "calibration drift": (columns["b2"] @ (CAL_DRIFT["b2"]*shape)
                              + columns["bs2"] @ (CAL_DRIFT["bs2"]*shape)),
    }
    return truths


def score_system(columns, system, kmax, context):
    """The errors, FoMs and break-test scores of one system at one cut.

    Returns:
      dict: 'sigma' ({w0, wa, omegam, As_1e9}), 'fom_w0wa', 'fom_om_as',
      'scores' ({truth: {'w0wa', 'omegam', 'As_1e9'}}), 'largest_shift',
      'dropped'; or {'status': reason} when the cut gives no fit.
    """
    tables = context["tables"]
    groups = context["groups"]
    zbar = tables["zbar"]
    D, names, tags = system_design(columns, system, groups, zbar,
                                   context["stencil_names"], context["prior_sigma"])
    m = list(CUT_KMAX).index(kmax)
    kept = context["kept"][m]
    shipped = context["kept"]["shipped"]
    Cinv_kept = context["inverse"][m]
    Cinv_shipped = context["inverse"]["shipped"]
    Dk = D[kept]
    F_data = Dk.T @ Cinv_kept @ Dk
    widths = prior_widths(system["model"], kmax)
    prior = np.zeros(len(names))
    for j, tag in enumerate(tags):
        if tag[0] == "gauss":
            prior[j] = 1.0/tag[1]**2
        elif tag[0] == "family":
            prior[j] = 1.0/widths[tag[1]]**2
    F = F_data + np.diag(prior)
    nbins = len(zbar)
    nx = len(names) - nbins
    spec = BIAS_PRIORS[system["prior"]]
    F[nx:, nx:] += group_bias_precision(zbar, groups, context["amplitude"],
                                        spec["L"], spec["sigma_b"])
    Ds = D[shipped]
    reference = np.einsum("ij,ij->j", Ds, Cinv_shipped @ Ds) + np.diag(F) - np.diag(F_data)
    live = []
    for j in range(len(names)):
        if F[j, j] > NO_DATA_RTOL*reference[j]:
            live.append(j)
    used = [names[j] for j in live]
    for p in COSMO:
        if p not in used:
            return {"status": "cosmology without data"}
    F_inv, null = stable_inverse(F[np.ix_(live, live)])
    iw = [used.index("w0"), used.index("wa")]
    ip = {}
    for p in SINGLE_PARAMS:
        ip[p] = used.index(p)
    scored = iw + list(ip.values())
    overlap = float(np.max(np.abs(null[scored]))) if null.shape[1] else 0.0
    if overlap > NULL_OVERLAP_MAX:
        return {"status": "scored parameters unconstrained"}
    Cw = F_inv[np.ix_(iw, iw)]
    Ci = np.linalg.inv(Cw)
    i_om_as = [ip["omegam"], ip["As_1e9"]]
    out = {
        "sigma": {"w0": float(np.sqrt(F_inv[iw[0], iw[0]])),
                  "wa": float(np.sqrt(F_inv[iw[1], iw[1]])),
                  "omegam": float(np.sqrt(F_inv[ip["omegam"], ip["omegam"]])),
                  "As_1e9": float(np.sqrt(F_inv[ip["As_1e9"], ip["As_1e9"]]))},
        "fom_w0wa": float(1.0/np.sqrt(np.linalg.det(Cw))),
        "fom_om_as": float(1.0/np.sqrt(np.linalg.det(F_inv[np.ix_(i_om_as, i_om_as)]))),
        "dropped": int(null.shape[1]),
        "scores": {},
    }
    truths = system_truths(columns, system, groups, zbar, kmax)
    largest = 0.0
    for truth, delta in truths.items():
        B = Dk[:, live].T @ (Cinv_kept @ delta[kept])
        dp = F_inv @ B
        dw = dp[iw]
        score = {"w0wa": float(np.sqrt(dw @ Ci @ dw))}
        for p in SINGLE_PARAMS:
            score[p] = float(abs(dp[ip[p]])/out["sigma"][p])
        out["scores"][truth] = score
        largest = max(largest, max(score.values()))
    out["largest_shift"] = largest
    return out


def compare_to(result, reference):
    """The changes of one system's numbers against the reference's.

    Returns:
      dict: 'sigma' (max |sigma/sigma_ref - 1|), 'fom' (max over both FoMs
      of |FoM/FoM_ref - 1|), 'shift' (max |score - score_ref| over truths
      and scores, in sigma), and 'converged' (all three within tolerance).
    """
    sigma = 0.0
    for p, value in result["sigma"].items():
        sigma = max(sigma, abs(value/reference["sigma"][p] - 1.0))
    fom = max(abs(result["fom_w0wa"]/reference["fom_w0wa"] - 1.0),
              abs(result["fom_om_as"]/reference["fom_om_as"] - 1.0))
    shift = 0.0
    for truth, scores in result["scores"].items():
        for score, value in scores.items():
            shift = max(shift, abs(value - reference["scores"][truth][score]))
    return {"sigma": sigma, "fom": fom, "shift": shift,
            "converged": bool(sigma < SIGMA_TOL and fom < FOM_TOL and shift < SHIFT_TOL)}


# the rules and step multiples the compare stage forms when the stored
# offsets allow them
RULE_STEPS = (("three", 0.5), ("three", 1.0), ("three", 2.0), ("three", 3.0),
              ("three", 4.0), ("three", 6.0), ("five", 0.5), ("five", 1.0),
              ("five", 2.0), ("five", 3.0))


def production_perbin(chains, layout, leaders):
    """The per-bin columns of the notebooks' caches, in this study's blocks.

    The model A notebook's derivative cache holds the partial columns of
    b1, b2, bs2 and alpha_0 and the SN0 templates; the model B notebook's
    template cache the alpha_2 columns and the SN2 templates. The bin
    amplitude column is the model A chain rule, (b1/B)[dm/db1 + b2L'(b1)
    dm/db2 - 4/7 dm/dbs2], b2L' the slope of the Lazeyras fit.

    Returns:
      dict: 'b1', 'b2', 'bs2', 'a0', 'a2', 'T0', 'T2' (each (n, nbins)).
    """
    import pickle
    number = list(LAYOUTS).index(layout) + 1
    with open(os.path.join(chains, f"EXAMPLE_EVALUATE_FISHER_MODEL_A_DERIVATIVES_{number}.pkl"),
              "rb") as handle:
        derivs = pickle.load(handle)["derivs"]
    with open(os.path.join(chains, f"EXAMPLE_EVALUATE_FISHER_MODEL_B_TEMPLATES_{number}.pkl"),
              "rb") as handle:
        templates = pickle.load(handle)["templates"]
    fiducial = dict(zip(derivs["names"], derivs["fiducial"]))
    b1 = derivs["b1"]
    amplitude = np.array([fiducial[leader] for leader in leaders])
    slope = -2.143 + 2.0*0.929*b1 + 3.0*0.008*b1**2
    ops = derivs["ops"]
    total = (ops["b1"] + ops["b2"]*slope[None, :] - 4.0/7.0*ops["bs2"])*(b1/amplitude)[None, :]
    return {"b1": total, "b2": ops["b2"], "bs2": ops["bs2"], "a0": ops["a0"],
            "a2": templates["a2"], "T0": derivs["T"], "T2": templates["T2"]}


def run_compare(layout, work, results, base=None, perbin_from=None):
    """Score every stored configuration of one layout against a base.

    Arguments:
      layout, work, results = as on the command line.
      base        = the configuration every other is compared with; None
                    for the reference (or reference_five).
      perbin_from = a chains folder whose notebook caches give the per-bin
                    columns of stencil-only configurations (production_perbin),
                    needed when the base itself is stencil-only.
    """
    import scipy.linalg
    tables = layout_tables(layout)
    leaders = bias_leaders(layout)
    groups = bias_groups(leaders)
    if base is None:
        reference_config = "reference"
        if not os.path.exists(work_file(work, layout, reference_config)):
            reference_config = "reference_five"
    else:
        reference_config = base
    reference_path = work_file(work, layout, reference_config)
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"{reference_path} is missing: run the derive stage "
                                f"with --config {reference_config} first")
    reference = np.load(reference_path)
    perbin = None
    if perbin_from is not None:
        perbin = production_perbin(perbin_from, layout, leaders)
    elif CONFIGS[reference_config]["blocks"] == "stencil":
        # a stencil-only base takes its per-bin columns from the reference
        # (as every stencil-only configuration does), or from the notebooks'
        # caches when --perbin-from names them
        full = None
        for name in ("reference", "reference_five"):
            if os.path.exists(work_file(work, layout, name)):
                full = name
                break
        if full is None:
            raise ValueError(f"the base {reference_config} holds the stencil block only "
                             "and no reference exists: pass --perbin-from <chains folder> "
                             "for its per-bin columns")
        perbin = base_columns(np.load(work_file(work, layout, full)), tables)
    cuts = np.load(cuts_file(work, layout))
    if cuts["mask"].shape[0] != len(tables["zbar"]) + len(tables["pairs"]):
        raise ValueError(f"{cuts_file(work, layout)} cuts {cuts['mask'].shape[0]} spectra, "
                         f"the dataset now has {len(tables['zbar']) + len(tables['pairs'])}: "
                         "it was made on an older layout (stale); rerun the cuts stage")
    if not np.allclose(cuts["kmax"], CUT_KMAX):
        raise ValueError(f"{cuts_file(work, layout)} holds the cuts {cuts['kmax']}, "
                         f"not {CUT_KMAX}: rerun the cuts stage")
    C = layout_covariance(layout, tables["n"])
    kept = {"shipped": np.where(tables["mask"])[0]}
    for m in range(len(CUT_KMAX)):
        mask = tables["mask"].copy()
        mask[tables["n_ee"]:] &= cuts["mask"][:, :, m].ravel() > 0
        kept[m] = np.where(mask)[0]
    inverse = {}
    for m, rows in kept.items():
        if m != "shipped" and CUT_KMAX[m] not in SCORED_KMAX:
            continue
        inverse[m] = scipy.linalg.inv(C[np.ix_(rows, rows)])*INVCOV_SCALE
    # the whitening of the noise estimate: C = L L^T on the shipped mask
    # (at the survey area), so |L^-1 x| is sqrt(x^T C^-1 x)
    shipped_cholesky = scipy.linalg.cholesky(
        C[np.ix_(kept["shipped"], kept["shipped"])]/INVCOV_SCALE, lower=True)
    del C

    def whiten(vector):
        return scipy.linalg.solve_triangular(shipped_cholesky,
                                             vector[kept["shipped"]], lower=True)
    amplitude = {}
    for i, leader in enumerate(leaders):
        amplitude[leader] = float(reference["b1_fiducial"][i])
    prior_sigma = {}
    for name, sigma in zip(reference["prior_names"], reference["prior_sigmas"]):
        prior_sigma[str(name)] = float(sigma)
    context = {"tables": tables, "groups": groups, "kept": kept, "inverse": inverse,
               "amplitude": amplitude, "prior_sigma": prior_sigma,
               "stencil_names": [str(x) for x in reference["stencil_names"]]}

    # the reference's own rules and steps, then every configuration at the
    # five-point rule and step 1
    # every configuration at the five-point rule and step d, then every other
    # rule and step its offsets allow (three-point at d/2 to 6d, five-point
    # at d/2 to 3d)
    variants = {}
    for config in CONFIGS:
        if config == "reference_five" and reference_config == "reference":
            continue
        if not os.path.exists(work_file(work, layout, config)):
            continue
        label = config
        variants[label] = (config, "five", 1.0)
        stored = OFFSETS[CONFIGS[config]["offsets"]]
        for rule, scale in RULE_STEPS:
            if rule == "five" and scale == 1.0:
                continue
            needed = [scale, -scale]
            if rule == "five":
                needed += [2.0*scale, -2.0*scale]
            if all(x in stored for x in needed):
                variants[f"{label}, {rule}-point, step x{scale:g}"] = (config, rule, scale)

    reference_columns = base_columns(reference, tables, fallback=perbin)
    # DerivKit variants: the five-point columns of the same configuration
    # with the stencil block and the DKIT_FAMILIES replaced by DerivKit's
    dkit_variants = {}
    for config in CONFIGS:
        if os.path.exists(dkit_file(work, layout, config)) and \
                os.path.exists(work_file(work, layout, config)):
            dkit_variants[f"DerivKit at {config}"] = config
    summary = {"layout": LAYOUTS[layout]["label"], "base": reference_config, "tolerance": {
        "sigma": SIGMA_TOL, "fom": FOM_TOL, "shift_sigma": SHIFT_TOL},
        "configurations": {}, "systems": list(SYSTEMS)}
    loaded = {reference_config: reference}
    for label, (config, rule, scale) in list(variants.items()) + \
            [(k, (v, "derivkit", 1.0)) for k, v in dkit_variants.items()]:
        if config not in loaded:
            loaded[config] = np.load(work_file(work, layout, config))
        data = loaded[config]
        if rule == "derivkit":
            columns = base_columns(data, tables, fallback=reference_columns)
            dkit = np.load(dkit_file(work, layout, config))
            if [str(x) for x in dkit["stencil_names"]] != [str(x) for x in data["stencil_names"]]:
                raise ValueError(f"{dkit_file(work, layout, config)}: other stencil names")
            columns_five = dict(columns)
            columns = dict(columns)
            columns["stencil"] = dkit["stencil"]
            for family in DKIT_FAMILIES:
                columns[family] = dkit[family]
            # the C^-1-weighted difference of every DerivKit column against
            # the five-point column of the same settings, in units of the
            # latter, on the shipped mask
            column_change = {}
            for j, name in enumerate(dkit["stencil_names"]):
                a = whiten(columns["stencil"][:, j])
                b = whiten(columns_five["stencil"][:, j])
                column_change[str(name)] = float(np.linalg.norm(a - b)/np.linalg.norm(b))
            for family in DKIT_FAMILIES:
                worst = 0.0
                for i in range(columns[family].shape[1]):
                    b = whiten(columns_five[family][:, i])
                    if np.linalg.norm(b) > 0.0:
                        a = whiten(columns[family][:, i])
                        worst = max(worst, float(np.linalg.norm(a - b)/np.linalg.norm(b)))
                column_change[f"{family} (largest bin)"] = worst
        else:
            columns = base_columns(data, tables, rule=rule, scale=scale,
                                   fallback=reference_columns)
        live = reference["m0"] != 0.0
        delta = (data["m0"] - reference["m0"])[kept["shipped"]]
        entry = {"config": config, "rule": rule, "step": scale,
                 "settings": json.loads(str(data["settings"])),
                 "evaluations": int(data["evaluations"]),
                 "seconds": float(data["seconds"]),
                 "fiducial_dchi2_shipped": float(delta @ inverse["shipped"] @ delta),
                 "max_rel_fiducial_change": float(np.max(np.abs(
                     data["m0"][live]/reference["m0"][live] - 1.0))),
                 "noise_per_point": stencil_noise(data, whiten),
                 "systems": {}}
        if rule == "derivkit":
            dkit = np.load(dkit_file(work, layout, config))
            entry["derivkit_column_change"] = column_change
            entry["derivkit_evaluations"] = json.loads(str(dkit["evaluations"]))
            entry["derivkit_seconds"] = json.loads(str(dkit["seconds"]))
        for name, system in SYSTEMS.items():
            entry["systems"][name] = {}
            for kmax in system["kmax"]:
                entry["systems"][name][f"{kmax:g}"] = score_system(columns, system, kmax, context)
        summary["configurations"][label] = entry
    # every configuration against the reference ("change") and, when the
    # candidate exists, against the candidate ("change_vs_candidate"): the
    # DerivKit and MCMC comparisons are made with the converged columns
    baselines = {"change": reference_config}
    if "candidate" in summary["configurations"]:
        baselines["change_vs_candidate"] = "candidate"
    for field, baseline in baselines.items():
        base_entry = summary["configurations"][baseline]
        for label, entry in summary["configurations"].items():
            for name in entry["systems"]:
                for kmax, result in entry["systems"][name].items():
                    ref = base_entry["systems"][name][kmax]
                    if "status" in result or "status" in ref:
                        continue
                    result[field] = compare_to(result, ref)
    os.makedirs(results, exist_ok=True)
    name = f"fisher_derivatives_{layout}.json"
    if reference_config not in ("reference", "reference_five"):
        name = f"fisher_derivatives_{layout}_vs_{reference_config}.json"
    path = os.path.join(results, name)
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=1)
    print_table(summary)
    print(f"wrote {path}")


def print_table(summary):
    """Print each configuration's changes against the reference."""
    print(f"{summary['layout']}: changes against {summary['base']} (tolerance: sigma "
          f"{SIGMA_TOL:.1%}, FoM {FOM_TOL:.0%}, shifts {SHIFT_TOL} sigma)")
    for name in summary["systems"]:
        for label, entry in summary["configurations"].items():
            line = f"  {name:30s} {label:42s}"
            for kmax, result in entry["systems"][name].items():
                if "change" not in result:
                    line += f" | k {kmax}: {result.get('status', 'reference')}"
                    if "fom_w0wa" in result:
                        line += (f" FoM {result['fom_w0wa']:.4g}/{result['fom_om_as']:.4g},"
                                 f" largest shift {result['largest_shift']:.3f}")
                    continue
                c = result["change"]
                flag = "ok" if c["converged"] else "MOVES"
                line += (f" | k {kmax}: sigma {c['sigma']:.2%} FoM {c['fom']:.2%} "
                         f"shift {c['shift']:.3f} {flag}")
            print(line)
    for label, entry in summary["configurations"].items():
        if "derivkit_column_change" not in entry:
            continue
        changes = entry["derivkit_column_change"]
        count = sum(entry["derivkit_evaluations"].values())
        print(f"{label}: {count} evaluations; |W (D_DerivKit - D_five)|/|W D_five| per column:")
        print("   " + ", ".join(f"{k} {v:.1e}" for k, v in changes.items()))
        for name in entry["systems"]:
            for kmax, result in entry["systems"][name].items():
                if "change_vs_candidate" in result:
                    c = result["change_vs_candidate"]
                    print(f"   against the candidate, {name}, k {kmax}: sigma {c['sigma']:.2%} "
                          f"FoM {c['fom']:.2%} shift {c['shift']:.3f}")
    print("noise of one evaluation [sqrt(chi2), shipped mask]: H0, omegam, w0, wa, "
          "largest sigma_v")
    for label, entry in summary["configurations"].items():
        if entry["rule"] != "five" or entry["step"] != 1.0:
            continue
        noise = entry["noise_per_point"]
        sigv = 0.0
        for name, value in noise.items():
            if name.startswith("DESI_SIGV_"):
                sigv = max(sigv, value)
        print(f"  {label:28s} {noise['H0']:.1e} {noise['omegam']:.1e} "
              f"{noise['w0']:.1e} {noise['wa']:.1e} {sigv:.1e}")


def run_agree(layout, work, results):
    """Check the study's per-bin route against the production caches.

    The model A notebook takes its per-bin columns by intercepting the
    compiled bias setter (the partial derivatives of b1, b2, bs2 and
    alpha_0); this study steps untied per-bin parameters through the
    likelihood. At the same settings the two routes compute the same
    numbers: the study's DESI_B1_<n> column is the model A chain rule of
    the notebook's columns, (b1/B)[dm/db1 + b2L'(b1) dm/db2 - 4/7
    dm/dbs2], and its DB2, DBS and A0 columns are the notebook's b2, bs2
    and a0 columns. The stencil columns of the cosmology and sigma_v are
    compared too. Each difference is the largest over the entries, in
    units of the largest entry of the column block.
    """
    import pickle
    candidate = np.load(work_file(work, layout, "candidate"))
    stem = LAYOUTS[layout]["dataset"]
    number = list(LAYOUTS).index(layout) + 1
    cache = os.path.join(PROJECT_DIR, "chains",
                         f"EXAMPLE_EVALUATE_FISHER_MODEL_A_DERIVATIVES_{number}.pkl")
    if not os.path.exists(cache):
        raise FileNotFoundError(f"{cache} is missing: run "
                                "EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb first")
    with open(cache, "rb") as handle:
        stored = pickle.load(handle)
    production = stored["derivs"]
    settings = stored["settings"]
    tables = layout_tables(layout)
    columns = base_columns(candidate, tables)
    ops = production["ops"]
    b1 = production["b1"]
    fiducial = dict(zip(production["names"], production["fiducial"]))
    leaders = bias_leaders(layout)
    amplitude = np.array([fiducial[leader] for leader in leaders])
    slope = -2.143 + 2.0*0.929*b1 + 3.0*0.008*b1**2
    chain_rule = (ops["b1"] + ops["b2"]*slope[None, :]
                  - 4.0/7.0*ops["bs2"])*(b1/amplitude)[None, :]
    pairs = {"DESI_B1 (chain rule)": (columns["b1"], chain_rule),
             "DESI_DB2": (columns["b2"], ops["b2"]),
             "DESI_DBS": (columns["bs2"], ops["bs2"]),
             "DESI_A0": (columns["a0"], ops["a0"])}
    names = [str(x) for x in candidate["stencil_names"]]
    stencil_study = columns["stencil"]
    stencil_production = np.zeros_like(stencil_study)
    for j, name in enumerate(names):
        stencil_production[:, j] = production["D"][:, production["names"].index(name)]
    pairs["stencil (COSMO, sigma_v)"] = (stencil_study, stencil_production)
    out = {"layout": LAYOUTS[layout]["label"], "production_cache": os.path.basename(cache),
           "production_settings": {k: v for k, v in settings.items() if k != "fingerprint"},
           "study_settings": json.loads(str(candidate["settings"])),
           "max_relative_difference": {}}
    print(f"{LAYOUTS[layout]['label']}: the study's untied per-bin route against "
          f"the setter route of {os.path.basename(cache)}")
    for label, (study, prod) in pairs.items():
        scale = float(np.max(np.abs(prod)))
        diff = float(np.max(np.abs(study - prod)))/scale
        out["max_relative_difference"][label] = diff
        print(f"  {label:28s} max |study - production| / max |production| = {diff:.1e}")
    path = os.path.join(results, f"fisher_derivatives_agreement_{layout}.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1, default=str)
    print(f"wrote {path}")


def run_plan():
    """Print every configuration's resolved settings and evaluation count."""
    for layout in LAYOUTS:
        tables = layout_tables(layout)
        nbins = len(tables["zbar"])
        ncolors = int(pair_coloring(nbins, tables["pairs"]).max()) + 1
        print(f"{layout} ({LAYOUTS[layout]['label']}): {nbins} bins, {ncolors} colors")
        for config in CONFIGS:
            s = resolve_settings(layout, config)
            K = len(OFFSETS[s["offsets"]])
            count = 2 + 14*K
            if s["blocks"] == "all":
                count += len(OPERATOR_STEPS)*ncolors*K + 2*ncolors
            print(f"  {config:24s} {count:4d} evaluations: {likelihood_overrides(s)} "
                  f"{camb_overrides(s)}")


def main():
    """Parse the command line and run one stage."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=["plan", "cuts", "derive", "dkit", "compare",
                                          "agree"])
    parser.add_argument("--layout", choices=list(LAYOUTS), default="dz0p05")
    parser.add_argument("--config", choices=list(CONFIGS), default="reference")
    parser.add_argument("--work", default=DEFAULT_WORK)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--base", default=None,
                        help="compare: the base configuration (default: the reference)")
    parser.add_argument("--perbin-from", default=None,
                        help="compare: a chains folder whose notebook caches give the "
                             "per-bin columns of stencil-only configurations")
    args = parser.parse_args()
    if args.stage == "plan":
        run_plan()
        return
    if ROOTDIR is None:
        raise RuntimeError("ROOTDIR is not set: source start_cocoa.sh in "
                           "cocoa/Cocoa/ first")
    os.chdir(ROOTDIR)
    sys.path.insert(0, os.path.join(ROOTDIR, "external_modules", "code", "cosmolike_core"))
    sys.path.insert(0, os.path.join(PROJECT_DIR, "interface"))
    if args.stage == "cuts":
        run_cuts(args.layout, args.work)
    elif args.stage == "derive":
        run_derive(args.layout, args.config, args.work)
    elif args.stage == "dkit":
        run_dkit(args.layout, args.config, args.work)
    elif args.stage == "agree":
        run_agree(args.layout, args.work, args.results)
    else:
        run_compare(args.layout, args.work, args.results, base=args.base,
                    perbin_from=args.perbin_from)


if __name__ == "__main__":
    main()
