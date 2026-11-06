"""Shared harness for the lssty1xdesi unit tests: the project's data
bound to the shared Cocoa test machinery.

The machinery itself (frozen-state verification, the chi2 pipeline,
worker-subprocess isolation, the race and baryon checks, the
N-random-models check, the EE2 modification comparison, and the
terminal reports) lives in
external_modules/code/cosmolike_core/cocoa_testing.py. This file
carries what is lssty1xdesi's alone: the examples table (the full
EE + gg + gg-cross Fourier vector and its EMUL2 emulator
configuration), the accuracy knobs - including the project's own
non-Limber controls - and the dataset names. It binds them to a
single cocoa_testing.CocoaTestHarness instance and re-exports the
instance's methods, and the core's functions, as module-level names,
so the test modules and generate_frozen_reference.py import
everything from this module.

The project declares no source nuisance parameters yet (intrinsic
alignment, shear calibration, source photo-z shifts), so the TATT
variants and the CFASTPT-vs-FASTPT comparison of the donor suite have
nothing to exercise: TATT_POINT is empty and no TATT vector is
frozen. Declaring those parameters, and porting the TATT and FASTPT
tests with them, is a follow-up of its own.

Frozen state: everything a test evaluates lives under tests/frozen/,
pinned byte for byte by tests/manifest_sha256.json and verified before
any model is built; refreshing the frozen state is a deliberate
maintainer action (generate_frozen_reference.py --overwrite).
"""

import os
import sys

# ---- tests/ paths -----------------------------------------------------------

# Everything the tests read or write lives relative to this folder, so
# the suite works no matter which directory pytest is launched from
# (__file__ is this module's own path; dirname strips the file name).
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(TESTS_DIR, "frozen")
MANIFEST_FILE = os.path.join(TESTS_DIR, "manifest_sha256.json")
REFERENCE_FILE = os.path.join(FROZEN_DIR, "reference_chi2.json")

# ---- the shared machinery ---------------------------------------------------

# The import is path-based (tests/ is three levels below Cocoa/, which
# holds external_modules/code/cosmolike_core) so it works before
# start_cocoa.sh's python-path setup runs.
_CORE_DIR = os.path.abspath(os.path.join(
    TESTS_DIR, "..", "..", "..", "external_modules", "code",
    "cosmolike_core"))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
import cocoa_testing as _cct
# cocoa_testing_suites holds the factories that build whole shared
# test classes bound to this module; the project's test files assign
# their return values to the module-level names pytest collects
import cocoa_testing_suites as _cct_suites
# cocoa_testing_freeze is the shared machinery behind the
# maintainer tool generate_frozen_reference.py
import cocoa_testing_freeze as _cct_freeze

# ---- the project data ------------------------------------------------------

# No source nuisance parameters are declared yet (see the module
# docstring), so there is no TATT variant to point at: the empty point
# keeps the harness constructible while every TATT code path stays
# unused by this suite.
TATT_POINT = {}

# The --mask machinery of the shared sweeps needs a table naming at
# least the frozen dataset; this project has no scale-cut variants of
# its measured Fourier dataset (the M1-M6 files under data/ are donor
# leftovers of the project transfer and describe the donor's layout).
FASTPT_MASK_DATASETS = {
    "frozen": "lssty1xdesi_desi_dr3.dataset",
}

# The frozen examples score Fourier band powers (init_data_fourier).
# The interface binds both spaces and keeps the generic getter names
# (compute_data_vector_masked, the 3x2pt sizes) on the real-space
# layout, so the shared suites must be told to read the Fourier twins.
BINNING_SPACE = "fourier"

# ---- configurations ---------------------------------------------------------

# The frozen configurations. Field meanings:
#   "likelihood"    = the cobaya component name, needed to reach that
#                     block inside the loaded info dictionary;
#   "provenance"    = the human-readable snapshot (never loaded);
#   "frozen_module" = the file, under frozen/, holding the frozen
#                     configuration and point;
#   "emulator"      = an EMUL2 configuration: machine-learning
#                     emulators replace the Boltzmann code. These are
#                     advisory (no pass/fail; see test_emul2.py);
#                     "exact_example" names the exact-physics
#                     counterpart whose frozen layout the harness
#                     copies, and "exact_reference" the reference
#                     chi2 their accuracy is judged against.
# The N-random-models check (test_accuracy.py, test_ax99_nmodels), one
# per DESI layout: the default settings against the pushed accuracy
# settings at 50 fixed points of the prior (seed 137 + m for model m).
# generate_frozen_reference.py --nmodels stores the pushed vectors in
# frozen/nmodels_<example>.npz and reads these same draw settings.
NMODELS_EXAMPLES = (
    ("example1", "example1 (DR3 comoving, NLA)"),
    ("example2", "example2 (Delta z = 0.02, NLA)"),
    ("example3", "example3 (Delta z = 0.05, NLA)"),
    ("example4", "example4 (Delta z = 0.2, NLA)"))
# every model must stay below the 0.2 budget of the reference tests
NMODELS_LIMIT = _cct.CHI2_TOLERANCE
# Omega_m is drawn in [0.2, 0.5], the box of the examples: above 0.5
# the pushed settings move the LSST Y1 3x2pt chi2 by up to 10^3 against
# the fiducial covariance. w0 <= -1/3 (an accelerating universe today)
# and w0 + wa >= -3 are the other boxes of the examples. The frozen
# configurations keep wider boxes; the draw narrows them.
NMODELS_BOX = {"omegam": (0.2, 0.5), "w": (-3.0, -1.0/3.0),
               "w0pwa": (-3.0, -0.01)}
# every odd model holds Omega_m, A_s (the primordial amplitude) and the
# DESI biases at the fiducial (the clustering amplitude A_s b_i^2 with
# them), so its delta chi2 shows what the other parameters do to the
# accuracy
NMODELS_HOLD = ("omegam", "As_1e9", "DESI_B1_")
# every drawn bin keeps (A_s/A_s,fid)(b_i/b_i,fid)^2 <= 8, the cut of
# the examples' prior and of every project: the DESI Fisher 2 sigma
# contours reach a ratio of about 2 (EXAMPLE_EVALUATE_FISHER.ipynb, section
# 8), while the bias boxes alone would reach about 50 through the ELG
# fiducial of 0.84
NMODELS_AMPLITUDE_CUT = 8.0

EXAMPLES = {
    "example1": {
        "frozen_module": "frozen_config_example1.py",
        "provenance": "EXAMPLE_EVALUATE1.yaml",
        "likelihood": "lssty1xdesi.desi_cl",
    },
    # the DESI DR1 fixed-width Delta z binnings (10, 36 and 90 lens bins;
    # scripts/make_desi_dz_case.py): the same likelihood on each, frozen
    # for the FKEM vs low-rank engine comparisons
    # Delta z = 0.02, 90 lens bins (case dr1_dz0p02)
    "example2": {
        "frozen_module": "frozen_config_example2.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "likelihood": "lssty1xdesi.desi_cl_dr1_dz0p02",
    },
    # Delta z = 0.05, 36 lens bins (case dr1_dz0p05)
    "example3": {
        "frozen_module": "frozen_config_example3.py",
        "provenance": "EXAMPLE_EVALUATE3.yaml",
        "likelihood": "lssty1xdesi.desi_cl_dr1_dz0p05",
    },
    # Delta z = 0.2, 10 lens bins (case dr1_dz0p20)
    "example4": {
        "frozen_module": "frozen_config_example4.py",
        "provenance": "EXAMPLE_EVALUATE4.yaml",
        "likelihood": "lssty1xdesi.desi_cl_dr1_dz0p20",
    },
    "emul2_example1": {
        "frozen_module": "frozen_config_emul2_example1.py",
        "provenance": "EXAMPLE_EMUL2_EVALUATE1.yaml",
        "likelihood": "lssty1xdesi.desi_cl",
        "emulator": True,
        "exact_example": "example1",
        "exact_reference": "example1_nla",
    },
}

# ---- accuracy knobs ---------------------------------------------------------

# High-accuracy settings for the accuracy advisory checks
# (test_accuracy.py): the same physics evaluated with the numerical
# knobs pushed far beyond the defaults. lmax is the project's Fourier
# bound (its default is 5000; in this binning it only caps the
# low-rank non-Limber cutoff).
HIGH_ACCURACY_LIKELIHOOD = {
    # this project's reference boost is 2 on every layout
    # (likelihood/*.yaml), so the all-knobs check pushes it to 4, twice
    # the reference, and the one-at-a-time scan keeps 6 as a deliberate
    # stress knob (other projects stop at 3: desy1xplanck breaks down
    # above it)
    "accuracyboost": 4.0,       # reference 2
    "internal_accuracyboost": 2.0, # default 1.0 (denser convolution grid)
    "integration_accuracy": 10,  # default 0
    "lmax": 20000,              # default 5000
    "kmax_boltzmann": 40.0,     # default 5.0
}

# The one-at-a-time scan of test_accuracy.py: each entry is (label,
# likelihood overrides, camb extra_args overrides), evaluated alone on
# the example1 configuration before the all-knobs checks, so a large
# all-knobs delta can be attributed to the knob causing it.
# Investigation order when several knobs move the chi2: raise the
# cosmolike accuracyboost first (cheap), then camb k_per_logint, and
# only then camb AccuracyBoost (expensive at run time). kmax_boltzmann
# and camb kmax are one physical cutoff seen from the two sides, so
# the scan moves them together.
#
# The last three knobs are this project's own: the chi grid of the
# low-rank non-Limber engine (the log-spaced grid makes the
# highest-redshift thin bins the narrowest in ln(chi): the default 64
# is the first nested doubling past their FoG-edge resolution
# threshold, so this is the accuracy-critical control), the kept
# SVD ranks, and the engine's own multipole cap.
ACCURACY_KNOBS = [
    ("accuracyboost -> 4", {"accuracyboost": 4.0}, {}),
    ("accuracyboost -> 6 (stress)", {"accuracyboost": 6.0}, {}),
    ("internal_accuracyboost 1->2", {"internal_accuracyboost": 2.0}, {}),
    ("integration_accuracy 0->10", {"integration_accuracy": 10}, {}),
    ("lmax 5000->20000", {"lmax": 20000}, {}),
    ("kmax_boltzmann -> 40 + camb kmax -> 50",
     {"kmax_boltzmann": 40.0}, {"kmax": 50.0}),
    ("camb AccuracyBoost 1.05->2", {}, {"AccuracyBoost": 2.0}),
    ("camb k_per_logint 15->50", {}, {"k_per_logint": 50}),
    ("nonlimber_accuracyboost -> 128",
     {"nonlimber_accuracyboost": 128.0}, {}),
    ("nonlimber_lowrank_rank 5->8", {"nonlimber_lowrank_rank": 8}, {}),
    ("nonlimber_lowrank_lmax 1000->2000",
     {"nonlimber_lowrank_lmax": 2000}, {}),
]

# The FASTPT settings exist only to satisfy the harness constructor:
# no FASTPT test module is bound (the module docstring says why), so
# nothing reads them. The values are the core's frozen defaults.
FASTPT_COMPARISON_TOLERANCE = 0.2
FASTPT_LOW_SETTINGS = {
    "accuracyboost": 1.0,
    "internal_accuracyboost": 1.0,
    "kmax_boltzmann": 7.5,
    "extrap_kmax": 250.0,
}
FASTPT_HIGH_SETTINGS = {
    "accuracyboost": 2.0,
    "internal_accuracyboost": 2.0,
    "kmax_boltzmann": 7.5,
    "extrap_kmax": 250.0,
}

# ---- project-independent constants ------------------------------------------

# These are identical in every project and live in the core module.
REQUIRED_OMP_THREADS = _cct.REQUIRED_OMP_THREADS
CHI2_TOLERANCE = _cct.CHI2_TOLERANCE
RACE_TOLERANCE = _cct.RACE_TOLERANCE
RACE_PERTURBATIONS = _cct.RACE_PERTURBATIONS
HIGH_ACCURACY_CAMB_EXTRA_ARGS = _cct.HIGH_ACCURACY_CAMB_EXTRA_ARGS
BARYON_METHODS = _cct.BARYON_METHODS
BARYON_POINT_OVERRIDES = _cct.BARYON_POINT_OVERRIDES
RANDOM_MODEL_SEED = _cct.RANDOM_MODEL_SEED
FASTPT_FROZEN_SETTINGS = _cct.FASTPT_FROZEN_SETTINGS

# The sampled-parameter prefix of this project's nuisance block is
# DESI; the comparison points are identical in every project.
FASTPT_COMPARISON_POINTS = _cct.fastpt_comparison_points("DESI")

# The 10 Halofit-vs-EE2 comparison cosmologies (omegam/ns/As draws,
# no project prefix); the values are identical in every project.
NONLINEAR_COMPARISON_POINTS = _cct.NONLINEAR_COMPARISON_POINTS

# ---- the harness -----------------------------------------------------------

# A single instance binds the shared machinery to this project's data;
# the assignments below re-export its methods, and the core's
# functions, under the module-level names the test modules import.
_H = _cct.CocoaTestHarness(
    worker_file=__file__,
    interface_module="cosmolike_lssty1xdesi_interface",
    examples=EXAMPLES,
    tatt_point=TATT_POINT,
    accuracy_knobs=ACCURACY_KNOBS,
    high_accuracy_likelihood=HIGH_ACCURACY_LIKELIHOOD,
    fastpt_low_settings=FASTPT_LOW_SETTINGS,
    fastpt_high_settings=FASTPT_HIGH_SETTINGS,
    fastpt_points=FASTPT_COMPARISON_POINTS,
    fastpt_masks=tuple(FASTPT_MASK_DATASETS),
)

# ---- module functions re-exported from the core (no project state) ----------
require_cocoa_environment = _cct.require_cocoa_environment
assert_omp_threads = _cct.assert_omp_threads
sha256_of = _cct.sha256_of
make_model = _cct.make_model
evaluate_chi2 = _cct.evaluate_chi2
_evaluate_cached = _cct._evaluate_cached
_load_datavector = _cct._load_datavector
_baryon_method = _cct._baryon_method
_baryon_dataset = _cct._baryon_dataset
build_original_ee2 = _cct.build_original_ee2
report_chi2_test = _cct.report_chi2_test
report_race_test = _cct.report_race_test
report_accuracy = _cct.report_accuracy
report_knob = _cct.report_knob
report_fastpt_comparison = _cct.report_fastpt_comparison
report_fastpt_test = _cct.report_fastpt_test
report_nonlinear_comparison = _cct.report_nonlinear_comparison
report_random_model = _cct.report_random_model
report_random_model_summary = _cct.report_random_model_summary
report_emul2_advisory = _cct.report_emul2_advisory
report_emul2_race = _cct.report_emul2_race

# ---- bound methods of the harness (the machinery, project-bound) ------------
compute_manifest = _H.compute_manifest
verify_frozen = _H.verify_frozen
load_reference = _H.load_reference
_frozen_module = _H._frozen_module
load_frozen_info = _H.load_frozen_info
load_frozen_point = _H.load_frozen_point
build_point = _H.build_point
_single_model_chi2_impl = _H._single_model_chi2_impl
_ten_in_a_row_impl = _H._ten_in_a_row_impl
_baryon_accuracy_delta_impl = _H._baryon_accuracy_delta_impl
_baryon_drift_chi2_impl = _H._baryon_drift_chi2_impl
single_model_chi2 = _H.single_model_chi2
ten_in_a_row_chi2 = _H.ten_in_a_row_chi2
baryon_accuracy_delta = _H.baryon_accuracy_delta
baryon_drift_chi2 = _H.baryon_drift_chi2
draw_uniform_point = _H.draw_uniform_point
nmodels_points = _H.nmodels_points
nmodels_config_hash = _H.nmodels_config_hash
ee2_original_vs_cocoa_dchi2s = _H.ee2_original_vs_cocoa_dchi2s
_ee2_comparison_block = _H._ee2_comparison_block
_worker = _H._worker
_run_isolated = _H._run_isolated
_fastpt_comparison_info = _H._fastpt_comparison_info
_fastpt_comparison_block = _H._fastpt_comparison_block
_run_fastpt_comparison_worker = _H._run_fastpt_comparison_worker
cfastpt_vs_fastpt_chi2s = _H.cfastpt_vs_fastpt_chi2s
_nonlinear_comparison_block = _H._nonlinear_comparison_block
halofit_vs_ee2_dchi2s = _H.halofit_vs_ee2_dchi2s

# ---- shared test-class factories (bound by the test files) ------------------
example_suite = _cct_suites.example_suite
accuracy_suite = _cct_suites.accuracy_suite
ee2_suite = _cct_suites.ee2_suite
nonlinear_suite = _cct_suites.nonlinear_suite
photoz_conventions_suite = _cct_suites.photoz_conventions_suite
nonlimber_suite = _cct_suites.nonlimber_suite
lowrank_gg_suite = _cct_suites.lowrank_gg_suite
cache_ladder_suite = _cct_suites.cache_ladder_suite
baryon_accuracy_suite = _cct_suites.baryon_accuracy_suite
baryon_drift_suite = _cct_suites.baryon_drift_suite
freeze_main = _cct_freeze.freeze_main
