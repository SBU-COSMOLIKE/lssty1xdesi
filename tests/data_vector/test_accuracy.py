"""Accuracy advisory checks: default vs high-accuracy settings.

Every reference in this suite is computed with the examples' default
numerical settings. These checks answer: how much numerical error do
those defaults carry? Each one re-evaluates a frozen configuration at
its frozen point with the numerical knobs pushed far beyond the
defaults (cosmolike: accuracyboost 3, internal_accuracyboost 2,
integration_accuracy 10, lmax 20000, kmax_boltzmann 40; CAMB:
AccuracyBoost 2.0, k_per_logint 50, kmax 50; the exact values live in
cocoa_test_utils.HIGH_ACCURACY_*) and reports

    delta chi2 = chi2(high accuracy) - chi2(default, frozen)

There is no pass/fail: how much numerical error an analysis tolerates
is a judgment call. One check per DESI layout covers its whole
vector (EE + gg + gg-cross, NLA):

  A1. example1, DR3 comoving bins (62 Mpc)
  A2. example2, DR1 Delta z = 0.02
  A3. example3, DR1 Delta z = 0.05
  A4. example4, DR1 Delta z = 0.2

The shipped data vector is the model at the frozen fiducial, so the
chi2 sits at a minimum and the delta is a stable, quadratic response
instead of a linear one.

Before the all-knobs checks, one scan (K) per layout evaluates each
accuracy knob alone on that example's configuration, so a large
all-knobs delta can be attributed to the knob causing it. The scan includes
accuracyboost 5 as a stress knob (in another survey that boost broke
the interface instead of refining it, and the one-at-a-time delta is
what tells such cases apart), and three knobs of this project's own:
the chi grid of the low-rank non-Limber engine
(nonlimber_accuracyboost, the accuracy-critical control of the
thin comoving bins), the kept SVD ranks and the engine's
multipole cap.

One check per layout extends the scan beyond the fiducial: the
N-random-models check evaluates the all-knobs delta at 50 fixed
points across the prior (seed 137 + m for model m; the draw settings
are NMODELS_BOX, NMODELS_HOLD and NMODELS_AMPLITUDE_CUT in
cocoa_test_utils.py) and fails when any |delta chi2| reaches 0.2.
The pushed-settings vectors of those points are stored in
frozen/nmodels_<example>.npz, so each check builds one model at the
default settings and compares its 50 vectors with them (see
test_ax99_nmodels). Another cosmology or prior means regenerating
the stored vectors:

    python ./projects/lssty1xdesi/tests/generate_frozen_reference.py --nmodels

A high-accuracy evaluation takes minutes, not seconds: the whole file
is far slower than the rest of the suite. To run only this file (from
the Cocoa/ folder, cocoa environment active, start_cocoa.sh sourced):

    python -m pytest ./projects/lssty1xdesi/tests/data_vector/test_accuracy.py

and to run the rest of the suite without it:

    python -m pytest ./projects/lssty1xdesi/tests --ignore \\
        ./projects/lssty1xdesi/tests/data_vector/test_accuracy.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the advisory class; assigning its return value to the usual
# module-level name is what makes pytest collect it here, with
# unchanged test ids. Each roster row is (method name, advisory
# label, example, TATT flag, report label); knob_scan binds the
# example the one-knob-at-a-time scan evaluates.
TestAccuracyAdvisory = u.accuracy_suite(
    u, name="TestAccuracyAdvisory",
    checks=(
        ("test_a1_full_vector_nla", "A1", "example1",
         False, "example1 (DR3 comoving, EE + gg + gg-cross, NLA)"),
        ("test_a2_full_vector_dz0p02_nla", "A2", "example2",
         False, "example2 (Delta z = 0.02, EE + gg + gg-cross, NLA)"),
        ("test_a3_full_vector_dz0p05_nla", "A3", "example3",
         False, "example3 (Delta z = 0.05, EE + gg + gg-cross, NLA)"),
        ("test_a4_full_vector_dz0p20_nla", "A4", "example4",
         False, "example4 (Delta z = 0.2, EE + gg + gg-cross, NLA)"),
    ),
    knob_scan=("example1", "example2", "example3", "example4"))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
