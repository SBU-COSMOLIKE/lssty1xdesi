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
is a judgment call. One check covers the measured vector:

  A1. EE + gg + gg-cross (example1), NLA

The shipped data vector is the model at the frozen fiducial, so the
chi2 sits at a minimum and the delta is a stable, quadratic response
instead of a linear one.

Before the all-knobs check, one scan (K) evaluates each accuracy
knob alone on the example1 configuration, so a large all-knobs
delta can be attributed to the knob causing it. The scan includes
accuracyboost 5 as a stress knob (in another survey that boost broke
the interface instead of refining it, and the one-at-a-time delta is
what tells such cases apart), and three knobs of this project's own:
the chi grid of the low-rank non-Limber engine
(nonlimber_accuracyboost, the accuracy-critical control of the
0.05-wide comoving bins), the kept SVD ranks and the engine's
multipole cap.

One opt-in check extends the scan beyond the fiducial: the
N-random-models check evaluates the same delta at N reproducible
random points across the prior, each against a synthetic data vector
generated at that point (see test_ax99_nmodels). It runs only when
the COCOA_ACCURACY_NMODELS environment variable is a positive
integer:

    COCOA_ACCURACY_NMODELS=10 python -m pytest \\
        ./projects/lssty1xdesi/tests/data_vector/test_accuracy.py -k nmodels

When this file runs as a script, --nmodels N sets the same option:

    python ./projects/lssty1xdesi/tests/data_vector/test_accuracy.py --nmodels 10

With the variable unset (or 0) the check prints how to enable it and
passes without evaluating anything.

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
         False, "example1 (EE + gg + gg-cross, NLA)"),
    ),
    knob_scan="example1",
    nmodels_example=("example1", "example1 (EE + gg + gg-cross, NLA)"))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    # --nmodels N is the script-run spelling of COCOA_ACCURACY_NMODELS;
    # it must leave sys.argv before unittest.main parses the arguments.
    # `in` scans the argument list for the flag, and .index returns
    # the position of its first occurrence
    if "--nmodels" in sys.argv:
        flag_index = sys.argv.index("--nmodels")
        if flag_index + 1 >= len(sys.argv):
            sys.exit("--nmodels requires a value, for example "
                     "--nmodels 10")
        os.environ["COCOA_ACCURACY_NMODELS"] = sys.argv[flag_index + 1]
        # del on the slice removes the flag and its value from the
        # list in place, so unittest.main never sees them
        del sys.argv[flag_index:flag_index + 2]
    unittest.main(verbosity=2)
