"""Unit tests 1-2: the full-vector likelihood on the frozen test data.

The measured vector holds 15 cosmic-shear (EE) spectra, the 76 DESI
thin-bin clustering autos and the 97 measured clustering cross pairs,
53 Fourier bandpowers each; here it is the lssty1xdesi.desi_cl
likelihood, evaluated on the frozen copy of example1's configuration
(see cocoa_test_utils for what "frozen" means and why). The two tests:

  1. chi2 at the frozen fiducial point, within CHI2_TOLERANCE (0.2) of
     the frozen reference value.
  2. race check: on one model, the fiducial evaluated fresh and again
     as the 10th of 10 cosmologies in a row must agree to
     RACE_TOLERANCE (1e-4). A disagreement means state leaked between
     evaluations or OpenMP threads raced.

The donor suite's TATT variants need declared source nuisance
parameters, which this project does not carry yet; they return with
that declaration.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/lssty1xdesi/tests/data_vector
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
# the chi2 + race test class for one frozen example; assigning its
# return value to the usual module-level name is what makes pytest
# collect it here, with unchanged test ids. The numbers name the
# methods and the printed reports; the label opens every report line.
TestExample1FullVector = u.example_suite(
    u, name="TestExample1FullVector", example="example1",
    label="example1 (EE + gg + gg-cross",
    numbers=("1", "2"))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
