"""Unit test: the fair FKEM vs low-rank SVD comparison.

test_nonlimber_lowrank_gg.py compares the two non-Limber clustering
engines at the likelihood defaults, where FKEM stops at its default
cap lmax_nonlimber = 150 and is pure Limber above it. On thin
spectroscopic bins that cap, not the method, dominates the
difference: the bins keep non-Limber power far past l = 150.

This test removes the cap from the comparison. It sets

    lmax_nonlimber: 550

in every build - above the highest band center, 539.5 - so FKEM is
non-Limber at every band, cross pairs included, exactly like the
engine. The low-rank engine ignores the key (it has its own cap,
nonlimber_lowrank_lmax), so the engine flag is the only difference
between the two runs. What remains is the methods' genuine
difference: FKEM keeps the nonlinear part of the density term in
Limber, while the engine projects the full nonlinear P_gg.

It evaluates the frozen fiducial (NLA) three times in one process
(low-rank SVD, FKEM, SVD again) and computes

    delta chi2 = delta^T C^-1 delta,
    delta = dv(FKEM) - dv(SVD),

with C^-1 the masked inverse covariance. Both engines run on the
frozen chi grid (nonlimber_accuracyboost: 64). Each engine's
steady-state evaluation time is reported (informational).

Assertions (the shared factory's five checks):
  1. delta chi2 is above a dead-flag floor: the engine flag reaches
     the C code and the clustering cache notices the change;
  2. only clustering entries change: cosmic shear and every other
     block are bitwise equal;
  3. switching back to the engine reproduces the first data vector
     bitwise;
  4. delta chi2 matches the value measured for this project
     (DCHI2_MEASURED below) to 10%: a change in either engine, the
     kernels, or the covariance shows up here;
  5. the engine (the default) reproduces the frozen reference chi2
     with the override in place (checked last, so a stale snapshot
     cannot hide checks 1-4).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/lssty1xdesi/tests/data_vector/test_nonlimber_fkem_fair.py
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

# The FKEM cap of the fair comparison: above the highest band center
# (539.5), so FKEM is non-Limber at every band
FAIR_OVERRIDES = {"lmax_nonlimber": 550}

# The engine delta chi2 of this project's frozen fiducial with FKEM at
# the fair cap, measured on 2026-11-04 (rank 5, the engine's l cap of
# 1000, nonlimber_accuracyboost 64): see assertion 4 of the module
# docstring.
DCHI2_MEASURED = 74.0168

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the class; assigning its return value to the usual module-level name
# is what makes pytest collect it here.
TestNonLimberFKEMFair = u.lowrank_gg_suite(
    u, name="TestNonLimberFKEMFair", measured=DCHI2_MEASURED,
    example="example1", default=1, overrides=FAIR_OVERRIDES)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
