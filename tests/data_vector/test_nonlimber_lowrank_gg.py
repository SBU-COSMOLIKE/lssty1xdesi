"""Unit test: FKEM vs low-rank SVD non-Limber clustering.

The likelihood yaml key nonlimber_lowrank_gg picks the engine behind
the non-Limber galaxy clustering C_l^gg (active when
adopt_limber_gg: 0):

  nonlimber_lowrank_gg: 0 - FKEM (Fang, Krause, Eifler & MacCrann,
      arXiv:1911.11947): an FFTLog integral of the separable linear
      spectrum plus, in Limber, what linear theory misses, below the
      cap lmax_nonlimber (default 150) and pure Limber above it; the
      installed cross pairs get the same split. The default of every
      project.
  nonlimber_lowrank_gg: 1 - the low-rank SVD engine
      (C_cl_tomo_lowrank in cosmo2D.c): the full nonlinear sqrt(P_g)
      decomposed into a short sum of separable terms, each projected
      by FFTLog, so the density-density term needs no Limber hand-off
      at any multipole. Built for exactly this project's thin
      spectroscopic lens bins, and its default here.

The engine difference - FKEM's Limber treatment of the nonlinear
density term and its multipole cap, against the SVD's full nonlinear
projection - matters here: the thin bins keep real non-Limber power
far past the default cap, which is why the SVD engine is the project
default. This test freezes the measured difference at the default
settings and both engines' runtimes.

It evaluates the frozen fiducial (NLA) three times in one
process (FKEM, SVD, FKEM again) and computes

    delta chi2 = delta^T C^-1 delta,
    delta = dv(FKEM) - dv(SVD),

with C^-1 the masked inverse covariance. It also reports each
engine's steady-state evaluation time (informational).

Assertions (the shared factory's five checks):
  1. delta chi2 is above a dead-flag floor: the flag reaches the C
     code and the clustering cache notices the change;
  2. only clustering entries change: cosmic shear, galaxy-galaxy
     lensing, and every other block are bitwise equal;
  3. switching back to FKEM reproduces the first data vector bitwise;
  4. delta chi2 matches the value measured for this project
     (DCHI2_MEASURED below) to 10%: a change in either non-Limber
     engine, the kernels, or the covariance shows up here;
  5. the FKEM default reproduces the frozen reference chi2 (checked
     last, so a stale snapshot cannot hide checks 1-4).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/lssty1xdesi/tests/data_vector/test_nonlimber_lowrank_gg.py
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

# The engine delta chi2 of this project's frozen fiducial, measured
# at OMP 4 with the likelihood defaults (rank 5 and the engine's l cap
# of 1000; FKEM at its cap of 150, cross pairs included; accuracyboost
# 2, measured on 2026-11-05): see assertion 4 of the module docstring.
DCHI2_MEASURED = 4.249476e+03

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the class; assigning its return value to the usual module-level name
# is what makes pytest collect it here, with unchanged test ids.
TestNonLimberLowrankGG = u.lowrank_gg_suite(
    u, name="TestNonLimberLowrankGG", measured=DCHI2_MEASURED,
    example="example1", default=1)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
