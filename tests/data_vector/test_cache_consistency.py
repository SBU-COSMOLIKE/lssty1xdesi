"""Unit test: sector-wise cache invalidation (the parameter ladder).

cosmolike keeps the result of every expensive stage in memory (a
cache) and recomputes it only when an input it depends on changed.
Each group of parameters (a sector) reaches the pipeline through its
own update path: cosmology (distances, growth, power-spectrum tables,
the non-Limber engine's tables), galaxy bias (set_nuisance_bias, with
each bin's amplitude divided by the growth at its mean redshift), and
the finger-of-god velocity dispersions (each bin's raw n(z) is
convolved and the whole lens sample is replaced through
set_lens_sample, which invalidates the lens kernels). A
partial-invalidation bug (one sector's update path failing to rebuild
a cached C static variable that another sector reads) produces wrong
data vectors only in mixed update sequences, which the per-point
tests never exercise.

The vector under test is the DESI galaxy clustering (autos and the
measured cross pairs). The LSST shear rows are masked and the
likelihoods sample no intrinsic-alignment, photo-z or
shear-calibration parameters, so the ladder walks the three sectors
the clustering depends on, evaluating the model after every step
(each sector's later steps keep the earlier sectors at their last
values, so the ladder ends at one well-defined point):

    3 x cosmology-only steps  (omegam, H0, As_1e9)
    3 x bias-only steps       (every sampled DESI_B1)
    3 x FoG-only steps        (every sampled DESI_SIGV, 5 km/s each)

It records the final data vector, then evaluates one scramble point
(every sector moved at once; the chi2 is discarded) and returns to the
ladder's final point: the pipeline must reproduce the recorded vector
bit for bit. A second model instance in the same process walks the
mirrored ladder (FoG -> bias -> cosmology) to the same final point:
the answer must depend on the point, never on the invalidation
history. Building the second model after the first also exercises
the reset of the model-defining statics (the cross-pair list among
them).

Assertions, on each of the four layouts (the cross-pair lists and bin
counts differ, so each one has its own static state):
  1. every ladder step changes the data vector (a dead sector flag
     would pass the later checks vacuously);
  2. a no-op update (re-evaluating the current point) leaves the
     vector bitwise unchanged;
  3. after the scramble, returning to the ladder's final point
     reproduces the recorded vector and chi2 bit for bit;
  4. the mirrored-order instance lands on the same final vector bit
     for bit.

The TATT variant of the shared class skips: no intrinsic alignment
enters a clustering-only vector.

Every evaluation forces a full recomputation (cobaya's cache is
bypassed), so each assertion tests cosmolike's own invalidation, not
cobaya's memoization.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/lssty1xdesi/tests/data_vector/test_cache_consistency.py
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
# the class; assigning its return value to the usual module-level name
# is what makes pytest collect it here, with unchanged test ids.
# the clustering ladder: cosmology, then galaxy bias, then the FoG
# velocity dispersions; one class per layout
WGG_SECTORS = ("cosmo", "bias", "fog")
TestCacheConsistency = u.cache_ladder_suite(
    u, name="TestCacheConsistency", example="example1",
    required_sectors=WGG_SECTORS, phases=WGG_SECTORS)
TestCacheConsistencyExample2 = u.cache_ladder_suite(
    u, name="TestCacheConsistencyExample2", example="example2",
    required_sectors=WGG_SECTORS, phases=WGG_SECTORS)
TestCacheConsistencyExample3 = u.cache_ladder_suite(
    u, name="TestCacheConsistencyExample3", example="example3",
    required_sectors=WGG_SECTORS, phases=WGG_SECTORS)
TestCacheConsistencyExample4 = u.cache_ladder_suite(
    u, name="TestCacheConsistencyExample4", example="example4",
    required_sectors=WGG_SECTORS, phases=WGG_SECTORS)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
