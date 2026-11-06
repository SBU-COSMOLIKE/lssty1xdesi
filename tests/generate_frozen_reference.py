"""Maintainer tool: (re)create the frozen state the unit tests run on.

Running this redefines what the tests protect, so it refuses to run
without the explicit --overwrite flag. Only run it when a change to
the data vectors, n(z), covariance, examples, or likelihood defaults
is deliberate, and review the printed chi2 values before committing:
they become the new references the tests compare against.

The machinery and its documentation live in
cosmolike_core/cocoa_testing_freeze.py (freeze_main lists the modes
and the stage order); this script keeps the project's data:
the generated-vector roster and the per-mask
descriptor variants.

Usage (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python ./projects/lssty1xdesi/tests/generate_frozen_reference.py --overwrite

Incremental modes on an existing frozen state (each verifies the
state against the manifest, then re-pins it):
  --tatt-masks rewrites the per-mask descriptors;
  --baryons regenerates the feedback vectors;
  --nmodels [example ...] computes the stored high-accuracy
    vectors of the N-random-models check (all of its examples
    when none is listed; hours on the thin DESI layouts).
"""

import os

# The references must be produced under the same OpenMP thread count
# the tests enforce, and before any cobaya/cosmolike import.
os.environ["OMP_NUM_THREADS"] = "4"

import sys

# __file__ is this script's own path; insert(0, ...) puts the tests/
# folder first on the module search path, so the import below finds
# the shim no matter where the script was launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

# The data vectors --overwrite regenerates:
# {descriptor dataset: (generating example, TATT flag)}. This project
# declares no source nuisance parameters yet, so there is no TATT
# vector to generate; the shipped fiducial data vector already is the
# model at the frozen point.
VECTORS = {}

# No per-mask descriptor variants: the project ships one measured
# Fourier dataset (see cocoa_test_utils.FASTPT_MASK_DATASETS).
TATT_MASK_VARIANTS = {}


if __name__ == "__main__":
    sys.exit(u.freeze_main(
        u, script_file=__file__, usage=__doc__,
        vectors=VECTORS,
        mask_variants=TATT_MASK_VARIANTS,
    ))
