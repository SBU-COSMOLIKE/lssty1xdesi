"""Compute the lensing-magnification part of the data vector of one layout.

What is computed. Lensing by the matter in front of a thin bin changes its
observed counts by C times the convergence kappa of the bin's galaxies, with
C = 5 s - 2 the magnification coefficient (s = dlog10 N(<m)/dm, the slope of
the cumulative counts at the sample's magnitude limit; the 5 s is the flux
brightening, the -2 the area dilution). The paper forecast sets C = 0 in
every bin. The shift test of notes/lensing_magnification asks where the fit
of that model lands when the data carry magnification, so it needs only

  delta d = m(C) - m(0)   (m the masked DESI band powers at the fiducial
                           point, C one value per bias chunk - a chunk is
                           the bin table's grouping of bins that share one
                           sampled bias amplitude: BGS, LRG1-3, ELG1-2),

and the existing derivative base. This script computes delta d for the
coefficient sets of COEFFICIENT_SETS, on one layout and one pair list:
"today" (EXAMPLE_EVALUATE<n>.yaml, data/<stem>_pairs.txt) or "pairmax"
(EXAMPLE_EVALUATE<n>_PAIRMAX.yaml, the maximal lists of the pair-axes study).

The evaluations, in order:

  1  the fiducial point with C = 0, compared bit for bit with the m0 of the
     layout's derivative cache (model_derivatives_<layout>_final): the
     magnification extension must leave the C = 0 model unchanged;
  k  one evaluation per coefficient set;
  1  the fiducial point with C = 0 again (the reset check: a coefficient set
     must not leak into the next evaluation).

The model is the derivative base's: the step classes of
compute_fisher_base_mpi_classes.py at the derivative tier's settings
(compute_fisher_base_mpi.layout_settings), with no step set. The
coefficients reach CosmoLike through desi_cl.set_magnification_coefficients.

Output. One cache, written with fm.write_fisher_cache in chains/:

  model_magnification_<layout>_final.npz
    'm0'       the fiducial vector with C = 0;
    'delta'    {set name: m(C) - m0} for each coefficient set;
    'reset'    max |m_after - m0| / max |m0| of the two C = 0 points;
    'seconds'  the wall time.

The stored settings are the derivative base's (yaml, dataset, accuracy,
tier), the coefficient sets and the fingerprint of the resolved paper yaml,
so the fits script can check that delta d belongs to the model of the base.
An existing cache is never replaced without --overwrite.

Run (from the Cocoa/ folder, cocoa environment active, start_cocoa.sh
sourced), one layout at a time (the 62 Mpc model needs about 14 GB):

    python ./projects/lssty1xdesi/scripts/compute_magnification_vectors.py \\
        --layout dz0p20_pairmax --threads 8
"""
import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CACHE_NAME = "model_magnification_{layout}_{tier}"
DERIVATIVES_NAME = "model_derivatives_{layout}_{tier}"
# The coefficient sets: C = 5 s - 2 per bias chunk. "fiducial" holds the
# measured DESI slopes (BGS s = 0.80-0.81, LRG s = 0.97-1.04; the ELG slope
# is unmeasured, so its central value is 0); "elg_plus" and "elg_minus"
# are the ELG bracket corners; "fiducial_negative" flips every sign, so
# the linear part (m(C) - m(-C))/2 and the quadratic part (m(C) + m(-C))/2 - m0
# of the response separate (the magnification x magnification term is
# quadratic in C, every other term linear).
COEFFICIENT_SETS = {
    "fiducial": {"BGS": 2.0, "LRG1": 3.0, "LRG2": 3.0, "LRG3": 3.0,
                 "ELG1": 0.0, "ELG2": 0.0},
    "elg_plus": {"BGS": 2.0, "LRG1": 3.0, "LRG2": 3.0, "LRG3": 3.0,
                 "ELG1": 2.0, "ELG2": 2.0},
    "elg_minus": {"BGS": 2.0, "LRG1": 3.0, "LRG2": 3.0, "LRG3": 3.0,
                  "ELG1": -2.0, "ELG2": -2.0},
    "fiducial_negative": {"BGS": -2.0, "LRG1": -3.0, "LRG2": -3.0, "LRG3": -3.0,
                          "ELG1": 0.0, "ELG2": 0.0},
}


def parse_arguments():
    """Return the command-line arguments (--layout, --threads, --overwrite)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layout", required=True,
                        help="a compute_fisher_base_mpi.LAYOUTS key with a paper or "
                             "pairmax yaml, e.g. dz0p20 or dz0p20_pairmax")
    parser.add_argument("--threads", type=int, default=8,
                        help="OpenMP threads of CosmoLike and CAMB")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing cache")
    return parser.parse_args()


def yaml_of(base, layout):
    """Return the example yaml of a layout key (the pairmax keys end in _pairmax).

    Arguments:
      base   = the module compute_fisher_base_mpi.
      layout = a base.LAYOUTS key.

    Returns:
      the absolute path of EXAMPLE_EVALUATE<n>.yaml or of its _PAIRMAX twin.
    """
    number = base.LAYOUTS[layout]["number"]
    # one simple ternary: the maximal lists have their own example yaml
    suffix = "_PAIRMAX" if layout.endswith("_pairmax") else ""
    return os.path.join(os.environ["ROOTDIR"], base.PROJECT,
                        f"EXAMPLE_EVALUATE{number}{suffix}.yaml")


def main():
    """Compute delta d for every coefficient set of one layout and save it.

    Side effects: changes into ROOTDIR; writes the cache in chains/.

    Raises:
      KeyError for an unknown layout; FileExistsError when the cache exists
      and --overwrite is not given (before any model is built);
      RuntimeError when the C = 0 vector differs from the derivative
      cache's m0 or the fingerprint changed during the run.
    """
    args = parse_arguments()
    # the compiled interface and the OpenMP runtime read the thread count
    # once, when they load
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import compute_fisher_base_mpi as base
    import numpy as np

    rootdir = os.environ["ROOTDIR"]
    os.chdir(rootdir)
    sys.path.insert(0, os.path.join(rootdir, "external_modules/code/cosmolike_core"))
    sys.path.insert(0, os.path.join(rootdir, base.PROJECT, "interface"))
    import cosmolike_lssty1xdesi_fisher as fm
    import compute_fisher_base_mpi_classes as step_classes

    if args.layout not in base.LAYOUTS:
        raise KeyError(f"layout {args.layout!r}: the layouts are {list(base.LAYOUTS)}")
    tier = base.DERIVATIVE_TIER
    name = CACHE_NAME.format(layout=args.layout, tier=tier)
    out_dir = os.path.join(rootdir, base.PROJECT, "chains")
    path = os.path.join(out_dir, name + ".npz")
    if os.path.exists(path) and not args.overwrite:
        raise FileExistsError(f"{path} exists; nothing was computed. Pass --overwrite "
                              "to replace it")
    yaml_path = yaml_of(base, args.layout)
    yaml_label = os.path.relpath(yaml_path, rootdir)
    settings = base.layout_settings(fm=fm, layout=args.layout, yaml_label=yaml_label)
    fingerprint = base.layout_fingerprint(fm=fm, yaml_path=yaml_path, settings=settings)
    fm.check_halofit_patch()
    # the derivative base's fiducial vector, read with the cache checks
    # (settings of its own kind are not compared here: the fingerprint
    # check below ties the two caches to one model)
    derivatives = fm.read_fisher_cache(DERIVATIVES_NAME.format(layout=args.layout,
                                                               tier=tier), tier=tier)
    header = fm.read_fisher_cache_header(DERIVATIVES_NAME.format(layout=args.layout,
                                                                 tier=tier))
    if header["tiers_and_fingerprints"]["settings/fingerprint"] != fingerprint:
        raise RuntimeError(f"{args.layout}: the derivative cache belongs to another "
                           "model than the current yaml, parameter and dataset files")
    m0_base = np.asarray(derivatives["m0"], dtype=float)

    start = time.time()
    model, like, build_seconds = base.build_step_model(
        fm=fm, step_classes=step_classes, yaml_path=yaml_path, settings=settings,
        check_input=True)
    print(f"{args.layout}: model built in {build_seconds:.0f} s, fingerprint "
          f"{fingerprint[:12]}", flush=True)

    t0 = time.time()
    like.set_magnification_coefficients(None)
    m0 = model.dv()
    if not np.array_equal(m0, m0_base):
        raise RuntimeError(f"{args.layout}: the C = 0 vector differs from the derivative "
                           f"cache's m0 (max |diff| {np.max(np.abs(m0 - m0_base)):.3e}): "
                           "the build does not reproduce the base")
    print(f"C = 0 vector equals the derivative cache's m0 bit for bit "
          f"({time.time() - t0:.0f} s)", flush=True)
    delta = {}
    for set_name, coefficients in COEFFICIENT_SETS.items():
        t0 = time.time()
        like.set_magnification_coefficients(coefficients)
        delta[set_name] = model.dv() - m0
        print(f"{set_name}: delta chi2 = {model.delta_chi2(delta[set_name]):.4g} on the "
              f"shipped mask ({time.time() - t0:.0f} s)", flush=True)
    # the reset check: back at C = 0, the vector must be m0 again, or a
    # coefficient set leaked into the next evaluation. The ratio is
    # stored in the cache and printed rather than enforced, so a nonzero
    # value is visible without discarding the computed deltas.
    like.set_magnification_coefficients(None)
    m_after = model.dv()
    reset = float(np.max(np.abs(m_after - m0))/np.max(np.abs(m0)))
    if base.layout_fingerprint(fm=fm, yaml_path=yaml_path, settings=settings) != fingerprint:
        raise RuntimeError("the fingerprint changed during the run (a yaml, parameter "
                           "or dataset file was edited); nothing was written")
    stored = dict(settings)
    stored["magnification_sets"] = COEFFICIENT_SETS
    stored["magnification_rule"] = ("delta d = m(C) - m(0) at the fiducial point, "
                                    "C per bias chunk via set_magnification_coefficients")
    stored["fingerprint"] = fingerprint
    result = {"m0": m0, "delta": delta, "reset": reset,
              "seconds": time.time() - start}
    fm.write_fisher_cache(result, name, stored, overwrite=args.overwrite,
                          folder=out_dir)
    print(f"reset check: max |m_after - m0| / max |m0| = {reset:.2e}; wall time "
          f"{time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
