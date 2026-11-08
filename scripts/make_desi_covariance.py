"""Write a DESI case's likelihood covariance file from its covariance archive.

compute_covariance.py saves the Gaussian (G), super-sample (SSC) and
connected (cNG) terms of one case in an .npz archive under covariance/.
The likelihood reads a sparse text file instead, data/<stem>_cov. This
script writes that file from the archive's Gaussian term, the covariance
every DESI case of the project uses:

  dr3          covariance/production_covariance.npz
               -> data/lssty1xdesi_desi_dr3_cov
  dr1_dz0p02   covariance/production_covariance_dr1_dz0p02.npz
               -> data/lssty1xdesi_desi_dr1_dz0p02_cov
  (and likewise dr1_dz0p05, dr1_dz0p20)

The file format is the cosmolike reader's: one "i j value" line per
nonzero entry of the upper triangle (i <= j) over the unmasked indices,
plus a 1.0 placeholder diagonal on every masked index. The reader
re-masks the inverse, so a placeholder never reaches the chi2.

From an activated Cocoa installation, in the folder cocoa/Cocoa:
    python projects/lssty1xdesi/scripts/make_desi_covariance.py --case dr3
"""

import argparse
import os
import sys

import numpy as np

# project = projects/lssty1xdesi, one level above this script's folder
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT, "data")


def archive_path(case):
    """Return the covariance archive of one case (see the module docstring)."""
    name = ("production_covariance.npz" if case == "dr3"
            else f"production_covariance_{case}.npz")
    return os.path.join(PROJECT, "covariance", name)


def main():
    parser = argparse.ArgumentParser(
        description="Write data/<stem>_cov from a case's covariance archive.")
    parser.add_argument("--case", required=True,
                        help="dr3, dr1_dz0p02, dr1_dz0p05 or dr1_dz0p20")
    args = parser.parse_args()

    # the adapter knows each case's data stem and row layout; it imports
    # the shared cosmolike_notebook_utils package from the cosmolike_core
    # checkout (compute_covariance.py's path rule)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(PROJECT)),
                                    "external_modules", "code",
                                    "cosmolike_core"))
    sys.path.insert(0, os.path.join(PROJECT, "covariance"))
    import lssty1xdesi_covariance as adapter

    if args.case not in adapter.CASES or adapter.CASES[args.case][1] != "desi":
        raise SystemExit(f"--case must be a DESI case of {sorted(adapter.CASES)}")
    stem = adapter.CASES[args.case][0]
    archive = np.load(archive_path(args.case), allow_pickle=False)
    if not np.array_equal(archive["rows"], adapter.observable_rows(args.case)):
        raise SystemExit("the archive's rows are not the case's layout")
    gauss = archive["gaussian"]
    mask = np.loadtxt(os.path.join(DATA, f"{stem}.mask"))[:, 1].astype(int)
    if gauss.shape != (mask.size, mask.size):
        raise SystemExit("archive and mask sizes differ")
    live = np.nonzero(mask)[0]
    sub = gauss[np.ix_(live, live)]
    # the masked Gaussian term must be a valid covariance: Cholesky
    # raises LinAlgError unless it is symmetric positive definite
    np.linalg.cholesky(sub)
    # Gaussian band powers couple no two bands, so most entries are exact
    # zeros: only the upper triangle's nonzeros are written
    a, b = np.nonzero(np.triu(sub))
    out_file = os.path.join(DATA, f"{stem}_cov")
    with open(out_file, "w") as out:
        for i, j, v in zip(live[a], live[b], sub[a, b]):
            out.write(f"{i} {j} {v:.12e}\n")
        for i in np.nonzero(mask == 0)[0]:
            out.write(f"{i} {i} 1.0\n")
    print(f"{args.case}: wrote {out_file} with {a.size} entries over "
          f"{live.size} unmasked rows (Gaussian term, positive definite)")


if __name__ == "__main__":
    main()
