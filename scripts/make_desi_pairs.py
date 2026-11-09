"""Extend a DESI layout's measured cross pairs by the Fingers-of-God overlap rule.

The clustering part of the data vector holds one auto spectrum per thin
bin and one cross spectrum per measured pair of bins, the gg-cross block,
in the order of the pair file data/<stem>_pairs.txt (one "i j" line per
pair, 0-based bins, i < j, sorted). Where the base lists come from:

  Delta d_com = 62 Mpc (<stem> = lssty1xdesi_desi_dr3): the DESI
      forecast's sacc file, which pairs bins by redshift proximity
      (scripts/make_desi_dataset.py imports it): 97 pairs, kept as
      data/lssty1xdesi_desi_dr3_linear_pairs.txt for the exploration
      examples;
  Delta z = 0.02, 0.05, 0.2: the raw-range rule of
      scripts/make_desi_dz_case.py (cross_pairs).

A bin's kernel in the likelihood is its raw n(z) convolved with the
Fingers-of-God (FoG) Gaussian of width sigma_z = (1 + zbar) sigma_v / c,
with zbar the bin's mean redshift and sigma_v the velocity dispersion of
its group. Two bins whose raw ranges just miss can still overlap after
the blur. With the normalized overlap integral of two kernels

    O_ij = integral n_i n_j dz / sqrt(integral n_i^2 dz integral n_j^2 dz)

(1 for identical shapes, 0 for disjoint ones; at high ell close to the
correlation coefficient of the two bins' spectra), the rule is:

    paper pairs = DESI's measured pairs (the base list, all of them,
                  including the BGS x LRG1 boundary pair of 62 Mpc, whose
                  blurred kernels overlap by 3e-7)
                  + every pair whose FoG-blurred kernels overlap at all,
                  O_ij >= OVERLAP_MIN = 1e-6 at the default sigma_v,

with the kernels blurred as the likelihood blurs them at the default
dispersions (the DESI_SIGV fiducials of scripts/make_desi_params.py).
OVERLAP_MIN is the threshold of notes/cross_correlations for "the blurred
kernels overlap": far above the 1e-14 that the truncated Gaussian tails
of two bins one apart reach, far below any measured neighbour pair.

What the rule adds: on 62 Mpc the ten LRG3 x ELG1 pairs whose bin edges
sit 0.001-0.002 apart in redshift (raw ranges 3 to 5 grid steps apart,
raw overlap 0), as 1-based bin numbers with their mean redshifts and
blurred overlaps:

    LRG3 bin (zbar)    ELG1 bin (zbar)    O_ij
    40 (0.8361)        50 (0.8118)        0.0277
    41 (0.8594)        51 (0.8351)        0.0344
    42 (0.8832)        52 (0.8585)        0.0287
    43 (0.9073)        53 (0.8823)        0.0346
    44 (0.9320)        54 (0.9063)        0.0289
    45 (0.9568)        55 (0.9308)        0.0295
    46 (0.9820)        56 (0.9560)        0.0317
    47 (1.0075)        57 (0.9809)        0.0340
    48 (1.0337)        58 (1.0068)        0.0309
    49 (1.0602)        59 (1.0328)        0.0308

and nothing on the three Delta z layouts, where no unmeasured pair
overlaps by more than 1e-6. No unmeasured pair of any layout lies between
1e-6 and 0.0277, so any threshold in [1e-6, 0.0277] gives the same lists:
the ten on 62 Mpc, none on the Delta z layouts.
A threshold at the smallest overlap of a measured LRG3 x ELG1 pair
(0.071 on 62 Mpc) would add nothing: the ten pairs overlap about half as
much as the measured ones.

Outputs, into data/ (or --out-dir), for a layout the rule extends:
  <stem>_pairs.txt = the extended list, sorted (i < j);
  <stem>.mask      = the mask on the extended layout: the shear and auto
                     rows as before, and each cross pair kept on the bands
                     where both of its autos are kept, the rule every pair
                     of the base mask follows (checked before writing).
The data vector and the covariance follow the pair list: rebuild them
with scripts/make_desi_fiducial.py and the covariance pipeline
(covariance/README.md, scripts/make_desi_covariance.py). The script only
reads data/<stem>_pairs.txt and data/<stem>.mask as the base, so a
second run finds nothing to add and leaves the files as they are.

Usage (from the Cocoa/ folder, cocoapy311 active):
    python ./projects/lssty1xdesi/scripts/make_desi_pairs.py --case dr3 --write
Without --write it prints the pairs the rule would add.
"""
import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SCRIPTS)
DATA = os.path.join(PROJECT, "data")
sys.path.insert(0, SCRIPTS)
import make_desi_params as parmod   # noqa: E402  the sigma_v groups and fiducials

# speed of light [km/s]: a line-of-sight velocity v shifts a measured
# redshift by (1 + z) v / c
C_KMS = 299792.458
# the blurred-overlap threshold of the rule (see the module docstring)
OVERLAP_MIN = 1.0e-6
# the layouts and their data stems
CASES = {"dr3": "lssty1xdesi_desi_dr3",
         "dr1_dz0p02": "lssty1xdesi_desi_dr1_dz0p02",
         "dr1_dz0p05": "lssty1xdesi_desi_dr1_dz0p05",
         "dr1_dz0p20": "lssty1xdesi_desi_dr1_dz0p20"}
# the bands of every spectrum (the band-center file of every DESI layout)
NCL = 53


def blurred_kernels(stem):
    """The FoG-blurred n(z) of every bin at the default dispersions.

    Arguments:
      stem = the layout's data stem.

    Returns:
      (z, kernels, samples, groups, zbar): the redshift grid [nz], the
      blurred n(z) [nz, nbin], and per bin its sample, group and mean
      redshift.
    """
    table = np.loadtxt(os.path.join(DATA, f"{stem}_bins.txt"), usecols=(1, 3),
                       dtype=str, ndmin=2)
    samples = [str(s) for s, g in table]
    groups = [str(g) for s, g in table]
    zbar = np.loadtxt(os.path.join(DATA, f"{stem}_bins.txt"), usecols=4, ndmin=1)
    nz = np.loadtxt(os.path.join(DATA, f"{stem}.nz"))
    z = nz[:, 0]
    dz = float(z[1] - z[0])
    fiducial = parmod.MODELS["params_desi.yaml"]["sigv_values"]
    kernels = np.empty_like(nz[:, 1:])
    for i in range(len(zbar)):
        sigma_v = fiducial[parmod.sigv_group_key(samples[i], groups[i], zbar[i])]
        sigma_z = (1.0 + zbar[i])*sigma_v/C_KMS
        # the likelihood's convolution (desi_cl.set_lens_related): a
        # Gaussian of sigma_z, in grid steps, zero beyond the grid
        kernels[:, i] = gaussian_filter1d(nz[:, i + 1], sigma=sigma_z/dz,
                                          mode="constant")
    return z, kernels, samples, groups, zbar


def overlap(n_a, n_b, z):
    """The normalized overlap integral O of two distributions (see above)."""
    cross = np.trapz(n_a*n_b, z)
    return float(cross/np.sqrt(np.trapz(n_a*n_a, z)*np.trapz(n_b*n_b, z)))


def added_pairs(stem, base):
    """The pairs the rule adds to a base list.

    Arguments:
      stem = the layout's data stem.
      base = set of (i, j) base pairs, 0-based, i < j.

    Returns:
      list of (i, j, O_ij) of every non-base pair with O_ij >= OVERLAP_MIN,
      sorted.
    """
    z, kernels, samples, groups, zbar = blurred_kernels(stem)
    nbin = kernels.shape[1]
    # the redshift support of each blurred kernel: pairs whose supports
    # do not meet have O = 0 and are skipped without an integral
    first = np.zeros(nbin)
    last = np.zeros(nbin)
    for i in range(nbin):
        live = np.nonzero(kernels[:, i] > 0.0)[0]
        first[i] = z[live[0]]
        last[i] = z[live[-1]]
    out = []
    for i in range(nbin):
        for j in range(i + 1, nbin):
            if (i, j) in base or first[j] > last[i] or first[i] > last[j]:
                continue
            value = overlap(kernels[:, i], kernels[:, j], z)
            if value >= OVERLAP_MIN:
                out.append((i, j, value))
    return out, groups, zbar


def extended_mask(old_mask, nbin, base_pairs, pairs):
    """The mask on the extended layout.

    Arguments:
      old_mask   = the base mask values, [n_old] (0 or 1).
      nbin       = the number of lens bins.
      base_pairs = the base pair list, in its order.
      pairs      = the extended pair list, in its order.

    Returns:
      int array [n_new]: the shear and auto rows of the base mask, then
      per pair the bands where both autos are kept.

    Raises:
      ValueError when a base pair's mask is not "both autos kept".
    """
    n_ee = old_mask.size - NCL*(nbin + len(base_pairs))
    autos = old_mask[n_ee:n_ee + NCL*nbin].reshape(nbin, NCL)
    cross_old = old_mask[n_ee + NCL*nbin:].reshape(len(base_pairs), NCL)
    for k, (i, j) in enumerate(base_pairs):
        if not np.array_equal(cross_old[k], autos[i]*autos[j]):
            raise ValueError(f"base pair ({i}, {j}) does not keep exactly the bands "
                             "both autos keep; the rule would change it")
    cross_new = np.array([autos[i]*autos[j] for i, j in pairs], dtype=int)
    return np.concatenate([old_mask[:n_ee + NCL*nbin], cross_new.ravel()])


def main():
    parser = argparse.ArgumentParser(
        description="Extend a DESI layout's cross pairs by the FoG overlap rule.")
    parser.add_argument("--case", default="dr3", choices=sorted(CASES))
    parser.add_argument("--write", action="store_true",
                        help="write the extended pair list and mask")
    parser.add_argument("--out-dir", default=DATA,
                        help="folder of the written files (default: data/)")
    args = parser.parse_args()
    stem = CASES[args.case]
    base_pairs = [tuple(int(v) for v in row) for row in
                  np.loadtxt(os.path.join(DATA, f"{stem}_pairs.txt"), dtype=int, ndmin=2)]
    added, groups, zbar = added_pairs(stem, set(base_pairs))
    print(f"{args.case}: {len(base_pairs)} base pairs; the rule adds {len(added)}")
    for i, j, value in added:
        print(f"  bins {i + 1} ({groups[i]}, zbar {zbar[i]:.4f}) x {j + 1} "
              f"({groups[j]}, zbar {zbar[j]:.4f}): O = {value:.4f}")
    if not args.write or not added:
        return 0
    pairs = sorted(base_pairs + [(i, j) for i, j, value in added])
    old_mask = np.loadtxt(os.path.join(DATA, f"{stem}.mask"))[:, 1].astype(int)
    mask = extended_mask(old_mask, nbin=len(zbar), base_pairs=base_pairs, pairs=pairs)
    with open(os.path.join(args.out_dir, f"{stem}_pairs.txt"), "w") as out:
        out.write("# bin_i  bin_j (i < j): the measured gg cross pairs and the\n")
        out.write("# FoG-overlap pairs of scripts/make_desi_pairs.py, sorted;\n")
        out.write("# row n is spectrum n of the gg-cross block\n")
        for i, j in pairs:
            out.write(f"{i} {j}\n")
    np.savetxt(os.path.join(args.out_dir, f"{stem}.mask"),
               np.column_stack([np.arange(mask.size), mask]), fmt="%d")
    print(f"wrote {len(pairs)} pairs and a mask of {mask.size} rows "
          f"({int(mask.sum())} unmasked; base {old_mask.size}, {int(old_mask.sum())})"
          f" into {args.out_dir}")
    return 0


if __name__ == "__main__":
    main()
