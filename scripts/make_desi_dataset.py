"""Convert the DESI DR3 sacc file into the cosmolike dataset files.

The student's forecast ships one sacc file (FITS serialization) holding
the DESI spectroscopic clustering bandpowers: 76 tracers (the thin
comoving bins), a galaxy_density_cl data table (auto spectra and
adjacent-bin cross spectra, each row one bandpower with its band
center), the bandpower windows (integer multipoles and weights), and
one Gaussian covariance over all rows. Cosmolike reads its own three
text files instead - data vector, mask and covariance, all indexed by
the position in ITS data-vector layout - plus the n(z) files named in a
.dataset descriptor. This script maps one onto the other.

Layout: the Fourier 3x2pt data vector is [EE | gs | gg | gg-cross],
each block Ncl entries per spectrum, where Ncl = the number of band
centers. The gg block holds one auto spectrum per clustering bin, in
the same positions as the auto-only import (existing masks and
fiducial vectors stay valid prefixes); the gg-cross block appends one
padded spectrum per measured cross pair, sorted by (bin_i, bin_j). The
pairs are whatever the sacc measured - the DESI forecast pairs bins by
REDSHIFT proximity, so besides the within-sample adjacent pairs it
holds LRG x ELG same-redshift crosses that are far apart in the global
bin index. The block sizes come from the compiled interface itself -
the ggl pair count depends on the lens/source overlap, and the
gg-cross count on the pair list this script passes first - so the
script initializes the interface with the project's n(z) files, sends
the pair list, and asks for the extended layout.

Outputs (into the project's data/ folder):
  lssty1xdesi_desi_dr3_ells.txt     = the union band-center list (one
                                      per line; every sacc spectrum
                                      uses a prefix of it)
  lssty1xdesi_desi_dr3_pairs.txt    = the measured cross pairs, one
                                      "bin_i bin_j" per line (i < j),
                                      sorted; the gg-cross block order
  lssty1xdesi_desi_dr3_sacc.datavector = [index, value], full layout,
                                      sacc auto and cross values in
                                      their gg blocks, zero elsewhere
  lssty1xdesi_desi_dr3.mask         = [index, 0/1], 1 exactly where a
                                      sacc bandpower exists (the
                                      per-spectrum scale cuts)
  lssty1xdesi_desi_dr3_sacc_cov     = [i, j, cov] for every imported
                                      row (auto and cross, with their
                                      correlations), plus a placeholder
                                      1.0 diagonal on every masked
                                      index (the reader demands a
                                      positive diagonal everywhere;
                                      masked rows never reach the chi2).
                                      The source forecast's covariance,
                                      kept for the comparison of the
                                      covariance guide; the likelihood
                                      reads lssty1xdesi_desi_dr3_cov,
                                      the project's own Gaussian term
                                      (scripts/make_desi_covariance.py)
  lssty1xdesi_desi_dr3_windows.npz  = per-spectrum integer multipoles
                                      and weights, kept for the
                                      exact-bandpower follow-up
  lssty1xdesi_desi_dr3.dataset      = the descriptor naming all of the
                                      above plus the n(z) files. It
                                      names the sacc data vector; the
                                      shipped descriptor names instead
                                      the model vector that
                                      make_desi_fiducial.py writes (the
                                      paper's model at its fiducial),
                                      and lssty1xdesi_desi_dr3_linear.dataset
                                      the growth-tied linear one

Usage (from the Cocoa/ folder, cocoapy311 active, start_cocoa.sh
sourced, the project compiled):

    python ./projects/lssty1xdesi/scripts/make_desi_dataset.py \
        --sacc /path/to/run_1x2pt_DESI_comoving_chocolate_non_limber.sacc
"""
import argparse
import os
import sys

import numpy as np
from astropy.io import fits

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT, "data")

# tracer prefix -> offset of that sample's first bin in the canonical
# bin order (BGS 0-17, LRG 18-48, ELG 49-75; make_desi_nz.py's order)
SAMPLE_OFFSET = {"spec_bgs": 0, "spec_lrg": 18, "spec_elg": 49}
NBIN = 76
# the LSST-Y1 source sample of the donor project (5 tomographic bins)
N_SOURCE = 5
# band centers agree across spectra when rounded here; they are
# (2l+1)-weighted means of integer windows, so they are not integers
CENTER_DECIMALS = 6


def center_key(x):
    """One rounding rule for every band-center comparison and lookup.

    np.round and Python's round can disagree in the last bit of a
    rounded double; a dictionary keyed by one and probed by the other
    then misses. Every center passes through this single function.

    Arguments:
      x = a band-center value (float or numpy scalar).

    Returns:
      the center rounded to CENTER_DECIMALS, as a plain float.
    """
    return float(np.round(np.float64(x), CENTER_DECIMALS))


def tracer_to_bin(name):
    """Map a sacc tracer name (e.g. 'spec_lrg7') to the canonical bin.

    Arguments:
      name = the tracer string from the sacc data table.

    Returns:
      the bin index in 0..75.

    Raises:
      KeyError/ValueError on a tracer outside the three DESI samples.
    """
    name = name.strip()
    for prefix, offset in SAMPLE_OFFSET.items():
        if name.startswith(prefix):
            return offset + int(name[len(prefix):])
    raise ValueError(f"unknown tracer {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Import the DESI sacc bandpowers, mask and "
                    "covariance into cosmolike dataset files.")
    parser.add_argument("--sacc", required=True,
                        help="path to the sacc FITS file")
    args = parser.parse_args()

    f = fits.open(args.sacc, memmap=True)
    d = f["data:galaxy_density_cl"].data
    tracer_0 = np.array([t.strip() for t in d["tracer_0"]])
    tracer_1 = np.array([t.strip() for t in d["tracer_1"]])
    values = np.asarray(d["value"], dtype=float)
    ells = np.asarray(d["ell"], dtype=float)
    cov = np.asarray(f["covariance"].data, dtype=float)
    nrows = len(values)
    if cov.shape != (nrows, nrows):
        raise AssertionError(
            f"covariance shape {cov.shape} does not match the "
            f"{nrows} data rows")

    # ---- the union band-center grid -------------------------------------
    # every spectrum's centers must be a prefix of one shared ascending
    # list (same binning, per-spectrum scale cut); verified row by row
    centers = np.unique([center_key(e) for e in ells])
    ncl = len(centers)
    center_index = {c: i for i, c in enumerate(centers)}
    auto = tracer_0 == tracer_1
    for name in np.unique(tracer_0[auto]):
        sel = auto & (tracer_0 == name)
        mine = np.sort([center_key(e) for e in ells[sel]])
        if not np.array_equal(mine, centers[:len(mine)]):
            raise AssertionError(
                f"{name}: band centers are not a prefix of the union "
                "grid - the shared-binning assumption is wrong")

    # ---- the measured cross pairs, in the canonical sorted order ---------
    # the same prefix check applies to every cross spectrum: one shared
    # band grid, a per-spectrum scale cut
    pair_rows = {}
    for row in range(nrows):
        if tracer_0[row] == tracer_1[row]:
            continue
        b1 = tracer_to_bin(tracer_0[row])
        b2 = tracer_to_bin(tracer_1[row])
        pair_rows.setdefault((min(b1, b2), max(b1, b2)), []).append(row)
    pairs = sorted(pair_rows)
    pair_index = {p: n for n, p in enumerate(pairs)}
    npairs = len(pairs)
    for p, rows_p in pair_rows.items():
        mine = np.sort([center_key(ells[r]) for r in rows_p])
        if not np.array_equal(mine, centers[:len(mine)]):
            raise AssertionError(
                f"cross pair {p}: band centers are not a prefix of the "
                "union grid - the shared-binning assumption is wrong")
    print(f"{nrows} sacc rows ({int(auto.sum())} autos, "
          f"{nrows - int(auto.sum())} cross bandpowers over {npairs} "
          f"pairs), Ncl = {ncl} band centers: "
          f"{centers[0]:.3f}..{centers[-1]:.3f}")

    # ---- the cosmolike layout, from the compiled interface ---------------
    # the ggl pair count depends on the lens/source overlap the C code
    # enumerates, so the layout is asked, never recomputed here
    sys.path.insert(0, os.path.join(PROJECT, "interface"))
    import cosmolike_lssty1xdesi_interface as ci
    ci.initial_setup()
    ci.init_accuracy_boost(accuracy_boost=1.0, integration_accuracy=0)
    ci.init_probes(possible_probes="3x2pt")
    ci.init_binning_fourier_ells(ells=centers,
                                 lmax_shear=int(np.ceil(centers[-1])))
    # the clustering-only likelihood excludes every (lens, source) ggl
    # pair (desi_cl.yaml: ggl_exclude all), so the layout asked below
    # must be built under the same exclusion
    ci.init_ggl_exclude(np.array(
        [[zl, zs] for zl in range(NBIN) for zs in range(N_SOURCE)],
        dtype=int).flatten())
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=os.path.join(DATA, "lssty1xdesi_desi_dr3.nz"),
        lens_ntomo=NBIN,
        source_multihisto_file=os.path.join(DATA, "lssty1xdesi_source.nz"),
        source_ntomo=N_SOURCE)
    # the pair list goes in BEFORE the layout is asked: the gg-cross
    # block size is Ncl per pair, appended after the auto gg block
    # (flattened [i0, j0, i1, j1, ...], the init_ggl_exclude convention)
    ci.init_gg_cross_pairs(
        pairs=np.asarray(pairs, dtype=int).flatten())
    sizes = ci.compute_data_vector_fourier_with_gg_cross_sizes()
    size_ee, size_gs = int(sizes[0]), int(sizes[1])
    size_gg, size_ggx = int(sizes[2]), int(sizes[3])
    if size_gg != ncl * NBIN:
        raise AssertionError(
            f"gg block is {size_gg}, expected Ncl*{NBIN} = {ncl*NBIN}")
    if size_ggx != ncl * npairs:
        raise AssertionError(
            f"gg-cross block is {size_ggx}, expected Ncl*{npairs} = "
            f"{ncl*npairs}")
    gg_start = size_ee + size_gs
    ggx_start = gg_start + size_gg
    ndata = size_ee + size_gs + size_gg + size_ggx
    print(f"fourier layout: EE {size_ee} + gs {size_gs} + gg {size_gg} "
          f"+ gg-cross {size_ggx} = {ndata} entries; gg starts at "
          f"{gg_start}, gg-cross at {ggx_start}")

    # ---- map the sacc rows into the gg and gg-cross blocks ---------------
    # cosmolike index of (bin, band)  = gg_start  + Ncl*bin + band
    # cosmolike index of (pair, band) = ggx_start + Ncl*pair + band
    dv = np.zeros(ndata)
    mask = np.zeros(ndata, dtype=int)
    sacc_row_of = {}
    n_auto_rows = 0
    for row in range(nrows):
        band = center_index[center_key(ells[row])]
        if tracer_0[row] == tracer_1[row]:
            b = tracer_to_bin(tracer_0[row])
            index = gg_start + ncl * b + band
            n_auto_rows += 1
        else:
            b1 = tracer_to_bin(tracer_0[row])
            b2 = tracer_to_bin(tracer_1[row])
            p = pair_index[(min(b1, b2), max(b1, b2))]
            index = ggx_start + ncl * p + band
        dv[index] = values[row]
        mask[index] = 1
        sacc_row_of[index] = row
    print(f"imported {n_auto_rows} auto and "
          f"{len(sacc_row_of) - n_auto_rows} cross bandpowers; "
          f"masked in {int(mask.sum())} of {ndata} entries")

    # ---- write data vector and mask --------------------------------------
    out_dv = os.path.join(DATA, "lssty1xdesi_desi_dr3_sacc.datavector")
    np.savetxt(out_dv, np.column_stack([np.arange(ndata), dv]),
               fmt=["%d", "%.12e"])
    out_mask = os.path.join(DATA, "lssty1xdesi_desi_dr3.mask")
    np.savetxt(out_mask, np.column_stack([np.arange(ndata), mask]),
               fmt="%d")

    # ---- write the covariance --------------------------------------------
    # imported entries carry the sacc covariance; every other index gets
    # a 1.0 placeholder diagonal (the reader checks the full diagonal is
    # positive before inverting, and re-masks the inverse afterwards, so
    # the placeholder can never reach the chi2). The file is the
    # comparison copy: the dataset names the project's own Gaussian
    # covariance, which scripts/make_desi_covariance.py writes
    out_cov = os.path.join(DATA, "lssty1xdesi_desi_dr3_sacc_cov")
    indices = sorted(sacc_row_of)
    with open(out_cov, "w") as out:
        for a_pos, i in enumerate(indices):
            for j in indices[a_pos:]:
                c = cov[sacc_row_of[i], sacc_row_of[j]]
                if i == j or c != 0.0:
                    out.write(f"{i} {j} {c:.12e}\n")
        for i in range(ndata):
            if mask[i] == 0:
                out.write(f"{i} {i} 1.0\n")
    print(f"wrote {out_cov}")

    # ---- the pair table, band-center list and dataset descriptor ---------
    out_pairs = os.path.join(DATA, "lssty1xdesi_desi_dr3_pairs.txt")
    with open(out_pairs, "w") as out:
        out.write("# bin_i  bin_j (i < j): the measured gg cross pairs,\n")
        out.write("# sorted; row n is spectrum n of the gg-cross block\n")
        for b1, b2 in pairs:
            out.write(f"{b1} {b2}\n")
    print(f"wrote {out_pairs} ({npairs} pairs)")
    np.savetxt(os.path.join(DATA, "lssty1xdesi_desi_dr3_ells.txt"),
               centers, fmt="%.8f")
    descriptor = os.path.join(DATA, "lssty1xdesi_desi_dr3.dataset")
    with open(descriptor, "w") as out:
        out.write("data_file = lssty1xdesi_desi_dr3_sacc.datavector\n")
        out.write("cov_file = lssty1xdesi_desi_dr3_cov\n")
        out.write("mask_file = lssty1xdesi_desi_dr3.mask\n")
        out.write("nz_lens_file = lssty1xdesi_desi_dr3.nz\n")
        out.write("nz_source_file = lssty1xdesi_source.nz\n")
        out.write(f"lens_ntomo = {NBIN}\n")
        out.write(f"source_ntomo = {N_SOURCE}\n")
        out.write("ell_file = lssty1xdesi_desi_dr3_ells.txt\n")
        out.write("lens_bin_table = lssty1xdesi_desi_dr3_bins.txt\n")
        out.write("gg_cross_pairs_file = lssty1xdesi_desi_dr3_pairs.txt\n")
    print(f"wrote {descriptor}")

    # ---- keep the exact bandpower windows for the follow-up --------------
    # the data table's window column holds the writer's window id (the
    # SACCNAME header of one window HDU); window_ind picks the row's
    # weight column inside that window
    window_of = {}
    for hdu in f:
        if hdu.name.startswith("window") and "SACCNAME" in hdu.header:
            window_of[int(hdu.header["SACCNAME"])] = hdu
    win = {}
    for row in range(nrows):
        hdu = window_of[int(d["window"][row])]
        band = center_index[center_key(ells[row])]
        if tracer_0[row] == tracer_1[row]:
            b = tracer_to_bin(tracer_0[row])
            key = f"bin{b}_band{band}"
        else:
            b1 = tracer_to_bin(tracer_0[row])
            b2 = tracer_to_bin(tracer_1[row])
            lo, hi = min(b1, b2), max(b1, b2)
            key = f"bin{lo}x{hi}_band{band}"
        w = np.asarray(hdu.data["weight"], dtype=float)
        if w.ndim == 2:
            w = w[:, int(d["window_ind"][row])]
        win[key + "_ells"] = np.asarray(hdu.data["values"], dtype=float)
        win[key + "_weights"] = w
    np.savez(os.path.join(DATA, "lssty1xdesi_desi_dr3_windows.npz"), **win)
    print(f"wrote the bandpower windows ({len(win)//2} bandpowers, "
          "auto and cross)")

    f.close()


if __name__ == "__main__":
    main()
