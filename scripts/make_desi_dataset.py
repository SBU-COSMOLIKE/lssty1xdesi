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

Layout: the Fourier 3x2pt data vector is [EE | gs | gg], each block
Ncl entries per spectrum, where Ncl = the number of band centers and
the gg block holds one auto spectrum per clustering bin (the cross-bin
spectra of the sacc have no cosmolike slot yet: the cross-bin wiring is
a held design discussion, so v0 imports the AUTO rows only). The block
sizes come from the compiled interface itself - the ggl pair count
depends on the lens/source overlap the C code enumerates - so this
script initializes the interface with the project's n(z) files and asks
compute_data_vector_3x2pt_fourier_sizes.

Outputs (into the project's data/ folder):
  lssty1xdesi_desi_dr3_ells.txt     = the union band-center list (one
                                      per line; every sacc spectrum
                                      uses a prefix of it)
  lssty1xdesi_desi_dr3_sacc.datavector = [index, value], full layout,
                                      sacc auto values in the gg block,
                                      zero elsewhere
  lssty1xdesi_desi_dr3.mask         = [index, 0/1], 1 exactly where a
                                      sacc auto bandpower exists (the
                                      per-bin scale cuts of the paper)
  lssty1xdesi_desi_dr3_cov          = [i, j, cov] for the imported auto
                                      rows, plus a placeholder 1.0
                                      diagonal on every masked index
                                      (the reader demands a positive
                                      diagonal everywhere; masked rows
                                      never reach the chi2)
  lssty1xdesi_desi_dr3_windows.npz  = per-auto-spectrum integer
                                      multipoles and weights, kept for
                                      the exact-bandpower follow-up
  lssty1xdesi_desi_dr3.dataset      = the descriptor naming all of the
                                      above plus the n(z) files

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
    print(f"{nrows} sacc rows ({int(auto.sum())} autos), "
          f"Ncl = {ncl} band centers: {centers[0]:.3f}..{centers[-1]:.3f}")

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
    sizes = ci.compute_data_vector_3x2pt_fourier_sizes()
    size_ee, size_gs, size_gg = int(sizes[0]), int(sizes[1]), int(sizes[2])
    if size_gg != ncl * NBIN:
        raise AssertionError(
            f"gg block is {size_gg}, expected Ncl*{NBIN} = {ncl*NBIN}")
    gg_start = size_ee + size_gs
    ndata = size_ee + size_gs + size_gg
    print(f"fourier layout: EE {size_ee} + gs {size_gs} + gg {size_gg} "
          f"= {ndata} entries; gg starts at {gg_start}")

    # ---- map the sacc auto rows into the gg block ------------------------
    # cosmolike index of (bin, band) = gg_start + Ncl*bin + band
    dv = np.zeros(ndata)
    mask = np.zeros(ndata, dtype=int)
    sacc_row_of = {}
    for row in range(nrows):
        if tracer_0[row] != tracer_1[row]:
            continue  # cross-bin rows wait for the cross wiring
        b = tracer_to_bin(tracer_0[row])
        band = center_index[center_key(ells[row])]
        index = gg_start + ncl * b + band
        dv[index] = values[row]
        mask[index] = 1
        sacc_row_of[index] = row
    print(f"imported {len(sacc_row_of)} auto bandpowers; "
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
    # the placeholder can never reach the chi2)
    out_cov = os.path.join(DATA, "lssty1xdesi_desi_dr3_cov")
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

    # ---- the band-center list and the dataset descriptor -----------------
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
        if tracer_0[row] != tracer_1[row]:
            continue
        hdu = window_of[int(d["window"][row])]
        b = tracer_to_bin(tracer_0[row])
        band = center_index[center_key(ells[row])]
        key = f"bin{b}_band{band}"
        w = np.asarray(hdu.data["weight"], dtype=float)
        if w.ndim == 2:
            w = w[:, int(d["window_ind"][row])]
        win[key + "_ells"] = np.asarray(hdu.data["values"], dtype=float)
        win[key + "_weights"] = w
    np.savez(os.path.join(DATA, "lssty1xdesi_desi_dr3_windows.npz"), **win)
    print(f"wrote the bandpower windows ({len(win)//2} auto bandpowers)")

    f.close()


if __name__ == "__main__":
    main()
