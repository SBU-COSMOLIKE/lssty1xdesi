"""Build the DESI clustering n(z) and bin table from the DR3 text files.

The DESI-DR3 forecast samples (BGS, LRG, ELG) arrive as one text file
per sample: the first row is a shared redshift grid (10^4 points,
uniform from z = 1e-4 to 3.5) and each later row is the UNCONVOLVED
redshift distribution of one thin comoving bin (equal comoving width,
about 62.3 Mpc). This script refines that grid five-fold by nested linear
interpolation (REFINE below says why and how), stacks the three
samples into the single multi-histogram n(z) file the cosmolike
likelihood reads (one z column followed by one column per tomographic
bin, 76 in all), and writes a bin table naming, for every bin, its
sample, its bias chunk and its mean redshift. The likelihood uses the table to map the handful of
sampled chunk parameters (one linear bias and one velocity dispersion
per chunk) onto the 76 bins at every point.

Bin order in the n(z) file and everywhere downstream: BGS 0-17, then
LRG 18-48, then ELG 49-75 (each sample in its file's row order, which
ascends in redshift).

Bias chunks follow the paper's Table 2, assigned by each bin's mean
redshift:
  BGS  = one chunk (0.1 < z < 0.4);
  LRG1 (z < 0.6), LRG2 (0.6 <= z < 0.8), LRG3 (z >= 0.8);
  ELG1 (z < 1.1), ELG2 (z >= 1.1).

Usage (from the Cocoa/ folder, cocoapy311 active, start_cocoa.sh
sourced; the input folder holds the three DR3 text files):

    python ./projects/lssty1xdesi/scripts/make_desi_nz.py \
        --input-dir /path/to/desi_dr3_inputs

Outputs (written into the project's data/ folder):
  lssty1xdesi_desi_dr3.nz       = z column + 76 n(z) columns
  lssty1xdesi_desi_dr3_bins.txt = bin index, sample, chunk index,
                                  chunk name, mean redshift
"""
import argparse
import os

import numpy as np

# the three samples, in the bin order every downstream consumer uses
SAMPLES = ("bgs", "lrg", "elg")

# chunk boundaries from the paper's Table 2, applied to the bin mean
# redshift; one (name, upper-edge) list per sample, last edge open
CHUNKS = {
    "bgs": (("BGS", np.inf),),
    "lrg": (("LRG1", 0.6), ("LRG2", 0.8), ("LRG3", np.inf)),
    "elg": (("ELG1", 1.1), ("ELG2", np.inf)),
}

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Grid refinement factor of the written n(z). On the input grid
# (dz = 3.5e-4) each thin bin holds 43-100 nonzero samples - the same
# per-bin count as a broad photometric bin - but the sharpest feature
# here is the Fingers-of-God-smoothed edge, sigma_z = (1 + zbar)
# sigma_v / c = 1.3e-3 to 2.4e-3, and that edge gets only 4-7 input
# samples per sigma. Refining five-fold gives every bin at least 215
# samples and every edge about 20-34 per sigma. The refinement is
# nested (new count = REFINE*(n - 1) + 1 over the same endpoints), so
# the original samples keep their exact values and the new points are
# linear interpolations: no information is invented beyond
# connecting the given samples.
REFINE = 5


def load_sample(input_dir, sample):
    """Read one sample's text file and split grid from bin rows.

    Arguments:
      input_dir = folder holding desi_<sample>_comoving_binned_
                  distribution_DR3.txt.
      sample    = "bgs", "lrg" or "elg".

    Returns:
      (z, rows): z = the shared redshift grid (length 10^4); rows =
      array [nbins][len(z)], one unconvolved n(z) per thin bin.

    Raises:
      AssertionError when the grid is not uniform (the cosmolike n(z)
      reader interpolates on a uniform grid) or a row is empty.
    """
    path = os.path.join(
        input_dir, f"desi_{sample}_comoving_binned_distribution_DR3.txt")
    table = np.loadtxt(path)
    z = table[0]
    rows = table[1:]
    dz = np.diff(z)
    # uniform grid check: the largest deviation from the first spacing
    # stays at rounding level, or the reader's direct-index lookup is off
    assert np.max(np.abs(dz - dz[0])) < 1.0e-10 * dz[0], \
        f"{path}: redshift grid is not uniform"
    for i, row in enumerate(rows):
        assert np.max(row) > 0.0, f"{path}: bin {i} is empty"
    # nested refinement: same endpoints, REFINE*(n - 1) + 1 points, so
    # every original node is kept and np.interp returns the original
    # value there exactly
    zf = np.linspace(z[0], z[-1], REFINE*(len(z) - 1) + 1)
    rows = np.array([np.interp(zf, z, row) for row in rows])
    return zf, rows


def main():
    parser = argparse.ArgumentParser(
        description="Stack the DESI DR3 thin-bin n(z) into one "
                    "cosmolike multi-histogram file plus a bin table.")
    parser.add_argument(
        "--input-dir", required=True,
        help="folder holding the three desi_*_comoving_binned_"
             "distribution_DR3.txt files")
    args = parser.parse_args()

    grids = []
    all_rows = []
    table_lines = []
    bin_index = 0
    for sample in SAMPLES:
        z, rows = load_sample(args.input_dir, sample)
        grids.append(z)
        for row in rows:
            # mean redshift of the bin: the weight is the distribution
            # itself, so empty tails do not move the mean
            zbar = float(np.sum(z * row) / np.sum(row))
            chunk_index = 0
            chunk_name = None
            for name, upper in CHUNKS[sample]:
                if zbar < upper:
                    chunk_name = name
                    break
                chunk_index += 1
            table_lines.append(
                f"{bin_index:4d}  {sample:4s}  {chunk_index:2d}  "
                f"{chunk_name:5s}  {zbar:.8f}")
            all_rows.append(row)
            bin_index += 1

    # every sample ships the same grid; one z column serves all 76 bins
    for z in grids[1:]:
        assert np.array_equal(grids[0], z), \
            "the three samples do not share one redshift grid"

    nz = np.column_stack([grids[0]] + all_rows)
    out_nz = os.path.join(PROJECT, "data", "lssty1xdesi_desi_dr3.nz")
    # full double precision: the uniform z grid must survive the text
    # round trip (the likelihood checks the spacing before building the
    # Fingers-of-God kernel in grid steps)
    np.savetxt(out_nz, nz, fmt="%.17e")
    print(f"wrote {out_nz}: {nz.shape[0]} redshifts x "
          f"{nz.shape[1] - 1} bins")

    out_table = os.path.join(PROJECT, "data",
                             "lssty1xdesi_desi_dr3_bins.txt")
    header = ("bin  sample  chunk_index  chunk_name  mean_redshift\n"
              "chunk_index counts chunks within the sample; the "
              "likelihood maps (sample, chunk_index) to its sampled "
              "bias and velocity-dispersion parameters")
    with open(out_table, "w") as f:
        f.write("# " + header.replace("\n", "\n# ") + "\n")
        f.write("\n".join(table_lines) + "\n")
    print(f"wrote {out_table}: {bin_index} bins")


if __name__ == "__main__":
    main()
