"""Fit the pair-axes study of the cross-correlations note on one DESI layout.

The question
------------
The DESI clustering part of the data vector holds one auto spectrum per thin
redshift bin and one cross spectrum per measured pair of bins. The measured
pairs are the pair list data/<stem>_pairs.txt, called the current list below.
The study asks how much the figures of merit grow when more pairs are
measured. It adds pairs along two axes, one layout at a time:

- the gap axis. Point "gap<=g" holds the current pairs plus every pair whose
  raw n(z) ranges lie at most g bin widths apart. The raw range of a bin is
  the redshift interval where its n(z) file is nonzero, before the
  Fingers-of-God blur. The gap of two bins is the empty interval between the
  two ranges (zero when they touch or overlap), divided by the bin width of
  the layout: Delta z for the fixed-width layouts, and for
  Delta d_com = 62 Mpc the median comoving width of its bins (the distances
  in Mpc come from a CAMB table, see Prerequisites).
- the sigma_v axis. Point "sigv=s" holds the current pairs plus every pair
  whose blurred kernels overlap by O_ij >= 1e-6 when every velocity
  dispersion is set to s km/s. The blur and the overlap O_ij are those of
  scripts/make_desi_pairs.py, the rule that makes the current list at the
  fiducial dispersions (its docstring defines O_ij). s only selects pairs:
  every fit keeps the model at the fiducial sigma_v.

Point "today" is the current list itself. Only the points whose pairs all lie
in the layout's maximal pair list, data/<stem>_pairmax_pairs.txt, are fitted:
the derivatives exist for those pairs only.

How a point is fitted: row selection
------------------------------------
One Fisher derivative base was computed per layout on its maximal list (the
archived caches model_derivatives_<tag>_pairmax_final and
model_templates_<tag>_pairmax_final). A smaller pair list is a subset of the
rows of that data vector: its autos plus its own pairs. The Fisher matrix of
the smaller list is built from the same derivative rows and the same
covariance block, so no new evaluation is needed:

    rows = (the entries the scale cut keeps) minus (the rows of every pair
           outside the point's list),
    G    = (A/A_cov) B[rows]^T C[rows, rows]^-1 B[rows],

with B the base columns (n x n_base), C the Gaussian covariance of the
maximal list and A/A_cov the ratio of the forecast area to the area of the
covariance file (INVCOV_SCALE of the notebook). C[rows, rows] is factorized
as L L^T (Cholesky) and W = L^-1 B[rows] is solved by forward substitution,
so G = W^T W without forming an inverse. The scale cut keeps a band of a
spectrum when the UV-fraction criterion of the notebook passes at
k_max/D(zbar_i) for each bin of it; k_max is quoted at z = 0 (0.075, 0.1
and 0.2 h/Mpc here).

The fits themselves are the notebook's: this script runs the code cells of
EXAMPLE_EVALUATE_FISHER_MODEL.ipynb in one Python namespace (see "The cell
runner" below), then puts each point's G in place of the notebook's matrix at
that cut (GRAM[key][m]) while the notebook's fit_system function builds and
inverts the Fisher matrix, and restores the notebook's matrix afterwards.
fit_system returns the figures of merit

    FoM(w0, wa)              = 1/sqrt(det C_(w0, wa)),
    FoM(Omega_m, 10^9 A_s)   = 1/sqrt(det C_(Omega_m, 10^9 A_s)),

with C_(a, b) the 2 x 2 block of the inverse Fisher matrix (the parameter
covariance marginalized over every other parameter).

The three fit families
----------------------
1. panel (figures pair_axes_gap and pair_axes_sigv): every point, two models
   on the pessimistic linear-bias prior, each with sigma_v fitted and with
   sigma_v fixed at the fiducial (its columns removed):
     paper     = 'B tilts noA2 sv x6', the notebook's SUMMARY_VARIANT: model
                 B + tilts with the alpha_2 counterterms fixed at 0 and every
                 Gaussian sigma_v prior width multiplied by 6;
     reference = 'B tilts': alpha_2 free, the sigma_v widths of the
                 exploration fits (the model of the pair_threshold figure).
2. attribution (figure pair_axes_attribution): the points today and gap<=g,
   the paper model with blocks of nuisance parameters held at their fiducial
   values (ATTRIBUTION_BLOCKS). Holding a block fixed deletes its rows and
   columns from the full Fisher matrix (data plus every prior) before the
   inversion; for a Gaussian likelihood this is exact conditioning on those
   parameters. Two more blocks split the per-bin linear biases: per bias
   group the bin biases are b = U a + E e, with U = (1, z - zbar) the free
   level and slope of the group, E the residual directions that the prior
   charges (U^T Q e = 0, Q the precision of the correlated prior), so
   keeping only U holds the residuals fixed and keeping only E holds the
   level and the slope fixed.
3. priors (figure pair_axes_priors): the points today and gap<=g, the paper
   model under three priors of the per-bin amplitudes beta_i = b_1,i D(zbar_i)
   (b_1,i the linear bias of bin i, D the linear growth factor):
     baseline    = per group a free level and slope in redshift plus
                   residuals of width 1% (BGS 3%) correlated as
                   exp(-|z_i - z_j|/L), L = 0.2;
     pessimistic = the same with 3% in every group and L = 0.05 (the paper
                   prior);
     free        = every beta_i free (the notebook's 'independent').

The cell runner
---------------
The notebook's code cells are run by index. Each index is checked against
the first line its cell must start with (FIRST_LINES), so a rebuilt notebook
with moved cells stops here instead of running the wrong code. The settings
cell (index 4) gets three text edits, each of which must match exactly once:
it must set CUT_ACCURACY = DERIVATIVE_TIER (the final-tier scale cuts); the
layout of this run gets its maximal-list dataset <stem>_pairmax and its yaml
EXAMPLE_EVALUATE<n>_PAIRMAX.yaml; and its cache tag becomes <tag>_pairmax,
so the notebook reads model_*_<tag>_pairmax_final for it. The other three
layouts stay on their current lists and caches (the notebook builds the
Fisher systems of every layout). Cell 13 stops before its comparison with
the linear notebook's cuts, which hold the current pairs only. The cut cell
9 is not run: this script reads the two cut caches itself (the current one
for the other layouts, the maximal-list one for this layout).

Prerequisites
-------------
This script never copies, loads or writes a cache by itself; it reads the
working copies in chains/ and stops with the command to run when one is
missing. From the folder Cocoa/, with Cocoa started:

1. Copy the archived final-tier caches to their working copies in chains/,
   one call per cache of fm = interface/cosmolike_lssty1xdesi_fisher.py
   (the call of the notebook's section 2b; a run with a missing copy stops
   and prints the full python -c command of each):
       fm.load_fisher_from_cache('model_derivatives_<tag>_pairmax_final')
       fm.load_fisher_from_cache('model_templates_<tag>_pairmax_final')
       fm.load_fisher_from_cache('model_scale_cuts_<tag>_pairmax_final')
       fm.load_fisher_from_cache('model_scale_cuts_final')
   and, for each of the three other layouts <other>:
       fm.load_fisher_from_cache('model_derivatives_<other>_final')
       fm.load_fisher_from_cache('model_templates_<other>_final')
2. Build the covariance file of the maximal list, data/<stem>_pairmax_cov,
   with the project's covariance command (covariance/README.md) on the
   layout's EXAMPLE_EVALUATE_COVARIANCE*_PAIRMAX.yaml, then install its
   Gaussian term:
       unset IGNORE_COSMOLIKE_LSSTY1XDESI_COVARIANCE
       source projects/lssty1xdesi/scripts/compile_lssty1xdesi.sh
       python projects/lssty1xdesi/covariance/compute_covariance.py \\
           projects/lssty1xdesi/EXAMPLE_EVALUATE_COVARIANCE<n>_PAIRMAX.yaml
       python projects/lssty1xdesi/scripts/make_desi_covariance.py \\
           --case <case>_pairmax
   (<n> is empty for dcom62 and 2, 3, 4 for dz0p02, dz0p05, dz0p20; <case>
   is dr3, dr1_dz0p02, dr1_dz0p05, dr1_dz0p20). The three other layouts
   need their current covariance files data/<stem>_cov, built the same way
   from EXAMPLE_EVALUATE_COVARIANCE<n>.yaml and --case <case>.
3. For dcom62 only: the CAMB table covariance/forecast_camb.npz (redshift
   z_1D and comoving distance chi in Mpc/h), which the covariance notebook
   EXAMPLE_EVALUATE_COVARIANCE.ipynb writes in its "Save the calculation"
   cell. The gap axis of that layout converts the raw ranges to Mpc with
   it. Some dcom62 pairs sit 1e-4 bin widths from a gap level, so another
   distance table (another z grid, another integration) can move a pair
   from one point to the next.

Usage
-----
From the folder Cocoa/, with Cocoa started; one layout per run, the runs one
after the other (wall time and peak memory on an Apple M2 Pro with 32 GB, on
AC power: dz0p20 19 s and 3.3 GB, dz0p05 27 s and 3.6 GB, dz0p02 61 s and
9.6 GB, dcom62 47 s and 7.6 GB):

    python projects/lssty1xdesi/scripts/compute_pair_axes_fits.py --layout dz0p20
    python projects/lssty1xdesi/scripts/compute_pair_axes_fits.py --layout dz0p05
    python projects/lssty1xdesi/scripts/compute_pair_axes_fits.py --layout dz0p02
    python projects/lssty1xdesi/scripts/compute_pair_axes_fits.py --layout dcom62

--dry-run builds and prints the axis points and checks the prerequisites
without running the notebook (seconds). Each run writes
chains/pair_axes_fits_<tag>.json (or --output); an existing file is replaced
only with --overwrite. The three data generators of
notes/cross_correlations/plots (make_pair_axes_data.py,
make_pair_axes_attribution_data.py, make_pair_axes_priors_data.py) read
these files.

Output
------
One JSON file per layout: 'layout', 'tag', 'kmax_h_per_Mpc', 'points'
({point: {'n_pairs', 'pairs'}}, every pair 0-based [i, j] with i < j), and
one entry per fit family, each with its model names and its 'cuts':
  panel:       cuts[k_max][point][model][sigma_v] = {fom, fom_om_as, status};
  attribution: cuts[k_max][point][block][sigma_v] = {fom, fom_om_as};
  priors:      cuts[k_max][point][prior][sigma_v] = {fom, fom_om_as, status};
with model 'paper' or 'reference', sigma_v 'sigma_v free' or 'sigma_v fixed',
fom = FoM(w0, wa), fom_om_as = FoM(Omega_m, 10^9 A_s), and status the
notebook's fit_system status ('fit', or why there is no fit; then the FoMs
are null).
"""
import argparse
import datetime
import json
import os
import sys
import time

# The notebook's first code cell sets OMP_NUM_THREADS = 8 before numpy loads.
# numpy's BLAS library (OpenBLAS) reads the thread count once, when it loads,
# and its matrix products and Cholesky factorizations add their terms in an
# order that depends on that count: the FoMs then change in their last digits
# (about 1e-10 relative). numpy loads here, before the notebook's cells run,
# so the notebook's value is set first; start_cocoa.sh exports 1.
NOTEBOOK_THREADS = "8"
os.environ["OMP_NUM_THREADS"] = NOTEBOOK_THREADS

import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

# The project modules imported below live in this repository: no bytecode
# caches are written next to them.
sys.dont_write_bytecode = True

# project = projects/lssty1xdesi, one level above this script's folder
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SCRIPTS)
DATA = os.path.join(PROJECT, "data")
CHAINS = os.path.join(PROJECT, "chains")
sys.path.insert(0, SCRIPTS)
# the Fingers-of-God overlap rule of the pair lists: its blur, its overlap
# integral, the speed of light and the sigma_v groups and fiducials
import make_desi_pairs as pair_rule  # noqa: E402

# the notebook whose code cells build and solve the Fisher systems
NOTEBOOK = os.path.join(PROJECT, "EXAMPLE_EVALUATE_FISHER_MODEL.ipynb")
# the CAMB tables of the covariance notebook's run; the gap axis of dcom62
# reads its comoving distances (Prerequisites, item 3)
CAMB_TABLE = os.path.join(PROJECT, "covariance", "forecast_camb.npz")

# The four layouts: tag -> the notebook's layout key, the number n of its
# paper yaml EXAMPLE_EVALUATE<n>.yaml, its data stem, its linear-bias
# parameter file (the paper model's pessimistic prior), the case name of
# its covariance yaml, and its bin width in redshift (None: the comoving
# layout, whose width is measured in Mpc).
LAYOUTS = {
    "dcom62": {
        "key": "Delta d_com = 62 Mpc",
        "number": 1,
        "stem": "lssty1xdesi_desi_dr3",
        "bias_params": "params_desi_bias_pessimistic.yaml",
        "case": "dr3",
        "width_z": None,
    },
    "dz0p02": {
        "key": "Delta z = 0.02",
        "number": 2,
        "stem": "lssty1xdesi_desi_dr1_dz0p02",
        "bias_params": "params_desi_dr1_dz0p02_bias_pessimistic.yaml",
        "case": "dr1_dz0p02",
        "width_z": 0.02,
    },
    "dz0p05": {
        "key": "Delta z = 0.05",
        "number": 3,
        "stem": "lssty1xdesi_desi_dr1_dz0p05",
        "bias_params": "params_desi_dr1_dz0p05_bias_pessimistic.yaml",
        "case": "dr1_dz0p05",
        "width_z": 0.05,
    },
    "dz0p20": {
        "key": "Delta z = 0.2",
        "number": 4,
        "stem": "lssty1xdesi_desi_dr1_dz0p20",
        "bias_params": "params_desi_dr1_dz0p20_bias_pessimistic.yaml",
        "case": "dr1_dz0p20",
        "width_z": 0.20,
    },
}
# the derivative tier of every cache read here (the archive holds the final
# tier only)
TIER = "final"

# The gap levels g [bin widths] of the gap axis. Levels beyond the maximal
# list of a layout are built and then skipped (every layout's maximal list
# ends at g = 2 or 3).
GAP_LEVELS = [0, 1, 2, 3, 4, 5, 6, 8]
# A gap counts as at most g when it is at most g + GAP_TOL bin widths: the
# n(z) files sample redshift on a grid, so a raw range ends up to one grid
# step away from the bin edge, and a pair g widths apart comes out within
# about 0.02 widths of g.
GAP_TOL = 0.02
# the sigma_v values [km/s] of the sigma_v axis: from about the fiducial
# group dispersions (289-371 km/s) up to 1200 km/s
SIGV_LEVELS = [450.0, 600.0, 750.0, 900.0, 1200.0]
# H0/(100 km/s/Mpc) of the forecast cosmology: the CAMB table holds chi in
# Mpc/h, and the comoving bin width of dcom62 is quoted in Mpc
HUBBLE_H = 0.6732

# the cuts of every fit [h/Mpc at z = 0], as the keys of the output
CUT_LABELS = ["0.075", "0.1", "0.2"]
# the linear-bias prior of the panel and attribution fits (the paper prior)
PRIOR = "pessimistic"
# the paper model: the notebook's SUMMARY_VARIANT, defined in cell 35
PAPER_VARIANT = "B tilts noA2 sv x6"
# the reference model: model B + tilts as the notebook defines it in cell 23
REFERENCE_VARIANT = "B tilts"
# the suffix of a variant with every sigma_v column removed (sigma_v held
# at the fiducial)
NO_SIGV = " noSV"
# The panel fits: (model, sigma_v treatment) -> the notebook variant.
PANEL_VARIANTS = {
    ("reference", "sigma_v free"): REFERENCE_VARIANT,
    ("reference", "sigma_v fixed"): REFERENCE_VARIANT + NO_SIGV,
    ("paper", "sigma_v free"): PAPER_VARIANT,
    ("paper", "sigma_v fixed"): PAPER_VARIANT + NO_SIGV,
}
# The nuisance blocks of the attribution fits: block name -> the prefixes of
# the parameter names it holds at the fiducial. DESI_B1_<n> is the amplitude
# of bin n; DESI_DB2_<g> and DESI_TB2_<g> the offset and redshift tilt of the
# second-order bias b_2 in group g; DESI_DBS_<g> and DESI_TBS_<g> the same
# for the tidal bias b_s2; DESI_A0_<g> and DESI_TA0_<g> the counterterm
# alpha_0 and its tilt; DESI_SN0_<g> and DESI_SN2_<g> the two shot-noise
# amplitudes. "all nuisance fixed" leaves the cosmology and sigma_v free.
ATTRIBUTION_BLOCKS = {
    "b1 per bin fixed": [
        "DESI_B1_",
    ],
    "b2 bs2 fixed": [
        "DESI_DB2",
        "DESI_TB2",
        "DESI_DBS",
        "DESI_TBS",
    ],
    "a0 fixed": [
        "DESI_A0",
        "DESI_TA0",
    ],
    "shot noise fixed": [
        "DESI_SN0",
        "DESI_SN2",
    ],
    "all nuisance fixed": [
        "DESI_B1_",
        "DESI_DB2",
        "DESI_TB2",
        "DESI_DBS",
        "DESI_TBS",
        "DESI_A0",
        "DESI_TA0",
        "DESI_SN0",
        "DESI_SN2",
    ],
}
# The two split blocks of the per-bin biases: block name -> the part of
# the b = U a + E e decomposition kept free (U: level and slope; E: the
# residuals).
SPLIT_BLOCKS = {
    "b1 residuals fixed (level, slope free)": "U",
    "b1 level and slope fixed (residuals free)": "E",
}
# The prior fits: output name -> the notebook's prior name in fit_system.
PRIOR_FITS = {
    "baseline": "baseline",
    "pessimistic": "pessimistic",
    "free": "independent",
}
# Two FoMs of one Fisher matrix computed two ways (fit_system's
# eigen-decomposition and a plain inverse) agree to this relative accuracy
# when the attribution rebuilds the matrix correctly.
REBUILD_RTOL = 1.0e-8

# The notebook cells this script runs, each with the first line its source
# must start with. 2: imports; 4: settings; 8: data tables and cut settings;
# 13: the cut list; 15-16: the derivative base and its caches; 20: the base
# checks and the priors; 23: the Fisher systems; 25: the FoM tables;
# 28, 31: the bias truths and residual covariances that cell 35 uses;
# 35: the summary model (the paper model, SUMMARY_VARIANT).
FIRST_LINES = {
    2: "import importlib",
    4: "PROJECT = 'projects/lssty1xdesi/'",
    8: "def read_rows(path):",
    13: "# The cached k_max column is (nk, 1): flatten it.",
    15: "def alpha2_setter():",
    16: "# Step 2: read the two caches of every layout",
    20: "def lazeyras_slope(b1):",
    23: "# the base block of each per-group parameter family",
    25: "FOM = {}",
    28: "def block_vector(key, amplitudes):",
    31: "def residual_covariance(key, prior):",
    35: "sys.path.insert(0, NOTE_PLOTS)",
}
SETUP_CELLS = [2, 4, 8]
FIT_CELLS = [13, 15, 16, 20, 23, 25, 28, 31, 35]
SETTINGS_CELL = 4
# the settings line that selects the final-tier scale cuts, which must be
# in the settings cell exactly once
CUT_TIER_LINE = "CUT_ACCURACY = DERIVATIVE_TIER\n"
# the settings-cell lines before which the layout and its tag are replaced
LAYOUT_ANCHOR = "# The linear parameter files of the exploration fits"
TAG_ANCHOR = "DERIVATIVES_NAME = {}\nTEMPLATES_NAME = {}"
# Cell 13 ends with a comparison against the linear notebook's cuts, which
# exist for the current pair lists only: the cell runs up to this line.
TRUNCATE_AT = {13: "# the linear notebook's cuts on the linear-bias spectra"}


def parse_arguments():
    """Read the command line.

    Returns:
      argparse.Namespace with layout (a LAYOUTS tag), output (the JSON
      path), overwrite and dry_run (booleans).
    """
    parser = argparse.ArgumentParser(
        description="Fit the pair-axes points of one DESI layout by row selection "
                    "from its maximal-list derivative base (see the module docstring).")
    parser.add_argument(
        "--layout", required=True, choices=list(LAYOUTS),
        help="dcom62 (Delta d_com = 62 Mpc), dz0p02, dz0p05 or dz0p20 "
             "(Delta z = 0.02, 0.05, 0.2)")
    parser.add_argument(
        "--output", default=None,
        help="the JSON file to write (default: chains/pair_axes_fits_<layout>.json)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing output file")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="build and print the axis points and check the prerequisites; no fit")
    return parser.parse_args()


def dataset_entries(stem):
    """Return the "key = value" lines of data/<stem>.dataset as a dict.

    The .dataset file is the small text file that tells the likelihood which
    data files a layout uses (data vector, covariance, mask, n(z), bin table,
    pair list), one "key = file name" line each.

    Arguments:
      stem = the data stem, e.g. lssty1xdesi_desi_dr1_dz0p20_pairmax.

    Returns:
      dict {key: value}, both stripped strings.

    Raises:
      FileNotFoundError when the file does not exist.
    """
    path = os.path.join(DATA, stem + ".dataset")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist: the layout's dataset file "
                                "is part of the repository's data/ folder")
    entries = {}
    with open(path) as stream:
        for line in stream:
            if "=" not in line:
                continue
            # split at the first "=" only: the value is a file name
            key, value = line.split("=", 1)
            entries[key.strip()] = value.strip()
    return entries


def read_pairs(path):
    """Return the pair list of a pair file as a list of (i, j) tuples.

    Arguments:
      path = a data/*_pairs.txt file: one "i j" line per pair, 0-based bins,
             lines starting with # skipped (np.loadtxt's default).

    Returns:
      list of (int, int), in the order of the file (the order of the
      gg-cross block of the data vector).
    """
    rows = np.loadtxt(path, dtype=int, ndmin=2)
    pairs = []
    for row in rows:
        pairs.append((int(row[0]), int(row[1])))
    return pairs


def read_layout(tag):
    """Read the bins, the raw n(z) and the two pair lists of one layout.

    The current and the maximal-list datasets must name the same n(z) file
    and bin table: the maximal list measures more pairs of the same bins.

    Arguments:
      tag = a LAYOUTS key.

    Returns:
      dict with 'samples' and 'groups' (per bin: the sample bgs/lrg/elg and
      the bias group BGS..ELG2), 'zbar' (per-bin mean redshift, [nbin]),
      'nz' (the raw n(z) table [n_z, 1 + nbin], column 0 the redshift grid),
      'current' (the current pair list) and 'maximal' (the maximal list).

    Raises:
      ValueError when the two datasets name other files, when the maximal
      list is not sorted or does not contain the current one.
    """
    stem = LAYOUTS[tag]["stem"]
    current = dataset_entries(stem)
    maximal = dataset_entries(stem + "_pairmax")
    # make_desi_pairs reads <stem>_bins.txt and <stem>.nz, so both datasets
    # must point at exactly those files
    expected = {"lens_bin_table": f"{stem}_bins.txt", "nz_lens_file": f"{stem}.nz"}
    for entry, name in expected.items():
        for label, entries in (("current", current), ("maximal-list", maximal)):
            if entries.get(entry) != name:
                raise ValueError(
                    f"the {label} dataset of {tag} has {entry} = {entries.get(entry)!r}; "
                    f"the pair axes need {name!r} in both datasets (the maximal list "
                    "measures more pairs of the same bins)")
    bin_table = os.path.join(DATA, current["lens_bin_table"])
    # columns 1 and 3 of the bin table: the sample and the bias group name
    table = np.loadtxt(bin_table, usecols=(1, 3), dtype=str, ndmin=2)
    samples = []
    groups = []
    for row in table:
        samples.append(str(row[0]))
        groups.append(str(row[1]))
    zbar = np.loadtxt(bin_table, usecols=4, ndmin=1)
    nz = np.loadtxt(os.path.join(DATA, current["nz_lens_file"]))
    current_pairs = read_pairs(os.path.join(DATA, current["gg_cross_pairs_file"]))
    maximal_pairs = read_pairs(os.path.join(DATA, maximal["gg_cross_pairs_file"]))
    if maximal_pairs != sorted(maximal_pairs):
        raise ValueError(f"{maximal['gg_cross_pairs_file']} is not sorted: the maximal "
                         "list is written in sorted order (i < j)")
    missing = set(current_pairs) - set(maximal_pairs)
    if missing:
        raise ValueError(f"{len(missing)} current pairs of {tag} are not in the maximal "
                         f"list, e.g. {sorted(missing)[:3]}: rebuild the maximal list "
                         "from the current one")
    return {"samples": samples, "groups": groups, "zbar": zbar, "nz": nz,
            "current": current_pairs, "maximal": maximal_pairs}


def blurred_kernels(layout, sigv_of_group):
    """Blur every bin's raw n(z) with its Fingers-of-God Gaussian.

    The likelihood convolves the n(z) of bin i with a Gaussian of width
    sigma_z = (1 + zbar_i) sigma_v / c, sigma_v the dispersion of the bin's
    group; scipy's gaussian_filter1d does it on the grid (sigma in grid
    steps, zero beyond the grid, truncated at 4 sigma), as in
    make_desi_pairs.blurred_kernels, which is this function at the fiducial
    dispersions.

    Arguments:
      layout        = read_layout's dict.
      sigv_of_group = {sigma_v group (make_desi_params.sigv_group_key):
                      sigma_v in km/s}.

    Returns:
      array [n_z, nbin]: the blurred kernels on the n(z) grid.
    """
    nz = layout["nz"]
    z = nz[:, 0]
    dz = float(z[1] - z[0])
    kernels = np.empty_like(nz[:, 1:])
    for i in range(len(layout["zbar"])):
        group = pair_rule.parmod.sigv_group_key(layout["samples"][i], layout["groups"][i],
                                                layout["zbar"][i])
        sigma_z = (1.0 + layout["zbar"][i])*sigv_of_group[group]/pair_rule.C_KMS
        kernels[:, i] = gaussian_filter1d(nz[:, i + 1], sigma=sigma_z/dz, mode="constant")
    return kernels


def overlapping_pairs(z, kernels, skip):
    """Return the overlap O_ij of every pair outside skip whose kernels meet.

    O_ij is make_desi_pairs.overlap. A pair whose kernel supports (the
    interval where a kernel is nonzero) do not meet has O_ij = 0 and is
    skipped without an integral.

    Arguments:
      z       = the redshift grid [n_z].
      kernels = the blurred kernels [n_z, nbin].
      skip    = set of (i, j) pairs left out (the current list).

    Returns:
      dict {(i, j): O_ij} with i < j, for the pairs with O_ij > 0.
    """
    nbin = kernels.shape[1]
    first = np.zeros(nbin)
    last = np.zeros(nbin)
    for i in range(nbin):
        live = np.nonzero(kernels[:, i] > 0.0)[0]
        first[i] = z[live[0]]
        last[i] = z[live[-1]]
    out = {}
    for i in range(nbin):
        for j in range(i + 1, nbin):
            if (i, j) in skip or first[j] > last[i] or first[i] > last[j]:
                continue
            value = pair_rule.overlap(kernels[:, i], kernels[:, j], z)
            if value > 0.0:
                out[(i, j)] = value
    return out


def gap_matrix(tag, layout):
    """Return the gap of every bin pair in bin widths (see the module docstring).

    The raw range of bin i runs from the first to the last grid node where
    its raw n(z) is positive, widened by half a grid step on each side. On
    the fixed-width layouts the gap is in redshift over Delta z. On dcom62
    the range ends are converted to comoving distance [Mpc] by linear
    interpolation of the CAMB table, and the unit is the median comoving
    width of the bins.

    Arguments:
      tag    = a LAYOUTS key.
      layout = read_layout's dict.

    Returns:
      array [nbin, nbin], symmetric, zero on the diagonal and for touching
      or overlapping ranges.

    Raises:
      FileNotFoundError when dcom62 has no CAMB table (Prerequisites, item 3).
    """
    nz = layout["nz"]
    z = nz[:, 0]
    step = float(z[1] - z[0])
    raw = nz[:, 1:]
    nbin = raw.shape[1]
    first = np.zeros(nbin, dtype=int)
    last = np.zeros(nbin, dtype=int)
    for i in range(nbin):
        positive = np.nonzero(raw[:, i] > 0)[0]
        first[i] = positive[0]
        last[i] = positive[-1]
    z_low = z[first] - step/2
    z_high = z[last] + step/2
    width_z = LAYOUTS[tag]["width_z"]
    if width_z is None:
        if not os.path.isfile(CAMB_TABLE):
            raise FileNotFoundError(
                f"{CAMB_TABLE} does not exist. The comoving gap axis of {tag} needs "
                "this CAMB distance table: run projects/lssty1xdesi/"
                "EXAMPLE_EVALUATE_COVARIANCE.ipynb up to its 'Save the calculation' "
                "cell, which writes it")
        table = np.load(CAMB_TABLE)
        chi_mpc = table["chi"]/HUBBLE_H
        low = np.interp(z_low, table["z_1D"], chi_mpc)
        high = np.interp(z_high, table["z_1D"], chi_mpc)
        width = float(np.median(high - low))
    else:
        low = z_low
        high = z_high
        width = width_z
    gap = np.zeros((nbin, nbin))
    for i in range(nbin):
        for j in range(nbin):
            gap[i, j] = max(0.0, low[j] - high[i], low[i] - high[j])/width
    return gap


def axis_points(tag, layout):
    """Build the pair list of every axis point that lies in the maximal list.

    Checks first that the blur of this script at the fiducial dispersions is
    make_desi_pairs.blurred_kernels exactly, and that at the fiducial no
    pair outside the current list reaches the overlap threshold (the rule
    that made the current list).

    Arguments:
      tag    = a LAYOUTS key.
      layout = read_layout's dict.

    Returns:
      dict {point: sorted list of (i, j)}: 'today' (the current list), then
      'gap<=g' for each GAP_LEVELS entry and 'sigv=s' for each SIGV_LEVELS
      entry whose list lies in the maximal list, in that order.

    Raises:
      RuntimeError when either check fails.
    """
    stem = LAYOUTS[tag]["stem"]
    current = sorted(layout["current"])
    current_set = set(current)
    maximal_set = set(layout["maximal"])
    z = layout["nz"][:, 0]
    fiducial = pair_rule.parmod.MODELS["params_desi.yaml"]["sigv_values"]
    kernels = blurred_kernels(layout=layout, sigv_of_group=fiducial)
    rule_kernels = pair_rule.blurred_kernels(stem)[1]
    if not np.array_equal(kernels, rule_kernels):
        raise RuntimeError(f"{tag}: the blur at the fiducial dispersions differs from "
                           "make_desi_pairs.blurred_kernels; the sigma_v axis would not "
                           "follow the pair rule")
    at_fiducial = overlapping_pairs(z=z, kernels=kernels, skip=current_set)
    for pair, value in at_fiducial.items():
        if value >= pair_rule.OVERLAP_MIN:
            raise RuntimeError(f"{tag}: pair {pair} overlaps by {value:.3g} >= "
                               f"{pair_rule.OVERLAP_MIN} at the fiducial dispersions but "
                               "is not in the current list: the current list does not "
                               "follow the pair rule")
    gap = gap_matrix(tag=tag, layout=layout)
    nbin = gap.shape[0]
    candidates = {}
    for level in GAP_LEVELS:
        added = []
        for i in range(nbin):
            for j in range(i + 1, nbin):
                if gap[i, j] <= level + GAP_TOL and (i, j) not in current_set:
                    added.append((i, j))
        candidates[f"gap<={level}"] = added
    for sigv in SIGV_LEVELS:
        # every dispersion group at the same sigma_v
        uniform = {}
        for group in fiducial:
            uniform[group] = sigv
        kernels = blurred_kernels(layout=layout, sigv_of_group=uniform)
        overlaps = overlapping_pairs(z=z, kernels=kernels, skip=current_set)
        added = []
        for pair, value in overlaps.items():
            if value >= pair_rule.OVERLAP_MIN:
                added.append(pair)
        candidates[f"sigv={int(sigv)}"] = added
    points = {"today": current}
    for name, added in candidates.items():
        # a point is fitted only when every added pair has derivatives
        if set(added) <= maximal_set:
            points[name] = sorted(current + added)
    return points


def required_caches(tag):
    """Return the names of the working copies a run of one layout reads.

    Arguments:
      tag = a LAYOUTS key.

    Returns:
      list of cache names (chains/<name>.npz): the maximal-list caches of
      the layout, the current cuts, and the current derivative bases of the
      three other layouts.
    """
    names = [f"model_derivatives_{tag}_pairmax_{TIER}",
             f"model_templates_{tag}_pairmax_{TIER}",
             f"model_scale_cuts_{tag}_pairmax_{TIER}",
             f"model_scale_cuts_{TIER}"]
    for other in LAYOUTS:
        if other == tag:
            continue
        names.append(f"model_derivatives_{other}_{TIER}")
        names.append(f"model_templates_{other}_{TIER}")
    return names


def covariance_commands(tag, suffix):
    """Return the shell commands that build data/<stem><suffix>_cov.

    Arguments:
      tag    = a LAYOUTS key.
      suffix = "_pairmax" for the maximal list, "" for the current list.

    Returns:
      str: the commands of Prerequisites item 2, one per line.
    """
    number = LAYOUTS[tag]["number"]
    yaml_number = "" if number == 1 else str(number)
    yaml_name = f"EXAMPLE_EVALUATE_COVARIANCE{yaml_number}{suffix.upper()}.yaml"
    case = LAYOUTS[tag]["case"] + suffix
    return ("  unset IGNORE_COSMOLIKE_LSSTY1XDESI_COVARIANCE\n"
            "  source projects/lssty1xdesi/scripts/compile_lssty1xdesi.sh\n"
            "  python projects/lssty1xdesi/covariance/compute_covariance.py "
            f"projects/lssty1xdesi/{yaml_name}\n"
            "  python projects/lssty1xdesi/scripts/make_desi_covariance.py "
            f"--case {case}")


def check_prerequisites(tag):
    """Stop with the commands to run when an input of the run is missing.

    Checks, before anything expensive: the notebook, the maximal-list paper
    yaml, every working copy of required_caches, the maximal-list covariance
    file of the layout and the current covariance files of the other three.
    The CAMB table of dcom62 is checked where it is read (gap_matrix).

    Arguments:
      tag = a LAYOUTS key.

    Raises:
      FileNotFoundError listing every missing file with its command.
    """
    problems = []
    if not os.path.isfile(NOTEBOOK):
        problems.append(f"{NOTEBOOK} does not exist (part of the repository)")
    number = LAYOUTS[tag]["number"]
    yaml_file = os.path.join(PROJECT, f"EXAMPLE_EVALUATE{number}_PAIRMAX.yaml")
    if not os.path.isfile(yaml_file):
        problems.append(f"{yaml_file} does not exist (part of the repository)")
    missing = []
    for name in required_caches(tag):
        if not os.path.isfile(os.path.join(CHAINS, name + ".npz")):
            missing.append(name)
    if missing:
        # the notebook's first code cell puts these two folders on sys.path
        # before it imports the Fisher module
        load = ("python -c \"import sys; "
                "sys.path.insert(0, 'external_modules/code/cosmolike_core'); "
                "sys.path.insert(0, 'projects/lssty1xdesi/interface'); "
                "import cosmolike_lssty1xdesi_fisher as fm; "
                "fm.load_fisher_from_cache('{}')\"")
        calls = []
        for name in missing:
            calls.append("  " + load.format(name))
        problems.append(
            "these working copies are not in chains/: " + ", ".join(missing) + ". Copy "
            "the archived caches, from the folder Cocoa/ with Cocoa started:\n"
            + "\n".join(calls))
    covariances = [(tag, "_pairmax")]
    for other in LAYOUTS:
        if other != tag:
            covariances.append((other, ""))
    for owner, suffix in covariances:
        path = os.path.join(DATA, LAYOUTS[owner]["stem"] + suffix + "_cov")
        if not os.path.isfile(path):
            problems.append(f"{path} does not exist. Build it from the folder Cocoa/ "
                            "(covariance/README.md):\n"
                            + covariance_commands(tag=owner, suffix=suffix))
    if problems:
        raise FileNotFoundError("the pair-axes fits of " + tag + " cannot start:\n- "
                                + "\n- ".join(problems))


def settings_substitutions(tag):
    """Return the text edits of the settings cell for one layout.

    Arguments:
      tag = a LAYOUTS key.

    Returns:
      list of (old, new) strings; each old string must occur exactly once in
      the settings cell (cell_source checks it). The first puts the layout on
      its maximal-list dataset and yaml just before the exploration
      parameter files are defined (the fingerprints, computed further down
      the cell, then cover the maximal-list files); the second gives the
      layout its maximal-list cache tag before the cache names are built.
    """
    layout = LAYOUTS[tag]
    key = layout["key"]
    number = layout["number"]
    dataset = layout["stem"] + "_pairmax"
    bias_params = layout["bias_params"]
    lines = [
        "# the pair-axes fits (scripts/compute_pair_axes_fits.py): this layout",
        "# on its maximal pair list",
        f"LAYOUTS[{key!r}] = paper_layout({number}, {dataset!r}, {bias_params!r})",
        f"LAYOUTS[{key!r}]['yaml'] = PROJECT + 'EXAMPLE_EVALUATE{number}_PAIRMAX.yaml'",
        LAYOUT_ANCHOR,
    ]
    layout_edit = (LAYOUT_ANCHOR, "\n".join(lines))
    tag_edit = (TAG_ANCHOR,
                f"LAYOUT_TAGS[{key!r}] = {tag + '_pairmax'!r}\n" + TAG_ANCHOR)
    return [layout_edit, tag_edit]


def cell_source(cells, index, substitutions):
    """Return the source of one notebook code cell, checked and edited.

    Arguments:
      cells         = the notebook's list of cells (its JSON "cells").
      index         = the cell index, a FIRST_LINES key.
      substitutions = settings_substitutions' list, applied to the settings
                      cell only.

    Returns:
      str: the cell's code; the settings cell with its edits, cell 13 up to
      its TRUNCATE_AT line.

    Raises:
      RuntimeError when the cell is not code, starts with another line, or
      an edit or truncation line does not occur exactly once.
    """
    cell = cells[index]
    if cell["cell_type"] != "code":
        raise RuntimeError(f"notebook cell {index} is a {cell['cell_type']} cell, "
                           "expected a code cell: the notebook changed; update "
                           "FIRST_LINES")
    # a cell's source is stored as a list of lines; "".join rebuilds the text
    source = "".join(cell["source"])
    if not source.startswith(FIRST_LINES[index]):
        raise RuntimeError(f"notebook cell {index} starts with {source[:60]!r}, "
                           f"expected {FIRST_LINES[index]!r}: the notebook changed; "
                           "update FIRST_LINES")
    if index == SETTINGS_CELL:
        count = source.count(CUT_TIER_LINE)
        if count != 1:
            raise RuntimeError(f"settings cell: {CUT_TIER_LINE.strip()!r} found {count} "
                               "times, expected once (the fits read the final-tier cuts)")
        for old, new in substitutions:
            count = source.count(old)
            if count != 1:
                raise RuntimeError(f"settings cell: {old[:40]!r} found {count} times, "
                                   "expected once: the notebook changed")
            source = source.replace(old, new)
    if index in TRUNCATE_AT:
        marker = TRUNCATE_AT[index]
        count = source.count(marker)
        if count != 1:
            raise RuntimeError(f"notebook cell {index}: the line {marker!r} found "
                               f"{count} times, expected once: the notebook changed")
        source = source[:source.index(marker)]
    return source


def run_cells(cells, indices, namespace, substitutions):
    """Run notebook cells in order in one namespace, as Jupyter would.

    Arguments:
      cells         = the notebook's list of cells.
      indices       = the cell indices to run, in order.
      namespace     = the dict the cells read and write (their globals).
      substitutions = settings_substitutions' list.

    Side effects: whatever the cells do; the cells run here read caches and
    data files, print tables and change the working directory to ROOTDIR
    (cell 2). They write no file.
    """
    for index in indices:
        start = time.time()
        source = cell_source(cells=cells, index=index, substitutions=substitutions)
        code = compile(source, f"<notebook cell {index}>", "exec")
        exec(code, namespace)
        print(f"[notebook cell {index}: {time.time() - start:.1f} s]", flush=True)


def notebook_namespace(tag):
    """Run the notebook with the layout on its maximal list and return its state.

    Runs SETUP_CELLS, reads the two cut caches (the current one, for the
    other layouts, with the tier and fingerprint checks; the layout's
    maximal-list one, also checked against the notebook's cut settings) in
    place of cell 9, then runs FIT_CELLS. Finally registers the two
    sigma_v-fixed variants of PANEL_VARIANTS.

    Arguments:
      tag = a LAYOUTS key.

    Returns:
      dict: the notebook's namespace (fit_system, GRAM, COLUMNS, TABLES, ...).

    Raises:
      RuntimeError when the notebook's paper model is not PAPER_VARIANT or a
      registered variant has no sigma_v column; the cache readers raise on a
      missing or mismatched cache.
    """
    key = LAYOUTS[tag]["key"]
    with open(NOTEBOOK) as stream:
        cells = json.load(stream)["cells"]
    substitutions = settings_substitutions(tag)
    namespace = {"__name__": "notebook_cells"}
    run_cells(cells=cells, indices=SETUP_CELLS, namespace=namespace,
              substitutions=substitutions)
    if os.environ["OMP_NUM_THREADS"] != NOTEBOOK_THREADS:
        raise RuntimeError(f"the notebook sets OMP_NUM_THREADS = "
                           f"{os.environ['OMP_NUM_THREADS']}, this script loaded numpy "
                           f"with {NOTEBOOK_THREADS}: update NOTEBOOK_THREADS")
    notebook_tier = namespace["DERIVATIVE_TIER"]
    if notebook_tier != TIER:
        raise RuntimeError(f"the notebook runs the {notebook_tier!r} tier; the archived "
                           f"caches are {TIER!r}")
    fm = namespace["fm"]
    current_cuts = fm.read_fisher_cache(f"model_scale_cuts_{TIER}", tier=TIER)
    maximal_cuts = fm.read_fisher_cache(f"model_scale_cuts_{tag}_pairmax_{TIER}",
                                        settings={key: namespace["CUT_SETTINGS"][key]},
                                        comparable=namespace["cut_comparable"], tier=TIER)
    cuts = {}
    for name in namespace["LAYOUTS"]:
        if name == key:
            cuts[name] = maximal_cuts[key]
        else:
            cuts[name] = current_cuts[name]
    namespace["CUTS"] = cuts
    run_cells(cells=cells, indices=FIT_CELLS, namespace=namespace,
              substitutions=substitutions)
    summary_variant = namespace["SUMMARY_VARIANT"]
    if summary_variant != PAPER_VARIANT:
        raise RuntimeError(f"the notebook's paper model is {summary_variant!r}, expected "
                           f"{PAPER_VARIANT!r}: update PAPER_VARIANT and the about texts "
                           "of the generators")
    # sigma_v fixed = every column whose name holds SIGV removed, so the fit
    # holds the dispersions at the fiducial (the notebook's rule for
    # 'B tilts noA2 noSV' in section 11)
    for variant in (REFERENCE_VARIANT, PAPER_VARIANT):
        columns, names, tags = namespace["COLUMNS"][(key, variant)]
        keep = []
        for j in range(len(names)):
            if "SIGV" not in names[j].upper():
                keep.append(j)
        if len(keep) == len(names):
            raise RuntimeError(f"{variant}: no sigma_v column to fix")
        kept_names = []
        kept_tags = []
        for j in keep:
            kept_names.append(names[j])
            kept_tags.append(tags[j])
        namespace["COLUMNS"][(key, variant + NO_SIGV)] = (columns[:, keep], kept_names,
                                                           kept_tags)
    return namespace


def cut_indices(namespace):
    """Return {CUT_LABELS entry: its index m in the notebook's KMAX_CUTS}.

    Raises:
      RuntimeError when a label is not exactly one cut of KMAX_CUTS.
    """
    out = {}
    for label in CUT_LABELS:
        found = []
        for m, kmax in enumerate(namespace["KMAX_CUTS"]):
            if abs(float(kmax) - float(label)) < 1e-9:
                found.append(m)
        if len(found) != 1:
            raise RuntimeError(f"k_max = {label} is not one cut of "
                               f"{namespace['KMAX_CUTS']}")
        out[label] = found[0]
    return out


def fom_pair(system, status):
    """Return {'fom', 'fom_om_as', 'status'} of a fit_system result (None FoMs
    when there is no fit)."""
    if system is None:
        return {"fom": None, "fom_om_as": None, "status": status}
    return {"fom": system["fom"], "fom_om_as": system["fom_om_as"], "status": status}


def panel_fits(namespace, key, m):
    """Fit every PANEL_VARIANTS entry at cut m with the namespace's GRAM.

    Arguments:
      namespace = the notebook's namespace, with the point's GRAM in place.
      key       = the layout key.
      m         = the cut index.

    Returns:
      {model: {sigma_v: fom_pair}}.
    """
    out = {}
    for (model, sigv), variant in PANEL_VARIANTS.items():
        system, status = namespace["fit_system"](key, m, variant, PRIOR)
        if model not in out:
            out[model] = {}
        out[model][sigv] = fom_pair(system=system, status=status)
    return out


def prior_fits(namespace, key, m):
    """Fit the paper model under every PRIOR_FITS prior at cut m.

    Arguments:
      namespace = the notebook's namespace, with the point's GRAM in place.
      key       = the layout key.
      m         = the cut index.

    Returns:
      {prior: {sigma_v: fom_pair}}, priors named as PRIOR_FITS' keys.
    """
    out = {}
    for label, prior in PRIOR_FITS.items():
        out[label] = {}
        for sigv, suffix in (("sigma_v free", ""), ("sigma_v fixed", NO_SIGV)):
            variant = PAPER_VARIANT + suffix
            system, status = namespace["fit_system"](key, m, variant, prior)
            out[label][sigv] = fom_pair(system=system, status=status)
    return out


def fixed_columns(prefixes, sigv, names):
    """Return the indices of the parameters a fit holds fixed.

    Arguments:
      prefixes = list of name prefixes (an ATTRIBUTION_BLOCKS entry, or []).
      sigv     = 'sigma_v free' or 'sigma_v fixed'; the latter also holds
                 every parameter whose name holds SIGV.
      names    = the parameter names.

    Returns:
      set of indices into names.
    """
    out = set()
    for j, name in enumerate(names):
        # str.startswith with a tuple is True when any prefix matches (and
        # False for the empty tuple)
        if name.startswith(tuple(prefixes)):
            out.add(j)
        elif sigv == "sigma_v fixed" and "SIGV" in name.upper():
            out.add(j)
    return out


def figures_of_merit(F, names, drop):
    """Return (FoM(w0, wa), FoM(Omega_m, 10^9 A_s)) of F without the columns drop.

    Deleting a parameter's row and column of the full Fisher matrix holds it
    at its fiducial value; the remaining matrix is inverted directly.

    Arguments:
      F     = the Fisher matrix [n_par, n_par], data plus every prior.
      names = its parameter names [n_par].
      drop  = set of indices to delete.

    Returns:
      (float, float).
    """
    keep = []
    for j in range(len(names)):
        if j not in drop:
            keep.append(j)
    kept_names = []
    for j in keep:
        kept_names.append(names[j])
    F_inv = np.linalg.inv(F[np.ix_(keep, keep)])
    i_w = [kept_names.index("w0"), kept_names.index("wa")]
    i_oa = [kept_names.index("omegam"), kept_names.index("As_1e9")]
    fom = float(1.0/np.sqrt(np.linalg.det(F_inv[np.ix_(i_w, i_w)])))
    fom_om_as = float(1.0/np.sqrt(np.linalg.det(F_inv[np.ix_(i_oa, i_oa)])))
    return fom, fom_om_as


def split_basis(namespace, key, names, parts):
    """Return the change of basis T that splits the per-bin biases.

    Every parameter that is not a per-bin amplitude DESI_B1_<n> keeps its
    own column. The amplitudes of each bias group are written as
    b = U a + E e: U = (1, z - zbar_group) spans the group's free level and
    slope, E spans the residual directions that the correlated prior
    charges (the null space of U^T Q, Q the inverse of the prior covariance
    sigma^2 exp(-|z_i - z_j|/L) of the pessimistic prior). A group with no
    more bins than U has columns keeps all its bins in U. The reduced Fisher
    matrix T^T F T then has one parameter per kept column.

    Arguments:
      namespace = the notebook's namespace.
      key       = the layout key.
      names     = the parameter names of F (the live columns of the fit).
      parts     = the parts kept, "U", "E" or both, in that order.

    Returns:
      array T [len(names), n_reduced]: first the non-amplitude columns in
      the order of names, then per group the columns of its kept parts.

    Raises:
      RuntimeError when the live amplitudes are not one per bin.
    """
    nbins = len(namespace["TABLES"][key]["bins"])
    position = {}
    for j, name in enumerate(names):
        if name.startswith("DESI_B1_"):
            # DESI_B1_<n> is bin n - 1 counted from 0
            position[int(name.rsplit("_", 1)[1]) - 1] = j
    if len(position) != nbins:
        raise RuntimeError(f"{len(position)} live b_1 columns, expected {nbins}")
    columns = []
    for j, name in enumerate(names):
        if name.startswith("DESI_B1_"):
            continue
        column = np.zeros(len(names))
        column[j] = 1.0
        columns.append(column)
    zbar = np.asarray(namespace["BIN_ZBAR"][key])
    fiducial = namespace["fiducial_of"](key)
    assumption = namespace["BIAS_ASSUMPTIONS"][PRIOR]
    for group, leader, members in namespace["bias_groups"](key):
        members = np.asarray(members)
        z = zbar[members]
        U = np.column_stack([np.ones(len(z)), z - z.mean()])
        if len(members) <= U.shape[1]:
            blocks = {"U": np.eye(len(members)), "E": np.zeros((len(members), 0))}
        else:
            width = assumption["sigma_b"][group]*fiducial[leader]
            # z[:, None] - z[None, :] is the matrix of redshift differences
            distance = np.abs(z[:, None] - z[None, :])
            Q = np.linalg.inv(width**2*np.exp(-distance/assumption["L"]))
            E = scipy.linalg.null_space(U.T @ Q)
            blocks = {"U": U, "E": E}
        for part in parts:
            for k in range(blocks[part].shape[1]):
                column = np.zeros(len(names))
                for b, value in zip(members, blocks[part][:, k]):
                    column[position[int(b)]] = value
                columns.append(column)
    return np.column_stack(columns)


def attribution_fits(namespace, key, m):
    """Fit the paper model at cut m with each nuisance block held fixed.

    Rebuilds the full Fisher matrix exactly as fit_system does (data, the
    diagonal priors, the correlated linear-bias prior) on fit_system's live
    parameters, checks that it gives fit_system's FoMs, then deletes each
    block of ATTRIBUTION_BLOCKS ("all free" deletes nothing) and evaluates
    the two SPLIT_BLOCKS in the basis of split_basis.

    Arguments:
      namespace = the notebook's namespace, with the point's GRAM in place.
      key       = the layout key.
      m         = the cut index.

    Returns:
      {block: {sigma_v: {'fom', 'fom_om_as'}}}, blocks in the order "all
      free", ATTRIBUTION_BLOCKS, SPLIT_BLOCKS.

    Raises:
      RuntimeError when the paper model has no fit at this cut, when the
      rebuilt matrix or the split basis changes the FoMs by more than
      REBUILD_RTOL.
    """
    system, status = namespace["fit_system"](key, m, PAPER_VARIANT, PRIOR)
    if system is None:
        raise RuntimeError(f"paper model at cut {m}: {status}")
    M, names, tags = namespace["COLUMNS"][(key, PAPER_VARIANT)]
    gram = namespace["GRAM"][key][m]
    # the model letter of the variant ('B') selects its prior widths
    model = PAPER_VARIANT.split()[0]
    prior_diagonal = namespace["prior_diagonal"](tags, model, namespace["KMAX_CUTS"][m])
    F = M.T @ gram @ M + np.diag(prior_diagonal)
    nbins = len(namespace["BIN_ZBAR"][key])
    # the per-bin amplitudes are the last nbins parameters of every variant
    n_other = len(names) - nbins
    F[n_other:, n_other:] += namespace["PRECISION"][(key, PRIOR)]
    if (key, PAPER_VARIANT) in namespace["EXTRA_PRECISION"]:
        F = F + namespace["EXTRA_PRECISION"][(key, PAPER_VARIANT)](
            key, namespace["KMAX_CUTS"][m])
    live = system["live"]
    F = F[np.ix_(live, live)]
    used = []
    for j in live:
        used.append(names[j])
    fom_all, fom_om_as_all = figures_of_merit(F=F, names=used, drop=set())
    if (abs(fom_all/system["fom"] - 1) > REBUILD_RTOL
            or abs(fom_om_as_all/system["fom_om_as"] - 1) > REBUILD_RTOL):
        raise RuntimeError(f"the rebuilt Fisher matrix gives FoM {fom_all} against "
                           f"fit_system's {system['fom']}")
    blocks = {"all free": []}
    for block, prefixes in ATTRIBUTION_BLOCKS.items():
        blocks[block] = prefixes
    out = {}
    for block, prefixes in blocks.items():
        out[block] = {}
        for sigv in ("sigma_v free", "sigma_v fixed"):
            drop = fixed_columns(prefixes=prefixes, sigv=sigv, names=used)
            fom, fom_om_as = figures_of_merit(F=F, names=used, drop=drop)
            out[block][sigv] = {"fom": fom, "fom_om_as": fom_om_as}
    n_amplitudes = 0
    other_names = []
    for name in used:
        if name.startswith("DESI_B1_"):
            n_amplitudes += 1
        else:
            other_names.append(name)
    for block, part in SPLIT_BLOCKS.items():
        T = split_basis(namespace=namespace, key=key, names=used, parts=(part,))
        F_reduced = T.T @ F @ T
        # the reduced parameters: the other names, then one placeholder name
        # per kept amplitude direction
        reduced = other_names + ["b1*"]*(T.shape[1] - len(other_names))
        out[block] = {}
        for sigv in ("sigma_v free", "sigma_v fixed"):
            drop = fixed_columns(prefixes=[], sigv=sigv, names=reduced)
            fom, fom_om_as = figures_of_merit(F=F_reduced, names=reduced, drop=drop)
            out[block][sigv] = {"fom": fom, "fom_om_as": fom_om_as}
    # both parts together span every amplitude, so the all-free FoM returns
    T = split_basis(namespace=namespace, key=key, names=used, parts=("U", "E"))
    reduced = other_names + ["b1*"]*n_amplitudes
    F_split = T.T @ F @ T
    fom_split = figures_of_merit(F=F_split, names=reduced, drop=set())[0]
    if abs(fom_split/fom_all - 1) > REBUILD_RTOL:
        raise RuntimeError(f"the U + E basis changes the FoM: {fom_split} against "
                           f"{fom_all}")
    return out


def point_gram(namespace, key, B, C, m, keep_pair):
    """Return the Fisher information matrix of the base columns for one point.

    Arguments:
      namespace = the notebook's namespace (TABLES, kept_rows, INVCOV_SCALE).
      key       = the layout key.
      B         = the base columns [n, n_base] of the maximal list.
      C         = its covariance [n, n].
      m         = the cut index.
      keep_pair = bool array [n_pairs]: the maximal-list pairs of the point.

    Returns:
      (G, n_rows): G = INVCOV_SCALE B[rows]^T C[rows, rows]^-1 B[rows]
      [n_base, n_base] (see "How a point is fitted"), n_rows = len(rows).
    """
    tables = namespace["TABLES"][key]
    nell = len(tables["ells"])
    n_ee = tables["n_ee"]
    nbins = len(tables["bins"])
    kept = namespace["kept_rows"](key, m)
    dropped = []
    for p in range(len(keep_pair)):
        if not keep_pair[p]:
            # the data vector is [EE | gs | gg autos | gg cross], one block of
            # nell bands per spectrum; pair p is spectrum nbins + p of the
            # clustering part
            first = n_ee + (nbins + p)*nell
            dropped.append(np.arange(first, first + nell))
    if dropped:
        rows = np.setdiff1d(kept, np.concatenate(dropped))
    else:
        rows = kept
    L = scipy.linalg.cholesky(C[np.ix_(rows, rows)], lower=True)
    W = scipy.linalg.solve_triangular(L, B[rows], lower=True)
    del L
    gram = namespace["INVCOV_SCALE"]*(W.T @ W)
    del W
    return gram, int(rows.size)


def fit_layout(tag, points):
    """Run the three fit families on every point of one layout.

    Arguments:
      tag    = a LAYOUTS key.
      points = axis_points' dict.

    Returns:
      dict {'panel': ..., 'attribution': ..., 'priors': ...}, each
      {'cuts': {k_max label: {point: ...}}} in the formats of the module
      docstring.

    Raises:
      RuntimeError when the notebook's pair list of the layout is not the
      maximal list the points were built from.
    """
    key = LAYOUTS[tag]["key"]
    namespace = notebook_namespace(tag)
    pairs = []
    for pair in namespace["TABLES"][key]["pairs"]:
        pairs.append(tuple(pair))
    all_points = set()
    for plist in points.values():
        all_points.update(plist)
    if not all_points <= set(pairs):
        raise RuntimeError(f"{key}: the notebook's pair list does not contain every "
                           "axis point's pairs")
    B = namespace["base_matrix"](key)[0]
    C = namespace["layout_covariance"](key, B.shape[0])
    print(f"{key}: {len(pairs)} pairs, n = {B.shape[0]}, {B.shape[1]} base columns",
          flush=True)
    cuts = cut_indices(namespace)
    panel = {}
    attribution = {}
    priors = {}
    for label, m in cuts.items():
        panel[label] = {}
        attribution[label] = {}
        priors[label] = {}
        for point, plist in points.items():
            members = set(plist)
            keep_pair = np.zeros(len(pairs), dtype=bool)
            for p in range(len(pairs)):
                keep_pair[p] = pairs[p] in members
            gram, n_rows = point_gram(namespace=namespace, key=key, B=B, C=C, m=m,
                                      keep_pair=keep_pair)
            # fit_system reads GRAM[key][m]: the point's matrix stands there
            # during its fits, and the notebook's matrix returns afterwards
            saved = namespace["GRAM"][key][m]
            namespace["GRAM"][key][m] = gram
            try:
                panel[label][point] = panel_fits(namespace=namespace, key=key, m=m)
                if point == "today" or point.startswith("gap<="):
                    attribution[label][point] = attribution_fits(namespace=namespace,
                                                                 key=key, m=m)
                    priors[label][point] = prior_fits(namespace=namespace, key=key, m=m)
            finally:
                namespace["GRAM"][key][m] = saved
            paper = panel[label][point]["paper"]["sigma_v free"]["fom"]
            print(f"k_max {label}, {point}: {len(plist)} pairs, {n_rows} rows, paper "
                  f"FoM(w0, wa) {paper}", flush=True)
    return {"panel": {"cuts": panel}, "attribution": {"cuts": attribution},
            "priors": {"cuts": priors}}


def main():
    """Build the axis points of one layout, fit them and write the JSON file.

    Side effects: writes the output file (only after every fit succeeded),
    reads the caches in chains/ and the files of data/.
    """
    args = parse_arguments()
    start = time.time()
    tag = args.layout
    output = args.output
    if output is None:
        output = os.path.join(CHAINS, f"pair_axes_fits_{tag}.json")
    # cell 2 changes the working directory: the output path is fixed first
    output = os.path.abspath(output)
    if os.path.exists(output) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{output} exists: pass --overwrite to replace it")
    if not os.path.isdir(os.path.dirname(output)):
        raise FileNotFoundError(f"the folder of {output} does not exist")
    if "ROOTDIR" not in os.environ:
        raise RuntimeError("ROOTDIR is not set: start Cocoa (source start_cocoa.sh) "
                           "in the folder Cocoa/ first")
    # the notebook reads the project under ROOTDIR: it must be this project
    notebook_project = os.path.join(os.environ["ROOTDIR"], "projects", "lssty1xdesi")
    if os.path.realpath(notebook_project) != os.path.realpath(PROJECT):
        raise RuntimeError(f"ROOTDIR = {os.environ['ROOTDIR']} holds another project "
                           f"folder than {PROJECT}: start Cocoa in the Cocoa/ folder "
                           "that holds this script")
    check_prerequisites(tag)
    layout = read_layout(tag)
    points = axis_points(tag=tag, layout=layout)
    for point, plist in points.items():
        print(f"{tag} {point}: {len(plist)} pairs", flush=True)
    if args.dry_run:
        return 0
    fits = fit_layout(tag=tag, points=points)
    point_table = {}
    for point, plist in points.items():
        pair_rows = []
        for i, j in plist:
            pair_rows.append([i, j])
        point_table[point] = {"n_pairs": len(plist), "pairs": pair_rows}
    result = {
        "about": ("Figures of merit of the pair-axes points of one DESI layout, by row "
                  "selection from its maximal-list derivative base; written by "
                  "scripts/compute_pair_axes_fits.py (its docstring defines every "
                  "entry)."),
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "layout": LAYOUTS[tag]["key"],
        "tag": tag,
        "dataset": LAYOUTS[tag]["stem"] + "_pairmax",
        "kmax_h_per_Mpc": CUT_LABELS,
        "kmax_unit": "h/Mpc at z = 0 (per-bin cut at k_max/D(zbar_i))",
        "points": point_table,
        "panel": {
            "prior": PRIOR,
            "models": {
                "paper": PAPER_VARIANT,
                "reference": REFERENCE_VARIANT,
            },
            "sigma_v fixed": "every sigma_v column removed (held at the fiducial)",
            "cuts": fits["panel"]["cuts"],
        },
        "attribution": {
            "model": PAPER_VARIANT,
            "prior": PRIOR,
            "blocks": ATTRIBUTION_BLOCKS,
            "split_blocks": SPLIT_BLOCKS,
            "cuts": fits["attribution"]["cuts"],
        },
        "priors": {
            "model": PAPER_VARIANT,
            "priors": PRIOR_FITS,
            "cuts": fits["priors"]["cuts"],
        },
        "seconds": time.time() - start,
    }
    partial = output + ".partial"
    with open(partial, "w") as stream:
        json.dump(result, stream, indent=1)
    os.replace(partial, output)
    print(f"wrote {output} ({result['seconds']:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
