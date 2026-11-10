"""Compute the Fisher derivative base of one lssty1xdesi layout on MPI ranks.

What is computed. Every Fisher matrix of the Fisher notebook of the paper
model (EXAMPLE_EVALUATE_FISHER_MODEL.ipynb) is built from a few base
columns per layout, the derivative base B. Its columns are derivatives
of the data vector m (the DESI band powers) at the fiducial point, the
parameter values of the paper yaml's evaluate block
(EXAMPLE_EVALUATE<n>.yaml: model B + tilts without alpha_2 on the
pessimistic linear-bias prior, every higher-order bias parameter at
zero):

  - stencil columns of the 20 sampled parameters the notebook varies:
    the seven cosmological parameters, the six group amplitudes
    DESI_B1_<g> (B_g, g = BGS, LRG1, LRG2, LRG3, ELG1, ELG2) and the
    seven velocity-dispersion leaders DESI_SIGV_<n>. Each is a
    five-point stencil,
    [-m(p + 2d) + 8 m(p + d) - 8 m(p - d) + m(p - 2d)]/(12 d), with d
    the parameter's step (ModelFisher.derivatives at h = 0.02: d = h
    times the fiducial value, or h for a zero fiducial, so no step
    depends on a prior);
  - per-bin columns of the bias operators b_1, b_2, b_{s^2} and alpha_0
    (dm/db_{1,i} at fixed b_{2,i} and b_{s^2,i}, and so on; b_{1,i} is
    the linear bias of bin i, b_{2,i} and b_{s^2,i} its quadratic and
    tidal biases, alpha_{0,i} its k^2 counterterm in h^-2 Mpc^2), by the
    same five-point stencil;
  - per-bin columns of the radial counterterm alpha_2, which enters m
    linearly, so the central difference at +-5 h^-2 Mpc^2 is exact;
  - the exact templates of the stochastic amplitudes SN0 and SN2 (the
    likelihood's stochastic_templates, read at the fiducial point).

A per-bin column moves a batch of bins at once: a greedy coloring of the
cross-pair graph puts two bins joined by a measured cross pair in
different batches ("colors"), so every spectrum involves at most one
shifted bin and one stencil gives the columns of every bin of a batch.
Every layout needs four colors.

The notebook forms G = f_A B_K^T C^-1 B_K at every scale cut K, with C
the covariance of the kept entries and f_A = 12300/14000 the rescaling
of the inverse covariance to the survey area. G costs seconds once B
exists, and no cache stores it, so this script stops at B.

Outputs. The two caches the notebook reads from chains/ (section 4,
"The derivative base"), in its format, with its settings and
fingerprint (fm.settings_fingerprint, the sha256 of the fully resolved
paper yaml: its options, parameter files, priors and fiducial point),
at the final derivative tier of fm.derivative_settings (the converged
settings of the derivative convergence study
tests/studies/run_fisher_derivatives.py):

  EXAMPLE_EVALUATE_FISHER_MODEL_DERIVATIVES_<n>.pkl  stencil columns,
      per-bin b_1, b_2, b_{s^2}, alpha_0 columns, SN0 templates, the
      fiducial vector and biases;
  EXAMPLE_EVALUATE_FISHER_MODEL_TEMPLATES_<n>.pkl    alpha_2 columns,
      SN0 and SN2 templates,

with n = 1, 2, 3, 4 for the layouts dcom62 (Delta d_com = 62 Mpc),
dz0p02, dz0p05 and dz0p20 (Delta z = 0.02, 0.05, 0.2). The notebook
compares a cache's settings, fingerprint included, with its own and
refuses a cache that differs. The 'seconds' entries hold the wall time
of this run.

The parallel layout. An MPI rank is one of the N processes mpirun
starts. The unit of work is one evaluation of the data vector. A layout
needs 154 of them, in this order:

  1   the fiducial point (its vector, templates and biases);
  64  per-bin operator points: 4 operators x 4 colors x 4 stencil points;
  80  stencil points: 20 parameters x 4 stencil points;
  8   alpha_2 points: 4 colors x 2 signs;
  1   the fiducial point again, after every step (the 'reset' check).

The evaluations are independent, so they are split statically and
deterministically: evaluation k goes to rank k mod N, except the last,
which goes to rank N - 1, so that with N > 1 it also compares two ranks;
the first evaluation, the fiducial point, is on rank 0, which reads the
templates right after it. Each kind of evaluation is contiguous in the
list, so every rank gets an equal share of each kind, to within one.
Every rank builds its own cobaya model once, evaluates its share in list
order and sends its vectors to rank 0, which assembles the caches with
the serial code's arithmetic: fm.five_point for the operator columns, and
ModelFisher.derivatives replayed on the gathered vectors for the stencil
columns (compute_fisher_base_mpi_classes.py explains the replay and the
per-bin steps). With N = 1 the evaluations run in the notebook's own
order (its fisher_base, section 4).

Bit for bit. A rank evaluates its points in another order than one
serial process. CosmoLike's setters store a value only when it differs
from the stored one by more than a relative 1e-13, and otherwise keep
the tables built on the old value, so a vector could in principle carry
a last-digit trace of the previous point. On Delta z = 0.2 the runs with
1 x 8, 2 x 2 and 2 x 4 ranks x threads reproduced the caches of the
serial runs (one process, 8 threads) bit for bit; with 4 x 2 the
Omega_b stencil column moved by 1e-13 of its largest entry and every
other array stayed bitwise equal. The 2 x 2 run has the same thread
count, so the evaluation order, not the thread count, makes that
difference.

Hybrid parallelism. Each rank runs CosmoLike's and CAMB's OpenMP loops
on --threads-per-rank threads (default 8), so N ranks use N times that
many cores; BLAS runs on one thread. Each rank holds a complete model
(covariance included) in memory: its resident memory peaks during the
first evaluation, near 14 GB on Delta d_com = 62 Mpc, 8 GB on
Delta z = 0.05 and 3 GB on Delta z = 0.2 (macOS, 8 threads), so 12
ranks on Delta d_com = 62 Mpc need about 165 GB. Each rank also pays
the cobaya start-up before its first evaluation: about 40 s on
Delta d_com = 62 Mpc and a few seconds on Delta z = 0.2. The ranks
start up at the same time, so the start-up adds to the wall time once;
with 154/N evaluations per rank it is a fixed cost that weighs more the
more ranks share the work.

Launch (from the Cocoa/ folder, with the cocoa environment active and
start_cocoa.sh sourced; --dry-run prints the split and the target files
without building any model, and --dry-run N plans for N ranks):

  laptop (macOS), 2 ranks x 4 threads:

    mpirun -n 2 --oversubscribe \\
      python ./projects/lssty1xdesi/scripts/compute_fisher_base_mpi.py \\
        --layout dz0p20 --threads-per-rank 4

  one 96-core HPC node (Linux), 12 ranks x 8 threads:

    export OMP_NUM_THREADS=8
    "${CONDA_PREFIX}"/bin/mpirun -n 12 --oversubscribe \\
      --mca pml ob1 --mca btl vader,tcp,self \\
      --bind-to core:overload-allowed --report-bindings \\
      --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \\
      python ./projects/lssty1xdesi/scripts/compute_fisher_base_mpi.py \\
        --layout dcom62 --threads-per-rank ${OMP_NUM_THREADS} \\
        --out ./projects/lssty1xdesi/chains/fisher_base_hpc

The binding flags are Cocoa's (README, "Running Examples"):
--map-by numa:pe=8 reserves eight cores for each rank, which must equal
--threads-per-rank. Where numa domains cause trouble, use slot instead
of numa. Across several nodes, export the environment to the remote
ranks (-x ROOTDIR -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH
-x CONDA_PREFIX), as the README's multi-node note ("Running ML
emulators") explains; every node must see the same files at the same
paths.

Cache rules. Saving is explicit: rank 0 writes the two files at the
end, only after every evaluation and every check has succeeded. A target
file that exists is never replaced without --overwrite; the run stops
before building any model, and says whether the existing file holds the
current fingerprint. Every rank computes the fingerprint from the files
it sees, and rank 0 recomputes it before saving; any mismatch (ranks
that see different files, or a yaml, parameter file or dataset file
edited during the run) stops the run without writing.
"""
import argparse
import hashlib
import inspect
import os
import pickle
import resource
import sys
import tempfile
import time
import traceback

# One BLAS thread per rank: a rank's cores belong to CosmoLike's and
# CAMB's OpenMP loops, which BLAS threads would compete with. The BLAS
# library reads these variables when numpy loads it, so they are set
# before the numpy import.
BLAS_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for blas_variable in BLAS_THREAD_VARIABLES:
    os.environ[blas_variable] = "1"

import numpy as np
from mpi4py import MPI

# the project folder relative to ROOTDIR (the Cocoa/ folder): the example
# yamls name their files relative to ROOTDIR, so the run changes into it
PROJECT = "projects/lssty1xdesi/"
# the folder of this script; it also holds CLASSES_MODULE, the module of
# the classes the ranks evaluate with, which cobaya imports from there
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES_MODULE = "compute_fisher_base_mpi_classes"

# The four layouts: the number n of their paper yaml
# (EXAMPLE_EVALUATE<n>.yaml) and of their caches (the notebook numbers the
# caches in this order), the dataset stem and the paper's linear-bias
# parameter file (the pessimistic prior), whose DESI_B1_<n> functions name
# the group amplitude DESI_B1_<g> each bin follows.
LAYOUTS = {
    "dcom62": {
        "number": 1,
        "label": "Delta d_com = 62 Mpc",
        "dataset": "lssty1xdesi_desi_dr3",
        "bias_params": "params_desi_bias_pessimistic.yaml",
    },
    "dz0p02": {
        "number": 2,
        "label": "Delta z = 0.02",
        "dataset": "lssty1xdesi_desi_dr1_dz0p02",
        "bias_params": "params_desi_dr1_dz0p02_bias_pessimistic.yaml",
    },
    "dz0p05": {
        "number": 3,
        "label": "Delta z = 0.05",
        "dataset": "lssty1xdesi_desi_dr1_dz0p05",
        "bias_params": "params_desi_dr1_dz0p05_bias_pessimistic.yaml",
    },
    "dz0p20": {
        "number": 4,
        "label": "Delta z = 0.2",
        "dataset": "lssty1xdesi_desi_dr1_dz0p20",
        "bias_params": "params_desi_dr1_dz0p20_bias_pessimistic.yaml",
    },
}

# The notebook's derivative settings, which the caches record and the
# notebook compares:
#   COSMO and EXTRA_PREFIXES = the varied parameters of the stencil block
#       (the cosmology, then every sampled DESI_B1_* and DESI_SIGV_*: the
#       group amplitudes B_g and the velocity-dispersion leaders);
#   INVCOV_SCALE = the rescaling of the inverse covariance from the
#       14,000 deg^2 of the covariance files to the 12,300 deg^2 of the
#       survey (ModelFisher keeps it; it does not enter any vector);
#   H_MAIN = the relative stencil step of the sampled parameters;
#   OPERATOR_STEPS = the steps d of the per-bin operators (five-point
#       stencils at +-d and +-2d): b1 relative, b2 and bs2 absolute,
#       alpha_0 in h^-2 Mpc^2;
#   N_STENCIL_OFFSETS = the four evaluated points of a five-point
#       stencil, at +2d, +d, -d and -2d (fm.FIVE_POINT_OFFSETS; the center
#       has weight zero and is never evaluated);
#   A2_STEP, ALPHA2_SIGNS = the central alpha_2 step [h^-2 Mpc^2] and its
#       two signs (exact for a term linear in alpha_2);
#   ALPHA2_SETTER, ALPHA2_KEYWORD = the compiled setter of alpha_2 and its
#       argument, as the template cache records them;
#   DERIVATIVE_TIER = 'final', the converged tier every paper figure uses;
#   STENCIL_RULE = the rule of the per-bin columns, as the caches record it.
COSMO = [
    "As_1e9",
    "ns",
    "H0",
    "omegab",
    "omegam",
    "w0",
    "wa",
]
EXTRA_PREFIXES = ("DESI_B1_", "DESI_SIGV_")
INVCOV_SCALE = 12300.0/14000.0
H_MAIN = 0.02
OPERATOR_STEPS = {
    "b1": 0.01,
    "b2": 0.02,
    "bs2": 0.02,
    "a0": 5.0,
}
N_STENCIL_OFFSETS = 4
A2_STEP = 5.0
ALPHA2_SIGNS = (1.0, -1.0)
ALPHA2_SETTER = "set_nuisance_bias_alpha2"
ALPHA2_KEYWORD = "A2"
DERIVATIVE_TIER = "final"
STENCIL_RULE = "five-point"

# the cache files, by kind, with {:d} = the layout number
CACHE_NAMES = {
    "derivatives": "EXAMPLE_EVALUATE_FISHER_MODEL_DERIVATIVES_{:d}.pkl",
    "templates": "EXAMPLE_EVALUATE_FISHER_MODEL_TEMPLATES_{:d}.pkl",
}

# the kinds of evaluation, in task-list order (the module docstring)
TASK_KINDS = (
    "fiducial",
    "operator",
    "stencil",
    "alpha2",
    "fiducial_after",
)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_arguments():
    """Read the command line.

    Returns:
      argparse.Namespace with layout, yaml, out, overwrite,
      threads_per_rank and dry_run (None for a real run; 0 to plan for
      the ranks of this launch; N > 0 to plan for N ranks).
    """
    parser = argparse.ArgumentParser(
        description="Compute the Fisher derivative base of one lssty1xdesi layout "
                    "on MPI ranks (the module docstring explains the split and "
                    "the launch lines).")
    parser.add_argument(
        "--layout", required=True, choices=list(LAYOUTS),
        help="dcom62 (Delta d_com = 62 Mpc), dz0p02, dz0p05 or dz0p20 "
             "(Delta z = 0.02, 0.05, 0.2)")
    parser.add_argument(
        "--yaml", default=None,
        help="the example yaml of the paper likelihood of the layout (default: "
             "projects/lssty1xdesi/EXAMPLE_EVALUATE<n>.yaml, the notebook's yaml)")
    parser.add_argument(
        "--out", default=None,
        help="folder of the two caches (default: projects/lssty1xdesi/chains, "
             "where the notebook reads them)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace cache files that exist (without it the run refuses)")
    parser.add_argument(
        "--threads-per-rank", type=int, default=8,
        help="OpenMP threads of each rank (default 8)")
    # nargs="?" makes the value optional: "--dry-run" alone gives const (0,
    # the ranks of this launch), "--dry-run 12" gives 12, no flag gives None
    parser.add_argument(
        "--dry-run", nargs="?", type=int, const=0, default=None, metavar="N",
        help="print the work split, the evaluations per rank and the target "
             "files, then stop (no model is built); N plans for N ranks "
             "instead of this launch's")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data files of a layout
# ---------------------------------------------------------------------------

def read_rows(path):
    """Return the data rows of a text file, split on blanks (# lines skipped)."""
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip() and not line.startswith("#"):
                rows.append(line.split())
    return rows


def dataset_entry(data_dir, stem, key):
    """Return the value of one "key = value" line of a .dataset file.

    A .dataset file is the small text file that lists which data files a
    likelihood reads, one "key = filename" line each.

    Arguments:
      data_dir = the project's data folder.
      stem     = the dataset stem (a LAYOUTS 'dataset').
      key      = the key of the line.

    Raises:
      KeyError when the file has no line for key.
    """
    with open(os.path.join(data_dir, stem + ".dataset")) as handle:
        for row in handle:
            if row.split("=")[0].strip() == key:
                return row.split("=", 1)[1].strip()
    raise KeyError(f"{key} is not in {stem}.dataset")


def layout_tables(data_dir, stem):
    """Read a layout's bins, cross pairs, bands and vector length.

    The data vector is [EE | gs | gg | gg-cross]: shear and galaxy-shear
    spectra first (empty here, n_ee entries), then one block of nell
    bands per clustering auto spectrum and per measured cross pair. The
    length n is the number of rows of the mask file, which lists every
    entry once; the run checks it against the first evaluated vector.

    Arguments:
      data_dir = the project's data folder.
      stem     = the dataset stem (a LAYOUTS 'dataset').

    Returns:
      dict: 'nbins', 'pairs' (list of (a, b) 0-based bin pairs), 'nell',
      'n' and 'n_ee'.

    Raises:
      ValueError when the blocks do not fit in the mask's length.
    """
    bins = read_rows(path=os.path.join(data_dir, stem + "_bins.txt"))
    pairs = []
    for row in read_rows(path=os.path.join(data_dir, stem + "_pairs.txt")):
        pairs.append((int(row[0]), int(row[1])))
    ell_file = dataset_entry(data_dir=data_dir, stem=stem, key="ell_file")
    ells = np.loadtxt(os.path.join(data_dir, ell_file)).ravel()
    mask_file = dataset_entry(data_dir=data_dir, stem=stem, key="mask_file")
    n = len(read_rows(path=os.path.join(data_dir, mask_file)))
    nbins = len(bins)
    nell = len(ells)
    n_ee = n - (nbins + len(pairs))*nell
    if n_ee < 0:
        raise ValueError(
            f"{stem}: {nbins} autos and {len(pairs)} pairs of {nell} bands need "
            f"{(nbins + len(pairs))*nell} entries, but the mask lists only {n}")
    return {
        "nbins": nbins,
        "pairs": pairs,
        "nell": nell,
        "n": n,
        "n_ee": n_ee,
    }


def pair_coloring(nbins, pairs):
    """Color the cross-pair graph greedily: no pair joins two bins of one color.

    Bin i takes the smallest color that none of its lower-numbered
    neighbors (the bins it shares a measured cross pair with) has. This
    is the notebook's coloring, so the batches are its own.

    Arguments:
      nbins = the number of lens bins.
      pairs = list of (a, b) 0-based bin pairs.

    Returns:
      list of int, the color of each bin (0, 1, ...).
    """
    neighbors = []
    for _ in range(nbins):
        neighbors.append(set())
    for a, b in pairs:
        neighbors[a].add(b)
        neighbors[b].add(a)
    color = []
    for i in range(nbins):
        used = set()
        for j in neighbors[i]:
            if j < i:
                used.add(color[j])
        c = 0
        while c in used:
            c += 1
        color.append(c)
    return color


def bins_of_color(color, c):
    """Return the 0-based indices of the bins of color c, as an int array."""
    members = []
    for i in range(len(color)):
        if color[i] == c:
            members.append(i)
    return np.array(members, dtype=int)


def spectrum_rows(tables):
    """List, for each bin, the vector entries of every spectrum involving it.

    A per-bin column keeps only these entries: the stencil of a batch
    moves several bins at once, and the coloring guarantees that each
    entry belongs to the spectra of at most one shifted bin.

    Arguments:
      tables = the dict of layout_tables.

    Returns:
      list (one per bin) of int arrays of data-vector indices: the bin's
      auto spectrum, then each cross pair that contains it.
    """
    nbins = tables["nbins"]
    nell = tables["nell"]
    n_ee = tables["n_ee"]
    rows = []
    for i in range(nbins):
        entries = list(range(n_ee + i*nell, n_ee + (i + 1)*nell))
        # enumerate yields (position, element): p numbers the pairs in the
        # order of their blocks in the vector
        for p, (a, b) in enumerate(tables["pairs"]):
            if i in (a, b):
                start = n_ee + (nbins + p)*nell
                entries += list(range(start, start + nell))
        rows.append(np.array(entries))
    return rows


def bias_leaders(bias_params):
    """Return, for each bin, the group amplitude DESI_B1_<g> it follows.

    The paper's linear-bias file writes each bin's DESI_B1_<n> as a
    function of its group's amplitude, tilt and residuals,
    "lambda DESI_B1_<g>, ...: DESI_B1_<g>*(1.0 + ...)"; the first lambda
    argument names the group. A sampled DESI_B1_<n> (no value) is its own
    leader. The six leaders name the six bias groups, as the notebook's
    bias_leaders reads them.

    Arguments:
      bias_params = path of the linear-bias parameter file (relative to
                    ROOTDIR).

    Returns:
      list of str, one leader name per bin, in bin order.
    """
    import re
    from cobaya.yaml import yaml_load_file
    params = yaml_load_file(bias_params)
    params = params.get("params", params)
    leader = {}
    for name, spec in params.items():
        # the per-bin amplitudes only, not the group amplitudes DESI_B1_<g>
        if not re.fullmatch(r"DESI_B1_\d+", name):
            continue
        value = ""
        if isinstance(spec, dict):
            value = str(spec.get("value", ""))
        # \w+? stops at the first comma or colon: the first argument
        found = re.search(r"lambda\s+(DESI_B1_\w+?)\s*[,:]", value)
        if found:
            leader[name] = found.group(1)
        else:
            leader[name] = name
    out = []
    for i in range(len(leader)):
        out.append(leader[f"DESI_B1_{i + 1}"])
    return out


# ---------------------------------------------------------------------------
# Settings, fingerprint and the cache files
# ---------------------------------------------------------------------------

def layout_settings(fm, layout, yaml_label):
    """Return the notebook's settings entry of one layout.

    The caches store this dict (plus a few keys of their own) and the
    notebook refuses a cache whose stored settings differ from its own,
    so every key and value is the notebook's (its paper_layout).

    Arguments:
      fm         = the module cosmolike_lssty1xdesi_fisher.
      layout     = a LAYOUTS key.
      yaml_label = the example yaml as stored (the notebook stores the
                   path relative to ROOTDIR; its comparison skips it).

    Returns:
      dict: 'yaml', 'dataset', 'bias_params', 'overrides' (likelihood
      options), 'camb' (CAMB extra arguments), 'accuracy_boost', 'tier'
      and 'stencil'.
    """
    spec = LAYOUTS[layout]
    accuracy = fm.derivative_settings(spec["dataset"], DERIVATIVE_TIER)
    return {
        "yaml": yaml_label,
        "dataset": spec["dataset"],
        "bias_params": PROJECT + "likelihood/" + spec["bias_params"],
        "overrides": accuracy["likelihood_overrides"],
        "camb": accuracy["camb_overrides"],
        "accuracy_boost": accuracy["accuracy_boost"],
        "tier": DERIVATIVE_TIER,
        "stencil": STENCIL_RULE,
    }


def layout_fingerprint(fm, yaml_path, settings):
    """Return the sha256 of the resolved model, as the notebook computes it.

    fm.settings_fingerprint hashes the fully resolved cobaya input (the
    example yaml merged with the likelihood's yaml and its parameter
    files, priors included), the fiducial point, the accuracy settings,
    the tier and the content of the dataset files.

    Arguments:
      fm        = the module cosmolike_lssty1xdesi_fisher.
      yaml_path = the example yaml.
      settings  = the dict of layout_settings.

    Returns:
      str, 64 hexadecimal digits.
    """
    return fm.settings_fingerprint(
        yaml_path,
        likelihood_overrides=settings["overrides"],
        accuracy_boost=settings["accuracy_boost"],
        camb_overrides=settings["camb"],
        tier=settings["tier"])


def cache_settings(settings, fingerprint):
    """Return the 'settings' entry of each cache, as the notebook writes it.

    Arguments:
      settings    = the dict of layout_settings.
      fingerprint = the layout's fingerprint.

    Returns:
      dict {kind: settings dict} for the kinds of CACHE_NAMES.
    """
    derivatives = dict(settings)
    derivatives["steps"] = dict(OPERATOR_STEPS)
    derivatives["h"] = H_MAIN
    derivatives["fingerprint"] = fingerprint
    templates = dict(settings)
    templates["a2_step"] = A2_STEP
    templates["alpha2_setter"] = ALPHA2_SETTER
    templates["alpha2_keyword"] = ALPHA2_KEYWORD
    templates["fingerprint"] = fingerprint
    return {
        "derivatives": derivatives,
        "templates": templates,
    }


def cache_paths(out_dir, layout):
    """Return {kind: path} of the two cache files of a layout in out_dir."""
    number = LAYOUTS[layout]["number"]
    paths = {}
    for kind, pattern in CACHE_NAMES.items():
        paths[kind] = os.path.join(out_dir, pattern.format(number))
    return paths


def existing_targets(paths, fingerprint):
    """Describe the cache files that already exist.

    Arguments:
      paths       = {kind: path} of cache_paths.
      fingerprint = the fingerprint of this run's model.

    Returns:
      list of str, one line per existing file: whether it holds the
      current fingerprint, another one, or cannot be read.
    """
    lines = []
    for path in paths.values():
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as handle:
                stored = pickle.load(handle)["settings"]["fingerprint"]
        except Exception as error:
            lines.append(f"{path} exists and cannot be read ({error!r})")
            continue
        if stored == fingerprint:
            lines.append(f"{path} exists and holds the current fingerprint "
                         "(nothing to recompute)")
        else:
            lines.append(f"{path} exists and holds another fingerprint "
                         f"({stored[:12]}..., current {fingerprint[:12]}...)")
    return lines


def save_caches(paths, payloads, overwrite):
    """Write the two caches, both in full or neither.

    Each payload goes to a temporary file in the target folder first;
    only when both are written are they moved onto their names
    (os.replace renames in one step on one file system), so a failure
    leaves no half-written cache under a cache name.

    Arguments:
      paths     = {kind: path} of cache_paths.
      payloads  = {kind: the dict to pickle}, the same kinds.
      overwrite = True to replace existing files.

    Raises:
      FileExistsError naming every existing target when overwrite is
      False (nothing is written then).

    Side effects: writes the two files (pickle at its default protocol,
    as the notebook writes them).
    """
    partial = {}
    try:
        for kind, path in paths.items():
            partial[kind] = f"{path}.partial.{os.getpid()}"
            with open(partial[kind], "wb") as handle:
                pickle.dump(payloads[kind], handle)
        if not overwrite:
            present = []
            for path in paths.values():
                if os.path.exists(path):
                    present.append(path)
            if present:
                raise FileExistsError(
                    f"{present} appeared during the run; nothing was written. "
                    "Rerun with --overwrite to replace them, or choose another --out")
        for kind, path in paths.items():
            os.replace(partial[kind], path)
            partial[kind] = None
    finally:
        # a temporary file left by a failure is removed; a moved one is None
        for name in partial.values():
            if name is not None and os.path.exists(name):
                os.remove(name)


# ---------------------------------------------------------------------------
# The model, built on the per-bin step class
# ---------------------------------------------------------------------------

def write_step_yaml(yaml_path, folder, step_classes):
    """Write a copy of the example yaml whose likelihood is the step class.

    The copy changes one thing: the likelihood block gains "class" (the
    subclass of compute_fisher_base_mpi_classes that can add a per-bin
    step) and "python_path" (the folder of that module). The block keeps
    its name and every option, so the model is the example's;
    check_same_input verifies that both resolve to the same cobaya input.

    Arguments:
      yaml_path    = the example yaml.
      folder       = where to write the copy (a temporary folder).
      step_classes = the module compute_fisher_base_mpi_classes.

    Returns:
      str, the path of the copy.

    Raises:
      ValueError when the yaml has not exactly one likelihood, or one
      without a step class (only the four paper likelihoods have one).
    """
    from cobaya.yaml import yaml_dump, yaml_load_file
    info = yaml_load_file(yaml_path)
    names = list(info["likelihood"])
    if len(names) != 1:
        raise ValueError(f"{yaml_path} has the likelihoods {names}; exactly one, a "
                         "paper likelihood, is needed")
    name = names[0]
    if name not in step_classes.STEP_CLASS_OF:
        raise ValueError(
            f"{yaml_path}: the likelihood {name} has no per-bin step class; the "
            f"supported likelihoods are {list(step_classes.STEP_CLASS_OF)} (the paper "
            "likelihoods the notebook differentiates)")
    block = dict(info["likelihood"][name])
    block["class"] = CLASSES_MODULE + "." + step_classes.STEP_CLASS_OF[name]
    block["python_path"] = SCRIPT_DIR
    info["likelihood"] = {name: block}
    step_yaml = os.path.join(folder, "step_" + os.path.basename(yaml_path))
    with open(step_yaml, "w") as handle:
        handle.write(yaml_dump(info))
    return step_yaml


def resolved_input(fm, yaml_path, settings):
    """Return the resolved cobaya input of a yaml at the derivative settings.

    Arguments:
      fm        = the module cosmolike_lssty1xdesi_fisher.
      yaml_path = an example yaml (or its step copy).
      settings  = the dict of layout_settings.

    Returns:
      (resolved, point): cobaya's update_info of fm.model_info's input
      (the likelihood's own yaml and parameter files merged in, nothing
      built), and the fiducial point.
    """
    from cobaya.input import update_info
    info, point = fm.model_info(
        yaml_path,
        likelihood_overrides=settings["overrides"],
        accuracy_boost=settings["accuracy_boost"],
        camb_overrides=settings["camb"])
    return update_info(info), point


def check_same_input(fm, yaml_path, step_yaml, settings):
    """Stop unless the step copy resolves to the example yaml's input.

    Compares the resolved likelihood options (without the "class" and
    "python_path" keys the copy adds), theory, parameters, priors and
    the fiducial point, each as sorted JSON text, the form
    fm.settings_fingerprint hashes.

    Arguments:
      fm        = the module cosmolike_lssty1xdesi_fisher.
      yaml_path = the example yaml.
      step_yaml = its step copy (write_step_yaml).
      settings  = the dict of layout_settings.

    Raises:
      RuntimeError naming the blocks that differ.
    """
    import json
    original, original_point = resolved_input(fm=fm, yaml_path=yaml_path,
                                              settings=settings)
    copy, copy_point = resolved_input(fm=fm, yaml_path=step_yaml, settings=settings)
    # the yaml's single likelihood (write_step_yaml refused any other count)
    name = list(original["likelihood"])[0]
    copy_block = dict(copy["likelihood"][name])
    copy_block.pop("class", None)
    copy_block.pop("python_path", None)
    pairs = {
        "likelihood": (original["likelihood"][name], copy_block),
        "theory": (original.get("theory"), copy.get("theory")),
        "params": (original.get("params"), copy.get("params")),
        "prior": (original.get("prior"), copy.get("prior")),
        "point": (original_point, copy_point),
    }
    different = []
    for block, (value_original, value_copy) in pairs.items():
        text_original = json.dumps(value_original, sort_keys=True, default=str)
        text_copy = json.dumps(value_copy, sort_keys=True, default=str)
        if text_original != text_copy:
            different.append(block)
    if different:
        raise RuntimeError(
            f"the step copy of {yaml_path} resolves to another cobaya input in "
            f"{different}: the per-bin step class would not evaluate the example's "
            "model")


def build_step_model(fm, step_classes, yaml_path, settings, check_input):
    """Build this rank's model of the example yaml on the per-bin step class.

    The step copy of the yaml (write_step_yaml) lives in a temporary
    folder that is removed once the model is built: cobaya reads the yaml
    only then.

    Arguments:
      fm           = the module cosmolike_lssty1xdesi_fisher.
      step_classes = the module compute_fisher_base_mpi_classes.
      yaml_path    = the example yaml.
      settings     = the dict of layout_settings.
      check_input  = True to run check_same_input first (one rank is
                     enough: the check reads files only).

    Returns:
      (model, like, seconds): the ReplayModelFisher, its likelihood (a
      per-bin step class) and the seconds the build took.

    Raises:
      RuntimeError when the likelihood cannot project SN2, or from
      check_same_input.
    """
    start = time.time()
    with tempfile.TemporaryDirectory(prefix="fisher_base_") as folder:
        step_yaml = write_step_yaml(yaml_path=yaml_path, folder=folder,
                                    step_classes=step_classes)
        if check_input:
            check_same_input(fm=fm, yaml_path=yaml_path, step_yaml=step_yaml,
                             settings=settings)
        model = step_classes.ReplayModelFisher(
            step_yaml, COSMO,
            extra_prefixes=EXTRA_PREFIXES,
            likelihood_overrides=settings["overrides"],
            invcov_scale=INVCOV_SCALE,
            accuracy_boost=settings["accuracy_boost"],
            camb_overrides=settings["camb"])
    seconds = time.time() - start
    like = model.model.likelihood[model.likelihood_name]
    template_arguments = inspect.signature(like.stochastic_templates).parameters
    if "order" not in template_arguments:
        raise RuntimeError(
            f"{inspect.getsourcefile(like.stochastic_templates)}: stochastic_templates "
            "has no order argument, so it cannot project SN2; install the model B "
            "version of likelihood/desi_cl.py")
    return model, like, seconds


def stencil_names_from_input(resolved):
    """Return the varied parameters ModelFisher would use, without building it.

    ModelFisher varies COSMO, then every sampled parameter starting with
    one of EXTRA_PREFIXES, in the model's order. A sampled parameter is
    one with a prior (cobaya's is_sampled_param), not a fixed value or a
    function of other parameters.

    Arguments:
      resolved = the resolved cobaya input (resolved_input).

    Returns:
      list of str, the varied parameters in order.
    """
    from cobaya.parameterization import is_sampled_param
    names = list(COSMO)
    for name, spec in resolved["params"].items():
        if name in names:
            continue
        if not name.startswith(EXTRA_PREFIXES):
            continue
        if is_sampled_param(spec):
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# The work split
# ---------------------------------------------------------------------------

def task_list(n_stencil_points, ncolors):
    """List every evaluation of a layout, in the order of the module docstring.

    Arguments:
      n_stencil_points = the number of stencil points (N_STENCIL_OFFSETS
                         per varied parameter).
      ncolors          = the number of bin colors.

    Returns:
      list of tuples, one per evaluation:
        ("fiducial",)
        ("operator", op, c, k)  op in OPERATOR_STEPS, color c, stencil
                                point k (offset fm.FIVE_POINT_OFFSETS[k])
        ("stencil", j)          the j-th recorded stencil point
        ("alpha2", c, s)        color c, sign ALPHA2_SIGNS[s]
        ("fiducial_after",)
    """
    tasks = [("fiducial",)]
    for op in OPERATOR_STEPS:
        for c in range(ncolors):
            for k in range(N_STENCIL_OFFSETS):
                tasks.append(("operator", op, c, k))
    for j in range(n_stencil_points):
        tasks.append(("stencil", j))
    for c in range(ncolors):
        for s in range(len(ALPHA2_SIGNS)):
            tasks.append(("alpha2", c, s))
    tasks.append(("fiducial_after",))
    return tasks


def task_owners(n_tasks, n_ranks):
    """Assign each evaluation to a rank, statically and deterministically.

    Evaluation k goes to rank k mod n_ranks, except the last (the
    fiducial point after every step), which goes to the last rank, so
    that with several ranks it is evaluated on another rank than the
    first one. The first evaluation (the fiducial point) is k = 0, on
    rank 0.

    Arguments:
      n_tasks = the number of evaluations (len(task_list(...))).
      n_ranks = the number of MPI ranks, >= 1.

    Returns:
      list of int, the rank of each evaluation.
    """
    owners = []
    for k in range(n_tasks):
        owners.append(k % n_ranks)
    owners[n_tasks - 1] = n_ranks - 1
    return owners


def split_report(tasks, owners, n_ranks):
    """Return a printable table of the evaluations of each kind on each rank."""
    counts = []
    for _ in range(n_ranks):
        row = {}
        for kind in TASK_KINDS:
            row[kind] = 0
        counts.append(row)
    # zip walks the two lists in step: one (task, owner) pair per turn
    for task, owner in zip(tasks, owners):
        counts[owner][task[0]] += 1
    header = (f"{'rank':>5} {'fiducial':>9} {'operator':>9} {'stencil':>8} "
              f"{'alpha2':>7} {'check':>6} {'total':>6}")
    lines = [header]
    for rank in range(n_ranks):
        row = counts[rank]
        total = 0
        for kind in TASK_KINDS:
            total += row[kind]
        lines.append(f"{rank:>5} {row['fiducial']:>9} {row['operator']:>9} "
                     f"{row['stencil']:>8} {row['alpha2']:>7} "
                     f"{row['fiducial_after']:>6} {total:>6}")
    return "\n".join(lines)


def task_label(task):
    """Return a short readable name of one evaluation, for the log."""
    if task[0] == "operator":
        return f"operator {task[1]} color {task[2]} point {task[3]}"
    if task[0] == "stencil":
        return f"stencil point {task[1]}"
    if task[0] == "alpha2":
        return f"alpha2 color {task[1]} sign {ALPHA2_SIGNS[task[2]]:+.0f}"
    return task[0]


def points_digest(points):
    """Return the sha256 of the stencil points, to compare ranks bit for bit."""
    digest = hashlib.sha256()
    for point in points:
        digest.update(np.ascontiguousarray(point, dtype=np.float64).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Evaluation on a rank
# ---------------------------------------------------------------------------

def peak_rss_mb():
    """Return the peak resident memory of this process so far, in MB (1e6 bytes).

    getrusage's ru_maxrss is the largest resident set size the process
    has had; macOS reports it in bytes, Linux in kilobytes (1024 bytes).
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak/1.0e6
    return peak*1024.0/1.0e6


def evaluate_task(fm, task, model, like, color, points):
    """Evaluate the data vector of one task on this rank's model.

    Arguments:
      fm     = the module cosmolike_lssty1xdesi_fisher.
      task   = one tuple of task_list.
      model  = this rank's ReplayModelFisher (in its "evaluate" mode).
      like   = its likelihood (a per-bin step class).
      color  = the bin colors (pair_coloring).
      points = the recorded stencil points.

    Returns:
      1D float64 array, the data vector of the task's point.
    """
    kind = task[0]
    if kind == "stencil":
        return model.dv(param=points[task[1]])
    if kind == "operator":
        op = task[1]
        members = bins_of_color(color=color, c=task[2])
        step = fm.FIVE_POINT_OFFSETS[task[3]]*OPERATOR_STEPS[op]
    elif kind == "alpha2":
        op = "a2"
        members = bins_of_color(color=color, c=task[1])
        step = ALPHA2_SIGNS[task[2]]*A2_STEP
    else:
        # "fiducial" and "fiducial_after": the fiducial point, no step
        return model.dv()
    like.set_per_bin_step(operator=op, bins=members, step=step)
    try:
        vector = model.dv()
    finally:
        like.clear_per_bin_step()
    return vector


def evaluate_share(fm, rank, tasks, owners, model, like, color, points, n_expected,
                   label, start, process_start):
    """Evaluate this rank's tasks in list order and keep their vectors.

    Right after the fiducial point (always rank 0's first task) the
    templates and the per-bin biases are read: both come from the last
    evaluated point (its kernels and its arrays).

    Arguments:
      fm, model, like, color, points = as in evaluate_task.
      rank       = this rank.
      tasks      = task_list.
      owners     = task_owners.
      n_expected = the vector length the dataset lays out.
      label      = the layout's name, for the messages.
      start      = the run's start time (time.time()).
      process_start = the time main() started on this rank.

    Returns:
      (results, fiducial_extras, timing): {task index: vector};
      None, or on rank 0 {"T0", "T2" ((n, nbins) templates of SN0 and
      SN2), "passed" (the biases of passed_biases), "seconds"}; and
      {"startup" (seconds from main() to this rank's first evaluation:
      imports, fingerprint, checks, model build), "evaluations",
      "evaluation_seconds", "peak_rss_mb" (peak_rss_mb after the last
      evaluation)}.

    Raises:
      ValueError when the fiducial vector's length is not the dataset's.
    """
    mine = []
    for k in range(len(tasks)):
        if owners[k] == rank:
            mine.append(k)
    results = {}
    fiducial_extras = None
    evaluation_start = time.time()
    # enumerate yields (position, element): done counts this rank's tasks
    for done, k in enumerate(mine):
        task_start = time.time()
        vector = evaluate_task(fm=fm, task=tasks[k], model=model, like=like,
                               color=color, points=points)
        results[k] = vector
        if tasks[k][0] == "fiducial":
            if len(vector) != n_expected:
                raise ValueError(
                    f"{label}: the vector has {len(vector)} entries, the dataset's "
                    f"mask lists {n_expected}: the derivatives and the dataset "
                    "disagree on the layout")
            fiducial_extras = {
                "T0": like.stochastic_templates(order=0),
                "T2": like.stochastic_templates(order=2),
                "passed": like.passed_biases(),
                "seconds": time.time() - start,
            }
        print(f"[rank {rank}] {done + 1}/{len(mine)} {task_label(tasks[k])}: "
              f"{time.time() - task_start:.1f} s", flush=True)
    timing = {
        "startup": evaluation_start - process_start,
        "evaluations": len(mine),
        "evaluation_seconds": time.time() - evaluation_start,
        "peak_rss_mb": peak_rss_mb(),
    }
    return results, fiducial_extras, timing


# ---------------------------------------------------------------------------
# Assembly on rank 0
# ---------------------------------------------------------------------------

def assemble_operator_columns(fm, vectors, task_index, color, rows, b1, n):
    """Combine the operator vectors into per-bin columns, as the notebook does.

    For operator op and color c the five-point rule fm.five_point on the
    four stencil vectors gives dm/d(op) for the whole batch; bin i of
    the batch keeps it on the entries of its own spectra (rows[i]). The
    b1 step is relative, so its column is divided by b_{1,i}.

    Arguments:
      fm         = the module cosmolike_lssty1xdesi_fisher.
      vectors    = the data vector of every task, by task index.
      task_index = {task tuple: index}.
      color      = the bin colors.
      rows       = spectrum_rows.
      b1         = the fiducial per-bin b_1.
      n          = the vector length.

    Returns:
      dict {op: (n, nbins) float64 array}: 'b1' per unit b_1, 'b2' and
      'bs2' per unit, 'a0' per h^-2 Mpc^2 of alpha_0.
    """
    nbins = len(color)
    ncolors = max(color) + 1
    ops = {}
    for op, step in OPERATOR_STEPS.items():
        Dop = np.zeros((n, nbins))
        for c in range(ncolors):
            members = bins_of_color(color=color, c=c)
            stencil_vectors = []
            for k in range(N_STENCIL_OFFSETS):
                stencil_vectors.append(vectors[task_index[("operator", op, c, k)]])
            d = fm.five_point(stencil_vectors, step)
            for i in members:
                # the b1 step is relative: (b1 dm/db1)/b1 = dm/db1
                Dop[rows[i], i] = d[rows[i]]/(b1[i] if op == "b1" else 1.0)
        ops[op] = Dop
    return ops


def assemble_alpha2_columns(vectors, task_index, color, rows, m0, label):
    """Form the alpha_2 columns and their linearity check, as the notebook does.

    alpha_2 enters the vector linearly, so the central difference
    [m(+A2_STEP) - m(-A2_STEP)]/(2 A2_STEP) is the exact column; the gap
    between the forward and the backward difference measures the
    linearity.

    Arguments:
      vectors, task_index, color, rows = as in assemble_operator_columns.
      m0    = the fiducial vector.
      label = the layout's name, for the error message.

    Returns:
      (A2, nonlinearity): A2 = (n, nbins) float64, dm/dalpha_{2,i} per
      h^-2 Mpc^2; nonlinearity = max over colors of max |forward -
      backward| / max |central|.

    Raises:
      RuntimeError when a step leaves the vector unchanged (the alpha_2
      setter has no effect).
    """
    nbins = len(color)
    ncolors = max(color) + 1
    A2 = np.zeros((len(m0), nbins))
    nonlinearity = 0.0
    for c in range(ncolors):
        members = bins_of_color(color=color, c=c)
        plus = vectors[task_index[("alpha2", c, 0)]]
        minus = vectors[task_index[("alpha2", c, 1)]]
        central = (plus - minus)/(2.0*A2_STEP)
        forward = (plus - m0)/A2_STEP
        backward = (m0 - minus)/A2_STEP
        size = float(np.max(np.abs(central)))
        if size == 0.0:
            raise RuntimeError(
                f"{label}: an alpha_2 step of {A2_STEP} on color {c} left the vector "
                f"unchanged: {ALPHA2_SETTER} has no effect, or the likelihood does "
                "not call it at every point")
        gap = float(np.max(np.abs(forward - backward)))/size
        nonlinearity = max(nonlinearity, gap)
        for i in members:
            A2[rows[i], i] = central[rows[i]]
    return A2, nonlinearity


def assemble_caches(fm, model, tables, color, points, tasks, gathered, fiducial_extras,
                    settings, fingerprint, label, start):
    """Assemble the two caches from every rank's vectors (rank 0).

    Arguments:
      fm              = the module cosmolike_lssty1xdesi_fisher.
      model           = rank 0's ReplayModelFisher (for the replay of
                        the stencil columns, the names and the priors).
      tables          = layout_tables.
      color           = the bin colors.
      points          = the recorded stencil points.
      tasks           = task_list.
      gathered        = one {task index: vector} dict per rank.
      fiducial_extras = rank 0's templates and biases (evaluate_share).
      settings        = the dict of layout_settings.
      fingerprint     = the layout's fingerprint.
      label           = the layout's name, for the messages.
      start           = the run's start time.

    Returns:
      (payloads, checks): {kind: the dict to pickle}, the notebook's
      cache dictionaries key by key; checks = {"nonlinearity", "reset"}.

    Raises:
      RuntimeError when an evaluation is missing, or from the replay and
      the alpha_2 assembly.
    """
    vectors = []
    for _ in range(len(tasks)):
        vectors.append(None)
    for rank_results in gathered:
        for k, vector in rank_results.items():
            vectors[k] = vector
    for k in range(len(tasks)):
        if vectors[k] is None:
            raise RuntimeError(
                f"no rank returned evaluation {k} ({task_label(tasks[k])})")
    task_index = {}
    for k in range(len(tasks)):
        task_index[tasks[k]] = k

    m0 = vectors[task_index[("fiducial",)]]
    passed = fiducial_extras["passed"]
    b1 = passed["B1"]
    rows = spectrum_rows(tables=tables)
    ops = assemble_operator_columns(fm=fm, vectors=vectors, task_index=task_index,
                                    color=color, rows=rows, b1=b1, n=len(m0))
    stencil_vectors = []
    for j in range(len(points)):
        stencil_vectors.append(vectors[task_index[("stencil", j)]])
    D = model.derivatives_from_vectors(h=H_MAIN, points=points,
                                       vectors=stencil_vectors)
    A2, nonlinearity = assemble_alpha2_columns(vectors=vectors, task_index=task_index,
                                               color=color, rows=rows, m0=m0,
                                               label=label)
    m_after = vectors[task_index[("fiducial_after",)]]
    reset = float(np.max(np.abs(m_after - m0))/np.max(np.abs(m0)))
    priors = {}
    for i, sigma in model.prior_sigmas().items():
        priors[model.names[i]] = sigma

    seconds = time.time() - start
    from cobaya.likelihoods.lssty1xdesi import desi_cl as desi_cl_module
    sn2_unit = dict(desi_cl_module.SN2_UNIT)
    stored = cache_settings(settings=settings, fingerprint=fingerprint)
    derivs = {
        "names": list(model.names),
        "fiducial": np.array(model.fiducial),
        "m0": m0,
        "D": D,
        "priors": priors,
        "b1": b1,
        "b2": passed["B2"],
        "bs2": passed["BS2"],
        "ops": ops,
        "T": fiducial_extras["T0"],
        "leaders": bias_leaders(bias_params=settings["bias_params"]),
        "colors": max(color) + 1,
        "seconds": seconds,
    }
    templates = {
        "m0": m0,
        "a2": A2,
        "T0": fiducial_extras["T0"],
        "T2": fiducial_extras["T2"],
        "sn2_unit": sn2_unit,
        "nonlinearity": nonlinearity,
        "reset": reset,
        "colors": max(color) + 1,
        "seconds": seconds,
    }
    payloads = {
        "derivatives": {"derivs": derivs, "settings": stored["derivatives"]},
        "templates": {"templates": templates, "settings": stored["templates"]},
    }
    checks = {
        "nonlinearity": nonlinearity,
        "reset": reset,
    }
    return payloads, checks


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def resolve_paths(args, rootdir):
    """Return the example yaml and the output folder, both absolute.

    A --yaml or --out given on the command line is read relative to the
    folder the run was launched from; the defaults are the notebook's
    files under ROOTDIR. The yaml is also returned as the notebook
    stores it: relative to ROOTDIR when it lies inside it.

    Arguments:
      args    = parse_arguments' result.
      rootdir = the Cocoa/ folder ($ROOTDIR).

    Returns:
      (yaml_path, yaml_label, out_dir).

    Raises:
      FileNotFoundError when the yaml does not exist.
    """
    number = LAYOUTS[args.layout]["number"]
    if args.yaml is None:
        yaml_path = os.path.join(rootdir, PROJECT, f"EXAMPLE_EVALUATE{number}.yaml")
    else:
        yaml_path = os.path.abspath(args.yaml)
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"the example yaml {yaml_path} does not exist")
    if yaml_path.startswith(rootdir + os.sep):
        yaml_label = os.path.relpath(yaml_path, rootdir)
    else:
        yaml_label = yaml_path
    if args.out is None:
        out_dir = os.path.join(rootdir, PROJECT, "chains")
    else:
        out_dir = os.path.abspath(args.out)
    return yaml_path, yaml_label, out_dir


def dry_run(args, comm, fm, step_classes, yaml_path, settings, out_dir, data_dir):
    """Print the work split and the target files, without building a model.

    Only rank 0 prints. The varied parameters come from the resolved
    yaml (stencil_names_from_input), the colors from the pair file.

    Arguments:
      args, comm = as in run.
      fm, step_classes = the modules cosmolike_lssty1xdesi_fisher and
                   compute_fisher_base_mpi_classes.
      yaml_path, settings, out_dir = the example yaml, layout_settings
                   and the output folder.
      data_dir   = the project's data folder.
    """
    if comm.Get_rank() != 0:
        return
    if args.dry_run > 0:
        n_ranks = args.dry_run
    else:
        n_ranks = comm.Get_size()
    spec = LAYOUTS[args.layout]
    tables = layout_tables(data_dir=data_dir, stem=spec["dataset"])
    color = pair_coloring(nbins=tables["nbins"], pairs=tables["pairs"])
    ncolors = max(color) + 1
    resolved, _ = resolved_input(fm=fm, yaml_path=yaml_path, settings=settings)
    name = list(resolved["likelihood"])[0]
    if name not in step_classes.STEP_CLASS_OF:
        print(f"note: the likelihood {name} has no per-bin step class; a real run "
              "would stop", flush=True)
    names = stencil_names_from_input(resolved=resolved)
    tasks = task_list(n_stencil_points=N_STENCIL_OFFSETS*len(names), ncolors=ncolors)
    owners = task_owners(n_tasks=len(tasks), n_ranks=n_ranks)
    fingerprint = layout_fingerprint(fm=fm, yaml_path=yaml_path, settings=settings)
    n_operator = len(OPERATOR_STEPS)*ncolors*N_STENCIL_OFFSETS
    n_alpha2 = ncolors*len(ALPHA2_SIGNS)
    print(f"layout {args.layout} ({spec['label']}): {tables['nbins']} bins, "
          f"{len(tables['pairs'])} cross pairs, {ncolors} bin colors, vector length "
          f"{tables['n']}", flush=True)
    print(f"yaml {yaml_path}\nfingerprint {fingerprint}", flush=True)
    print(f"{len(names)} stencil parameters: {names}", flush=True)
    print(f"evaluations: 1 fiducial + {n_operator} operator ({len(OPERATOR_STEPS)} "
          f"operators x {ncolors} colors x {N_STENCIL_OFFSETS} points) + "
          f"{N_STENCIL_OFFSETS*len(names)} stencil ({len(names)} parameters x "
          f"{N_STENCIL_OFFSETS} points) + {n_alpha2} alpha_2 ({ncolors} colors x "
          f"{len(ALPHA2_SIGNS)} signs) + 1 fiducial check = {len(tasks)}", flush=True)
    print(f"{n_ranks} ranks x {args.threads_per_rank} threads = "
          f"{n_ranks*args.threads_per_rank} cores", flush=True)
    print(split_report(tasks=tasks, owners=owners, n_ranks=n_ranks), flush=True)
    print(f"output folder {out_dir}", flush=True)
    lines = existing_targets(paths=cache_paths(out_dir=out_dir, layout=args.layout),
                             fingerprint=fingerprint)
    if not lines:
        print("  no cache of this layout exists there", flush=True)
        return
    for line in lines:
        print("  " + line, flush=True)
    if not args.overwrite:
        print("  a real run would stop here: pass --overwrite to replace them",
              flush=True)


def check_before_build(fm, ci, comm, fingerprint, paths, overwrite, out_dir):
    """Run the checks that must pass before any model is built (rank 0).

    The final tier needs CAMB with the halofit tolerance patch, the
    alpha_2 columns need the compiled alpha_2 setter, every rank must
    have computed the same fingerprint (ranks on other nodes must see
    the same files), and no target may exist unless overwrite is set.

    Arguments:
      fm, ci      = the modules cosmolike_lssty1xdesi_fisher and
                    cosmolike_lssty1xdesi_interface.
      comm        = the MPI communicator.
      fingerprint = this rank's fingerprint.
      paths       = cache_paths.
      overwrite   = --overwrite.
      out_dir     = the output folder.

    Raises:
      RuntimeError or FileExistsError naming the failed check.

    Side effects: rank 0 creates out_dir when every check passes.
    """
    fingerprints = comm.allgather(fingerprint)
    if comm.Get_rank() != 0:
        return
    fm.check_halofit_patch()
    if not hasattr(ci, ALPHA2_SETTER):
        raise RuntimeError(
            f"the compiled interface {ci.__file__} has no {ALPHA2_SETTER}: the "
            "alpha_2 columns need the model B counterterm in CosmoLike; rebuild the "
            "interface with it")
    for other_rank in range(comm.Get_size()):
        if fingerprints[other_rank] != fingerprint:
            raise RuntimeError(
                f"rank {other_rank} computed the fingerprint "
                f"{fingerprints[other_rank]}, rank 0 {fingerprint}: the ranks see "
                "different yaml, parameter or dataset files; give every node the "
                "same files at the same paths")
    lines = existing_targets(paths=paths, fingerprint=fingerprint)
    if lines and not overwrite:
        raise FileExistsError(
            "the caches exist:\n  " + "\n  ".join(lines)
            + "\nnothing was computed; pass --overwrite to replace them, or choose "
              "another --out")
    os.makedirs(out_dir, exist_ok=True)


def run(args, comm, process_start):
    """Compute and save the two caches of one layout on every rank of comm.

    The steps, in order: the checks before any model is built
    (check_before_build); the model build on every rank; the stencil
    points recorded on every rank and compared; the evaluations of each
    rank's share; the gather to rank 0; the assembly, the final
    fingerprint check and the save on rank 0.

    Arguments:
      args = parse_arguments' result.
      comm = the MPI communicator of the ranks.
      process_start = the time main() started (for the start-up and wall
             times of the final table).

    Raises:
      RuntimeError, ValueError, FileExistsError or FileNotFoundError
      naming the failed check.

    Side effects: changes into ROOTDIR; rank 0 writes the two cache
    files of the layout.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    rootdir = os.environ["ROOTDIR"]
    yaml_path, yaml_label, out_dir = resolve_paths(args=args, rootdir=rootdir)
    # the example yamls name their data and codes relative to ROOTDIR
    os.chdir(rootdir)
    sys.path.insert(0, os.path.join(rootdir, "external_modules/code/cosmolike_core"))
    sys.path.insert(0, os.path.join(rootdir, PROJECT, "interface"))
    sys.path.insert(0, SCRIPT_DIR)
    # these load the compiled interface, which reads OMP_NUM_THREADS
    import cosmolike_lssty1xdesi_fisher as fm
    import cosmolike_lssty1xdesi_interface as ci
    import compute_fisher_base_mpi_classes as step_classes

    spec = LAYOUTS[args.layout]
    data_dir = os.path.join(rootdir, PROJECT, "data")
    settings = layout_settings(fm=fm, layout=args.layout, yaml_label=yaml_label)
    if args.dry_run is not None:
        dry_run(args=args, comm=comm, fm=fm, step_classes=step_classes,
                yaml_path=yaml_path, settings=settings, out_dir=out_dir,
                data_dir=data_dir)
        return

    start = time.time()
    paths = cache_paths(out_dir=out_dir, layout=args.layout)
    fingerprint = layout_fingerprint(fm=fm, yaml_path=yaml_path, settings=settings)
    check_before_build(fm=fm, ci=ci, comm=comm, fingerprint=fingerprint, paths=paths,
                       overwrite=args.overwrite, out_dir=out_dir)
    if rank == 0:
        print(f"layout {args.layout} ({spec['label']}), {size} ranks x "
              f"{args.threads_per_rank} threads; yaml {yaml_path}; fingerprint "
              f"{fingerprint}", flush=True)
    tables = layout_tables(data_dir=data_dir, stem=spec["dataset"])
    color = pair_coloring(nbins=tables["nbins"], pairs=tables["pairs"])
    model, like, build_seconds = build_step_model(
        fm=fm, step_classes=step_classes, yaml_path=yaml_path, settings=settings,
        check_input=(rank == 0))

    # the stencil points: computed from the fiducial point on every rank,
    # so they must be the same floats everywhere
    points = model.record_stencil_points(h=H_MAIN)
    digests = comm.allgather(points_digest(points=points))
    for other_rank in range(size):
        if digests[other_rank] != digests[0]:
            raise RuntimeError(
                f"rank {other_rank} recorded other stencil points than rank 0: the "
                "ranks built different models")
    tasks = task_list(n_stencil_points=len(points), ncolors=max(color) + 1)
    owners = task_owners(n_tasks=len(tasks), n_ranks=size)
    if rank == 0:
        print(f"{len(model.names)} stencil parameters: {model.names}", flush=True)
        print(split_report(tasks=tasks, owners=owners, n_ranks=size), flush=True)

    results, fiducial_extras, timing = evaluate_share(
        fm=fm, rank=rank, tasks=tasks, owners=owners, model=model, like=like,
        color=color, points=points, n_expected=tables["n"], label=spec["label"],
        start=start, process_start=process_start)
    timing["build"] = build_seconds
    gathered = comm.gather(results, root=0)
    timings = comm.gather(timing, root=0)
    if rank != 0:
        return

    payloads, checks = assemble_caches(
        fm=fm, model=model, tables=tables, color=color, points=points, tasks=tasks,
        gathered=gathered, fiducial_extras=fiducial_extras, settings=settings,
        fingerprint=fingerprint, label=spec["label"], start=start)
    # the files must not have changed while the ranks evaluated
    final_fingerprint = layout_fingerprint(fm=fm, yaml_path=yaml_path,
                                           settings=settings)
    if final_fingerprint != fingerprint:
        raise RuntimeError(
            f"the fingerprint changed during the run ({fingerprint} at the start, "
            f"{final_fingerprint} now): a yaml, parameter or dataset file was edited, "
            "so the vectors may mix two models; nothing was written")
    save_caches(paths=paths, payloads=payloads, overwrite=args.overwrite)

    print(f"alpha_2: max |forward - backward| / max |central| = "
          f"{checks['nonlinearity']:.2e}; the fiducial point after every step (rank "
          f"{size - 1}) against rank 0's: max |m - m0| / max |m0| = "
          f"{checks['reset']:.2e}", flush=True)
    # start-up = main() to the first evaluation (build included); the
    # evaluation rate is per rank; peak RSS is each rank's largest
    # resident memory up to its last evaluation
    print(f"{'rank':>5} {'startup [s]':>12} {'build [s]':>10} {'evaluations':>12} "
          f"{'evaluating [s]':>15} {'per minute':>11} {'peak RSS [MB]':>14}",
          flush=True)
    for other_rank in range(size):
        t = timings[other_rank]
        rate = 60.0*t["evaluations"]/t["evaluation_seconds"]
        print(f"{other_rank:>5} {t['startup']:>12.1f} {t['build']:>10.1f} "
              f"{t['evaluations']:>12} {t['evaluation_seconds']:>15.1f} "
              f"{rate:>11.2f} {t['peak_rss_mb']:>14.0f}", flush=True)
    wall = time.time() - process_start
    print(f"wall time {wall:.0f} s for {len(tasks)} evaluations "
          f"({60.0*len(tasks)/wall:.2f} per minute); wrote:", flush=True)
    for path in paths.values():
        print(f"  {path}", flush=True)


def main():
    """Parse the command line, set the threads, and run on MPI.COMM_WORLD.

    OMP_NUM_THREADS is set before anything imports the compiled
    interface: the OpenMP runtime and the likelihood base read it once,
    when they load.

    Raises:
      ValueError for a thread count below 1 or a negative --dry-run N;
      RuntimeError when ROOTDIR is unset (start_cocoa.sh sets it).
    """
    process_start = time.time()
    args = parse_arguments()
    if args.threads_per_rank < 1:
        raise ValueError(f"--threads-per-rank must be at least 1; got "
                         f"{args.threads_per_rank}")
    if args.dry_run is not None and args.dry_run < 0:
        raise ValueError(f"--dry-run N needs N >= 1 ranks (or no N); got "
                         f"{args.dry_run}")
    if not os.environ.get("ROOTDIR"):
        raise RuntimeError("ROOTDIR is not set: source start_cocoa.sh from the Cocoa/ "
                           "folder first")
    os.environ["OMP_NUM_THREADS"] = str(args.threads_per_rank)
    run(args=args, comm=MPI.COMM_WORLD, process_start=process_start)


if __name__ == "__main__":
    # A failure on one rank would leave the others waiting for its vectors
    # forever; with several ranks the failing one prints its error and
    # stops every rank (MPI Abort).
    try:
        main()
    except Exception:
        world = MPI.COMM_WORLD
        if world.Get_size() == 1:
            raise
        sys.stderr.write(f"[rank {world.Get_rank()}] failed:\n{traceback.format_exc()}")
        sys.stderr.flush()
        world.Abort(1)
