"""Compute the per-bin velocity-dispersion columns of one layout on MPI ranks.

What is computed. The per-bin sigma_v study (notes/velocity_dispersion_per_bin)
gives each thin bin's Fingers-of-God dispersion sigma_{v,i} its own freedom
on top of today's one value per dispersion group (sigma_{v,i} is the
line-of-sight velocity spread, in km/s, that blurs bin i's redshift
distribution n(z); a dispersion group is a set of consecutive bins whose
sampled dispersion is one parameter). In linear response every such model
is a combination of the per-bin columns

  dm/dsigma_{v,i}   (m the DESI band powers, sigma_{v,i} = DESI_SIGV_<i> in
                     km/s, every other parameter at the fiducial point),

so this script computes those columns and nothing else. Each is a
five-point stencil at relative steps +-d and +-2d of sigma_{v,i}: the rule
[-m(+2d) + 8 m(+d) - 8 m(-d) + m(-2d)]/(12 d) (fm.five_point; the center
has weight zero and is never evaluated), with the bins moved in batches:
the greedy coloring of the cross-pair graph that the derivative base uses
puts two bins joined by a measured cross pair in different batches
("colors"), so every spectrum involves at most one shifted bin and one
stencil gives the columns of every bin of a batch
(compute_fisher_base_mpi.py explains the coloring; every layout needs four
colors). A step of sigma_{v,i} only changes bin i's blurred n(z)
(compute_fisher_base_mpi_classes.py, the "sigv" operator), so bin i keeps
the stencil on the entries of its own spectra.

Several steps d can be computed in one run (--steps); the cache keeps one
set of columns per step, so the step convergence of the columns is read
from one file. The study uses the step of the group dispersions'
stencil columns (H_MAIN = 0.02, relative), so the sum of a group's
per-bin columns can be compared with the group column of the derivative
base (model_derivatives_<layout>_final.npz, column DESI_SIGV_<leader>).

The evaluations of a layout, in order:

  1                 the fiducial point (its vector and per-bin sigma_v);
  4 x 4 x n_steps   per-bin sigma_v points: 4 colors x 4 stencil points
                    per step;
  1                 the fiducial point again, after every step (the reset
                    check).

They are split over the ranks as in compute_fisher_base_mpi.py: evaluation
k on rank k mod N, the last on rank N - 1.

Output. One cache, written with fm.write_fisher_cache (an .npz file of
plain arrays with a JSON header):

  model_sigv_bins_<layout>_final.npz
    'm0'       the fiducial vector;
    'sigma_v'  the fiducial per-bin dispersions [km/s];
    'zbar'     the bin mean redshifts;
    'color'    the bin colors;
    'D_sigv'   {"<d>": (n, nbins) array of dm/dsigma_{v,i} per km/s} for
               each step d of --steps (keys as repr(d), e.g. "0.02");
    'reset'    max |m_after - m0| / max |m0| of the two fiducial points;
    'seconds'  the wall time.

The settings stored with it are the derivative base's settings of the
layout (layout_settings: yaml, dataset, accuracy, tier), the steps and the
fingerprint of the resolved paper yaml, so a reader can check that the
columns belong to the same model as the base. An existing cache is never
replaced without --overwrite. The script writes working copies in chains/
only; archiving is a separate, explicit call (fm.save_fisher_to_cache).

Launch (from the Cocoa/ folder, with the cocoa environment active and
start_cocoa.sh sourced):

    mpirun -n 2 --oversubscribe \\
      python ./projects/lssty1xdesi/scripts/compute_sigv_bins_mpi.py \\
        --layout dz0p20 --threads-per-rank 4 --steps 0.01 0.02 0.04

Memory per rank is the derivative base's (near 14 GB on Delta d_com =
62 Mpc, 3 GB on Delta z = 0.2): run Delta d_com = 62 Mpc on one rank.
"""
import argparse
import os
import sys
import time
import traceback

# Importing compute_fisher_base_mpi runs its top-level code, which sets
# one BLAS thread per rank before numpy loads (the BLAS library reads the
# thread variables once, when numpy loads it); this import must therefore
# stay above the numpy import. The module also provides the layout table,
# the coloring, the work split and the cache plumbing reused here.
import compute_fisher_base_mpi as base
import numpy as np
from mpi4py import MPI

# the layouts of today's pair lists (the paper yamls EXAMPLE_EVALUATE1-4)
SIGV_LAYOUTS = ("dcom62", "dz0p02", "dz0p05", "dz0p20")
# the step of the group dispersions' stencil columns (relative): --steps
# defaults to it so the sum check of the fits compares equal steps
MAIN_STEP = base.H_MAIN
# the cache name: what it holds, the layout and the derivative tier
CACHE_NAME = "model_sigv_bins_{layout}_{tier}"


def parse_arguments():
    """Read the command line.

    Returns:
      argparse.Namespace with layout, steps, out, overwrite and
      threads_per_rank.
    """
    parser = argparse.ArgumentParser(
        description="Per-bin sigma_v columns of one lssty1xdesi layout "
                    "(module docstring).")
    parser.add_argument("--layout", required=True, choices=list(SIGV_LAYOUTS),
                        help="the layout (its paper yaml EXAMPLE_EVALUATE<n>.yaml)")
    parser.add_argument("--steps", type=float, nargs="+", default=[MAIN_STEP],
                        help="relative stencil steps d of sigma_v (default: "
                             f"{MAIN_STEP}); the first is not special")
    parser.add_argument("--out", default=None,
                        help="output folder (default: the project's chains/)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing cache")
    parser.add_argument("--threads-per-rank", type=int, default=8,
                        help="OpenMP threads of each rank (default 8)")
    return parser.parse_args()


def task_list(steps, ncolors):
    """List every evaluation, in the order of the module docstring.

    Arguments:
      steps   = the relative stencil steps of --steps.
      ncolors = the number of bin colors of the layout.

    Returns:
      list of tuples: ("fiducial",), ("sigv", s, c, k) for step index s,
      color c and stencil point k (offset fm.FIVE_POINT_OFFSETS[k]), and
      ("fiducial_after",).
    """
    tasks = [("fiducial",)]
    for s in range(len(steps)):
        for c in range(ncolors):
            for k in range(base.N_STENCIL_OFFSETS):
                tasks.append(("sigv", s, c, k))
    tasks.append(("fiducial_after",))
    return tasks


def evaluate_task(fm, task, model, like, color, steps):
    """Return the data vector of one task on this rank's model.

    A "fiducial" or "fiducial_after" task evaluates the point as it is.
    A "sigv" task first registers a per-bin step with the likelihood:
    every bin of the task's color gets its DESI_SIGV_<i> scaled by
    (1 + offset*d), with offset one of the four stencil offsets (+2, +1,
    -1, -2) and d the task's relative step.

    Arguments:
      fm    = the module cosmolike_lssty1xdesi_fisher.
      task  = one tuple of task_list.
      model = this rank's ReplayModelFisher (in its "evaluate" mode).
      like  = its likelihood (a per-bin step class of
              compute_fisher_base_mpi_classes).
      color = the bin colors (base.pair_coloring).
      steps = the relative steps d of --steps.

    Returns:
      1D float64 array, the data vector of the task's point.
    """
    if task[0] != "sigv":
        return model.dv()
    # tuple unpacking: task = (kind, step index, color, stencil point);
    # the leading _ discards the kind, tested just above
    _, s, c, k = task
    members = base.bins_of_color(color=color, c=c)
    like.set_per_bin_step(operator="sigv", bins=members,
                          step=fm.FIVE_POINT_OFFSETS[k]*steps[s])
    # try/finally guarantees the step is cleared even when the evaluation
    # fails, so no later task of this rank inherits it
    try:
        vector = model.dv()
    finally:
        like.clear_per_bin_step()
    return vector


def assemble(fm, tables, color, steps, tasks, vectors, sigma_v):
    """Combine the stencil vectors into per-bin columns per km/s.

    For step d and color c, fm.five_point on the four vectors gives the
    derivative with respect to the relative offset x of the batch
    (sigma_v -> sigma_v (1 + x)), which equals dm/d(ln sigma_v); bin i
    keeps it on the entries of its own spectra (the coloring guarantees
    no other shifted bin touches them) and divides by its fiducial
    sigma_{v,i}, turning the column into dm/dsigma_{v,i} per km/s.

    Shape flow (one step):
      4 vectors [n] -> five_point [n] -> per-bin columns D [n, nbins]
      legend: n = data-vector entries, nbins = lens bins

    Arguments:
      fm      = the module cosmolike_lssty1xdesi_fisher.
      tables  = base.layout_tables' dict of the layout.
      color   = the bin colors (base.pair_coloring).
      steps   = the relative steps d of --steps.
      tasks   = task_list's list.
      vectors = the data vector of every task, by task index.
      sigma_v = the fiducial per-bin dispersions [km/s].

    Returns:
      dict {repr(d): (n, nbins) float64 array}, one entry per step. The
      keys are repr strings so the cache's JSON header and the fits
      script (compute_sigv_bins_fits.py, its STEP) can compare them as
      text.
    """
    # a tuple is hashable, so a task tuple can be a dict key: index maps
    # each task to its position in the evaluation list
    index = {}
    for k in range(len(tasks)):
        index[tasks[k]] = k
    # rows[i] = the data-vector entries of every spectrum involving bin i
    # (its auto spectrum and each measured cross pair that contains it)
    rows = base.spectrum_rows(tables=tables)
    n = len(vectors[index[("fiducial",)]])
    nbins = len(color)
    out = {}
    # enumerate yields (position, element): s numbers the steps
    for s, step in enumerate(steps):
        D = np.zeros((n, nbins))
        for c in range(max(color) + 1):
            stencil = []
            for k in range(base.N_STENCIL_OFFSETS):
                stencil.append(vectors[index[("sigv", s, c, k)]])
            # the five-point rule on the whole batch of color c
            d = fm.five_point(stencil, step)
            for i in base.bins_of_color(color=color, c=c):
                # rows[i] is an int array: the assignment writes only
                # those rows of column i (the rest of the column stays 0)
                D[rows[i], i] = d[rows[i]]/sigma_v[i]
        out[repr(float(step))] = D
    return out


def run(args, comm, process_start):
    """Compute and save the per-bin sigma_v cache of one layout.

    The steps, in order: the fingerprint and target checks on rank 0
    before any model is built; the model build on every rank; the
    evaluations of each rank's share; the gather to rank 0; the
    assembly, the final fingerprint check and the save on rank 0.

    Arguments:
      args = parse_arguments' result.
      comm = the MPI communicator of the ranks.
      process_start = the time main() started (for the wall time).

    Raises:
      RuntimeError, ValueError or FileExistsError naming the failed
      check.

    Side effects: changes into ROOTDIR; rank 0 writes the cache file.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    rootdir = os.environ["ROOTDIR"]
    number = base.LAYOUTS[args.layout]["number"]
    yaml_path = os.path.join(rootdir, base.PROJECT, f"EXAMPLE_EVALUATE{number}.yaml")
    yaml_label = os.path.relpath(yaml_path, rootdir)
    # one simple ternary: the default output folder is the project's
    # chains/, where the fits script reads the cache
    out_dir = (os.path.join(rootdir, base.PROJECT, "chains") if args.out is None
               else os.path.abspath(args.out))
    # the example yamls name their data and codes relative to ROOTDIR
    os.chdir(rootdir)
    sys.path.insert(0, os.path.join(rootdir, "external_modules/code/cosmolike_core"))
    sys.path.insert(0, os.path.join(rootdir, base.PROJECT, "interface"))
    sys.path.insert(0, base.SCRIPT_DIR)
    # these load the compiled interface, which reads OMP_NUM_THREADS
    import cosmolike_lssty1xdesi_fisher as fm
    import compute_fisher_base_mpi_classes as step_classes

    spec = base.LAYOUTS[args.layout]
    name = CACHE_NAME.format(layout=args.layout, tier=base.DERIVATIVE_TIER)
    path = os.path.join(out_dir, name + ".npz")
    settings = base.layout_settings(fm=fm, layout=args.layout, yaml_label=yaml_label)
    # the sha256 of the resolved paper yaml (options, parameter files,
    # priors, fiducial point, accuracy settings and dataset files): the
    # cache stores it, and the fits script refuses a cache whose
    # fingerprint is not the notebook's
    fingerprint = base.layout_fingerprint(fm=fm, yaml_path=yaml_path, settings=settings)
    # allgather sends this rank's value to every rank and returns the
    # list of every rank's value, ordered by rank
    fingerprints = comm.allgather(fingerprint)
    if rank == 0:
        fm.check_halofit_patch()
        # ranks on other nodes must see the same yaml, parameter and
        # dataset files, or their vectors would mix two models
        mismatch = False
        for other_fingerprint in fingerprints:
            if other_fingerprint != fingerprint:
                mismatch = True
        if mismatch:
            raise RuntimeError("the ranks computed different fingerprints: they see "
                               "different yaml, parameter or dataset files")
        if os.path.exists(path) and not args.overwrite:
            raise FileExistsError(f"{path} exists; nothing was computed. Pass "
                                  "--overwrite to replace it")
        os.makedirs(out_dir, exist_ok=True)
        print(f"layout {args.layout} ({spec['label']}), {size} ranks x "
              f"{args.threads_per_rank} threads, steps {args.steps}; fingerprint "
              f"{fingerprint}", flush=True)
    # every rank waits here, so none builds a model when rank 0's checks
    # stop the run
    comm.Barrier()

    start = time.time()
    data_dir = os.path.join(rootdir, base.PROJECT, "data")
    tables = base.layout_tables(data_dir=data_dir, stem=spec["dataset"])
    color = base.pair_coloring(nbins=tables["nbins"], pairs=tables["pairs"])
    model, like, build_seconds = base.build_step_model(
        fm=fm, step_classes=step_classes, yaml_path=yaml_path, settings=settings,
        check_input=(rank == 0))
    tasks = task_list(steps=args.steps, ncolors=max(color) + 1)
    owners = base.task_owners(n_tasks=len(tasks), n_ranks=size)
    if rank == 0:
        print(f"{tables['nbins']} bins, {len(tables['pairs'])} cross pairs, "
              f"{max(color) + 1} colors, {len(tasks)} evaluations; model built in "
              f"{build_seconds:.0f} s", flush=True)

    results = {}
    extras = None
    # this rank's share of the evaluations, in list order
    mine = []
    for k in range(len(tasks)):
        if owners[k] == rank:
            mine.append(k)
    evaluation_start = time.time()
    # enumerate yields (position, element): done counts this rank's tasks
    for done, k in enumerate(mine):
        t0 = time.time()
        vector = evaluate_task(fm=fm, task=tasks[k], model=model, like=like,
                               color=color, steps=args.steps)
        results[k] = vector
        if tasks[k][0] == "fiducial":
            # the fiducial point is task 0, always on rank 0 (the task
            # owner rule), so the extras exist where the cache is built
            if len(vector) != tables["n"]:
                raise ValueError(f"{spec['label']}: the vector has {len(vector)} "
                                 f"entries, the dataset's mask lists {tables['n']}")
            # passed_sigv returns the per-bin DESI_SIGV_<i> [km/s] of the
            # point just evaluated, before any step; desi_zbar the bin
            # mean redshifts the likelihood read from the bin table
            extras = {"sigma_v": like.passed_sigv(),
                      "zbar": np.array(like.desi_zbar, dtype=float)}
        print(f"[rank {rank}] {done + 1}/{len(mine)} {tasks[k]}: "
              f"{time.time() - t0:.1f} s", flush=True)
    timing = {"evaluations": len(mine),
              "seconds": time.time() - evaluation_start,
              "peak_rss_mb": base.peak_rss_mb()}
    # gather collects one entry per rank into a list on rank 0 (ordered
    # by rank); every other rank receives None
    gathered = comm.gather(results, root=0)
    timings = comm.gather(timing, root=0)
    if rank != 0:
        return

    # one slot per evaluation, filled from every rank's share
    vectors = []
    for _ in range(len(tasks)):
        vectors.append(None)
    for part in gathered:
        for k, vector in part.items():
            vectors[k] = vector
    missing = []
    for k in range(len(tasks)):
        if vectors[k] is None:
            missing.append(k)
    if missing:
        raise RuntimeError(f"no rank returned the evaluations {missing}")
    sigma_v = extras["sigma_v"]
    if not np.all(sigma_v > 0.0):
        raise ValueError(f"the fiducial per-bin sigma_v must be positive; got {sigma_v}")
    columns = assemble(fm=fm, tables=tables, color=color, steps=args.steps,
                       tasks=tasks, vectors=vectors, sigma_v=sigma_v)
    # the reset check: the fiducial point evaluated again after every
    # step (the last task, on rank N - 1) must reproduce the first one,
    # or a step leaked into the model's state
    m0 = vectors[0]
    m_after = vectors[-1]
    reset = float(np.max(np.abs(m_after - m0))/np.max(np.abs(m0)))
    # the files must not have changed while the ranks evaluated
    if base.layout_fingerprint(fm=fm, yaml_path=yaml_path, settings=settings) != fingerprint:
        raise RuntimeError("the fingerprint changed during the run (a yaml, parameter "
                           "or dataset file was edited); nothing was written")
    # the stored settings: the derivative base's, plus this run's steps
    # (fully resolved, so a reader never substitutes a default)
    stored = dict(settings)
    stored["sigv_steps"] = [float(s) for s in args.steps]
    stored["sigv_step_rule"] = "relative five-point stencil, batched by the greedy coloring"
    stored["fingerprint"] = fingerprint
    result = {
        "m0": m0,
        "sigma_v": sigma_v,
        "zbar": extras["zbar"],
        "color": np.array(color, dtype=int),
        "D_sigv": columns,
        "reset": reset,
        "seconds": time.time() - start,
    }
    fm.write_fisher_cache(result, name, stored, overwrite=args.overwrite,
                          folder=out_dir)
    print(f"reset check: max |m_after - m0| / max |m0| = {reset:.2e}", flush=True)
    for other in range(size):
        t = timings[other]
        print(f"rank {other}: {t['evaluations']} evaluations in {t['seconds']:.0f} s, "
              f"peak RSS {t['peak_rss_mb']:.0f} MB", flush=True)
    print(f"wall time {time.time() - process_start:.0f} s; wrote {path}", flush=True)


def main():
    """Parse the command line, set the threads, and run on MPI.COMM_WORLD.

    OMP_NUM_THREADS is set before anything imports the compiled
    interface: the OpenMP runtime and the likelihood base read it once,
    when they load.

    Raises:
      ValueError for a thread count below 1 or a step that is not
      positive; RuntimeError when ROOTDIR is unset (start_cocoa.sh sets
      it).
    """
    process_start = time.time()
    args = parse_arguments()
    if args.threads_per_rank < 1:
        raise ValueError(f"--threads-per-rank must be at least 1; got "
                         f"{args.threads_per_rank}")
    # "not step > 0.0" also catches a NaN, which no comparison can pass
    bad_step = False
    for step in args.steps:
        if not step > 0.0:
            bad_step = True
    if bad_step:
        raise ValueError(f"every step must be positive; got {args.steps}")
    if not os.environ.get("ROOTDIR"):
        raise RuntimeError("ROOTDIR is not set: source start_cocoa.sh from the Cocoa/ "
                           "folder first")
    os.environ["OMP_NUM_THREADS"] = str(args.threads_per_rank)
    run(args=args, comm=MPI.COMM_WORLD, process_start=process_start)


if __name__ == "__main__":
    # a failure on one rank would leave the others waiting forever; with
    # several ranks the failing one prints its error and stops every rank
    try:
        main()
    except Exception:
        world = MPI.COMM_WORLD
        if world.Get_size() == 1:
            raise
        sys.stderr.write(f"[rank {world.Get_rank()}] failed:\n{traceback.format_exc()}")
        sys.stderr.flush()
        world.Abort(1)
