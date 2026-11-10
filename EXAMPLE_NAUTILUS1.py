"""Runs Nautilus nested sampling on the paper's model, with CAMB.

Nested sampling estimates the Bayesian evidence and weighted posterior
samples: a set of live points shrinks from the full prior toward high
likelihood, and neural networks learn the likelihood boundary to propose
new points (the Nautilus package). This file samples the model of
EXAMPLE_EVALUATE1.yaml - the paper's model, model B + tilts with alpha_2 = 0
on the pessimistic linear-bias prior (likelihood
lssty1xdesi.desi_cl_model_b), on the Delta d_com = 62 Mpc main dataset,
with its CAMB theory block - whose sampler.evaluate.override block is the
fiducial point.

The structure and the options follow the hybrid-emulator driver of the
other projects (cosmolike_core/cocoa_hybrid_sampling.py, its nautilus
mode), which accepts only emulator likelihoods (use_emulator: 2); this
file holds the same steps for a CAMB model. Every MPI rank builds its own
cobaya model, and only parameter arrays and scores cross the worker pool.
Nautilus samples cobaya's independent one-parameter priors directly; the
joint prior factor (the amplitude cut) enters the likelihood once.

Options: --input (evaluate yaml, default EXAMPLE_EVALUATE1.yaml beside this
file), --root (folder whose chains/ subfolder receives the outputs, default
this project), --outroot (default EXAMPLE_NAUTILUS1), --seed (default 42),
--check, and the Nautilus settings --nlive (1000), --maxfeval (100000),
--neff (10000), --flive (0.01) and --nnetworks (4), the defaults of the
hybrid driver. The model samples 151 parameters: 1000 live points are about
seven per dimension, and a run that reaches --maxfeval stops without
convergence, which the JSON record states.

From the Cocoa/ folder, with Cocoa activated (start_cocoa.sh), check the
setup without sampling - the model builds, the fiducial point is evaluated
and the Nautilus prior and sampler are built:

    python ./projects/lssty1xdesi/EXAMPLE_NAUTILUS1.py --check

A run writes projects/lssty1xdesi/chains/<outroot>.json (the run record),
<outroot>_checkpoint.hdf5 and, when posterior samples exist,
<outroot>.1.txt and <outroot>.paramnames; an existing result is never
overwritten. The project README gives the MPI commands.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path

# Set before numpy, CAMB or cobaya load, on every MPI rank: one BLAS thread
# per rank leaves the cores to the ranks and cosmolike's OpenMP threads, and
# COBAYA_NOMPI makes each rank's cobaya act alone, since every rank owns its
# own model.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"

import numpy as np

PROJECT = Path(__file__).resolve().parent

# The model of this process. Each MPI rank sets it before it serves
# likelihood calls; nested_likelihood reads it, because the worker pool
# sends only parameter arrays.
_MODEL = None


def load_model(filename):
    """Build the cobaya model of an evaluate yaml, without its outputs.

    Arguments:
      filename = Path of the evaluate yaml.

    Returns:
      (cobaya model, sampled parameter names in sampled order, fiducial
      point [n_sampled] from sampler.evaluate.override).

    Raises:
      ValueError when the override block misses a sampled parameter.
    """
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file

    info = yaml_load_file(str(filename))
    point = info.get("sampler", {}).get("evaluate", {}).get("override", {})
    info.pop("sampler", None)
    info.pop("output", None)
    for options in info["likelihood"].values():
        options["print_datavector"] = False
    model = get_model(info)
    names = list(model.parameterization.sampled_params())
    missing = [name for name in names if name not in point]
    if missing:
        raise ValueError(f"{filename}: sampler.evaluate.override must set {missing}")
    start = np.array([point[name] for name in names], dtype=float)
    return model, names, start


def evaluate(model, values):
    """The log prior and log likelihood of one point in sampled order.

    Raises:
      ValueError for a non-finite point or an invalid score.
    """
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("sampled parameters must be finite")
    # zip pairs each sampled name with its value; strict=True refuses a
    # point of the wrong length
    point = dict(zip(model.parameterization.sampled_params(), values, strict=True))
    prior = float(model.logprior(params_values=point))
    if prior == -np.inf:
        return prior, -np.inf
    likelihood = float(model.loglike(params_values=point, cached=False,
                                     return_derived=False))
    if np.isnan(likelihood) or likelihood == np.inf:
        raise ValueError("the model returned an invalid log likelihood")
    return prior, likelihood


def nested_likelihood(values):
    """The log likelihood Nautilus sees: the likelihood plus the joint prior.

    Nautilus samples the independent one-parameter priors itself, so the
    score is cobaya's log likelihood plus its full log prior minus that
    independent part, which leaves the joint factor (the amplitude cut).
    """
    prior, likelihood = evaluate(model=_MODEL, values=values)
    if prior == -np.inf:
        return -np.inf
    independent = float(_MODEL.prior.logps_internal(np.asarray(values)))
    return likelihood + prior - independent


def nautilus_sampler(model, names, pool, args, filepath):
    """The Nautilus prior and sampler over the model's independent priors.

    Arguments:
      model    = cobaya model; model.prior.pdf holds the one-parameter priors.
      names    = sampled parameter names in sampled order.
      pool     = schwimmbad MPIPool on the master rank, or None.
      args     = parsed options (nlive, nnetworks, seed).
      filepath = checkpoint file, or None to write none.

    Returns:
      the nautilus Sampler, not yet run.
    """
    from nautilus import Prior, Sampler

    prior = Prior()
    for name, distribution in zip(names, model.prior.pdf, strict=True):
        prior.add_parameter(key=name, dist=distribution)
    # pool=(pool, None): the MPI workers evaluate likelihoods; Nautilus's own
    # work, such as network training, stays on this rank
    return Sampler(prior=prior, likelihood=nested_likelihood, pass_dict=False,
                   pool=(pool, None), n_live=args.nlive, n_networks=args.nnetworks,
                   filepath=filepath, resume=False, seed=args.seed)


def run_nautilus(model, pool, args, prefix, names):
    """Sample, and save weighted GetDist rows.

    Returns:
      dict for the JSON run record: converged, log_evidence, evaluations,
      posterior_samples and effective_samples.

    Side effects: writes <prefix>_checkpoint.hdf5 and, when posterior samples
    exist, <prefix>.1.txt (weight, -log likelihood including the joint prior
    factor, then the sampled parameters) and <prefix>.paramnames.
    """
    sampler = nautilus_sampler(model=model, names=names, pool=pool, args=args,
                               filepath=str(prefix) + "_checkpoint.hdf5")
    converged = sampler.run(f_live=args.flive, n_eff=args.neff,
                            n_like_max=args.maxfeval, verbose=True,
                            discard_exploration=True)
    if sampler.n_eff == 0:
        print("Budget exhausted before retaining posterior samples; the "
              "checkpoint is saved, but there is no posterior chain.", flush=True)
        return {"converged": False, "log_evidence": None,
                "evaluations": int(sampler.n_like), "posterior_samples": 0,
                "effective_samples": 0.0}
    points, log_weight, log_like = sampler.posterior()
    np.savetxt(str(prefix) + ".1.txt",
               np.column_stack((np.exp(log_weight), -log_like, points)),
               header="weight neg_log_likelihood " + " ".join(names))
    params = model.info()["params"]
    with open(str(prefix) + ".paramnames", "w") as output:
        for name in names:
            label = params[name].get("latex", name)
            output.write(f"{name} {label}\n")
    return {"converged": bool(converged), "log_evidence": float(sampler.log_z),
            "evaluations": int(sampler.n_like), "posterior_samples": len(points),
            "effective_samples": float(sampler.n_eff)}


def main():
    """Parse the options, build one model per rank, check or sample."""
    global _MODEL
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=PROJECT/"EXAMPLE_EVALUATE1.yaml")
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--outroot", default="EXAMPLE_NAUTILUS1")
    parser.add_argument("--check", action="store_true",
                        help="evaluate the fiducial and build the sampler; no sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nlive", type=int, default=1000)
    parser.add_argument("--maxfeval", type=int, default=100000)
    parser.add_argument("--neff", type=int, default=10000)
    parser.add_argument("--flive", type=float, default=0.01)
    parser.add_argument("--nnetworks", type=int, default=4)
    args = parser.parse_args()
    for key in ("nlive", "maxfeval", "neff", "nnetworks"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    if not 0 < args.flive < 1:
        parser.error("--flive must lie between zero and one")
    np.random.seed(args.seed)
    _MODEL, names, start = load_model(filename=args.input)
    # Open MPI exports OMPI_COMM_WORLD_SIZE to every rank; a plain python run
    # lacks it and stays serial (nullcontext then yields pool = None)
    ranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    context = nullcontext(None)
    if ranks > 1:
        from schwimmbad import MPIPool
        context = MPIPool()
    with context as pool:
        # worker ranks serve evaluations here until the master closes the pool
        if pool is not None and not pool.is_master():
            pool.wait()
            return
        prior, likelihood = evaluate(model=_MODEL, values=start)
        if not np.isfinite(prior + likelihood):
            raise ValueError("the yaml fiducial has a non-finite prior or likelihood")
        print(f"{len(names)} sampled parameters:", dict(enumerate(names)), flush=True)
        print(f"Fiducial log prior={prior:.10g}, log likelihood={likelihood:.10g}, "
              f"chi2={-2.0*likelihood:.6g}", flush=True)
        if args.check:
            sampler = nautilus_sampler(model=_MODEL, names=names, pool=None,
                                       args=args, filepath=None)
            print(f"Nautilus sampler built: {sampler.n_dim} dimensions, "
                  f"{sampler.n_live} live points, {args.nnetworks} networks; "
                  "no sampling (--check)", flush=True)
            _MODEL.close()
            return
        prefix = args.root/"chains"/args.outroot
        if (Path(str(prefix) + ".json").exists()
                or Path(str(prefix) + "_checkpoint.hdf5").exists()):
            raise FileExistsError(f"{prefix} already exists; select a new --outroot")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        record = {"input": str(args.input.resolve()), "parameters": names,
                  "mode": "nautilus", "seed": args.seed,
                  "omp_threads": os.environ.get("OMP_NUM_THREADS"), "mpi_ranks": ranks}
        record.update(run_nautilus(model=_MODEL, pool=pool, args=args,
                                   prefix=prefix, names=names))
        Path(str(prefix) + ".json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Wrote {prefix}.json", flush=True)
        _MODEL.close()


# __name__ is "__main__" only when this file runs as a script
if __name__ == "__main__":
    main()
