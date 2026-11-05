"""Generate the per-bin DESI nuisance-parameter yaml fragments.

The likelihood declares one bias amplitude and one velocity dispersion
per thin bin (DESI_B1_<n>, DESI_SIGV_<n>, n = 1..76 in the bin-table
order), and knows nothing about parameter groupings: a GROUP is a yaml
statement. The first bin of a group is its leader and carries the
prior, reference and proposal; every other member is tied to the
leader with a cobaya value function,

    DESI_B1_2:
      value: 'lambda DESI_B1_1: DESI_B1_1'

so the sampler sees exactly one degree of freedom per group, tied
members never appear in chain output, and changing the grouping (or
freeing every bin) is an edit to this configuration, never to code.

Two fragments are written into likelihood/:

  params_desi.yaml              = the project default: bias grouped
                                  by the Table-2 chunks (BGS, LRG1-3,
                                  ELG1-2) and velocity dispersions by
                                  the same chunks with BGS split in
                                  two (SIGV_BGS_SPLIT_Z), at the
                                  forecast's simulation-fit fiducials:
                                  the values the committed fiducial
                                  data vector and the covariance
                                  kernels assume.
  params_desi_dr3_forecast.yaml = the DR3 forecast model: the same
                                  seven-group velocity dispersions
                                  over the forecast pipeline's own
                                  bias fiducials.

Groups are read off the dataset's bin table, never hardcoded: a bin
belongs to the group of its (sample, chunk) row.

Usage (from the Cocoa/ folder):
    python ./projects/lssty1xdesi/scripts/make_desi_params.py

With --case <name> (a fixed-width Delta z case of
scripts/make_desi_dz_case.py, e.g. dr1_dz0p02) it writes instead the
single fragment likelihood/params_desi_<name>.yaml - the project
default model on that case's bin table.
"""
import argparse
import os

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_TABLE = os.path.join(PROJECT, "data", "lssty1xdesi_desi_dr3_bins.txt")
OUT_DIR = os.path.join(PROJECT, "likelihood")

# the two generated models: for each, the bias grouping key and the
# fiducial value per group (velocity dispersions always use the
# seven sigv_group_key groups)
MODELS = {
    "params_desi.yaml": {
        "comment": ("the project default: Table-2 bias chunks and the "
                    "seven-group velocity dispersions (BGS split at "
                    "SIGV_BGS_SPLIT_Z), at the forecast's "
                    "simulation-fit fiducials - keep the dispersions "
                    "in sync with the covariance adapter's "
                    "CHUNK_SIGMA_V"),
        "bias_groups": "chunk",
        "bias_values": {"BGS": 1.34, "LRG1": 1.70, "LRG2": 1.70,
                        "LRG3": 1.70, "ELG1": 0.84, "ELG2": 0.84},
        "sigv_values": {"BGS0": 371.0, "BGS1": 358.0, "LRG1": 357.0,
                        "LRG2": 348.0, "LRG3": 343.0, "ELG1": 304.0,
                        "ELG2": 289.0},
    },
    "params_desi_dr3_forecast.yaml": {
        "comment": ("the DR3 forecast model: the same seven-group "
                    "velocity dispersions over the forecast pipeline's "
                    "own bias fiducials"),
        "bias_groups": "chunk",
        "bias_values": {"BGS": 1.375, "LRG1": 1.448, "LRG2": 1.403,
                        "LRG3": 1.388, "ELG1": 0.704, "ELG2": 0.704},
        "sigv_values": {"BGS0": 371.0, "BGS1": 358.0, "LRG1": 357.0,
                        "LRG2": 348.0, "LRG3": 343.0, "ELG1": 304.0,
                        "ELG2": 289.0},
    },
}


# The measured sigma_v(zbar) of the forecast's simulations steps
# inside BGS: the first ten bins arc around 371 km/s and the bins
# above this redshift sit on a flat 358 km/s plateau (the step read
# off the measured figure, between the bin means 0.247 and 0.263).
# The velocity dispersions therefore carry one more group than the
# bias. Keep the groups and values in sync with the covariance
# adapter's CHUNK_SIGMA_V.
SIGV_BGS_SPLIT_Z = 0.255


def sigv_group_key(sample, chunk, zbar):
    """The velocity-dispersion group of one bin.

    Arguments:
      sample = "bgs", "lrg" or "elg".
      chunk  = the bin's bias chunk name (BGS, LRG1..ELG2).
      zbar   = the bin's mean redshift.

    Returns:
      the group name: the bias chunk, except BGS, which splits at
      SIGV_BGS_SPLIT_Z into BGS0 and BGS1.
    """
    if sample == "bgs":
        return "BGS0" if zbar < SIGV_BGS_SPLIT_Z else "BGS1"
    return chunk


B1_PRIOR = (0.8, 4.0)       # the Table-2 bias range
B1_SCALE = 0.02             # reference scatter and proposal width
SIGV_SCALE = 5.0

# Gaussian prior widths [km/s] per velocity-dispersion group: 5x the
# median per-bin error bar of the simulations' measured sigma_v(zbar)
# figure (digitized, with the y scale calibrated on the seven group
# means; residuals < 5 km/s). A flat prior would let the sampler
# wander to dispersions the simulations exclude; the Gaussian keeps
# the chains inside the measured band. The prior is truncated at zero
# only to remove the unphysical negative tail (the BGS0 fiducial sits
# 2.6 prior sigma above zero).
SIGV_PRIOR_SIGMA = {
    "BGS0": 145.0,
    "BGS1": 105.0,
    "LRG1": 70.0,
    "LRG2": 50.0,
    "LRG3": 50.0,
    "ELG1": 40.0,
    "ELG2": 30.0,
}


def group_key(sample, chunk, mode):
    """The group a bin belongs to, under one grouping mode."""
    if mode == "chunk":
        return chunk
    return sample


def emit_blocks(out, prefix, groups, values, prior, scale, latex_of):
    """Write the parameter blocks of one quantity.

    Arguments:
      out      = the open output file.
      prefix   = "DESI_B1" or "DESI_SIGV".
      groups   = per-bin group key, in bin order (length = bin count).
      values   = group key -> fiducial value.
      prior    = (min, max) of a flat prior shared by every leader,
                 or {group: sigma} of a Gaussian prior centered on the
                 group's fiducial (truncated at zero: scipy truncnorm
                 reads its bounds in units of scale from loc).
      scale    = reference scatter and proposal width of every leader.
      latex_of = group key -> latex label of the leader.
    """
    leader_of = {}
    for n, g in enumerate(groups, start=1):
        name = f"{prefix}_{n}"
        if g not in leader_of:
            leader_of[g] = name
            out.write(f"  {name}: # leader of group {g}\n")
            out.write("    prior:\n")
            if isinstance(prior, dict):
                out.write("      dist: truncnorm\n")
                out.write(f"      a: {-values[g]/prior[g]:.6f}\n")
                out.write("      b: .inf\n")
                out.write(f"      loc: {values[g]}\n")
                out.write(f"      scale: {prior[g]}\n")
            else:
                out.write(f"      min: {prior[0]}\n")
                out.write(f"      max: {prior[1]}\n")
            out.write("    ref:\n")
            out.write("      dist: norm\n")
            out.write(f"      loc: {values[g]}\n")
            out.write(f"      scale: {scale}\n")
            out.write(f"    proposal: {scale}\n")
            out.write(f"    latex: {latex_of[g]}\n")
        else:
            leader = leader_of[g]
            out.write(f"  {name}: # tied to {leader} (group {g})\n")
            out.write(f"    value: 'lambda {leader}: {leader}'\n")


def main():
    parser = argparse.ArgumentParser(
        description="Write the per-bin DESI nuisance-parameter fragments.")
    parser.add_argument(
        "--case", default=None,
        help="a Delta z case (e.g. dr1_dz0p02): write only its "
             "params_desi_<case>.yaml from its bin table")
    args = parser.parse_args()
    bin_table = BIN_TABLE
    models = MODELS
    if args.case is not None:
        bin_table = os.path.join(
            PROJECT, "data", f"lssty1xdesi_desi_{args.case}_bins.txt")
        models = {f"params_desi_{args.case}.yaml":
                  MODELS["params_desi.yaml"]}

    table = np.loadtxt(bin_table, usecols=(1, 3), dtype=str, ndmin=2)
    samples = [s for s, c in table]
    chunks = [c for s, c in table]
    zbars = np.loadtxt(bin_table, usecols=4, ndmin=1)

    for fname, model in models.items():
        bias_groups = [group_key(s, c, model["bias_groups"])
                       for s, c in zip(samples, chunks)]
        sigv_groups = [sigv_group_key(s, c, z)
                       for s, c, z in zip(samples, chunks, zbars)]
        bias_latex = {g: "b_\\mathrm{" + g + "}"
                      for g in dict.fromkeys(bias_groups)}
        sigv_latex = {g: "\\sigma_v^\\mathrm{" + g.upper() + "}"
                      for g in dict.fromkeys(sigv_groups)}

        path = os.path.join(OUT_DIR, fname)
        with open(path, "w") as out:
            out.write(f"# Generated by scripts/make_desi_params.py: "
                      f"{model['comment']}.\n")
            out.write("# One bias amplitude (b1 at z = 0 under passive\n")
            out.write("# evolution, b1(z) = amplitude / D(z)) and one\n")
            out.write("# velocity dispersion [km/s] per thin bin, in the\n")
            out.write("# bin-table order; groups are pure configuration\n")
            out.write("# (leaders sampled, members tied by value).\n")
            emit_blocks(out, "DESI_B1", bias_groups,
                        model["bias_values"], B1_PRIOR, B1_SCALE,
                        bias_latex)
            emit_blocks(out, "DESI_SIGV", sigv_groups,
                        model["sigv_values"], SIGV_PRIOR_SIGMA,
                        SIGV_SCALE, sigv_latex)
        nlead = len(set(bias_groups)) + len(set(sigv_groups))
        print(f"wrote {path}: {2*len(samples)} parameters, "
              f"{nlead} sampled leaders")
    return 0


if __name__ == "__main__":
    main()
