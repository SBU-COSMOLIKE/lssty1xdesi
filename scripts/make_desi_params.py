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

  params_desi.yaml              = the project default: bias grouped by
                                  the Table-2 chunks (BGS, LRG1-3,
                                  ELG1-2), velocity dispersions grouped
                                  by sample, with the fiducial values
                                  the committed fiducial data vector
                                  assumes. Value-for-value identical to
                                  the pre-refactor suffix parameters.
  params_desi_dr3_forecast.yaml = the DR3 forecast model: bias AND
                                  velocity dispersions grouped by
                                  chunk, with the forecast pipeline's
                                  fiducial values. The forecast splits
                                  the BGS velocity dispersion into two
                                  groups whose bin boundary its
                                  configuration does not state; until
                                  that boundary is pinned down, BGS
                                  keeps one group at the first value.

Groups are read off the dataset's bin table, never hardcoded: a bin
belongs to the group of its (sample, chunk) row.

Usage (from the Cocoa/ folder):
    python ./projects/lssty1xdesi/scripts/make_desi_params.py
"""
import os

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_TABLE = os.path.join(PROJECT, "data", "lssty1xdesi_desi_dr3_bins.txt")
OUT_DIR = os.path.join(PROJECT, "likelihood")

# the two generated models: for each, the bias and velocity-dispersion
# grouping key ("chunk" or "sample") and the fiducial value per group
MODELS = {
    "params_desi.yaml": {
        "comment": ("the project default: Table-2 bias chunks, one "
                    "velocity dispersion per sample - value-for-value "
                    "the pre-refactor suffix parameters"),
        "bias_groups": "chunk",
        "bias_values": {"BGS": 1.34, "LRG1": 1.70, "LRG2": 1.70,
                        "LRG3": 1.70, "ELG1": 0.84, "ELG2": 0.84},
        "sigv_groups": "sample",
        "sigv_values": {"bgs": 357.0, "lrg": 326.0, "elg": 200.0},
    },
    "params_desi_dr3_forecast.yaml": {
        "comment": ("the DR3 forecast model: bias and velocity "
                    "dispersions grouped by chunk, forecast fiducials; "
                    "the forecast's two-group BGS velocity split awaits "
                    "its bin boundary and stays one group here"),
        "bias_groups": "chunk",
        "bias_values": {"BGS": 1.375, "LRG1": 1.448, "LRG2": 1.403,
                        "LRG3": 1.388, "ELG1": 0.704, "ELG2": 0.704},
        "sigv_groups": "chunk",
        "sigv_values": {"BGS": 371.0, "LRG1": 357.0, "LRG2": 348.0,
                        "LRG3": 343.0, "ELG1": 304.0, "ELG2": 289.0},
    },
}

B1_PRIOR = (0.8, 4.0)       # the Table-2 bias range
SIGV_PRIOR = (100.0, 1000.0)  # km/s, the forecast's range
B1_SCALE = 0.02             # reference scatter and proposal width
SIGV_SCALE = 5.0


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
      prior    = (min, max) of every leader's flat prior.
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
    table = np.loadtxt(BIN_TABLE, usecols=(1, 3), dtype=str)
    samples = [s for s, c in table]
    chunks = [c for s, c in table]

    for fname, model in MODELS.items():
        bias_groups = [group_key(s, c, model["bias_groups"])
                       for s, c in zip(samples, chunks)]
        sigv_groups = [group_key(s, c, model["sigv_groups"])
                       for s, c in zip(samples, chunks)]
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
                        model["sigv_values"], SIGV_PRIOR, SIGV_SCALE,
                        sigv_latex)
        nlead = len(set(bias_groups)) + len(set(sigv_groups))
        print(f"wrote {path}: {2*len(samples)} parameters, "
              f"{nlead} sampled leaders")
    return 0


if __name__ == "__main__":
    main()
