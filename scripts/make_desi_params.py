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

With --model-a (alone or with --case) it writes the model A fragments
of the higher-order bias note instead: params_desi[_<case>]_model_a.yaml
(one offset per chunk for b2, bs2, alpha_0 and SN0, the tilts fixed at
zero) and params_desi[_<case>]_model_a_pessimistic.yaml (the b2, bs2 and
alpha_0 tilts free). The desi_cl*_model_a likelihoods read them next to
the linear fragment.
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


# Model A (notes/higher_order_bias, its recommendation section): the
# analysis cut, and the Gaussian prior widths of the per-chunk offsets,
# all centered on zero. alpha_0 follows the DESI rule rescaled to this
# cut - a 50% correction to the linear spectrum at k_max - and SN0 is in
# units of the bin's Poisson power (twice the Poisson power).
MODEL_A_KMAX = 0.075  # h/Mpc
MODEL_A_FAMILIES = (
    # (per-bin prefix, tilt prefix or None, Gaussian width, latex symbol)
    ("DESI_DB2", "DESI_TB2", 1.0, "\\Delta_{2}"),
    ("DESI_DBS", "DESI_TBS", 0.5, "\\Delta_{s}"),
    ("DESI_A0", "DESI_TA0", 0.5/MODEL_A_KMAX**2, "\\alpha_{0}"),
    ("DESI_SN0", None, 2.0, "s_{0}"),
)
# the flat range of a free tilt (pessimistic variant), per unit redshift
MODEL_A_TILT_RANGE = {"DESI_TB2": 50.0, "DESI_TBS": 50.0,
                      "DESI_TA0": 5000.0}


def emit_model_a(out, chunks, zbars, pessimistic):
    """Write the model A parameters of one bin table.

    Each chunk g has one sampled offset per family (DESI_DB2_<g>,
    DESI_DBS_<g>, DESI_A0_<g>, DESI_SN0_<g>) and, for the first three, a
    tilt in redshift (DESI_TB2_<g>, ...): fixed at zero in the baseline
    variant, free in the pessimistic one. Every bin n of the chunk takes
    its per-bin value from them,

        DESI_DB2_<n> = DESI_DB2_<g> + DESI_TB2_<g> (z_n - zbar_g),

    with z_n the bin's mean redshift and zbar_g the mean of the chunk's
    bin redshifts; the stochastic amplitudes never tilt (the Poisson
    part is known per bin and sits in the covariance).

    Arguments:
      out         = the open output file.
      chunks      = per-bin chunk name, in bin order.
      zbars       = per-bin mean redshift, in bin order.
      pessimistic = True frees the tilts.
    """
    groups = list(dict.fromkeys(chunks))
    zbar_of = {g: float(np.mean([z for c, z in zip(chunks, zbars)
                                 if c == g])) for g in groups}
    for prefix, tilt, width, symbol in MODEL_A_FAMILIES:
        for g in groups:
            out.write(f"  {prefix}_{g}: # offset of chunk {g}\n")
            out.write("    prior:\n")
            out.write("      dist: norm\n")
            out.write("      loc: 0.0\n")
            out.write(f"      scale: {width:.6g}\n")
            out.write("    ref:\n")
            out.write("      dist: norm\n")
            out.write("      loc: 0.0\n")
            out.write(f"      scale: {0.1*width:.6g}\n")
            out.write(f"    proposal: {0.1*width:.6g}\n")
            out.write(f"    latex: {symbol}^{{\\mathrm{{{g}}}}}\n")
            if tilt is None:
                continue
            if pessimistic:
                span = MODEL_A_TILT_RANGE[tilt]
                out.write(f"  {tilt}_{g}: # tilt of chunk {g} [per unit z]\n")
                out.write("    prior:\n")
                out.write(f"      min: {-span:g}\n")
                out.write(f"      max: {span:g}\n")
                out.write("    ref:\n")
                out.write("      dist: norm\n")
                out.write("      loc: 0.0\n")
                out.write(f"      scale: {0.1*width:.6g}\n")
                out.write(f"    proposal: {0.1*width:.6g}\n")
                out.write(f"    latex: t_{{{symbol}}}^{{\\mathrm{{{g}}}}}\n")
            else:
                out.write(f"  {tilt}_{g}: # tilt of chunk {g}, fixed "
                          "(baseline)\n")
                out.write("    value: 0.0\n")
        for n, (g, z) in enumerate(zip(chunks, zbars), start=1):
            leader = f"{prefix}_{g}"
            if tilt is None:
                out.write(f"  {prefix}_{n}: # chunk {g}\n")
                out.write(f"    value: 'lambda {leader}: {leader}'\n")
            else:
                t = f"{tilt}_{g}"
                out.write(f"  {prefix}_{n}: # chunk {g}\n")
                out.write(f"    value: 'lambda {leader}, {t}: {leader} + "
                          f"{t}*({float(z) - zbar_of[g]:+.8f})'\n")


def main():
    parser = argparse.ArgumentParser(
        description="Write the per-bin DESI nuisance-parameter fragments.")
    parser.add_argument(
        "--case", default=None,
        help="a Delta z case (e.g. dr1_dz0p02): write only its "
             "params_desi_<case>.yaml from its bin table")
    parser.add_argument(
        "--model-a", action="store_true",
        help="write the model A fragments instead: "
             "params_desi[_<case>]_model_a.yaml (offsets per chunk, "
             "tilts fixed) and ..._model_a_pessimistic.yaml (tilts free)")
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

    if args.model_a:
        stem = ("params_desi" if args.case is None
                else f"params_desi_{args.case}")
        for suffix, pessimistic in (("_model_a", False),
                                    ("_model_a_pessimistic", True)):
            path = os.path.join(OUT_DIR, f"{stem}{suffix}.yaml")
            with open(path, "w") as out:
                out.write("# Generated by scripts/make_desi_params.py "
                          "--model-a: the model A parameters of "
                          "notes/higher_order_bias")
                out.write(" (pessimistic variant: free tilts).\n"
                          if pessimistic else
                          " (baseline: one offset per chunk).\n")
                out.write("# Units: DESI_A0 in h^-2 Mpc^2, DESI_SN0 in "
                          "units of each bin's Poisson power; the\n"
                          "# per-bin values are tied to the chunk "
                          "parameters (see emit_model_a).\n")
                emit_model_a(out, chunks, zbars, pessimistic)
            print(f"wrote {path}")
        return 0

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
            out.write("# One bias amplitude (b1 at z = 0 under a constant\n")
            out.write("# clustering amplitude, b1(z) = amplitude / D(z)) and one\n")
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
