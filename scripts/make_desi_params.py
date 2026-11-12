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

Two fragments are written:

  likelihood/params_desi.yaml   = the project default, the growth-tied
                                  linear bias: bias grouped by the
                                  Table-2 chunks (BGS, LRG1-3, ELG1-2)
                                  and velocity dispersions by the same
                                  chunks with BGS split in two
                                  (SIGV_BGS_SPLIT_Z), at the forecast's
                                  simulation-fit fiducials: the values
                                  the committed fiducial data vector
                                  and the covariance kernels assume.
  likelihood/exploration/params_desi_dr3_forecast.yaml
                                = the DR3 forecast model: the same
                                  seven-group velocity dispersions
                                  over the forecast pipeline's own
                                  bias fiducials (no likelihood or
                                  example reads it).

Groups are read off the dataset's bin table, never hardcoded: a bin
belongs to the group of its (sample, chunk) row.

Usage (from the Cocoa/ folder):
    python ./projects/lssty1xdesi/scripts/make_desi_params.py

With --case <name> (a fixed-width Delta z case of
scripts/make_desi_dz_case.py, e.g. dr1_dz0p02) it writes instead the
single fragment likelihood/params_desi_<name>.yaml - the project
default model on that case's bin table.

With --bias-pessimistic (alone or with --case) it writes instead the
linear fragment of the paper model,
likelihood/params_desi[_<case>]_bias_pessimistic.yaml: the pessimistic
linear-bias prior of notes/correlated_bias_prior_v2 (a free amplitude
and tilt per group, and a correlated Gaussian residual per bin; see
emit_bias_pessimistic), with the velocity dispersions of
params_desi.yaml at PAPER_SIGV_WIDTH_FACTOR times their prior widths.
The desi_cl*_model_b likelihoods read it.

With --model-b (alone or with --case; the flag keeps its historical
name) it writes the higher-order fragment of the fiducial model of
notes/higher_order_bias: likelihood/params_desi[_<case>]_model_b_tilts.yaml
(one offset per chunk for b2, bs2, alpha_0, SN0 and SN2 at the DESI DR1
prior widths, the b2, bs2 and alpha_0 tilts free, and alpha_2 fixed to 0
in every bin; read by the desi_cl*_model_b likelihoods, next to the
pessimistic linear fragment).
"""
import argparse
import os

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_TABLE = os.path.join(PROJECT, "data", "lssty1xdesi_desi_dr3_bins.txt")
OUT_DIR = os.path.join(PROJECT, "likelihood")
# the fragments no likelihood of the paper model reads (the DR3 forecast
# fiducials) live in the subpackage likelihood/exploration/
EXPLORATION_DIR = os.path.join(OUT_DIR, "exploration")

# the two generated models: for each, the output folder, the bias
# grouping key and the fiducial value per group (velocity dispersions
# always use the seven sigv_group_key groups)
MODELS = {
    "params_desi.yaml": {
        "comment": ("the project default: Table-2 bias chunks and the "
                    "seven-group velocity dispersions (BGS split at "
                    "SIGV_BGS_SPLIT_Z), at the forecast's "
                    "simulation-fit fiducials - keep the dispersions "
                    "in sync with the covariance adapter's "
                    "CHUNK_SIGMA_V"),
        "folder": OUT_DIR,
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
        "folder": EXPLORATION_DIR,
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
# 2.6 prior sigma above zero). These are the widths of the exploration
# fragments (params_desi.yaml and its cases); the paper model widens
# them by PAPER_SIGV_WIDTH_FACTOR.
SIGV_PRIOR_SIGMA = {
    "BGS0": 145.0,
    "BGS1": 105.0,
    "LRG1": 70.0,
    "LRG2": 50.0,
    "LRG3": 50.0,
    "ELG1": 40.0,
    "ELG2": 30.0,
}

# The paper model has no radial counterterm alpha_2 (PAPER_FAMILIES) and
# multiplies every width of SIGV_PRIOR_SIGMA by this one factor, on every
# layout. On these thin-bin kernels alpha_2 is about 99% degenerate with
# the velocity dispersions sigma_v: both damp the line-of-sight modes
# (notes/fog_parameter_degeneracy_study). A fit without alpha_2 whose
# sigma_v priors are wide enough absorbs an alpha_2 of the DESI DR1 prior
# size (12.5 h^-2 Mpc^2 in every group) at about the same dark-energy
# figure of merit for k_max >= 0.1 h/Mpc (the z = 0 value of a per-bin
# cut). The factor covers the most demanding case of the sigma_v
# prior-width scan (section 11 of EXAMPLE_EVALUATE_FISHER_MODEL.ipynb), the
# Delta z = 0.02 layout at
# k_max = 0.1 h/Mpc, where the cosmological shifts of that test fall
# below a third of their error bars only from about 6x. At 6x the
# truncation at zero is no longer a
# formality: the fiducials sit 0.4 (BGS0) to 1.6 (ELG2) widened prior
# sigma above zero.
PAPER_SIGV_WIDTH_FACTOR = 6.0


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


# The pessimistic linear-bias prior of notes/correlated_bias_prior_v2
# (section "The linear-bias prior"). Bin i of group g has
#     b_{1,i} = B_g (1 + delta_i) / D(zbar_i),
#     delta_i = s_g (zbar_i - zbar_g) + eps_i,
# with B_g the group amplitude (b1 at z = 0 under a constant clustering
# amplitude b D), D the linear growth factor, s_g the tilt of the group's
# line per unit redshift, zbar_g the mean of the group's bin redshifts,
# and eps_i the residual of bin i around that line. The residuals are
# Gaussian, correlated inside a group and independent across groups,
#     <eps_i eps_j> = sigma_g^2 exp(-|zbar_i - zbar_j| / L),
# with sigma_g the width of one bin's residual and L the redshift
# separation over which two residuals forget each other. The pessimistic
# prior takes sigma_g = 3% in every group and L = 0.05: local selection
# features as large as the largest derived curvature, on the redshift
# scale over which the 4000 A break crosses a filter edge. The likelihood
# divides each DESI_B1_<n> by D(zbar_n), so the fragment sets
# DESI_B1_<n> = B_g (1 + delta_n).
PESSIMISTIC_SIGMA = 0.03
PESSIMISTIC_L = 0.05

# The flat prior of a tilt s_g lets the group's line move its outermost
# bin by at most this fraction of B_g, either way:
# |s_g| <= TILT_EDGE_FRACTION / max_i |zbar_i - zbar_g|. The largest
# derived drift, the BGS flux limit, changes the bias by 22% across its
# group (about 11% at either edge), less than a quarter of the range, so
# the bound does not bind. It also keeps every bias positive: at the
# bound the line leaves 1 + delta_i >= 0.5 - |eps_i|, and a 3% residual
# reaches 0.5 only at 17 standard deviations (the likelihood stops on a
# b1 <= 0).
TILT_EDGE_FRACTION = 0.5
# The reference scatter and proposal width of a tilt move the outermost
# bin by 1% of B_g, close to what B1_SCALE moves the LRG amplitude
# (0.02/1.7). A whitened residual has a unit-width prior; its reference
# scatter and proposal width are a tenth of that, like the higher-order
# offsets below.
TILT_EDGE_SCALE = 0.01
RESIDUAL_SCALE = 0.1


def residual_coefficients(z, sigma, length):
    """The matrix that turns a group's whitened residuals into its residuals.

    The residuals eps of one group have the covariance
    sigma^2 K, K_ij = exp(-|z_i - z_j| / length). With E the lower
    triangular Cholesky factor of that covariance (E E^T = sigma^2 K) and
    u a vector of independent unit Gaussians, eps = E u has the
    covariance E <u u^T> E^T = E E^T = sigma^2 K. Sampling u instead of eps
    turns a correlated Gaussian prior into independent unit Gaussians,
    which cobaya states one parameter at a time; the correlation moves
    into the fixed coefficients of E. Lower triangular means bin n of the
    group reads only the residuals of the group's first n bins.

    Arguments:
      z      = the mean redshifts of the group's bins, in bin order
               (numpy array, length >= 3 where it is used).
      sigma  = the width of one bin's residual (fractional, e.g. 0.03).
      length = the correlation length L in redshift.

    Returns:
      (n, n) float64 array E, lower triangular, E E^T = sigma^2 K.
    """
    # separation[i, j] = |z_i - z_j|: numpy broadcasting of a column
    # against a row gives every pair at once
    separation = np.abs(z[:, None] - z[None, :])
    covariance = sigma**2*np.exp(-separation/length)
    return np.linalg.cholesky(covariance)


def emit_bias_pessimistic(out, chunks, zbars, values):
    """Write the DESI_B1 blocks of the pessimistic linear-bias prior.

    Each group g gets a sampled amplitude DESI_B1_<g> = B_g (the flat
    B1_PRIOR box and the reference of the growth-tied fragment), a sampled
    tilt DESI_TB1_<g> = s_g with a flat prior (TILT_EDGE_FRACTION), and
    one sampled whitened residual DESI_UB1_<n> ~ N(0, 1) per bin. Every
    bin n of the group then takes the value

        DESI_B1_<n> = B_g (1 + s_g (z_n - zbar_g) + sum_m E_nm u_m),

    with E = residual_coefficients(z, PESSIMISTIC_SIGMA, PESSIMISTIC_L)
    and the sum over the group's bins m up to n. Each coefficient is
    written with 16 significant digits, so the prior the yaml states
    equals the kernel to double-precision round-off.

    Groups of one or two bins: there B_g alone (one bin) or B_g and s_g
    (two bins) already set every bin's bias freely, so a residual cannot
    be told apart from them and its parameters would only be exact
    degeneracies. Such a group carries no residuals, and a one-bin group
    no tilt (z_n - zbar_g = 0); its bins are free, as in the Fisher
    notebooks, whose projected prior precision is zero for a group of one
    or two bins. At the fiducial (s_g = 0, every u = 0) every bin takes
    exactly B_g, the growth-tied value.

    Arguments:
      out    = the open output file.
      chunks = per-bin group (chunk) name, in bin order.
      zbars  = per-bin mean redshift, in bin order.
      values = group name -> fiducial B_g (the reference center).
    """
    groups = list(dict.fromkeys(chunks))
    for g in groups:
        # the 0-based positions of the group's bins in the bin table
        members = []
        for i, chunk in enumerate(chunks):
            if chunk == g:
                members.append(i)
        z = np.array([float(zbars[i]) for i in members])
        offsets = z - np.mean(z)
        nbin = len(members)
        amplitude = f"DESI_B1_{g}"
        tilt = f"DESI_TB1_{g}"

        out.write(f"  {amplitude}: # amplitude B_g of group {g}\n")
        out.write("    prior:\n")
        out.write(f"      min: {B1_PRIOR[0]}\n")
        out.write(f"      max: {B1_PRIOR[1]}\n")
        out.write("    ref:\n")
        out.write("      dist: norm\n")
        out.write(f"      loc: {values[g]}\n")
        out.write(f"      scale: {B1_SCALE}\n")
        out.write(f"    proposal: {B1_SCALE}\n")
        out.write(f"    latex: B_\\mathrm{{{g}}}\n")

        if nbin >= 2:
            edge = float(np.max(np.abs(offsets)))
            span = TILT_EDGE_FRACTION/edge
            scale = TILT_EDGE_SCALE/edge
            out.write(f"  {tilt}: # tilt s_g of group {g} [per unit z]: "
                      f"+-{TILT_EDGE_FRACTION:g} of B_g at the outermost "
                      "bin\n")
            out.write("    prior:\n")
            out.write(f"      min: {-span:.4g}\n")
            out.write(f"      max: {span:.4g}\n")
            out.write("    ref:\n")
            out.write("      dist: norm\n")
            out.write("      loc: 0.0\n")
            out.write(f"      scale: {scale:.4g}\n")
            out.write(f"    proposal: {scale:.4g}\n")
            out.write(f"    latex: s_\\mathrm{{{g}}}\n")
        else:
            out.write(f"  # group {g} has one bin: no tilt and no residual "
                      "(both are degenerate with B_g)\n")

        residuals = []
        if nbin >= 3:
            for i in members:
                name = f"DESI_UB1_{i + 1}"
                residuals.append(name)
                out.write(f"  {name}: # whitened residual of bin {i + 1} "
                          f"(group {g})\n")
                out.write("    prior:\n")
                out.write("      dist: norm\n")
                out.write("      loc: 0.0\n")
                out.write("      scale: 1.0\n")
                out.write("    ref:\n")
                out.write("      dist: norm\n")
                out.write("      loc: 0.0\n")
                out.write(f"      scale: {RESIDUAL_SCALE}\n")
                out.write(f"    proposal: {RESIDUAL_SCALE}\n")
                out.write(f"    latex: u_{{{i + 1}}}\n")
            coefficients = residual_coefficients(
                z, sigma=PESSIMISTIC_SIGMA, length=PESSIMISTIC_L)
        elif nbin == 2:
            out.write(f"  # group {g} has two bins: no residuals (B_g and "
                      "the tilt already free both bins)\n")

        for position, i in enumerate(members):
            name = f"DESI_B1_{i + 1}"
            arguments = [amplitude]
            delta = ""
            if nbin >= 2:
                arguments.append(tilt)
                delta = f" + {tilt}*({offsets[position]:+.15e})"
            if nbin >= 3:
                # the lower-triangular row: bin `position` reads the
                # residuals of the group's bins 0..position
                terms = []
                for m in range(position + 1):
                    arguments.append(residuals[m])
                    terms.append(f"{coefficients[position, m]:+.15e}"
                                 f"*{residuals[m]}")
                delta = delta + " " + " ".join(terms)
            out.write(f"  {name}: # group {g}\n")
            if nbin == 1:
                out.write(f"    value: 'lambda {amplitude}: {amplitude}'\n")
            else:
                out.write(f"    value: 'lambda {', '.join(arguments)}: "
                          f"{amplitude}*(1.0{delta})'\n")


# The fiducial model (notes/higher_order_bias, its recommendation section;
# the MODEL_B names are historical): the shipped scale
# cut, and the Gaussian prior widths of the DESI DR1 full-shape analysis,
# all centered on zero. The shipped masks keep a band while
# l + 1/2 <= k chi(zbar), with chi the comoving distance to the bin's
# mean redshift zbar and k = 0.1185/Mpc at every redshift (no growth
# scaling). The rule is the same in every case (KMAX of
# make_desi_dz_case.py; the Delta d_com = 62 Mpc sacc mask matches it
# within one band), and at the fiducial h = 0.6732 it reads
# k_max = 0.176 h/Mpc. The two counterterms take the DESI rule at that
# cut, a 50% correction to the linear spectrum at k_max: a width of
# 0.5/k_max^2 = 16.1 h^-2 Mpc^2 (DESI DR1 quotes 12.5 at its
# k_max = 0.2 h/Mpc). SN0 is in units of the bin's Poisson power and SN2
# in units of f_sat sigma_v^2 times it (f_sat the satellite fraction,
# sigma_v the satellites' velocity dispersion in Mpc/h; SN2_UNIT of
# likelihood/desi_cl.py).
MODEL_B_KMAX = 0.1185/0.6732  # h/Mpc
MODEL_B_FAMILIES = (
    # (per-bin prefix, tilt prefix or None, Gaussian width, latex symbol)
    ("DESI_DB2", "DESI_TB2", 2.0, "\\Delta_{2}"),
    ("DESI_DBS", "DESI_TBS", 1.0, "\\Delta_{s}"),
    ("DESI_A0", "DESI_TA0", 0.5/MODEL_B_KMAX**2, "\\alpha_{0}"),
    ("DESI_A2", "DESI_TA2", 0.5/MODEL_B_KMAX**2, "\\alpha_{2}"),
    ("DESI_SN0", None, 2.0, "s_{0}"),
    ("DESI_SN2", None, 5.0, "s_{2}"),
)
# the flat range of a free tilt, per unit redshift: 50 prior widths for
# Delta_2, 100 for Delta_s and about 60 for the counterterms, rounded
MODEL_B_TILT_RANGE = {
    "DESI_TB2": 100.0,
    "DESI_TBS": 100.0,
    "DESI_TA0": 1000.0,
    "DESI_TA2": 1000.0,
}
# The paper model: model B + tilts without the radial counterterm.
# DESI_A2 is fixed to 0 in every bin (a width of None, no offset and no
# tilt), which the one-loop likelihood reads as the term switched off;
# sigma_v takes over the line-of-sight damping, on the priors widened by
# PAPER_SIGV_WIDTH_FACTOR.
PAPER_FAMILIES = tuple(
    (prefix, None, None, symbol) if prefix == "DESI_A2"
    else (prefix, tilt, width, symbol)
    for prefix, tilt, width, symbol in MODEL_B_FAMILIES)


def emit_offsets(out, chunks, zbars, tilts, families, tilt_range):
    """Write the higher-order bias parameters of one model and bin table.

    Each chunk g has one sampled offset per family (DESI_DB2_<g>,
    DESI_DBS_<g>, ...) and, for the families with a tilt prefix, a tilt
    in redshift (DESI_TB2_<g>, ...): fixed at zero when tilts is False,
    free when tilts is True. Every bin n of the chunk takes
    its per-bin value from them,

        DESI_DB2_<n> = DESI_DB2_<g> + DESI_TB2_<g> (z_n - zbar_g),

    with z_n the bin's mean redshift and zbar_g the mean of the chunk's
    bin redshifts; the stochastic amplitudes never tilt (the Poisson
    part is known per bin and sits in the covariance). A family with no
    width is not sampled: every bin's value is fixed to 0. The tilt of a
    one-bin chunk is fixed to 0 even with free tilts: its only bin has
    z_n - zbar_g = 0, so the tilt multiplies zero, and a sampled tilt
    would be a direction no data can constrain (on Delta z = 0.2, LRG1
    and LRG2 have one bin each).

    Arguments:
      out         = the open output file.
      chunks      = per-bin chunk name, in bin order.
      zbars       = per-bin mean redshift, in bin order.
      tilts       = True frees the tilts.
      families    = the model's (per-bin prefix, tilt prefix or None,
                    Gaussian width or None, latex symbol) rows, in output
                    order (MODEL_B_FAMILIES or PAPER_FAMILIES).
      tilt_range  = tilt prefix -> half width of the flat tilt prior,
                    per unit redshift (read only when tilts is True).
    """
    groups = list(dict.fromkeys(chunks))
    zbar_of = {g: float(np.mean([z for c, z in zip(chunks, zbars)
                                 if c == g])) for g in groups}
    # the number of bins of each chunk (list.count counts the bins whose
    # chunk name equals g)
    nbin_of = {}
    for g in groups:
        nbin_of[g] = list(chunks).count(g)
    for prefix, tilt, width, symbol in families:
        if width is None:
            # a family the model does not have: each bin's value is a
            # fixed 0, which cobaya passes to the likelihood unsampled
            for n, g in enumerate(chunks, start=1):
                out.write(f"  {prefix}_{n}: # chunk {g}, fixed (not in "
                          "this model)\n")
                out.write("    value: 0.0\n")
            continue
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
            if tilts and nbin_of[g] == 1:
                # the only bin has z_n - zbar_g = 0: the tilt multiplies
                # zero, so a sampled tilt would be a pure degeneracy
                out.write(f"  {tilt}_{g}: # tilt of chunk {g}, fixed (one "
                          "bin: z_n - zbar_g = 0)\n")
                out.write("    value: 0.0\n")
            elif tilts:
                span = tilt_range[tilt]
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
                          "(no tilts)\n")
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


def emit_model_b(out, chunks, zbars, tilts, families=MODEL_B_FAMILIES):
    """Write the model B parameters of one bin table.

    Model B has six families: DESI_DB2, DESI_DBS, DESI_A0 and DESI_A2,
    each with a tilt, and DESI_SN0 and DESI_SN2 (MODEL_B_FAMILIES). The
    paper model (PAPER_FAMILIES) is the same with DESI_A2 fixed to 0 in
    every bin. The offsets, tilts and per-bin ties follow emit_offsets.

    Arguments:
      out, chunks, zbars, tilts = as in emit_offsets.
      families = MODEL_B_FAMILIES, or PAPER_FAMILIES for the paper model.
    """
    emit_offsets(out, chunks, zbars, tilts,
                 families=families, tilt_range=MODEL_B_TILT_RANGE)


def main():
    parser = argparse.ArgumentParser(
        description="Write the per-bin DESI nuisance-parameter fragments.")
    parser.add_argument(
        "--case", default=None,
        help="a Delta z case (e.g. dr1_dz0p02): write only its "
             "params_desi_<case>.yaml from its bin table")
    # the pessimistic linear fragment and the two higher-order models
    # write different files: one run writes one of them (argparse refuses
    # two of these flags together)
    model_flags = parser.add_mutually_exclusive_group()
    model_flags.add_argument(
        "--bias-pessimistic", action="store_true",
        help="write the linear fragment of the paper model instead: "
             "params_desi[_<case>]_bias_pessimistic.yaml (the pessimistic "
             "linear-bias prior: amplitude and tilt per group, correlated "
             "residual per bin)")
    model_flags.add_argument(
        "--model-b", action="store_true",
        help="write the fiducial model's higher-order fragment instead: "
             "likelihood/params_desi[_<case>]_model_b_tilts.yaml (offsets "
             "and tilts per chunk, alpha_2 fixed to 0)")
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

    if args.bias_pessimistic:
        stem = ("params_desi" if args.case is None
                else f"params_desi_{args.case}")
        default = MODELS["params_desi.yaml"]
        sigv_groups = [sigv_group_key(s, c, z)
                       for s, c, z in zip(samples, chunks, zbars)]
        sigv_latex = {g: "\\sigma_v^\\mathrm{" + g.upper() + "}"
                      for g in dict.fromkeys(sigv_groups)}
        path = os.path.join(OUT_DIR, f"{stem}_bias_pessimistic.yaml")
        with open(path, "w") as out:
            out.write("# Generated by scripts/make_desi_params.py "
                      "--bias-pessimistic: the pessimistic linear-bias "
                      "prior of\n"
                      "# notes/correlated_bias_prior_v2 (section \"The "
                      "linear-bias prior\"), sigma_g = "
                      f"{PESSIMISTIC_SIGMA:g} in every group and "
                      f"L = {PESSIMISTIC_L:g}.\n"
                      "# Bin n of group g: DESI_B1_<n> = B_g (1 + delta_n), "
                      "delta_n = s_g (z_n - zbar_g) + eps_n, with\n"
                      "# B_g = DESI_B1_<g> (b1 at z = 0 under a constant "
                      "clustering amplitude; the likelihood\n"
                      "# divides DESI_B1_<n> by D(z_n)), s_g = DESI_TB1_<g> "
                      "(flat prior), zbar_g the mean of the group's\n"
                      "# bin redshifts, and eps = sigma_g C u, C the "
                      "Cholesky factor of exp(-|z_i - z_j|/L) in the\n"
                      "# group and u_n = DESI_UB1_<n> ~ N(0, 1). Groups "
                      "of one or two bins carry no residuals\n"
                      "# (and one bin no tilt): their bins are free (see "
                      "emit_bias_pessimistic).\n"
                      "# Velocity dispersions [km/s]: the seven groups of "
                      "params_desi.yaml, leaders sampled,\n"
                      "# members tied by value, at "
                      f"{PAPER_SIGV_WIDTH_FACTOR:g}x the prior widths of "
                      "params_desi.yaml (PAPER_SIGV_WIDTH_FACTOR:\n"
                      "# without alpha_2, sigma_v carries the "
                      "line-of-sight damping alone).\n")
            emit_bias_pessimistic(out, chunks, zbars,
                                  values=default["bias_values"])
            # the paper model's widened velocity-dispersion widths
            paper_sigma = {g: PAPER_SIGV_WIDTH_FACTOR*w
                           for g, w in SIGV_PRIOR_SIGMA.items()}
            emit_blocks(out, "DESI_SIGV", sigv_groups,
                        default["sigv_values"], paper_sigma,
                        SIGV_SCALE, sigv_latex)
        print(f"wrote {path}")
        return 0

    if args.model_b:
        stem = ("params_desi" if args.case is None
                else f"params_desi_{args.case}")
        # the counterterm width of MODEL_B_FAMILIES, quoted in the header
        width = 0.5/MODEL_B_KMAX**2
        # the fiducial model's higher-order fragment: the offsets and
        # tilts with alpha_2 fixed to 0 (PAPER_FAMILIES). The header text
        # below is part of the generated file, which the archived caches
        # fingerprint: change it only together with recomputing them
        path = os.path.join(OUT_DIR, f"{stem}_model_b_tilts.yaml")
        with open(path, "w") as out:
            out.write("# Generated by scripts/make_desi_params.py "
                      "--model-b: the model B parameters of "
                      "notes/higher_order_bias")
            out.write(" for the paper model:\n"
                      "# model B + tilts with free tilts on "
                      "Delta_2, Delta_s and alpha_0, and DESI_A2 "
                      "(alpha_2) fixed to 0 in\n"
                      "# every bin (the velocity dispersions of "
                      "the linear fragment absorb it).\n")
            out.write("# Units: DESI_A0 in h^-2 Mpc^2, DESI_SN0 in "
                      "units of each bin's Poisson power,\n")
            out.write("# DESI_SN2 in units of f_sat sigma_v^2 times "
                      "it (f_sat the satellite fraction, sigma_v the\n"
                      "# satellites' velocity dispersion in Mpc/h); "
                      "the per-bin values are tied to the chunk\n"
                      "# parameters (see emit_model_b).\n")
            out.write("# Scale cut: the shipped mask, k_max = 0.1185/Mpc "
                      f"= {MODEL_B_KMAX:.3f} h/Mpc at every redshift "
                      "(h = 0.6732),\n")
            out.write("# the same in every case; the alpha_0 prior "
                      "width is 0.5/k_max^2 = "
                      f"{width:.1f} h^-2 Mpc^2.\n")
            emit_model_b(out, chunks, zbars, True, families=PAPER_FAMILIES)
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

        path = os.path.join(model["folder"], fname)
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
