"""Fit the per-bin sigma_v study of notes/velocity_dispersion_per_bin.

The question
------------
Today each dispersion group g has one Fingers-of-God dispersion sigma_{v,g}
(the line-of-sight velocity spread, in km/s, that blurs the redshift
distribution n(z) of every bin of the group), shared by its thin bins. The
study lets every bin's dispersion move, with the freedom put on the halo
part of sigma_v only:

    sigma_{v,i}^2 = sigma_lin,i^2 + sigma_HOD,i^2,
    sigma_HOD,i  -> sigma_HOD,i [1 + s_g (zbar_i - zbar_g) + eps_i],

sigma_lin the linear-theory rms line-of-sight velocity at zbar_i (fixed by
the fiducial cosmology), sigma_HOD the halo (virial motion) part, s_g a
slope per group and eps_i a residual with a correlated Gaussian prior. To
first order, at fixed sigma_lin,

    delta sigma_{v,i} = w_i sigma_{v,g} [s_g (zbar_i - zbar_g) + eps_i],
    w_i = sigma_HOD,i^2 / sigma_{v,g}^2 = 1 - sigma_lin,i^2 / sigma_{v,g}^2,

so every variant is a linear combination of the per-bin columns
dm/dsigma_{v,i} (scripts/compute_sigv_bins_mpi.py, the caches
model_sigv_bins_<layout>_final). w_i is called the halo share below.
zbar_g is the mean of the group's zbar_i.

What this script computes, per layout
-------------------------------------
1. Checks of the per-bin columns. The sum of a group's per-bin columns must
   reproduce the group's DESI_SIGV stencil column of the derivative base
   (both at the relative step 0.02); and the columns at the steps 0.01 and
   0.04 must agree with those at 0.02 (the step test). Both are measured in
   the whitened norm |C^-1/2 x| = sqrt(x^T C^-1 x) on the shipped mask
   (whitening rescales the data by their correlated noise C, so every
   entry counts in units of its error bar), relative to the norm of the
   column itself, and the step test also as the change of the FoMs of the
   freest variant (V4).
2. sigma_lin(zbar_i) and w_i from one CAMB linear P(k) at the fiducial (the
   computation of the note's generators, plots/sigv_common.py).
3. The figures of merit FoM(w0, wa) and FoM(Omega_m, 10^9 A_s) of the paper
   model (the notebook's 'B tilts noA2 sv x6' on the pessimistic linear-bias
   prior) with the sigma_v freedom of each variant added, at every cut of the
   notebook's KMAX_CUTS (FoM(a, b) = 1/sqrt(det C_(a, b)), C_(a, b) the
   2 x 2 block of the inverse Fisher matrix of the two parameters):
     V0           today's model (one sigma_v per group);
     V1 <width>   + the halo slope s_g, Gaussian of width 0.5, 1.0 or 2.0 per
                  unit z (BGS always 2.0);
     V2           + slope (1.0) + residual A (10% of sigma_HOD, L = 0.2);
     V3           + slope (1.0) + residual B (20%, BGS 30%, L = 0.05);
     V3 white     + slope (1.0) + residual B + a white term (5%, L = 0.02);
     V4           every per-bin sigma_v free (flat), the bound.
   A residual is parameterized only in the directions its group's level and
   slope cannot reach: eps = E eta, with E spanning the vectors e with
   U^T Q e = 0 (U = [1, z - zbar_g], Q the inverse of the residual
   covariance; the set of such vectors is the null space of U^T Q). This is
   the projected precision of the linear bias (group_bias_precision of the
   notebook) restricted to its non-null directions, so the residual charges
   neither the level nor the slope, which have their own parameters.
4. The accuracy test: eta = sqrt(<chi^2>/n_dof) of each variant's fit when
   the true dispersions differ from the fiducial, for (w0, wa) (n_dof = 2),
   Omega_m and A_s (n_dof = 1), computed as the notebook's honesty: the
   expected chi^2 of the recovered-minus-true parameters under the fit's own
   covariance, with the true linear-bias residuals drawn from the paper
   prior (the notebook's convention, so V0 without a sigma_v truth gives the
   paper's own value) and the sigma_v truths below:
     T_lin       the linear flow's own redshift dependence: sigma_HOD flat in
                 each group, sigma_lin following zbar_i, i.e.
                 sigma_{v,i} = sqrt(sigma_lin,i^2 + sigma_{v,g}^2
                 - sigma_lin(zbar_g)^2); no prior choice;
     A, B,       random truths: slope (width 1.0, BGS 2.0) plus the full
     B white     Gaussian residual of A, B or B + white, drawn in the
                 sigma_HOD parameterization above;
     T_lin + B   both.
   The noise term of a fit uses its Fisher matrix without the priors whose
   population the truth draws (the linear-bias precision and the variant's
   sigma_v slope and residual priors), as the notebook's honesty does.

Output: <out>/results_<tag>.json (every number above) and a log on stdout.

Run (cocoa environment, start_cocoa.sh sourced, from any folder):
    python compute_sigv_bins_fits.py --layouts dz0p20 dz0p05 dz0p02 dcom62 \\
        --out <folder>
"""
import argparse
import json
import os
import sys
import time

# the notebook's BLAS thread count, before numpy loads (compute_pair_axes_fits
# explains it: the BLAS library reads the count once, when numpy loads it,
# and the summation order of its matrix products depends on it)
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402

# no bytecode caches next to the repository's modules
sys.dont_write_bytecode = True
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
# the notebook cell runner and the layout table of the pair-axes fits
import compute_pair_axes_fits as pa  # noqa: E402

# the derivative tier of every cache read here ('final', the converged tier)
TIER = pa.TIER
# the paper model, the notebook's SUMMARY_VARIANT: 'B tilts noA2 sv x6'
PAPER_VARIANT = pa.PAPER_VARIANT
# the paper's linear-bias prior
PRIOR = "pessimistic"
# The main step and the test steps of the per-bin columns. The cache's
# D_sigv entries are keyed by repr(step) strings (compute_sigv_bins_mpi,
# assemble), so the steps are compared as text here.
STEP = repr(0.02)
TEST_STEPS = (repr(0.01), repr(0.04))
CACHE_NAME = "model_sigv_bins_{tag}_{tier}"

# the priors on the halo part (notes/velocity_dispersion_per_bin, Table 5)
SLOPE_WIDTHS = {"0.5": 0.5, "1.0": 1.0, "2.0": 2.0}
BGS_SLOPE_WIDTH = 2.0
BASE_SLOPE = "1.0"
# each residual: the per-bin width sigma (a fraction of sigma_HOD; the BGS
# group has its own) and the correlation length L in redshift of the kernel
# sigma^2 exp(-|z_i - z_j|/L); "B white" stacks two kernels
RESIDUALS = {
    "A": [{"sigma": 0.10, "sigma_bgs": 0.10, "L": 0.2}],
    "B": [{"sigma": 0.20, "sigma_bgs": 0.30, "L": 0.05}],
    "B white": [{"sigma": 0.20, "sigma_bgs": 0.30, "L": 0.05},
                {"sigma": 0.05, "sigma_bgs": 0.05, "L": 0.02}],
}
# the variants: name -> (slope width key or None, residual key or None, free)
VARIANTS = {
    "V0": (None, None, False),
    "V1 0.5": ("0.5", None, False),
    "V1 1.0": ("1.0", None, False),
    "V1 2.0": ("2.0", None, False),
    "V2": (BASE_SLOPE, "A", False),
    "V3": (BASE_SLOPE, "B", False),
    "V3 white": (BASE_SLOPE, "B white", False),
    "V4": (None, None, True),
}
# the truths: name -> (T_lin included, slope width key or None, residual key)
TRUTHS = {
    "T_lin": (True, None, None),
    "A": (False, BASE_SLOPE, "A"),
    "B": (False, BASE_SLOPE, "B"),
    "B white": (False, BASE_SLOPE, "B white"),
    "T_lin + B": (True, BASE_SLOPE, "B"),
}
# The fiducial cosmology of the paper yamls (the note's fiducial section):
# H0 [km/s/Mpc], the total matter and baryon fractions Omega_m and Omega_b,
# the spectral index n_s, 10^9 A_s (A_s the primordial amplitude), the
# neutrino mass m_nu [eV] and the optical depth tau.
H0, OMEGA_M, OMEGA_B = 67.32, 0.3, 0.04
NS, AS_1E9, MNU, TAU = 0.96605, 2.1, 0.06, 0.0697186
# the wavenumber [1/Mpc] where the growth factor is read off the linear
# P(k) ratio (the likelihood's convention, far inside the linear regime)
KREF = 5.0e-3


def parse_arguments():
    """Read the command line.

    Returns:
      argparse.Namespace with layouts (a list of pa.LAYOUTS tags) and out
      (the folder of the result files).
    """
    parser = argparse.ArgumentParser(description="Per-bin sigma_v study fits.")
    parser.add_argument("--layouts", nargs="+", default=["dz0p20", "dz0p05", "dz0p02",
                                                          "dcom62"],
                        choices=list(pa.LAYOUTS))
    parser.add_argument("--out", required=True, help="folder of the result files")
    return parser.parse_args()


class LinearVelocity:
    """sigma_lin(z) = a H(z) f(z) sigma_d(z) from one CAMB linear P(k).

    This is the linear-theory rms line-of-sight velocity: a = 1/(1 + z)
    the scale factor, H(z) the Hubble rate [km/s/Mpc], f = dln D/dln a
    the linear growth rate (D the growth factor, D(0) = 1) and sigma_d
    the rms displacement [Mpc] of the linear density field. The result
    is in km/s, as the note's plots/sigv_common.py computes it, so the
    fits and the note's figures share one sigma_lin.
    """

    def __init__(self):
        """Run CAMB once at the fiducial cosmology and keep its results.

        Side effects: a CAMB run of a few seconds; the results, the P(k)
        interpolator and the k grid of the sigma_d integral are stored on
        the instance.
        """
        import camb
        h = H0/100.0
        # CAMB takes physical densities omega h^2. The cold-dark-matter
        # one is total matter minus baryons minus massive neutrinos, with
        # omega_nu h^2 = m_nu [eV]/94.0708 scaled by (N_eff/3)^(3/4) for
        # one massive state under N_eff = 3.046 (the same expression as
        # the note's plots/sigv_common.py)
        omch2 = (OMEGA_M - OMEGA_B)*h**2 - (MNU*(3.046/3.0)**0.75)/94.0708
        pars = camb.set_params(H0=H0, ombh2=OMEGA_B*h**2, omch2=omch2, ns=NS,
                               As=AS_1E9*1e-9, mnu=MNU, tau=TAU, num_massive_neutrinos=1,
                               nnu=3.046, kmax=100.0, k_per_logint=30, AccuracyBoost=2.0)
        # the redshift nodes of the P(k) interpolator: steps of 0.025 from
        # 0 to 2 cover every bin; the [::-1] slice (step -1) reverses the
        # array, the order CAMB expects
        zgrid = np.linspace(0.0, 2.0, 81)
        pars.set_matter_power(redshifts=zgrid[::-1], kmax=100.0, k_per_logint=30,
                              nonlinear=False)
        self.res = camb.get_results(pars)
        self.pk = self.res.get_matter_power_interpolator(
            nonlinear=False, hubble_units=False, k_hunit=False, var1="delta_tot",
            var2="delta_tot")
        # the ln k grid of the sigma_d integral: 4000 points from
        # k = 1e-5 to 80 /Mpc, wide enough that the integrand has decayed
        # at both ends
        self.lnk = np.linspace(np.log(1e-5), np.log(80.0), 4000)
        self.k = np.exp(self.lnk)

    def growth(self, z):
        """Return the linear growth factor D(z), normalized to D(0) = 1.

        D is read off the linear P(k) ratio at the reference wavenumber
        KREF, the likelihood's convention: D(z) = sqrt(P(z, k)/P(0, k)).

        Arguments:
          z = one redshift.

        Returns:
          float.
        """
        return float(np.sqrt(self.pk.P(z, KREF)/self.pk.P(0.0, KREF)))

    def growth_rate(self, z):
        """Return the linear growth rate f(z) = dln D/dln a.

        With a = 1/(1 + z), f = -(1 + z) dln D/dz. The derivative is a
        five-point stencil of step dz:
        [-g(+2dz) + 8 g(+dz) - 8 g(-dz) + g(-2dz)]/(12 dz) with
        g = ln D. np.clip floors the stencil points at z = 0, where a
        point would land below the P(k) table; every bin of these
        layouts sits far above 2 dz, so the guard never acts here.

        Arguments:
          z = one redshift.

        Returns:
          float.
        """
        dz = 0.01
        # the four stencil points, from +2dz down to -2dz
        zs = np.clip(z + dz*np.array([2.0, 1.0, -1.0, -2.0]), 0.0, None)
        growth_values = []
        for z_point in zs:
            growth_values.append(self.growth(z_point))
        lnD = np.log(growth_values)
        return -(1.0 + z)*(-lnD[0] + 8.0*lnD[1] - 8.0*lnD[2] + lnD[3])/(12.0*dz)

    def sigma_d(self, z):
        """Return the rms displacement of the linear density field [Mpc].

        sigma_d^2 = (1/6 pi^2) int P(k, z) dk; on the stored ln k grid
        the measure is dk = k dln k, so the integrand is P k and np.trapz
        (the trapezoid rule) integrates it over lnk.

        Arguments:
          z = one redshift.

        Returns:
          float [Mpc].
        """
        integrand = self.pk.P(z, self.k)*self.k
        return float(np.sqrt(np.trapz(integrand, self.lnk)/(6.0*np.pi**2)))

    def sigma_lin(self, z):
        """Return sigma_lin(z) [km/s] (the class docstring's product).

        hubble_parameter returns H(z) in km/s/Mpc and sigma_d is in Mpc,
        so the product a H f sigma_d is in km/s.

        Arguments:
          z = one redshift.

        Returns:
          float [km/s].
        """
        return (float(self.res.hubble_parameter(z))/(1.0 + z)*self.growth_rate(z)
                *self.sigma_d(z))


def build_namespace():
    """Run the paper notebook's cells on today's pair lists (all layouts).

    compute_pair_axes_fits.run_cells executes the notebook's code cells
    by index in one plain dict, as Jupyter would (each cell's first line
    is checked, so a rebuilt notebook with moved cells stops here). The
    setup cells run first (imports, settings, data tables); the final
    scale-cut cache is then read in place of the notebook's cut cell;
    the fit cells follow (the derivative base, the priors, the Fisher
    systems of every layout, the paper model). substitutions=[] applies
    no text edit: every layout stays on its current pair list (the
    pair-axes script's own edits would move one layout to its maximal
    list).

    Returns:
      dict, the notebook's namespace: the functions this script calls
      (fit_system, prior_diagonal, honesty, base_matrix, layout_grams,
      stable_inverse, residual_covariance, fiducial_of) and the tables
      it reads (fm, BASE, GRAM, COLUMNS, TABLES, BIN_ZBAR, PRECISION,
      EXTRA_PRECISION, KMAX_CUTS, FINGERPRINT, BIAS_ASSUMPTIONS, COSMO,
      SINGLE_PARAMS, NO_DATA_RTOL, NULL_OVERLAP_MAX, derivs).

    Raises:
      RuntimeError when the notebook runs another derivative tier or its
      paper model is not PAPER_VARIANT; run_cells raises when a cell
      moved; the cache reader raises on a missing or mismatched cut
      cache.
    """
    # a notebook file is JSON; "cells" is the list of its cells
    with open(pa.NOTEBOOK) as stream:
        cells = json.load(stream)["cells"]
    # the dict the cells read and write (their globals)
    ns = {"__name__": "notebook_cells"}
    pa.run_cells(cells=cells, indices=pa.SETUP_CELLS, namespace=ns, substitutions=[])
    if ns["DERIVATIVE_TIER"] != TIER:
        raise RuntimeError(f"the notebook runs the {ns['DERIVATIVE_TIER']!r} tier")
    fm = ns["fm"]
    # in place of the notebook's cut cell: read the archived final-tier
    # scale cuts (read_fisher_cache checks their settings and raises on
    # a mismatch, never recomputing)
    ns["CUTS"] = fm.read_fisher_cache(f"model_scale_cuts_{TIER}", tier=TIER)
    pa.run_cells(cells=cells, indices=pa.FIT_CELLS, namespace=ns, substitutions=[])
    if ns["SUMMARY_VARIANT"] != PAPER_VARIANT:
        raise RuntimeError(f"the notebook's paper model is {ns['SUMMARY_VARIANT']!r}")
    return ns


def sigv_groups(ns, key):
    """List the dispersion groups of one layout.

    A dispersion group is the set of consecutive bins that share one
    sampled velocity dispersion. DESI_SIGV_<n> (n 1-based) leads the
    group whose first bin is bin n - 1 (the notebook's rule); a group
    runs to the bin before the next leader. The group names count per
    sample: BGS0, BGS1, LRG0, ...

    Arguments:
      ns  = the notebook namespace (build_namespace).
      key = the notebook's layout key (pa.LAYOUTS[tag]["key"]).

    Returns:
      list of tuples (name, leader, bins, sample): the group name, its
      leader parameter DESI_SIGV_<n>, the 0-based bin indices (int
      array) and the sample ('bgs', 'lrg' or 'elg').
    """
    # the stencil block holds the cosmology and the sampled dispersions;
    # keep the leaders and turn each trailing number (the text after the
    # last underscore) into the 0-based index of the group's first bin
    names = ns["BASE"][key]["stencil_names"]
    starts = []
    for name in names:
        if name.startswith("DESI_SIGV_"):
            starts.append(int(name.split("_")[-1]) - 1)
    starts = sorted(starts)
    nbins = len(ns["BIN_ZBAR"][key])
    out = []
    counter = {}
    # enumerate yields (position, element): j numbers the groups
    for j, s in enumerate(starts):
        # one simple ternary: a group ends where the next one starts;
        # the last group runs to the last bin
        stop = starts[j + 1] if j + 1 < len(starts) else nbins
        # the bin table's column 1 is the sample of the group's first bin
        sample = str(ns["TABLES"][key]["bins"][s][1]).lower()
        # dict.get(key, default) returns default when the key is missing,
        # so the first group of a sample gets the number 0
        counter[sample] = counter.get(sample, -1) + 1
        out.append((f"{sample.upper()}{counter[sample]}", f"DESI_SIGV_{s + 1}",
                    np.arange(s, stop), sample))
    return out


def projected_basis(z, kernel_cov):
    """Build the residual directions a group's level and slope cannot reach.

    The residual eps of a group carries a correlated Gaussian prior with
    covariance kernel_cov. Its level (a constant shift) and its slope (a
    straight line in redshift) have their own free parameters, so the
    prior must charge neither. Writing eps = E eta with the columns of E
    spanning the null space of U^T Q (the null space of a matrix A is
    every vector e with A e = 0; scipy.linalg.null_space returns an
    orthonormal basis of it), every direction a level or a slope can
    produce is removed, and the prior precision of eta is E^T Q E.

    Arguments:
      z          = the bin mean redshifts of the group [nbin_g].
      kernel_cov = the residual covariance [nbin_g, nbin_g].

    Returns:
      (E, P_eta): E = (nbin_g, nbin_g - 2) orthonormal columns; P_eta =
      (nbin_g - 2, nbin_g - 2), the prior precision of eta.
    """
    # U's columns are the level (all ones) and the slope about the
    # group's mean redshift; np.column_stack glues 1D arrays as columns
    U = np.column_stack([np.ones(len(z)), z - z.mean()])
    Q = np.linalg.inv(kernel_cov)
    # @ is numpy's matrix product; U.T @ Q is (2, nbin_g)
    E = scipy.linalg.null_space(U.T @ Q)
    return E, E.T @ Q @ E


def residual_cov(z, specs, bgs):
    """Build the residual covariance of a group, in fractions of sigma_HOD.

    Each spec adds one exponential kernel sigma^2 exp(-|z_i - z_j|/L):
    sigma the per-bin width (a fraction of the halo dispersion
    sigma_HOD; the BGS group has its own) and L the correlation length
    in redshift. The "B white" residual stacks a correlated kernel and a
    nearly white one (a short L).

    Arguments:
      z     = the bin mean redshifts of the group [nbin_g].
      specs = a RESIDUALS entry: list of {"sigma", "sigma_bgs", "L"}.
      bgs   = True for the BGS group (sigma_bgs applies).

    Returns:
      (nbin_g, nbin_g) float64 array.
    """
    # broadcasting: z[:, None] is a column, z[None, :] a row; their
    # difference is the matrix of every pairwise separation
    dz = np.abs(z[:, None] - z[None, :])
    K = np.zeros((len(z), len(z)))
    for spec in specs:
        # one simple ternary: the BGS width or the common one
        sigma = spec["sigma_bgs"] if bgs else spec["sigma"]
        K += sigma**2*np.exp(-dz/spec["L"])
    return K


def variant_extras(groups, zbar, scale, variant, nbase, offset):
    """Build the extra parameters of a variant in the augmented base.

    The augmented base is [the notebook's base columns | one per-bin
    sigma_v block per step], so a coordinate vector of length nbase
    addresses both. Each extra parameter is one column that is zero
    outside the per-bin sigma_v block of the main step, plus a Gaussian
    prior precision:

      - the slope s_g moves bin i by scale_i (zbar_i - zbar_g) km/s per
        unit s_g (scale_i = w_i sigma_{v,g}, the km/s one unit of
        fractional halo change gives); its prior is a 1 x 1 precision
        1/width^2;
      - a residual is parameterized as eps = E eta (projected_basis), so
        its columns are scale_i E[:, a] and its precision is E^T Q E;
      - V4 (free) instead frees every bin but the group leader (the
        leader's own dispersion is already a stencil column of the
        base, so together they span every per-bin dispersion of the
        group), with a flat prior: a zero precision block.

    Arguments:
      groups  = sigv_groups' list.
      zbar    = the bin mean redshifts [nbins].
      scale   = w_i sigma_{v,g} per bin [km/s per unit fractional halo
                change].
      variant = a VARIANTS entry: (slope width key or None, residual key
                or None, free).
      nbase   = the number of augmented base columns.
      offset  = the index of the first per-bin sigma_v column (the main
                step's block) in the augmented base.

    Returns:
      (X, P, names): X = (nbase, k) columns, P = (k, k) prior precision
      (block diagonal over the groups), names = k strings.
    """
    # tuple unpacking: a VARIANTS entry is (slope, residual, free)
    slope_key, residual_key, free = variant
    cols = []
    names = []
    blocks = []
    # tuple unpacking again: each group is (name, leader, bins, sample)
    for name, leader, idx, sample in groups:
        z = zbar[idx]
        if free:
            # idx[1:] drops the first entry, the leader bin; each other
            # bin gets a unit column into its own per-bin slot
            for i in idx[1:]:
                col = np.zeros(nbase)
                col[offset + i] = 1.0
                cols.append(col)
                names.append(f"free_{i + 1}")
            # a zero precision block: the flat prior
            blocks.append(np.zeros((len(idx) - 1, len(idx) - 1)))
            continue
        if slope_key is not None and len(idx) > 1:
            width = BGS_SLOPE_WIDTH if sample == "bgs" else SLOPE_WIDTHS[slope_key]
            col = np.zeros(nbase)
            # offset + idx is an int array: the assignment writes only
            # the group's slots of the per-bin block
            col[offset + idx] = scale[idx]*(z - z.mean())
            cols.append(col)
            names.append(f"slope_{name}")
            blocks.append(np.array([[1.0/width**2]]))
        if residual_key is not None and len(idx) > 2:
            # a residual needs at least three bins: the projection removes
            # two directions (the level and the slope)
            E, P_eta = projected_basis(z, residual_cov(z, RESIDUALS[residual_key],
                                                       sample == "bgs"))
            for a in range(E.shape[1]):
                col = np.zeros(nbase)
                col[offset + idx] = scale[idx]*E[:, a]
                cols.append(col)
                names.append(f"res_{name}_{a}")
            blocks.append(P_eta)
    if not cols:
        # V0 adds nothing: empty columns and an empty precision
        return np.zeros((nbase, 0)), np.zeros((0, 0)), []
    # np.column_stack glues the 1D columns into (nbase, k); the * in
    # block_diag(*blocks) unpacks the list so each block becomes one
    # argument, and the blocks are placed along the diagonal
    return np.column_stack(cols), scipy.linalg.block_diag(*blocks), names


def truth_population(groups, zbar, scale, sigma_v, sigma_lin, zg_lin, truth):
    """Build the per-bin dispersion population of one TRUTHS entry.

    A truth is what the universe does to the dispersions while the fit
    believes its own model: a deterministic shift delta0 plus a random
    part of covariance S. T_lin keeps each bin's halo part at the group
    value while sigma_lin,i follows the bin's redshift, so
    sigma_{v,i} = sqrt(sigma_lin,i^2 + sigma_{v,g}^2 -
    sigma_lin(zbar_g)^2) and delta0 is that minus the fiducial. The
    random part draws the slope and the residual in the sigma_HOD
    parameterization (fractions of the halo part) and maps them to km/s
    with scale.

    Arguments:
      groups    = sigv_groups' list.
      zbar      = the bin mean redshifts [nbins].
      scale     = w_i sigma_{v,g} per bin [km/s per unit fractional halo
                  change].
      sigma_v   = the fiducial per-bin dispersions [km/s].
      sigma_lin = sigma_lin(zbar_i) per bin [km/s].
      zg_lin    = {group name: sigma_lin at the group's mean redshift}.
      truth     = a TRUTHS entry: (T_lin included, slope width key or
                  None, residual key or None).

    Returns:
      (delta0, S): delta0 = (nbins,) shift [km/s]; S = (nbins, nbins)
      covariance [(km/s)^2], block diagonal over the groups.
    """
    # tuple unpacking: a TRUTHS entry is (T_lin, slope, residual)
    with_lin, slope_key, residual_key = truth
    n = len(zbar)
    delta0 = np.zeros(n)
    S = np.zeros((n, n))
    for name, leader, idx, sample in groups:
        z = zbar[idx]
        if with_lin:
            # the group's one fiducial dispersion, read from its first
            # bin (fit_layout checked the group is uniform)
            sv = sigma_v[idx[0]]
            delta0[idx] = np.sqrt(sigma_lin[idx]**2 + sv**2 - zg_lin[name]**2) - sv
        # frac = the covariance of the fractional halo change of the
        # group's bins
        frac = np.zeros((len(idx), len(idx)))
        if slope_key is not None:
            width = BGS_SLOPE_WIDTH if sample == "bgs" else SLOPE_WIDTHS[slope_key]
            t = z - z.mean()
            # np.outer(t, t) is the rank-one matrix t t^T: the covariance
            # a Gaussian slope of this width spreads over the bins
            frac += width**2*np.outer(t, t)
        if residual_key is not None:
            frac += residual_cov(z, RESIDUALS[residual_key], sample == "bgs")
        # np.ix_(idx, idx) selects the square block with rows and columns
        # idx; the broadcasting multiplies row i by scale_i and column j
        # by scale_j, mapping the fractional covariance to (km/s)^2
        S[np.ix_(idx, idx)] = scale[idx][:, None]*frac*scale[idx][None, :]
    return delta0, S


def fit(ns, key, m, gram, gram_shipped, M, tags, X, Px):
    """Build the paper model's Fisher system at cut m with extra columns X.

    Replicates the notebook's fit_system on the parameter map [M | X]:
    the data block A^T G A, the diagonal Gaussian priors of the paper
    model, the pessimistic linear-bias precision on the per-bin
    amplitudes (the last nbins parameters of M), the variant's precision
    Px on the extras, the no-data rule (a parameter whose information at
    this cut falls below NO_DATA_RTOL of its information on the shipped
    mask is dropped as having no data), the eigenvalue pseudo-inverse of
    stable_inverse, and the check that no dropped null direction
    overlaps a scored parameter.

    This function is a numerical kernel: the matrix algebra stays
    vectorized, with the shapes in the comments.

    Arguments:
      ns           = the notebook namespace (build_namespace).
      key          = the notebook's layout key.
      m            = the index of the cut in the notebook's KMAX_CUTS.
      gram         = G = B_aug^T C^-1 B_aug at the cut [naug, naug].
      gram_shipped = the same on the shipped mask (the no-data
                     reference).
      M            = the paper variant's parameter map, padded to the
                     augmented base [naug, npar].
      tags         = the prior tags of M's parameters (the notebook's
                     variant_columns).
      X            = the variant's extra columns [naug, k].
      Px           = their prior precision [k, k].

    Returns:
      (system, status): status is "fit", "cosmology without data" or
      "scored parameters unconstrained"; system is None unless "fit",
      and then a dict with "A" (the live columns), "F_inv", "F0" (data
      plus diagonal priors only, the noise reference of eta), "iw" and
      "ip" (positions of w0, wa and of the SINGLE_PARAMS among the live
      parameters), "Cw" and "Ci" (the marginalized (w0, wa) covariance
      and its inverse), "sigma" (marginalized errors), "fom" and
      "fom_om_as" (1/sqrt(det C) of the two parameter pairs) and "gram".
    """
    nbins = len(ns["BIN_ZBAR"][key])
    npar = M.shape[1]
    # np.hstack glues the columns: A maps [paper parameters | extras]
    A = np.hstack([M, X])
    # the data block, (npar + k, npar + k); gram @ A is (naug, npar + k)
    F_data = A.T @ gram @ A
    prior = np.zeros(A.shape[1])
    # the paper model's diagonal Gaussian priors. "B" is the model family
    # of the paper variant 'B tilts noA2 sv x6' (the notebook's
    # fit_system passes variant.split()[0]); the widened sigma_v widths
    # ride in tags.
    prior[:npar] = ns["prior_diagonal"](tags, "B", ns["KMAX_CUTS"][m])
    # F0 = data + diagonal priors only: eta's noise term uses it, so the
    # correlated priors added below (whose populations the truths draw)
    # stay out of the noise
    F0 = F_data + np.diag(prior)
    F = F0.copy()
    # the pessimistic linear-bias precision charges the per-bin
    # amplitudes, the last nbins of the paper parameters
    nx = npar - nbins
    F[nx:npar, nx:npar] += ns["PRECISION"][(key, PRIOR)]
    F[npar:, npar:] += Px
    # the paper variant's correlated higher-order priors, when the
    # notebook registered one for this layout
    if (key, PAPER_VARIANT) in ns["EXTRA_PRECISION"]:
        F[:npar, :npar] += ns["EXTRA_PRECISION"][(key, PAPER_VARIANT)](key,
                                                                     ns["KMAX_CUTS"][m])
    # each parameter's information on the shipped mask with this fit's
    # priors. einsum("ij,ij->j", A, Y) multiplies the two arrays entry by
    # entry and sums each column: column j gives A[:, j]^T Y[:, j], the
    # diagonal of A^T gram_shipped A without forming the whole matrix.
    reference = np.einsum("ij,ij->j", A, gram_shipped @ A) + np.diag(F) - np.diag(F_data)
    # the no-data rule: keep parameter j only when its information at
    # this cut is a measurable share of its shipped-mask reference
    live = []
    for j in range(A.shape[1]):
        if F[j, j] > ns["NO_DATA_RTOL"]*reference[j]:
            live.append(j)
    # the parameter names: the paper variant's, then one x<j> per extra
    base_names = ns["COLUMNS"][(key, PAPER_VARIANT)][1]
    extra_names = []
    for j in range(X.shape[1]):
        extra_names.append(f"x{j}")
    names = base_names + extra_names
    used = []
    for j in live:
        used.append(names[j])
    for p in ns["COSMO"]:
        if p not in used:
            return None, "cosmology without data"
    # the eigenvalue pseudo-inverse of the live block; null holds the
    # dropped directions as columns (zero columns: none dropped)
    F_inv, null = ns["stable_inverse"](F[np.ix_(live, live)])
    iw = [used.index("w0"), used.index("wa")]
    ip = {}
    for p in ns["SINGLE_PARAMS"]:
        ip[p] = used.index(p)
    # a python dict keeps insertion order, so list(ip.values()) follows
    # SINGLE_PARAMS' order
    scored = iw + list(ip.values())
    # one simple ternary: with no dropped direction the overlap is zero
    overlap = float(np.max(np.abs(null[scored]))) if null.shape[1] else 0.0
    if overlap > ns["NULL_OVERLAP_MAX"]:
        return None, "scored parameters unconstrained"
    # Cw = the marginalized (w0, wa) covariance: the 2 x 2 block of the
    # inverse Fisher matrix (np.ix_ selects rows and columns iw)
    Cw = F_inv[np.ix_(iw, iw)]
    ioa = [ip["omegam"], ip["As_1e9"]]
    sigma = {}
    for p, i in ip.items():
        sigma[p] = float(np.sqrt(F_inv[i, i]))
    return {"A": A[:, live], "F_inv": F_inv, "F0": F0[np.ix_(live, live)], "iw": iw,
            "ip": ip, "Cw": Cw, "Ci": np.linalg.inv(Cw),
            "sigma": sigma,
            "fom": float(1.0/np.sqrt(np.linalg.det(Cw))),
            "fom_om_as": float(1.0/np.sqrt(np.linalg.det(F_inv[np.ix_(ioa, ioa)]))),
            "gram": gram}, "fit"


def eta(system, Cb, Sb, Cs, delta0, Ss):
    """Score one fit against truths: eta = sqrt(<chi^2>/n_dof).

    The statistic is the notebook's honesty: chi^2 = Delta p^T C_p^-1
    Delta p with Delta p the recovered-minus-true parameters and C_p the
    fit's own marginalized covariance, so eta = 1 means the quoted error
    bars cover where the fit lands. The truths move the data linearly:
    the linear-bias residuals through the per-bin amplitude columns Cb
    (zero mean, covariance Sb) and the dispersions through the per-bin
    sigma_v columns Cs (shift delta0, covariance Ss). The fit's own
    noise enters through F_inv F0 F_inv; F0 holds the data and the
    diagonal priors only, so the priors whose populations the truths
    draw are not counted as noise as well.

    This function is a numerical kernel: the response algebra stays
    vectorized, with the shapes in the comments.

    Arguments:
      system = a fit result (fit's dict).
      Cb     = the per-bin amplitude columns [naug, nbins] (per unit
               fractional linear-bias residual).
      Sb     = the linear-bias residual covariance [nbins, nbins].
      Cs     = the per-bin sigma_v columns [naug, nbins] (per km/s).
      delta0 = the deterministic dispersion shift [nbins] [km/s].
      Ss     = the dispersion covariance [nbins, nbins] [(km/s)^2].

    Returns:
      dict: "w0wa" and each single parameter p hold eta; the
      "_shift_only" keys hold the shift of delta0 alone, in sigma.
    """
    # R maps a data change (in base coordinates) onto the recovered
    # parameters: dp = F^-1 A^T G c; R is (n_live, naug)
    R = system["F_inv"] @ system["A"].T @ system["gram"]
    # the parameter responses per unit linear-bias residual and per km/s
    # of each bin's dispersion, (n_live, nbins) each
    Rb = R @ Cb
    Rs = R @ Cs
    # the fit's own noise, (n_live, n_live)
    noise = system["F_inv"] @ system["F0"] @ system["F_inv"]
    iw = system["iw"]
    # the (w0, wa) shift of the deterministic part, a length-2 vector
    d_w = Rs[iw] @ delta0
    # the (w0, wa) error matrix: the shift squared (np.outer gives the
    # rank-one matrix d d^T), the two random populations propagated as
    # R S R^T, and the noise block
    K_w = (np.outer(d_w, d_w) + Rs[iw] @ Ss @ Rs[iw].T + Rb[iw] @ Sb @ Rb[iw].T
           + noise[np.ix_(iw, iw)])
    out = {"w0wa": float(np.sqrt(np.trace(system["Ci"] @ K_w)/2.0)),
           "w0wa_shift_only": float(np.sqrt(d_w @ system["Ci"] @ d_w))}
    for p, i in system["ip"].items():
        s = system["sigma"][p]
        # Rs[i] is parameter i's response row; drift and spread are in
        # units of the parameter's marginalized sigma
        drift = float(Rs[i] @ delta0)/s
        spread = float(Rs[i] @ Ss @ Rs[i] + Rb[i] @ Sb @ Rb[i] + noise[i, i])/s**2
        out[p] = float(np.sqrt(drift**2 + spread))
        out[p + "_shift_only"] = abs(drift)
    return out


def whitened_norms(gram, cols):
    """Return the whitened norm of each column of cols.

    Whitening rescales the data by their correlated noise so every
    direction counts in units of its error bar: |x|_C = sqrt(x^T C^-1 x).
    For x = B_aug c and G = B_aug^T C^-1 B_aug that is sqrt(c^T G c),
    computed per column by einsum ("ij,ij->j" multiplies the arrays
    entry by entry and sums each column). np.maximum(..., 0.0) floors
    round-off: a squared norm cannot be negative, but a zero column can
    give a tiny negative number.

    Arguments:
      gram = G [naug, naug].
      cols = the columns, in augmented base coordinates [naug, k].

    Returns:
      (k,) float64 array.
    """
    return np.sqrt(np.maximum(np.einsum("ij,ij->j", cols, gram @ cols), 0.0))


def fit_layout(ns, tag, lin, out_dir):
    """Run every check, fit and accuracy test of one layout, and save them.

    The steps, in order: read the per-bin sigma_v cache and check that it
    belongs to the notebook's model; build the augmented base and its
    Gram matrices; the sum check and the step test (module docstring,
    item 1); sigma_lin and the halo share (item 2); the variants and the
    accuracy test at every cut (items 3 and 4); the JSON file and the
    log.

    Arguments:
      ns      = the notebook namespace (build_namespace).
      tag     = a pa.LAYOUTS key.
      lin     = the LinearVelocity instance.
      out_dir = the folder of the result files.

    Raises:
      RuntimeError when the cache belongs to another model, its bin
      redshifts differ from the notebook's, a group's fiducial
      dispersions are unequal, or sigma_lin exceeds sigma_v somewhere
      (the halo share w_i would be negative and the parameterization
      breaks down); the cache readers raise on a missing or mismatched
      cache.

    Side effects: writes <out_dir>/results_<tag>.json and prints the
    log.
    """
    key = pa.LAYOUTS[tag]["key"]
    fm = ns["fm"]
    start = time.time()
    # read_fisher_cache checks the stored settings and tier against the
    # module's and raises on any difference
    cache = fm.read_fisher_cache(CACHE_NAME.format(tag=tag, tier=TIER), tier=TIER)
    # the cache stores the sha256 of the resolved paper yaml it was
    # computed on; it must equal the notebook's, or the columns belong to
    # another model (read_fisher_cache_header reads the JSON header only)
    header = fm.read_fisher_cache_header(CACHE_NAME.format(tag=tag, tier=TIER))
    stored_fingerprint = header["tiers_and_fingerprints"]["settings/fingerprint"]
    if ns["FINGERPRINT"][key] != stored_fingerprint:
        raise RuntimeError(f"{tag}: the per-bin sigma_v cache belongs to another model "
                           "than the notebook's derivative base")
    zbar = ns["BIN_ZBAR"][key]
    nbins = len(zbar)
    # rtol=0: the cached redshifts must be the notebook's own floats
    if not np.allclose(cache["zbar"], zbar, rtol=0, atol=1e-12):
        raise RuntimeError(f"{tag}: the cache's bin redshifts differ from the notebook's")
    sigma_v = np.asarray(cache["sigma_v"], dtype=float)
    # the steps of this run: the main step, then the test steps the cache
    # holds (the D_sigv keys are repr(step) strings)
    steps = [STEP]
    for s in TEST_STEPS:
        if s in cache["D_sigv"]:
            steps.append(s)
    B, info = ns["base_matrix"](key)
    nb = B.shape[1]
    # the augmented base: the notebook's columns, then one (n, nbins)
    # per-bin sigma_v block per step
    pieces = [B]
    for s in steps:
        pieces.append(cache["D_sigv"][s])
    B_aug = np.hstack(pieces)
    # free the large intermediates: only B_aug is used from here on
    del pieces
    del B
    # the Gram matrices G = B_aug^T C^-1 B_aug at every cut and on the
    # shipped mask; the second return (the kept-entry counts) is unused
    grams, _ = ns["layout_grams"](key, B_aug)
    del B_aug
    naug = nb + nbins*len(steps)
    # the column offset of each step's per-bin block in the augmented base
    offsets = {}
    for k, s in enumerate(steps):
        offsets[s] = nb + k*nbins
    groups = sigv_groups(ns, key)
    G_ship = grams["shipped"]

    # 1. the sum check and the step test, in the whitened norm
    stencil = info["stencil_names"]
    sum_check = {}
    for name, leader, idx, sample in groups:
        # the sum check and the truths assume one dispersion per group
        if not np.allclose(sigma_v[idx], sigma_v[idx[0]]):
            raise RuntimeError(f"{tag}: group {name} has unequal fiducial sigma_v")
        # e = the group's stencil column dm/dsigma_{v,g} in augmented
        # base coordinates (list.index finds the leader's position in the
        # stencil block); both e and the per-bin columns are per km/s and
        # the group's dispersions are equal, so they compare directly
        e = np.zeros(naug)
        e[info["blocks"]["stencil"][stencil.index(leader)]] = 1.0
        # c = the group column minus the sum of its per-bin columns: zero
        # when the per-bin columns add up to the group's
        c = e.copy()
        c[offsets[STEP] + idx] -= 1.0
        # c[:, None] turns the vector into a one-column matrix, the shape
        # whitened_norms takes
        sum_check[name] = float(whitened_norms(G_ship, c[:, None])[0]
                                / whitened_norms(G_ship, e[:, None])[0])
    step_check = {}
    # steps[1:] skips the main step: each test step is compared with it
    for s in steps[1:]:
        diffs = []
        for i in range(nbins):
            c = np.zeros(naug)
            c[offsets[s] + i] = 1.0
            c[offsets[STEP] + i] -= 1.0
            ref = np.zeros(naug)
            ref[offsets[STEP] + i] = 1.0
            norm = whitened_norms(G_ship, ref[:, None])[0]
            # a bin whose spectra keep no entry has norm 0 and is skipped
            if norm > 0:
                diffs.append(float(whitened_norms(G_ship, c[:, None])[0]/norm))
        step_check[s] = max(diffs)

    # 2. sigma_lin and the halo share
    sigma_lin_values = []
    for z in zbar:
        sigma_lin_values.append(lin.sigma_lin(z))
    sigma_lin = np.array(sigma_lin_values)
    # sigma_lin at each group's mean redshift (the T_lin truth subtracts
    # it, so a group's leader bin keeps its fiducial value)
    zg_lin = {}
    for name, leader, idx, sample in groups:
        zg_lin[name] = lin.sigma_lin(float(zbar[idx].mean()))
    # the halo share: the fraction of sigma_v^2 the halo part contributes
    w = 1.0 - sigma_lin**2/sigma_v**2
    if not np.all(w > 0):
        # np.where(condition)[0] lists the indices where it holds
        raise RuntimeError(f"{tag}: sigma_lin exceeds sigma_v in bins "
                           f"{np.where(w <= 0)[0].tolist()}")
    # the km/s one unit of fractional halo change moves sigma_v by
    scale = w*sigma_v

    # 3 and 4. the variants and the accuracy test at every cut
    M0, _, tags = ns["COLUMNS"][(key, PAPER_VARIANT)]
    # pad the paper map with zero rows for the appended sigma_v blocks:
    # the paper parameters do not move them
    M = np.vstack([M0, np.zeros((naug - nb, M0.shape[1]))])
    # beta_i = the fiducial group amplitude of each bin's leader: the
    # linear-bias truths are drawn as fractions of it (the notebook's
    # honesty convention)
    beta_values = []
    for x in ns["derivs"][key]["leaders"]:
        beta_values.append(ns["fiducial_of"](key)[x])
    beta = np.array(beta_values)
    # the per-bin amplitude columns per unit fractional residual: the
    # amplitudes are the last nbins parameters of the paper map, and the
    # broadcasting scales column i by beta_i
    Cb = M[:, -nbins:]*beta[None, :]
    Sb = ns["residual_covariance"](key, ns["BIAS_ASSUMPTIONS"][PRIOR])
    # the per-bin sigma_v columns at the main step, one unit vector per
    # bin: the paired index arrays write Cs[offset + i, i] = 1 for each i
    Cs = np.zeros((naug, nbins))
    Cs[offsets[STEP] + np.arange(nbins), np.arange(nbins)] = 1.0
    truths = {}
    for t, spec in TRUTHS.items():
        truths[t] = truth_population(groups, zbar, scale, sigma_v, sigma_lin,
                                     zg_lin, spec)
    # the no-dispersion-truth case: zero shift, zero covariance
    no_sigv = (np.zeros(nbins), np.zeros((nbins, nbins)))
    extras = {}
    for v, spec in VARIANTS.items():
        extras[v] = variant_extras(groups, zbar, scale, spec, naug, offsets[STEP])
    per_cut = {}
    v0_check = 0.0
    eta_check = 0.0
    # enumerate yields (position, element): m indexes the cuts
    for m, kmax in enumerate(ns["KMAX_CUTS"]):
        # the output's cut label: "%g" drops trailing zeros ("0.075")
        label = f"{float(kmax):g}"
        gram = grams[m]
        if gram is None:
            per_cut[label] = {"status": "no band kept"}
            continue
        entry = {"variants": {}, "eta": {}}
        # tuple unpacking: each variant's (columns, precision, names)
        for v, (X, Px, xnames) in extras.items():
            system, status = fit(ns, key, m, gram, G_ship, M, tags, X, Px)
            if system is None:
                entry["variants"][v] = {"status": status}
                continue
            entry["variants"][v] = {"status": "fit", "fom": system["fom"],
                                    "fom_om_as": system["fom_om_as"],
                                    "n_extra": len(xnames)}
            if v == "V0":
                # V0 adds nothing, so its FoM must reproduce the
                # notebook's own paper fit, and eta without a sigma_v
                # truth must reproduce the notebook's honesty
                nb_system, nb_status = ns["fit_system"](key, m, PAPER_VARIANT, PRIOR)
                if nb_system is not None:
                    v0_check = max(v0_check, abs(system["fom"]/nb_system["fom"] - 1.0))
                    ref = ns["honesty"](nb_system, np.zeros(nbins), Sb)["w0wa"][
                        "error_inflation"]
                    # *no_sigv unpacks the (shift, covariance) pair into
                    # eta's last two arguments
                    mine = eta(system, Cb, Sb, Cs, *no_sigv)["w0wa"]
                    eta_check = max(eta_check, abs(mine/ref - 1.0))
            # fit's F0 holds the data and the diagonal priors only: the
            # linear-bias precision and the variant's sigma_v priors are the
            # populations the truths draw, so they stay out of the noise term
            scores = {"none": eta(system, Cb, Sb, Cs, *no_sigv)}
            for t, (d0, Ss) in truths.items():
                scores[t] = eta(system, Cb, Sb, Cs, d0, Ss)
            entry["eta"][v] = scores
            # the step test on the freest variant: its FoMs with the columns
            # of the other steps
            if v == "V4":
                entry["step_fom"] = {}
                for s in steps[1:]:
                    # rebuild each unit column in the test step's block:
                    # V4's columns are unit vectors, so np.argmax (the
                    # row of the largest entry) finds each column's bin
                    Xs = np.zeros_like(X)
                    for j in range(X.shape[1]):
                        i = int(np.argmax(X[:, j])) - offsets[STEP]
                        Xs[offsets[s] + i, j] = 1.0
                    other, _ = fit(ns, key, m, gram, G_ship, M, tags, Xs, Px)
                    if other is not None:
                        entry["step_fom"][s] = {
                            "fom": other["fom"]/system["fom"] - 1.0,
                            "fom_om_as": other["fom_om_as"]/system["fom_om_as"] - 1.0}
        per_cut[label] = entry
    # the groups of the output, with 1-based bin numbers (the parameter
    # names count bins from 1)
    group_records = []
    for name, leader, idx, sample in groups:
        bin_numbers = []
        for i in idx:
            bin_numbers.append(int(i) + 1)
        group_records.append({"name": name, "leader": leader, "bins": bin_numbers,
                              "sample": sample, "sigma_v": float(sigma_v[idx[0]]),
                              "zbar_g": float(zbar[idx].mean()),
                              "sigma_lin_g": zg_lin[name]})
    result = {
        "layout": tag, "key": key, "nbins": nbins,
        "groups": group_records,
        "zbar": zbar.tolist(), "sigma_v": sigma_v.tolist(),
        "sigma_lin": sigma_lin.tolist(), "w": w.tolist(),
        "t_lin_shift_km_s": truths["T_lin"][0].tolist(),
        "sum_check": sum_check, "step_check": step_check,
        "v0_vs_notebook_fom": v0_check, "eta_vs_notebook": eta_check,
        "cuts": per_cut,
        "priors": {"slope_widths": SLOPE_WIDTHS, "bgs_slope_width": BGS_SLOPE_WIDTH,
                   "residuals": RESIDUALS},
        "seconds": time.time() - start,
    }
    path = os.path.join(out_dir, f"results_{tag}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=1)
    print(f"{tag}: sum check (whitened, relative) max {max(sum_check.values()):.2e}; "
          f"step test {step_check}; V0 vs notebook FoM {v0_check:.1e}, eta "
          f"{eta_check:.1e}; {time.time() - start:.0f} s -> {path}", flush=True)
    for label, entry in per_cut.items():
        if "variants" not in entry:
            continue
        # one FoM field per variant; dict.get returns the default (here
        # NaN) when a failed fit stored no FoM
        fom_fields = []
        for v, e in entry["variants"].items():
            fom_fields.append(f"{v} {e.get('fom', float('nan')):.1f}")
        foms = "  ".join(fom_fields)
        # V0's eta per truth ("none" first); .get covers a cut where V0
        # itself did not fit
        eta_fields = []
        for t in entry["eta"].get("V0", {}):
            eta_fields.append(f"{t} {entry['eta']['V0'][t]['w0wa']:.3f}")
        etas = "  ".join(eta_fields)
        print(f"  k_max {label}: FoM(w0,wa) {foms}\n      V0 eta(w0,wa) {etas}",
              flush=True)


def main():
    """Parse the command line, build the shared state once, and fit.

    The notebook namespace and the CAMB run are shared by every layout:
    the notebook's cells build the Fisher systems of all four layouts in
    one namespace, and sigma_lin depends only on the fiducial cosmology.

    Side effects: creates --out; fit_layout writes one
    results_<tag>.json per layout.
    """
    args = parse_arguments()
    os.makedirs(args.out, exist_ok=True)
    out_dir = os.path.abspath(args.out)
    ns = build_namespace()
    lin = LinearVelocity()
    for tag in args.layouts:
        fit_layout(ns, tag, lin, out_dir)


if __name__ == "__main__":
    main()
