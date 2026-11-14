"""The classes compute_fisher_base_mpi.py evaluates the Fisher base with.

compute_fisher_base_mpi.py splits the evaluations of the Fisher
derivative base over MPI ranks. Two things there need behavior that the
project's own classes do not offer, and each is a subclass here, not a
replacement of a function at run time:

  1. Per-bin bias steps. The base holds per-bin columns of the one-loop
     bias operators: dm/db_{1,i} at fixed b_{2,i} and b_{s^2,i},
     dm/db_{2,i}, dm/db_{s^2,i}, dm/dalpha_{0,i} and dm/dalpha_{2,i},
     with m the data vector (the band powers), b_{1,i} the linear bias
     of bin i, b_{2,i} and b_{s^2,i} its quadratic and tidal biases,
     alpha_{0,i} the k^2 counterterm and alpha_{2,i} the radial
     (redshift-space) counterterm. No sampled parameter moves b_{1,i}
     alone (the paper's parameter files tie b_2 and b_{s^2} to b_1
     through their calibrations), so a column needs the per-bin arrays
     the likelihood hands to CosmoLike, shifted for a set of bins. The
     four *_steps classes below are subclasses of the four paper
     likelihoods (desi_cl_model_b and its Delta z siblings: model B +
     tilts without alpha_2) that can add such a step (PerBinStepMixin).
  2. Replaying stencil points. The stencil columns of the sampled
     parameters come from ModelFisher.derivatives, which evaluates its
     points one after the other. ReplayModelFisher can list those points
     without evaluating them, and later run derivatives again with every
     evaluation answered from vectors computed elsewhere (on other
     ranks). The points and the arithmetic that combines the vectors are
     then ModelFisher's own, unchanged.

The step arithmetic is the one of the Fisher notebook of the paper model
(EXAMPLE_EVALUATE_FISHER_MODEL.ipynb, section 4, its fisher_base), term
by term, so a shifted array holds the same floats as the notebook's:

  b1   b_{1,i} -> b_{1,i} (1 + step)            (a relative step)
  b2   b_{2,i} -> b_{2,i} + step
  bs2  b_{s^2,i} -> b_{s^2,i} + step
  a0   b_{K,i} -> b_{K,i} + step/(2 b_{1,i} (c/H0)^2), step in h^-2 Mpc^2:
       CosmoLike's k^2 term is 2 b_1 b_K k^2 P with k in units of H0/c
  a2   alpha_{2,i} -> alpha_{2,i} + step, step in h^-2 Mpc^2
  sigv sigma_{v,i} -> sigma_{v,i} (1 + step)   (a relative step of the
       velocity dispersion DESI_SIGV_<i> that blurs bin i's n(z); used by
       compute_sigv_bins_mpi.py, the per-bin sigma_v study)

How a step reaches CosmoLike. desi_cl's set_lens_related computes the
per-bin arrays of a point and passes them to the compiled interface,
among them ci.set_nuisance_bias_tidal (b_1, b_2, b_{s^2} and b_K, the
slot that carries alpha_0) and ci.set_nuisance_bias_alpha2 (alpha_2).
With a step set, the *_steps classes run the parent's set_lens_related
in full (every array, the Fingers-of-God n(z), every other setter), then
call the stepped operator's setter once more with the step added. Each
compiled setter stores a value only when it differs from the stored one
(by more than a relative 1e-13), then renews the cache key of the bias
tables, and computes nothing else, so the second call leaves CosmoLike
in the state that one call with the shifted arrays leaves it in.
Without a step the classes evaluate exactly what their parents do.

Cobaya finds the *_steps classes through the "class" and "python_path"
keys of a likelihood block; compute_fisher_base_mpi.py writes that block
into a temporary copy of the example yaml and checks that the copy
resolves to the same cobaya input. A cobaya class without a yaml file of
its own inherits its parent's defaults (options, parameter files,
dataset), so every model detail comes from the parent.

This module imports the compiled interface, so it is imported only after
OMP_NUM_THREADS is set (the OpenMP runtime and the likelihood base read
it once, when they load).
"""
import numpy as np

import cosmolike_lssty1xdesi_fisher as fm
import cosmolike_lssty1xdesi_interface as ci
from cobaya.likelihoods.lssty1xdesi.desi_cl import C_OVER_H0
from cobaya.likelihoods.lssty1xdesi.desi_cl_model_b import desi_cl_model_b
from cobaya.likelihoods.lssty1xdesi.desi_cl_dr1_dz0p02_model_b import (
    desi_cl_dr1_dz0p02_model_b,
)
from cobaya.likelihoods.lssty1xdesi.desi_cl_dr1_dz0p05_model_b import (
    desi_cl_dr1_dz0p05_model_b,
)
from cobaya.likelihoods.lssty1xdesi.desi_cl_dr1_dz0p20_model_b import (
    desi_cl_dr1_dz0p20_model_b,
)

# The operators a step can move: the four of the tidal setter (the bias
# operators b_1, b_2, b_{s^2}, alpha_0) and alpha_2 (the radial
# counterterm, fixed to 0 in the paper model).
TIDAL_OPERATORS = (
    "b1",
    "b2",
    "bs2",
    "a0",
)
ALPHA2_OPERATOR = "a2"
# the velocity dispersion of a bin's Fingers-of-God blur (DESI_SIGV_<i>,
# km/s, i 1-based). Unlike the bias operators, it is not a compiled
# setter's argument: it shapes the lens n(z) the likelihood builds, so its
# step is applied to the point's DESI_SIGV_<i> before the likelihood
# builds the blurred n(z)
SIGV_OPERATOR = "sigv"


# ---------------------------------------------------------------------------
# Per-bin steps
# ---------------------------------------------------------------------------

class PerBinStepMixin:
    """Adds an optional per-bin bias step to a desi_cl likelihood.

    A mixin is a class that is never used alone: listed before a
    likelihood class in a subclass's bases, its methods run first and
    reach the likelihood's own through super(). Cobaya reads defaults
    only from the bases that are cobaya components, so the mixin adds
    no option and no parameter (it defines methods only, no public class
    attribute, which cobaya would read as an option).

    State kept on the instance:
      _step   = None, or a dict {"operator": one of TIDAL_OPERATORS,
                ALPHA2_OPERATOR or SIGV_OPERATOR, "bins": int array of
                0-based bin indices, "step": float}; set by
                set_per_bin_step.
      _passed = the per-bin arrays the likelihood computed at the last
                evaluated point, before any step: {"B1", "B2", "BS2",
                "BK"} (one float64 array per bias, length = number of
                lens bins), or None before the first point.
      _passed_sigv = the per-bin velocity dispersions DESI_SIGV_<i> [km/s]
                of the last evaluated point, before any step, or None
                before the first point.
    """

    def initialize(self):
        """Run the likelihood's initialize, then start with no step.

        Side effects: everything the parent's initialize does (reading
        the dataset, setting up CosmoLike).
        """
        super().initialize()
        self._step = None
        self._passed = None
        self._passed_sigv = None

    def set_per_bin_step(self, operator, bins, step):
        """Add a step to one operator of some bins at every next point.

        The step stays until clear_per_bin_step, so a caller sets it,
        evaluates one point and clears it (try/finally).

        Arguments:
          operator = "b1", "b2", "bs2", "a0", "a2" or "sigv" (module
                     docstring).
          bins     = 0-based indices of the bins to shift (a non-empty
                     integer sequence without repeats).
          step     = the step, finite: relative for "b1" and "sigv",
                     absolute for "b2" and "bs2", in h^-2 Mpc^2 for "a0"
                     and "a2".

        Raises:
          ValueError naming the operator, the bins or the step that is
          not valid, before anything is stored.
        """
        known = TIDAL_OPERATORS + (ALPHA2_OPERATOR, SIGV_OPERATOR)
        if operator not in known:
            raise ValueError(
                f"per-bin step: operator {operator!r} is not one of {known}")
        nbins = int(self.lens_ntomo)
        indices = np.asarray(bins, dtype=int)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(
                "per-bin step: bins must be a non-empty 1D list of bin indices; "
                f"got {bins!r}")
        if indices.min() < 0 or indices.max() >= nbins:
            raise ValueError(
                f"per-bin step: bin indices must lie in [0, {nbins - 1}]; "
                f"got {indices.tolist()}")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"per-bin step: bin indices repeat in {indices.tolist()}")
        if not np.isfinite(step):
            raise ValueError(f"per-bin step: the step must be finite; got {step!r}")
        self._step = {
            "operator": operator,
            "bins": indices,
            "step": float(step),
        }

    def clear_per_bin_step(self):
        """Remove the step: the next point is the likelihood's own."""
        self._step = None

    def passed_biases(self):
        """Return copies of the per-bin biases of the last evaluated point.

        These are the arrays the likelihood computed from the point's
        parameters, before any step; at the fiducial point they are the
        'b1', 'b2' and 'bs2' of the derivative cache.

        Returns:
          dict {"B1", "B2", "BS2", "BK"} of float64 arrays (length =
          number of lens bins).

        Raises:
          RuntimeError before the first evaluated point.
        """
        if self._passed is None:
            raise RuntimeError(
                "passed_biases: no point has been evaluated yet; evaluate the "
                "fiducial point first")
        out = {}
        for key, values in self._passed.items():
            out[key] = values.copy()
        return out

    def passed_sigv(self):
        """Return a copy of the per-bin dispersions of the last evaluated point.

        These are the DESI_SIGV_<i> [km/s] the point's parameters give
        (group members tied to their leaders), before any step.

        Returns:
          float64 array (length = number of lens bins).

        Raises:
          RuntimeError before the first evaluated point.
        """
        if self._passed_sigv is None:
            raise RuntimeError(
                "passed_sigv: no point has been evaluated yet; evaluate the "
                "fiducial point first")
        return self._passed_sigv.copy()

    def one_loop_biases(self, b1, params):
        """Return the parent's (b2, bs2, bk), and record the four arrays.

        set_lens_related calls this with the point's b_1 array and hands
        b_1 and the three returned arrays to ci.set_nuisance_bias_tidal
        right after, so the recorded copies are what CosmoLike receives
        when no step is set.

        Arguments:
          b1     = the per-bin linear biases of the point.
          params = the point's parameters.

        Returns:
          (b2, bs2, bk): the parent's arrays, unchanged.
        """
        b2, bs2, bk = super().one_loop_biases(b1, params)
        self._passed = {
            "B1": np.array(b1, dtype=float),
            "B2": np.array(b2, dtype=float),
            "BS2": np.array(bs2, dtype=float),
            "BK": np.array(bk, dtype=float),
        }
        return b2, bs2, bk

    def set_lens_related(self, **params):
        """Send the point's lens nuisances to CosmoLike, plus the step if set.

        The parent sends every array of the point; with a step set, the
        setter of the stepped operator is called again with the step
        added to the shifted bins (the arithmetic of each operator is in
        the module docstring). A "sigv" step works one stage earlier:
        the shifted bins' DESI_SIGV_<i> are scaled by (1 + step) in a copy
        of the point's parameters, and the parent builds every array and
        the blurred n(z) from that copy, so only those bins' n(z) (and the
        stochastic kernels the likelihood keeps with them) change.

        Arguments:
          **params = the point's parameters (logp's keyword arguments).

        Side effects: CosmoLike's nuisance parameters and lens n(z), as
        in the parent; with a step, the shifted values replace the
        stepped operator's.
        """
        # record the point's own dispersions before any step:
        # per_bin_values resolves DESI_SIGV_<i> for every bin from the
        # point's parameters (a group member reads its leader's value)
        self._passed_sigv = np.array(self.per_bin_values("DESI_SIGV", params),
                                     dtype=float)
        if self._step is not None and self._step["operator"] == SIGV_OPERATOR:
            # dict(params) is a shallow copy: the step edits the copy's
            # entries, and the caller's params stay unchanged
            shifted = dict(params)
            for i in self._step["bins"]:
                # the parameters count bins from 1, the step's indices from 0
                name = f"DESI_SIGV_{i + 1}"
                shifted[name] = float(params[name])*(1.0 + self._step["step"])
            # **shifted passes the dict entries back as keyword arguments;
            # the early return skips the re-call below, which serves the
            # bias operators only
            super().set_lens_related(**shifted)
            return
        super().set_lens_related(**params)
        if self._step is None:
            return
        operator = self._step["operator"]
        members = self._step["bins"]
        step = self._step["step"]
        nbins = int(self.lens_ntomo)
        zeros = np.zeros(nbins)
        if operator == ALPHA2_OPERATOR:
            # the alpha_2 the likelihood passed at this point (zeros under
            # the paper model), plus a step that is nonzero on the shifted
            # bins only
            alpha2 = np.array(self.per_bin_values("DESI_A2", params), dtype=float)
            shift = np.zeros(nbins)
            shift[members] = step
            alpha2 = alpha2 + shift
            ci.set_nuisance_bias_alpha2(A2=alpha2)
            return
        # fresh copies, so the recorded arrays stay those of the point
        B1 = np.array(self._passed["B1"], dtype=float)
        B2 = np.array(self._passed["B2"], dtype=float)
        BS2 = np.array(self._passed["BS2"], dtype=float)
        BK = np.array(self._passed["BK"], dtype=float)
        if operator == "b1":
            # a relative step of b1 alone: b2, bs2 and bK keep the point's
            # values (here b2 and bs2 do not follow b1)
            B1[members] *= 1.0 + step
        elif operator == "b2":
            B2[members] += step
        elif operator == "bs2":
            BS2[members] += step
        else:
            # alpha_0 rides CosmoLike's 2 b1 bK k^2 P slot with k in H0/c
            # units: an alpha_0 step [h^-2 Mpc^2] is a bK step of
            # step/(2 b1 (c/H0)^2), with c/H0 in Mpc/h
            BK[members] += step/(2.0*B1[members]*C_OVER_H0**2)
        # re-send the likelihood's magnification coefficients (zeros
        # unless a study set them with set_magnification_coefficients):
        # the setter replaces every array it names, so passing zeros here
        # would silently switch magnification off in the stepped call
        bmag = np.array(self.desi_bmag, dtype=float)
        ci.set_nuisance_bias_tidal(B1=B1, B2=B2, BS2=BS2, B_MAG=bmag,
                                   B3nl=zeros, BK=BK)


# One class per paper likelihood: the second base decides the layout (its
# dataset, parameter files and options).

class desi_cl_model_b_steps(PerBinStepMixin, desi_cl_model_b):
    """desi_cl_model_b (Delta d_com = 62 Mpc) with an optional per-bin step."""


class desi_cl_dr1_dz0p02_model_b_steps(PerBinStepMixin, desi_cl_dr1_dz0p02_model_b):
    """desi_cl_dr1_dz0p02_model_b (Delta z = 0.02) with an optional per-bin step."""


class desi_cl_dr1_dz0p05_model_b_steps(PerBinStepMixin, desi_cl_dr1_dz0p05_model_b):
    """desi_cl_dr1_dz0p05_model_b (Delta z = 0.05) with an optional per-bin step."""


class desi_cl_dr1_dz0p20_model_b_steps(PerBinStepMixin, desi_cl_dr1_dz0p20_model_b):
    """desi_cl_dr1_dz0p20_model_b (Delta z = 0.2) with an optional per-bin step."""


# The cobaya name of each paper likelihood (as an example yaml writes it)
# -> the name of its step class in this module.
STEP_CLASS_OF = {
    "lssty1xdesi.desi_cl_model_b": "desi_cl_model_b_steps",
    "lssty1xdesi.desi_cl_dr1_dz0p02_model_b": "desi_cl_dr1_dz0p02_model_b_steps",
    "lssty1xdesi.desi_cl_dr1_dz0p05_model_b": "desi_cl_dr1_dz0p05_model_b_steps",
    "lssty1xdesi.desi_cl_dr1_dz0p20_model_b": "desi_cl_dr1_dz0p20_model_b_steps",
}


# ---------------------------------------------------------------------------
# Stencil points listed, then replayed
# ---------------------------------------------------------------------------

class ReplayModelFisher(fm.ModelFisher):
    """A ModelFisher whose stencil points can be listed and replayed.

    ModelFisher.derivatives(h) builds one five-point column per varied
    parameter: for each it asks dv for the vector at four points (the
    parameter at +2d, +d, -d and -2d around the fiducial, d its step)
    and combines them. This subclass overrides dv only, with three
    modes:

      "evaluate" = the parent's dv (the default; every evaluation of a
                   rank runs here);
      "record"   = store the requested point and return a dummy vector,
                   so derivatives(h) lists its points without
                   evaluating anything (the dummy columns are dropped);
      "replay"   = answer the k-th request with the k-th given vector,
                   after checking that the requested point is the k-th
                   recorded point bit for bit.

    A recorded list and a replay therefore reproduce the points and the
    arithmetic of a serial derivatives(h) call exactly; only where each
    vector was computed changes.

    State kept on the instance: _mode (one of the three above),
    _recorded (the points of the last record), _replay_points,
    _replay_vectors and _replay_next (the replay's inputs and position).
    """

    def __init__(self, *args, **kwargs):
        """Build the ModelFisher (same arguments), starting in "evaluate"."""
        super().__init__(*args, **kwargs)
        self._mode = "evaluate"
        self._recorded = []
        self._replay_points = None
        self._replay_vectors = None
        self._replay_next = 0

    def dv(self, param=None, AccuracyBoost=1.0):
        """Return the data vector at param in "evaluate" mode (see the class).

        Arguments:
          param = 1D array in the order of self.names, or None for the
                  fiducial (as in ModelFisher.dv).
          AccuracyBoost = accepted and ignored, as in ModelFisher.dv.

        Returns:
          1D float64 array: the masked vector ("evaluate"), a length-1
          zero array ("record") or the replayed vector ("replay").

        Raises:
          RuntimeError in "replay" when the requested point is not the
          recorded one or more points are requested than were given.
        """
        if self._mode == "evaluate":
            return super().dv(param=param, AccuracyBoost=AccuracyBoost)
        if param is None:
            param = self.fiducial
        point = np.array(param, dtype=np.float64, copy=True)
        if self._mode == "record":
            self._recorded.append(point)
            return np.zeros(1)
        position = self._replay_next
        if position >= len(self._replay_vectors):
            raise RuntimeError(
                f"replay: derivatives asked for point {position + 1}, but only "
                f"{len(self._replay_vectors)} vectors were given")
        expected = self._replay_points[position]
        if not np.array_equal(point, expected):
            raise RuntimeError(
                f"replay: point {position} differs from the recorded one "
                f"(requested {point.tolist()}, recorded {expected.tolist()}): the "
                "stencil changed between the record and the replay")
        self._replay_next = position + 1
        return np.array(self._replay_vectors[position], dtype=np.float64)

    def record_stencil_points(self, h):
        """List the points derivatives(h) evaluates, without evaluating them.

        Arguments:
          h = the relative stencil step of derivatives (ModelFisher's).

        Returns:
          list of 1D float64 arrays (each in the order of self.names), in
          the order derivatives(h) requests them: four per varied
          parameter, parameter by parameter.
        """
        self._mode = "record"
        self._recorded = []
        try:
            self.derivatives(h=h)
        finally:
            self._mode = "evaluate"
        points = []
        for point in self._recorded:
            points.append(point.copy())
        return points

    def derivatives_from_vectors(self, h, points, vectors):
        """Run derivatives(h) with every evaluation answered from vectors.

        Arguments:
          h       = the step given to record_stencil_points.
          points  = the list record_stencil_points returned.
          vectors = one data vector per point, in the same order.

        Returns:
          2D float64 array (n, len(self.names)): ModelFisher.derivatives'
          D, one column per varied parameter.

        Raises:
          ValueError when points and vectors differ in number;
          RuntimeError when the replay does not consume every vector or
          meets a point other than the recorded one.
        """
        if len(points) != len(vectors):
            raise ValueError(
                f"replay: {len(points)} points but {len(vectors)} vectors")
        self._mode = "replay"
        self._replay_points = points
        self._replay_vectors = vectors
        self._replay_next = 0
        try:
            D = self.derivatives(h=h)
        finally:
            self._mode = "evaluate"
        if self._replay_next != len(vectors):
            raise RuntimeError(
                f"replay: derivatives used {self._replay_next} of the "
                f"{len(vectors)} vectors: the stencil changed between the record "
                "and the replay")
        return D
