"""Fisher matrices on this project's cobaya models.

The shared Fisher machinery (cosmolike_notebook_utils.fisher: the
five-point-stencil derivative, F = D^T C^-1 D, Gaussian priors, the
figure of merit, the triangle plots) needs one function: the data vector
at a flat parameter vector. This module builds it on the cobaya model of
an EXAMPLE yaml, so a Fisher matrix differentiates exactly the vector the
likelihood scores - its CAMB settings, desi_cl's per-point FoG
convolution and bias rules, its masked layout.

The Gaussian priors come from the same model: every varied parameter
whose cobaya prior is a normal (or truncated normal) adds 1/sigma^2 to
its diagonal, so a Fisher matrix and a chain share their priors. Flat
priors add nothing.

The dark-energy pair. The likelihoods sample w (= w0) and w0pwa
(= w0 + wa); a Fisher matrix may vary "w0" and "wa" instead, mapped onto
those two before every evaluation.

Stencil steps. A parameter with a nonzero fiducial p0 takes the relative
steps p0 h and 2 p0 h of cnu.get_ddv. A parameter whose fiducial is zero
(photo-z shifts, shear calibrations, wa) has no scale to be relative to:
its absolute step is h s, with s = sigma_prior/0.02 for a Gaussian prior
(the step equals the prior width at the default h = 0.02) and s = 1
otherwise. Every step scales with h, so an h scan probes the stability
of all derivatives at once.

Every derivative is a five-point stencil. The step d cannot be small:
the vector carries numerical noise that a difference divides by d. At
such a step the three-point (central) difference [m(p + d) - m(p - d)]/
(2 d) is off by a term of order d^2 times the third derivative of the
vector; the five-point combination FIVE_POINT_WEIGHTS of the offsets
FIVE_POINT_OFFSETS (in units of d) cancels that term and leaves one of
order d^4. The per-bin columns of the notebooks (the bias operators of
the one-loop model) use the same two constants. Terms that enter the
vector linearly (the stochastic amplitudes SN0 and SN2, the counterterm
alpha_2) have exact templates and need no stencil.

Evaluations ignore the priors. Each stencil point runs the likelihoods
directly (Model.loglike), so a point just outside a flat prior's edge
still yields its data vector; a likelihood that fails (a CAMB error, a
non-finite chi2) raises instead of leaving the previous point's vector
in the interface.

Fisher bias. The vector of an alternative model at the same point
(ModelFisher built with likelihood_overrides, its dv()) against the
reference model's gives the linear-order shift of the best fit when
the reference model is fitted to the alternative's vector:
ModelFisher.bias_vector, then fisher_bias, which also sums blocks.

One model at a time. The compiled interface holds one dataset (mask,
covariance, data vector) and one lens sample per process, so building a
second ModelFisher replaces the first one's state. Compute each Fisher
matrix right after building its model; a ModelFisher whose state was
replaced refuses to evaluate instead of returning another model's
vectors.
"""
import numpy as np
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file

import cosmolike_lssty1xdesi_interface as ci
from cosmolike_notebook_utils import fisher as cnu

# the ModelFisher whose model owns the interface's state right now
_ACTIVE = [None]

# the default relative step of cnu.get_ddv
H_DEFAULT = 0.02

# The five-point stencil of cnu.get_ddv, for derivatives taken outside it
# (the notebooks' per-bin bias columns): with d the step,
#   dm/dp = sum_k FIVE_POINT_WEIGHTS[k] m(p + FIVE_POINT_OFFSETS[k] d)/d,
# that is [-m(p + 2d) + 8 m(p + d) - 8 m(p - d) + m(p - 2d)]/(12 d), exact
# for a vector up to quartic in p. The center point has weight zero and is
# never evaluated.
FIVE_POINT_OFFSETS = (2.0, 1.0, -1.0, -2.0)
FIVE_POINT_WEIGHTS = (-1.0/12.0, 8.0/12.0, -8.0/12.0, 1.0/12.0)


# The accuracy of the DESI Fisher derivatives, in two tiers. A derivative
# differences nearby vectors and so amplifies every numerical error that
# is not smooth in the parameter; the MCMC settings of the example yamls
# converge one chi2 (to 0.2), not a derivative.
#
#   "exploration": fast settings for exploration notebooks and quick
#       checks - the ones the Fisher notebooks used before the convergence
#       study (each layout's yaml settings, with the chi grid of Delta z =
#       0.05 raised to 2 x 32 and Delta z = 0.2 at fm.boost_accuracy 3).
#       Their FoMs are not converged: with three-point columns these
#       settings gave FoMs too high by about half at k_max = 0.075 on
#       Delta z = 0.05 and by a factor of about 3 on Delta d_com = 62 Mpc.
#   "final": the converged settings, for every figure that goes into a
#       paper or into a note's final version. The convergence study
#       tests/studies/run_fisher_derivatives.py chose them on the paper
#       model's errors, figures of merit and break-test shifts
#       (tests/studies/README.md); each is at or above the yamls' MCMC
#       value, CAMB included. The final tier needs CAMB built with the
#       halofit tolerance patch (check_halofit_patch).
#
# Per dataset: likelihood options ("likelihood_overrides"), CAMB extra
# arguments ("camb_overrides") and the overall boost of boost_accuracy
# ("accuracy_boost"), passed to ModelFisher (derivative_settings). The
# chi grid is the combined factor accuracyboost x nonlimber_accuracyboost,
# rounded up to a power of two in the C code; CAMB's transfer kmax is the
# likelihood's request kmax_boltzmann x accuracyboost [1/Mpc]. The final
# tier's chi grid is the same on every layout, 3 x 64 = 192 rounded up to
# 256: the coarsest grid at which the figures of merit do not depend on
# the stencil step.
DERIVATIVE_DATASETS = ("lssty1xdesi_desi_dr3", "lssty1xdesi_desi_dr1_dz0p02",
                       "lssty1xdesi_desi_dr1_dz0p05", "lssty1xdesi_desi_dr1_dz0p20")
DERIVATIVE_EXPLORATION = {
    "lssty1xdesi_desi_dr3": {
        "likelihood_overrides": {}, "camb_overrides": {}, "accuracy_boost": 1.0},
    "lssty1xdesi_desi_dr1_dz0p02": {
        "likelihood_overrides": {}, "camb_overrides": {}, "accuracy_boost": 1.0},
    "lssty1xdesi_desi_dr1_dz0p05": {
        "likelihood_overrides": {"nonlimber_accuracyboost": 32.0},
        "camb_overrides": {}, "accuracy_boost": 1.0},
    "lssty1xdesi_desi_dr1_dz0p20": {
        "likelihood_overrides": {"nonlimber_accuracyboost": 64.0},
        "camb_overrides": {}, "accuracy_boost": 3.0},
}
DERIVATIVE_OVERRIDES = {
    "lssty1xdesi_desi_dr3": {
        "accuracyboost": 3.0, "nonlimber_accuracyboost": 64.0,
        "integration_accuracy": 1, "kmax_boltzmann": 20.0},
    "lssty1xdesi_desi_dr1_dz0p02": {
        "accuracyboost": 3.0, "nonlimber_accuracyboost": 64.0,
        "integration_accuracy": 1, "kmax_boltzmann": 20.0},
    "lssty1xdesi_desi_dr1_dz0p05": {
        "accuracyboost": 3.0, "nonlimber_accuracyboost": 64.0,
        "integration_accuracy": 1, "kmax_boltzmann": 20.0},
    "lssty1xdesi_desi_dr1_dz0p20": {
        "accuracyboost": 3.0, "nonlimber_accuracyboost": 64.0,
        "integration_accuracy": 1, "kmax_boltzmann": 20.0},
}
DERIVATIVE_CAMB = {"AccuracyBoost": 2.0, "k_per_logint": 30}
DERIVATIVE_TIERS = ("exploration", "final")


def derivative_settings(dataset, tier):
    """The ModelFisher accuracy arguments of one dataset in one tier.

    Arguments:
      dataset = a DERIVATIVE_DATASETS stem.
      tier    = "exploration" or "final" (see DERIVATIVE_TIERS above).

    Returns:
      dict: "likelihood_overrides" (the tier's options plus data_file, the
      layout's paper dataset), "camb_overrides" and "accuracy_boost"
      (fresh copies), and "tier".

    Raises:
      KeyError naming the unknown tier or dataset.
    """
    if tier not in DERIVATIVE_TIERS:
        raise KeyError(f"tier {tier!r}: the tiers are {DERIVATIVE_TIERS}")
    if dataset not in DERIVATIVE_DATASETS:
        raise KeyError(f"dataset {dataset!r}: the datasets are {DERIVATIVE_DATASETS}")
    if tier == "exploration":
        spec = DERIVATIVE_EXPLORATION[dataset]
        out = {"likelihood_overrides": dict(spec["likelihood_overrides"]),
               "camb_overrides": dict(spec["camb_overrides"]),
               "accuracy_boost": float(spec["accuracy_boost"])}
    else:
        out = {"likelihood_overrides": dict(DERIVATIVE_OVERRIDES[dataset]),
               "camb_overrides": dict(DERIVATIVE_CAMB),
               "accuracy_boost": 1.0}
    # every Fisher forecast runs on the layout's paper dataset, whatever
    # example yaml provides the likelihood class (the *_linear datasets of
    # the exploration yamls serve only those examples' chi2)
    out["likelihood_overrides"]["data_file"] = dataset + ".dataset"
    out["tier"] = tier
    return out


def check_halofit_patch(camb_dir=None):
    """Stop unless the CAMB that cobaya runs carries the halofit tolerance patch.

    Stock CAMB stops halofit's bisection for the nonlinear scale at
    |sigma(R) - 1| <= 1e-3; where it stops jumps between nearby
    cosmologies, so the nonlinear P(k) jitters by about 0.3% and every
    finite-difference derivative of the cosmology moves with the step.
    Cocoa's PATCH_CAMB_HALOFIT_TOLERANCE key of set_installation_options.sh
    installs the 1e-7 tolerance; it is on by default in current Cocoa. An
    older install, or a CAMB clone made with the key off, lacks the patch.
    The final derivative tier requires it.

    The check reads the CAMB source cobaya builds from (the example yamls'
    camb path, external_modules/code/CAMB): its halofit.f90 must hold the
    1e-7 bisection test, and the compiled camb/camblib.so must be newer
    than that file, so a patched source that was never rebuilt fails too.
    A numerical smoothness test of P_NL would need steps large enough to
    cross a bisection stop, so it is slower and less certain than reading
    the source the library was built from.

    Arguments:
      camb_dir = the CAMB folder; None for $ROOTDIR/external_modules/code/CAMB.

    Returns:
      the path of the checked halofit.f90.

    Raises:
      RuntimeError naming the missing patch, or the library older than the
      patched source, and the repair.
    """
    import os
    if camb_dir is None:
        camb_dir = os.path.join(os.environ["ROOTDIR"], "external_modules", "code", "CAMB")
    source = os.path.join(camb_dir, "fortran", "halofit.f90")
    library = os.path.join(camb_dir, "camb", "camblib.so")
    with open(source) as handle:
        text = handle.read()
    if "abs(diff).le.1.d-7" not in text:
        raise RuntimeError(
            f"{source} has the stock halofit bisection tolerance (1e-3), so the "
            "final derivative tier cannot run. PATCH_CAMB_HALOFIT_TOLERANCE is "
            "on by default in current Cocoa; an older install, or a CAMB clone "
            "made without it, lacks the patch. Re-clone CAMB with "
            "OVERWRITE_EXISTING_CAMB_CODE=1 (installation_scripts/setup_camb.sh), "
            "then installation_scripts/compile_camb.sh")
    if not os.path.exists(library) or os.path.getmtime(library) < os.path.getmtime(source):
        raise RuntimeError(
            f"{library} is missing or older than the patched {source}: "
            "recompile CAMB (installation_scripts/compile_camb.sh)")
    return source


def five_point(vectors, step):
    """Combine the four stencil vectors into a five-point derivative.

    Arguments:
      vectors = the vectors at p + FIVE_POINT_OFFSETS[k]*step, in that
                order (four 1D arrays of one length).
      step    = the step d, in the units of the parameter.

    Returns:
      1D float64 array: dm/dp at p.

    Raises:
      ValueError when four vectors are not given or the step is not
      positive.
    """
    if len(vectors) != len(FIVE_POINT_OFFSETS):
        raise ValueError(f"five_point needs {len(FIVE_POINT_OFFSETS)} "
                         f"vectors (offsets {FIVE_POINT_OFFSETS}); got "
                         f"{len(vectors)}")
    if not step > 0.0:
        raise ValueError(f"five_point: the step must be positive; got {step}")
    total = np.zeros(len(vectors[0]), dtype=np.float64)
    for weight, vector in zip(FIVE_POINT_WEIGHTS, vectors):
        total += weight*np.asarray(vector, dtype=np.float64)
    return total/step


def boost_accuracy(info, block, boost):
    """Raise a model's accuracy the way the lsst_y1 notebook wrappers do.

    One overall knob AB, split between the two codes as
    cnu.get_camb_cosmology splits it:

      CAMB:      AccuracyBoost x c, kmax x (1 + 3 (c - 1)) and
                 k_per_logint + 30 (c - 1), with c = 1 + (AB - 1)/3
                 (a third of the excess: CAMB is the costly side);
      cosmolike: accuracyboost x AB (the likelihood's z and k table
                 grids refine with it too) and integration_accuracy
                 + |3 (AB - 1)|.

    nonlimber_accuracyboost, the chi grid of the thin bins, is a
    separate control with its own measured ladder in each likelihood
    yaml, and stays as set; the grid itself refines with the product
    accuracyboost x nonlimber_accuracyboost, so AB refines it too.

    In this project the CAMB kmax line does nothing: the likelihood asks
    CAMB for P(k) up to k_max = kmax_boltzmann x accuracyboost (40/Mpc
    at the yamls' 20 and 2), and cobaya runs CAMB to the larger of that
    request and the kmax extra argument. CAMB's k range therefore grows
    with AB through the cosmolike accuracyboost. To set the CAMB
    accuracy directly, pass ModelFisher's camb_overrides instead.

    Arguments:
      info  = the yaml's info dictionary (its camb block is changed).
      block = its likelihood block (changed in place).
      boost = AB; 1 leaves both untouched.
    """
    if boost == 1.0:
        return
    c = 1.0 + (boost - 1.0)/3.0
    camb = info["theory"]["camb"].setdefault("extra_args", {})
    camb["AccuracyBoost"] = float(camb.get("AccuracyBoost", 1.0))*c
    camb["kmax"] = float(camb.get("kmax", 10.0))*(1.0 + 3.0*(c - 1.0))
    camb["k_per_logint"] = int(camb.get("k_per_logint", 10)) + \
        int(30*(c - 1.0))
    block["accuracyboost"] = float(block.get("accuracyboost", 1.0))*boost
    block["integration_accuracy"] = int(block.get("integration_accuracy", 0)) \
        + int(abs(3*(boost - 1.0)))


def model_info(yaml_file, likelihood_overrides=None, accuracy_boost=1.0,
               camb_overrides=None, sampled_overrides=None):
    """The cobaya input and the fiducial point ModelFisher builds a model from.

    The yaml's info with the sampler and output blocks removed, then, in
    this order, boost_accuracy, the likelihood overrides, the CAMB
    overrides and the sampled overrides.

    Arguments:
      yaml_file and the four overrides = as for ModelFisher (its
      docstring explains each).

    Returns:
      (info, point): the input dictionary for cobaya's get_model and the
      fiducial point {name: value} (the evaluate override block, with the
      sampled overrides' values added).
    """
    info = yaml_load_file(yaml_file)
    point = dict(info["sampler"]["evaluate"]["override"])
    for key in ("sampler", "output", "force", "resume", "debug"):
        info.pop(key, None)
    names = list(info["likelihood"])
    if len(names) != 1:
        raise ValueError(f"{yaml_file} has the likelihoods {names}; "
                         "ModelFisher needs exactly one")
    block = info["likelihood"][names[0]]
    boost_accuracy(info, block, float(accuracy_boost))
    block.update(likelihood_overrides or {})
    if camb_overrides:
        camb = info["theory"]["camb"].setdefault("extra_args", {})
        camb.update(camb_overrides)
    if sampled_overrides:
        params_info = info.setdefault("params", {})
        for name, value in sampled_overrides.items():
            value = float(value)
            # a box of +-1e3 around the fiducial: wide enough for any
            # stencil step, and only a declaration (no prior is read)
            params_info[name] = {
                "prior": {"min": value - 1.0e3, "max": value + 1.0e3},
                "ref": value,
                "proposal": 1.0e-3,
            }
            point[name] = value
    return info, point


# The .dataset entries whose files do not enter a derivative: the data
# vector (the model vector is computed, not read) and the covariance (the
# Fisher matrices read it as it is when they are built).
DATASET_ENTRIES_NOT_HASHED = ("data_file", "cov_file")


def dataset_digest(path, data_file):
    """The sha256 of one likelihood's dataset, for cache checks.

    A layout's dataset is a .dataset file of "key = value" lines naming
    further files: the cross-pair list, the bin table, the mask, the n(z)
    and ell files. A derivative cache depends on their content, which can
    change under the same names (a pair added to the pair list), so this
    hashes, for every line, its key and either the bytes of the file it
    names or, for a plain value (a bin count), the value itself. File
    names do not enter: two datasets whose files hold the same bytes under
    other names (the paper dataset and its *_linear sibling) have one
    digest. The lines of DATASET_ENTRIES_NOT_HASHED are skipped, file and
    line alike.

    Arguments:
      path      = the likelihood's data folder, absolute or relative to
                  $ROOTDIR (the likelihood option "path").
      data_file = the .dataset file name in that folder.

    Returns:
      str: 64 hexadecimal digits.
    """
    import hashlib
    import os
    folder = path if os.path.isabs(path) else os.path.join(os.environ["ROOTDIR"], path)
    digest = hashlib.sha256()
    with open(os.path.join(folder, data_file)) as handle:
        lines = handle.read().splitlines()
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = (part.strip() for part in text.split("=", 1))
        if key in DATASET_ENTRIES_NOT_HASHED:
            continue
        named = os.path.join(folder, value)
        if os.path.isfile(named):
            with open(named, "rb") as handle:
                content = "file " + hashlib.sha256(handle.read()).hexdigest()
        else:
            content = "value " + value
        digest.update(f"{key} = {content}\n".encode())
    return digest.hexdigest()


def settings_fingerprint(yaml_file, likelihood_overrides=None,
                         accuracy_boost=1.0, camb_overrides=None,
                         sampled_overrides=None, tier=None):
    """A sha256 of everything that sets a model's vectors, for cache checks.

    A derivative cache is valid only for the model that made it. Comparing
    file names is not enough (a yaml edited under the same name, or a
    parameter fragment it names, changes the model silently), so the
    caches store this fingerprint: the sha256 of the fully resolved cobaya
    input (cobaya's update_info merges the likelihood's own yaml and its
    parameter fragments into the example yaml, as get_model does, without
    building anything) together with the fiducial point and the content of
    every likelihood's dataset files (dataset_digest: the pair list, the
    bin table, the mask, the n(z) and ell files). Any change of a
    likelihood option, a CAMB argument, a prior, a tie between parameters,
    the fiducial point or a dataset file changes it; moving or renaming a
    file whose content stays the same does not.

    Arguments:
      the first five arguments of ModelFisher, with the same meaning;
      tier = the derivative tier the settings belong to ("exploration" or
             "final", see DERIVATIVE_TIERS), hashed with them so a cache
             states which tier made it; None for settings of no tier.

    Returns:
      str: 64 hexadecimal digits.
    """
    import hashlib
    import json
    from cobaya.input import update_info
    info, point = model_info(yaml_file,
                             likelihood_overrides=likelihood_overrides,
                             accuracy_boost=accuracy_boost,
                             camb_overrides=camb_overrides,
                             sampled_overrides=sampled_overrides)
    resolved = update_info(info)
    # each likelihood's data_file enters through the content of its dataset
    # (dataset_digest), not through its name
    datasets = {}
    likelihoods = {}
    for name, options in (resolved.get("likelihood") or {}).items():
        if isinstance(options, dict) and options.get("data_file") and options.get("path"):
            datasets[name] = dataset_digest(options["path"], options["data_file"])
            options = {key: value for key, value in options.items() if key != "data_file"}
        likelihoods[name] = options
    content = {"likelihood": likelihoods,
               "theory": resolved.get("theory"),
               "params": resolved.get("params"),
               "prior": resolved.get("prior"),
               "point": point,
               "tier": tier,
               "datasets": datasets}
    # sort_keys fixes the order; default=str writes numpy scalars and any
    # other non-JSON value by its text
    text = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


class ModelFisher:
    """The Fisher machinery bound to one cobaya model of this project."""

    def __init__(self, yaml_file, params, extra_prefixes=(),
                 likelihood_overrides=None, invcov_scale=1.0,
                 accuracy_boost=1.0, camb_overrides=None,
                 sampled_overrides=None):
        """Build the model of an EXAMPLE yaml at its fiducial point.

        Arguments:
          yaml_file = path of the EXAMPLE yaml; its evaluate sampler's
                      override block is the fiducial point.
          params    = names of the varied parameters, in the Fisher
                      matrix order: sampled parameters of the model, or
                      "w0" and "wa" (see the module docstring).
          extra_prefixes = name prefixes (e.g. ("DESI_",)): every other
                      sampled parameter starting with one of them is
                      varied too, after params, in the model's order -
                      the way to marginalize over all nuisances.
          likelihood_overrides = optional {option: value} set in the
                      yaml's single likelihood block (e.g. a
                      nonlimber_accuracyboost).
          invcov_scale = factor on the inverse covariance. A Gaussian
                      covariance scales as 1/f_sky, so moving a survey to
                      another footprint multiplies the inverse by
                      area_new/area_old.
          accuracy_boost = the overall accuracy of the evaluations (see
                      boost_accuracy); derivatives amplify numerical
                      noise, so they want more than a single chi2.
          camb_overrides = optional {argument: value} set in the camb
                      theory's extra_args after boost_accuracy (e.g.
                      {"AccuracyBoost": 1.5, "k_per_logint": 30}): the
                      CAMB accuracy of the evaluations, set directly.
          sampled_overrides = optional {name: fiducial value}: each named
                      parameter becomes a sampled parameter at that
                      value, replacing a fixed value or a value function
                      of the likelihood's parameter files (cobaya's
                      merge drops the old entry). The per-bin parameters
                      of the one-loop model (DESI_B1_<n>, DESI_DB2_<n>,
                      ...) are tied to group parameters there; sampled,
                      a derivative steps one bin, or one set of bins,
                      through the likelihood itself. The flat prior they
                      get is never evaluated (dv calls Model.loglike).
        """
        info, point = model_info(yaml_file,
                                 likelihood_overrides=likelihood_overrides,
                                 accuracy_boost=accuracy_boost,
                                 camb_overrides=camb_overrides,
                                 sampled_overrides=sampled_overrides)
        (self.likelihood_name,) = info["likelihood"]
        self.accuracy_boost = float(accuracy_boost)
        self.model = get_model(info)
        _ACTIVE[0] = self
        sampled = list(self.model.parameterization.sampled_params())
        missing = [k for k in sampled if k not in point]
        if missing:
            raise ValueError(f"{yaml_file}: the override block has no "
                             f"value for {missing}")
        self.point = {k: point[k] for k in sampled}
        self.names = list(params)
        self.names += [k for k in sampled if k not in self.names
                       and k.startswith(tuple(extra_prefixes))]
        for name in self.names:
            if name not in sampled and name not in ("w0", "wa"):
                raise ValueError(f"{name} is not a sampled parameter of "
                                 f"{self.likelihood_name}")
        self.fiducial = np.array([self._get(self.point, n)
                                  for n in self.names])
        self.invcov_scale = float(invcov_scale)
        sigma = self.prior_sigmas()
        self.zero_scale = np.array([
            sigma[i]/H_DEFAULT if i in sigma else 1.0
            for i in range(len(self.names))])
        like = self.model.likelihood[self.likelihood_name]
        if getattr(like, "binning_space", "real") != "fourier":
            self._getter = ci.compute_data_vector_masked
        elif getattr(like, "gg_cross_pairs", None) is not None:
            self._getter = ci.compute_data_vector_fourier_gg_cross_masked
        else:
            self._getter = ci.compute_data_vector_fourier_masked

    def _require_active(self):
        """Refuse when another ModelFisher owns the interface's state."""
        if _ACTIVE[0] is not self:
            raise RuntimeError(
                "another ModelFisher replaced this model's interface "
                "state; rebuild this one before using it")

    @staticmethod
    def _get(point, name):
        """A varied parameter's value at a point (w0/wa from w, w0pwa)."""
        if name == "w0":
            return point["w"]
        if name == "wa":
            return point["w0pwa"] - point["w"]
        return point[name]

    def _point_at(self, param):
        """The model point with the varied parameters set to param."""
        point = dict(self.point)
        values = dict(zip(self.names, param))
        for name, value in values.items():
            if name not in ("w0", "wa"):
                point[name] = float(value)
        if "w0" in values or "wa" in values:
            w0 = values.get("w0", self._get(self.point, "w0"))
            wa = values.get("wa", self._get(self.point, "wa"))
            point["w"] = float(w0)
            point["w0pwa"] = float(w0 + wa)
        return point

    def dv(self, param=None, AccuracyBoost=1.0):
        """The masked data vector at a flat parameter vector.

        Arguments:
          param = 1D array in the order of self.names, or None for the
                  fiducial.
          AccuracyBoost = accepted for the shared machinery's calling
                  convention and ignored: the model's accuracy is fixed
                  by its yaml (likelihood_overrides at construction).

        Returns:
          1D float64 array: the masked vector (masked entries 0).
        """
        self._require_active()
        if param is None:
            param = self.fiducial
        point = self._point_at(param)
        loglike = self.model.loglike(point, return_derived=False,
                                     cached=False)
        if not np.isfinite(loglike):
            raise RuntimeError(f"{self.likelihood_name}: the likelihood "
                               f"failed at {point}")
        # a likelihood that adds terms to the compiled-interface vector
        # (desi_cl's model A stochastic term) keeps its full vector in
        # last_datavector; the others are the interface vector itself
        like = self.model.likelihood[self.likelihood_name]
        stored = getattr(like, "last_datavector", None)
        if stored is not None:
            return np.array(stored, dtype=np.float64)
        return np.array(self._getter(), dtype=np.float64)

    def ddv(self, index=0, h=H_DEFAULT, CV=None, AccuracyBoost=1.0):
        """Five-point-stencil derivative along one parameter.

        cnu.get_ddv on the parameter measured in units of its zero
        scale (see the module docstring): a zero fiducial then takes
        the absolute step h s, and the result is divided by s.
        """
        CV = self.fiducial if CV is None else np.asarray(CV, dtype=float)
        s = self.zero_scale[index] if CV[index] == 0.0 else 1.0
        scaled = np.array(CV, dtype=np.float64, copy=True)
        scaled[index] /= s

        def dv_scaled(param, AccuracyBoost=1.0):
            p = np.array(param, dtype=np.float64, copy=True)
            p[index] *= s
            return self.dv(param=p, AccuracyBoost=AccuracyBoost)

        return cnu.get_ddv(dv_scaled, index=index, h=h, CV=scaled,
                           AccuracyBoost=AccuracyBoost)/s

    def prior_sigmas(self):
        """{index: sigma} of the varied parameters' normal priors."""
        params_info = self.model.info()["params"]
        out = {}
        for i, name in enumerate(self.names):
            prior = (params_info.get(name) or {}).get("prior")
            if isinstance(prior, dict) and \
                    prior.get("dist") in ("norm", "truncnorm"):
                out[i] = float(prior["scale"])
        return out

    def priors(self):
        """{index: (mean, sigma)} of the normal priors (cnu's format)."""
        return {i: (float(self.fiducial[i]), sigma)
                for i, sigma in self.prior_sigmas().items()}

    def invcov(self):
        """The masked inverse covariance, times invcov_scale."""
        self._require_active()
        return np.array(ci.get_inv_cov_masked())*self.invcov_scale

    def derivatives(self, h=H_DEFAULT):
        """D: one column of d(data vector)/d(parameter) per name."""
        return np.column_stack([self.ddv(index=i, h=h)
                                for i in range(len(self.names))])

    def fisher(self, h=H_DEFAULT, D=None, priors=True):
        """F = D^T C^-1 D, plus the model's normal priors.

        Arguments:
          h      = relative stencil step (see the module docstring).
          D      = optional derivative matrix from derivatives(h), to
                   reuse one set of evaluations.
          priors = False leaves the Gaussian priors out (the data-only
                   information, e.g. to combine data sets that share a
                   parameter and its prior only once).

        Returns:
          2D array (len(names), len(names)) in the order of self.names.
        """
        if D is None:
            D = self.derivatives(h=h)
        F = D.T @ (self.invcov() @ D)
        if priors and self.prior_sigmas():
            F = cnu.add_gaussian_priors(F=F, priors=self.priors())
        return F

    def bias_vector(self, D, delta):
        """B = D^T C^-1 delta, the numerator of the Fisher bias.

        Arguments:
          D     = derivative matrix from derivatives(h).
          delta = m_alt - m_ref: the masked vector of an alternative
                  model minus this model's, both at the fiducial point.

        Returns:
          1D array, one entry per name (see fisher_bias).
        """
        return D.T @ (self.invcov() @ delta)

    def delta_chi2(self, delta):
        """delta^T C^-1 delta: the chi2 an alternative model's vector
        scores against this model's."""
        return float(delta @ (self.invcov() @ delta))


def fisher_bias(F, names, *shifts):
    """Parameter shifts from fitting a model to another model's vector.

    Fitting this module's model to data drawn from an alternative model
    (a different non-Limber engine, Limber instead of non-Limber) moves
    the best fit, to linear order in the vector difference delta, by

        delta_p = F^-1 B,  B = D^T C^-1 delta

    (Huterer et al. 2006; Amara & Refregier 2008), with F the Fisher
    matrix of the fitted model. The nuisance parameters are part of F,
    so they absorb what they can of delta and the shift of each
    cosmological parameter is the marginalized one. Gaussian priors
    centered on the fiducial enter F but not B. The linear order holds
    while delta_p stays well inside the 1-sigma errors; a larger shift
    signals the alternative is excluded, with a size only indicative.

    Arguments:
      F      = Fisher matrix over names (one block or combine's sum).
      names  = its parameter names.
      *shifts = (B, block_names) pairs from ModelFisher.bias_vector:
               the blocks' numerators add, as their information does
               with a block-diagonal covariance.

    Returns:
      {name: delta_p} over names.
    """
    index = {n: k for k, n in enumerate(names)}
    B = np.zeros(len(names))
    for b, block_names in shifts:
        B[[index[n] for n in block_names]] += b
    shift = np.linalg.solve(0.5*(F + F.T), B)
    return dict(zip(names, (float(x) for x in shift)))


def combine(*blocks):
    """Sum Fisher matrices of independent data sets on their shared names.

    With a block-diagonal covariance the joint Fisher matrix is the sum
    of the separate ones: a parameter several blocks vary (cosmology)
    collects every block's information, a parameter only one block
    varies (its nuisances) keeps that block's rows. A Gaussian prior
    belongs in exactly one block, or it is counted twice.

    Arguments:
      *blocks = (F, names) pairs.

    Returns:
      (F, names): the joint matrix over the union of the names, in
      order of first appearance.
    """
    names = []
    for _, block_names in blocks:
        names.extend(n for n in block_names if n not in names)
    index = {n: k for k, n in enumerate(names)}
    total = np.zeros((len(names), len(names)))
    for F, block_names in blocks:
        where = [index[n] for n in block_names]
        total[np.ix_(where, where)] += F
    return total, names


def marginalized_sigma(F, names, keep):
    """1-sigma errors of the parameters `keep`, marginalized over the rest."""
    cov = np.linalg.inv(0.5*(F + F.T))
    return {n: float(np.sqrt(cov[names.index(n), names.index(n)]))
            for n in keep}
