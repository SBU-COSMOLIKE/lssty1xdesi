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

Caches. The notebooks keep their expensive vectors and derivatives in
cache files, and four functions at the end of this module move them,
each called explicitly in its own notebook cell: load_fisher_from_cache
copies an archived final-tier cache (fisher_caches/, git-lfs) to its
working copy in chains/, read_fisher_cache reads a working copy after
checking its settings and fingerprints against the notebook's,
write_fisher_cache writes a freshly computed result, and
save_fisher_to_cache archives a final-tier working copy. Nothing loads,
computes or writes a cache on its own, and nothing replaces a file
without overwrite=True. The files are .npz archives of plain numpy
arrays with a JSON header: no pickles, so reading a cache runs no code.
The header carries a provenance record (the resolved cobaya input of
every model, the dataset files' checksums, the code's git state), which
the archive also keeps as fisher_caches/<name>.json.
"""
import datetime
import hashlib
import json
import os
import re
import shutil
import uuid

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


def _resolve_model(yaml_file, likelihood_overrides=None, accuracy_boost=1.0,
                   camb_overrides=None, sampled_overrides=None, tier=None):
    """Return the resolved cobaya input of one model and its datasets.

    cobaya's update_info merges the likelihood's own yaml and its
    parameter fragments into the example yaml, as get_model does, and fills
    in every component's defaults (the theory code's options among them),
    without building anything.

    Arguments:
      the arguments of settings_fingerprint.

    Returns:
      (content, datasets):
        content  = dict, what settings_fingerprint hashes: "likelihood"
                   (every likelihood's options, without data_file),
                   "theory", "params" (every parameter: sampled with its
                   prior, fixed with its value, derived), "prior", "point"
                   (the fiducial point), "tier" and "datasets" (each
                   likelihood's dataset_digest);
        datasets = {likelihood name: {"path": ..., "data_file": ...}}.
    """
    from cobaya.input import update_info
    info, point = model_info(yaml_file,
                             likelihood_overrides=likelihood_overrides,
                             accuracy_boost=accuracy_boost,
                             camb_overrides=camb_overrides,
                             sampled_overrides=sampled_overrides)
    resolved = update_info(info)
    # each likelihood's data_file enters through the content of its dataset
    # (dataset_digest), not through its name
    digests = {}
    datasets = {}
    likelihoods = {}
    for name, options in (resolved.get("likelihood") or {}).items():
        if isinstance(options, dict) and options.get("data_file") and options.get("path"):
            digests[name] = dataset_digest(options["path"], options["data_file"])
            datasets[name] = {"path": options["path"], "data_file": options["data_file"]}
            options = {key: value for key, value in options.items() if key != "data_file"}
        likelihoods[name] = options
    content = {"likelihood": likelihoods,
               "theory": resolved.get("theory"),
               "params": resolved.get("params"),
               "prior": resolved.get("prior"),
               "point": point,
               "tier": tier,
               "datasets": digests}
    return content, datasets


def _content_sha256(content):
    """Return the sha256 of a resolved input's sorted JSON text."""
    # sort_keys fixes the order; default=str writes numpy scalars and any
    # other non-JSON value by its text
    text = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def settings_fingerprint(yaml_file, likelihood_overrides=None,
                         accuracy_boost=1.0, camb_overrides=None,
                         sampled_overrides=None, tier=None):
    """A sha256 of everything that sets a model's vectors, for cache checks.

    A derivative cache is valid only for the model that made it. Comparing
    file names is not enough (a yaml edited under the same name, or a
    parameter fragment it names, changes the model silently), so the
    caches store this fingerprint: the sha256 of the fully resolved cobaya
    input (_resolve_model: cobaya's update_info merges the likelihood's own
    yaml and its parameter fragments into the example yaml, as get_model
    does, without building anything) together with the fiducial point and
    the content of every likelihood's dataset files (dataset_digest: the
    pair list, the bin table, the mask, the n(z) and ell files). Any change
    of a likelihood option, a CAMB argument, a prior, a tie between
    parameters, the fiducial point or a dataset file changes it; moving or
    renaming a file whose content stays the same does not.

    Arguments:
      the first five arguments of ModelFisher, with the same meaning;
      tier = the derivative tier the settings belong to ("exploration" or
             "final", see DERIVATIVE_TIERS), hashed with them so a cache
             states which tier made it; None for settings of no tier.

    Returns:
      str: 64 hexadecimal digits.
    """
    content, _ = _resolve_model(yaml_file,
                                likelihood_overrides=likelihood_overrides,
                                accuracy_boost=accuracy_boost,
                                camb_overrides=camb_overrides,
                                sampled_overrides=sampled_overrides,
                                tier=tier)
    return _content_sha256(content)


def dataset_files(path, data_file):
    """Return every file of one likelihood's dataset with its sha256.

    The provenance record of a cache lists them: the .dataset file itself
    and, for each of its "key = value" lines, the file it names (pair list,
    bin table, mask, n(z), ell file, data vector, covariance) with its
    sha256, or the plain value. "in_fingerprint" says whether the line
    enters dataset_digest (the data vector and the covariance do not:
    DATASET_ENTRIES_NOT_HASHED).

    Arguments:
      path      = the likelihood's data folder, absolute or relative to
                  $ROOTDIR (the likelihood option "path").
      data_file = the .dataset file name in that folder.

    Returns:
      dict: "dataset_file" {"name", "sha256"} and "entries" {key: {"file",
      "sha256", "in_fingerprint"} or {"value", "in_fingerprint"}}.
    """
    folder = path if os.path.isabs(path) else os.path.join(os.environ["ROOTDIR"], path)
    dataset_path = os.path.join(folder, data_file)
    entries = {}
    with open(dataset_path) as handle:
        lines = handle.read().splitlines()
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = (part.strip() for part in text.split("=", 1))
        named = os.path.join(folder, value)
        hashed = key not in DATASET_ENTRIES_NOT_HASHED
        if os.path.isfile(named):
            entries[key] = {"file": value, "sha256": _sha256(named),
                            "in_fingerprint": hashed}
        else:
            entries[key] = {"value": value, "in_fingerprint": hashed}
    return {"dataset_file": {"name": data_file, "sha256": _sha256(dataset_path)},
            "entries": entries}


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


# ---------------------------------------------------------------------------
# Caches: copied, read, written and archived only by explicit calls
# ---------------------------------------------------------------------------
#
# The Fisher notebooks cache their expensive vectors and derivatives. No
# code here decides on its own whether to load or to compute: a notebook
# cell calls one of four functions, so every load and every save shows
# intent, and a changed setting can never reuse or overwrite a cache
# unnoticed.
#
#   load_fisher_from_cache(name)   copies the archived final-tier cache
#                                  fisher_caches/<name>.npz (git-lfs) to the
#                                  working copy chains/<name>.npz;
#   read_fisher_cache(name, ...)   reads the working copy, checks its
#                                  settings, tier and fingerprints against
#                                  the notebook's, and raises on a mismatch;
#   write_fisher_cache(result, name, settings)
#                                  writes a freshly computed result to the
#                                  working copy chains/<name>.npz;
#   save_fisher_to_cache(name)     copies a final-tier working copy into the
#                                  archive fisher_caches/.
#
# A copy or a write refuses to replace an existing file unless the call
# passes overwrite=True. Only save_fisher_to_cache changes the archive, and
# fisher_caches/checksums.json records the sha256 of every archived file,
# which load_fisher_from_cache verifies. (sha256 is a 64-digit checksum of a
# file's bytes: two files with the same checksum hold the same bytes.)
#
# The file format, for the archive and the working copies alike: an .npz
# file (a zip archive of numpy arrays, np.savez_compressed) of plain arrays,
# read with allow_pickle=False, plus one JSON header array. A pickle can
# run code when it is loaded and depends on Python's object layout; plain
# arrays and JSON text do neither, so reading a cache runs no code. A cache
# is a nested structure of dicts, lists, tuples, arrays and scalars. Every
# array, numpy scalars included, is stored under a flat key, its path in
# the structure joined by "/" (result/ops/b1, settings/Delta z = 0.2/...);
# the header's "structure" tree holds everything else and rebuilds the
# structure exactly: the same dict keys in the same order, the same Python
# and numpy types, every float bit for bit. For a reader of the file alone,
# the header also lists the tier and fingerprint entries of the settings,
# the parameter names, the shape and dtype of every array, the cache's name
# and the date it was written.
#
# The provenance record. Every header also holds a record of how the cache
# was made, and save_fisher_to_cache writes the same record next to the
# archived file as fisher_caches/<name>.json (plain git, readable on
# GitHub): for every model the settings name with a fingerprint, the fully
# resolved cobaya input (the example yaml merged with the likelihood's
# options and every parameter file, the theory block with all its options,
# every parameter's prior or value, the fiducial point, the accuracy
# overrides) and the sha256 of every dataset file; the stored settings
# (tier, stencil, steps); the git commit and dirty flag of the
# repositories, the CAMB version, the halofit patch, the Python and numpy
# versions; the date. write_fisher_cache builds it from the files at write
# time and refuses to write when a stored fingerprint no longer matches
# them.

# the project folder: this file is interface/cosmolike_lssty1xdesi_fisher.py
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the archive of final-tier caches (its .npz files go through git-lfs) and
# the table of their checksums (plain git)
CACHE_ARCHIVE = os.path.join(PROJECT_DIR, "fisher_caches")
CACHE_CHECKSUMS = os.path.join(CACHE_ARCHIVE, "checksums.json")
# the working copies the notebooks read and write (chains/ is gitignored)
CACHE_WORKING = os.path.join(PROJECT_DIR, "chains")
# the format tag of every cache header; a reader refuses any other tag
CACHE_FORMAT = "lssty1xdesi-fisher-cache-1"
# the .npz entry of the JSON header; every array key starts with "result"
# or "settings", so no array can take this name
CACHE_HEADER_KEY = "__header__"
# A cache name is lower case letters, digits and underscores, e.g.
# model_derivatives_dz0p05_final: what it holds, the layout, the tier.
CACHE_NAME_PATTERN = r"[a-z0-9_]+"
# The first bytes of a git-lfs pointer: the small text file git leaves in
# place of a large file that was never downloaded (git lfs pull fetches it).
LFS_POINTER_START = b"version https://git-lfs.github.com/spec/v1"
# the settings entries the header lists for a reader of the file alone
HEADER_MARK_KEYS = ("tier", "fingerprint", "fingerprints")
HEADER_NAME_KEYS = ("names",)
# the format tag of the provenance record (header and fisher_caches/*.json)
PROVENANCE_FORMAT = "lssty1xdesi-fisher-provenance-1"
# The record's notes when the caller gives none: a cache computed and
# written in one session. A cache converted from an older file says how.
PROVENANCE_NOTES = {
    "computed": "in the session that wrote the cache, before the write",
    "resolved_input": "resolved from the files at write time; every stored "
                      "fingerprint verified equal",
    "code": "the repositories at write time (code_at_write)",
}
# a git call that takes longer than this many seconds is reported, not
# waited for
GIT_TIMEOUT_SECONDS = 60


def _cache_path(folder, name):
    """Return the path folder/<name>.npz of one cache.

    Arguments:
      folder = the folder of the file.
      name   = the bare cache name (CACHE_NAME_PATTERN).

    Raises:
      ValueError when name is not a bare lower-case cache name.
    """
    if not isinstance(name, str) or re.fullmatch(CACHE_NAME_PATTERN, name) is None:
        raise ValueError(
            f"cache name {name!r}: a cache name is lower case letters, digits and "
            "underscores, without a folder or the .npz suffix (for example "
            "'model_derivatives_dz0p05_final')")
    # fisher_caches/checksums.json shares the folder with the records
    # fisher_caches/<name>.json
    if name == "checksums":
        raise ValueError("the cache name 'checksums' is taken by "
                         "fisher_caches/checksums.json: choose another name")
    return os.path.join(folder, name + ".npz")


def _short_path(path):
    """Return path relative to the project folder when it lies inside it."""
    if path.startswith(PROJECT_DIR + os.sep):
        return os.path.relpath(path, PROJECT_DIR)
    return path


def _encode_key(key, where):
    """Return a dict key as a JSON pair [kind, value].

    The kinds are none, bool, int, float, str and, for a numpy scalar key,
    [numpy, dtype, text]. A float is stored as repr(x), the shortest text
    that reads back as the same float bit for bit.

    Arguments:
      key   = the dict key.
      where = its path in the structure, for the error message.

    Raises:
      TypeError for any other key type.
    """
    if key is None:
        return ["none", None]
    # numpy scalars first: np.float64 is a subclass of float and np.bool_
    # behaves like bool, so the plain checks below would lose the dtype
    if isinstance(key, np.generic):
        kind = key.dtype.kind
        if kind == "f":
            return ["numpy", key.dtype.str, repr(float(key))]
        if kind in ("i", "u"):
            return ["numpy", key.dtype.str, str(int(key))]
        if kind == "b":
            return ["numpy", key.dtype.str, "1" if key else "0"]
        if kind == "U":
            return ["numpy", key.dtype.str, str(key)]
        raise TypeError(f"cache entry {where}: a dict key of numpy type "
                        f"{key.dtype} cannot be stored")
    # bool before int: True and False are ints to isinstance
    if isinstance(key, bool):
        return ["bool", key]
    if isinstance(key, int):
        return ["int", key]
    if isinstance(key, float):
        return ["float", repr(key)]
    if isinstance(key, str):
        return ["str", key]
    raise TypeError(f"cache entry {where}: a dict key of type {type(key).__name__} "
                    "cannot be stored (keys are str, int, float, bool or None)")


def _decode_key(pair):
    """Return the dict key of one JSON pair of _encode_key."""
    kind = pair[0]
    if kind == "none":
        return None
    if kind == "bool":
        return bool(pair[1])
    if kind == "int":
        return int(pair[1])
    if kind == "float":
        return float(pair[1])
    if kind == "str":
        return str(pair[1])
    if kind == "numpy":
        numpy_type = np.dtype(pair[1]).type
        text = pair[2]
        if np.dtype(pair[1]).kind == "f":
            return numpy_type(float(text))
        if np.dtype(pair[1]).kind == "b":
            return numpy_type(text == "1")
        if np.dtype(pair[1]).kind == "U":
            return numpy_type(text)
        return numpy_type(int(text))
    raise ValueError(f"the cache header has a dict key of unknown kind {kind!r}")


def _array_key(path, arrays):
    """Return the flat .npz key of the array at path, unique among arrays.

    The key is the path joined by "/"; a "/" inside a dict key becomes "|".
    Two paths that give one key (keys "a/b" and "a|b") get "#2", "#3", ...
    appended, and the structure tree records the key actually used.
    """
    parts = []
    for part in path:
        parts.append(part.replace("/", "|"))
    base = "/".join(parts)
    key = base
    count = 1
    while key in arrays:
        count += 1
        key = f"{base}#{count}"
    return key


def _encode(value, path, arrays):
    """Return the structure node of value; its arrays go into `arrays`.

    Arguments:
      value  = the cached structure: dicts, lists, tuples, numpy arrays,
               numpy scalars, None, bool, int, float, str.
      path   = list of str, the path of value in the structure.
      arrays = {flat key: numpy array}, filled in place.

    Returns:
      dict, the JSON node: {"t": kind, ...}.

    Raises:
      TypeError naming the path of an entry that cannot be stored (an
      object array, an array subclass, a set, a custom object).
    """
    where = "/".join(path)
    if isinstance(value, dict):
        keys = []
        values = []
        for key, item in value.items():
            keys.append(_encode_key(key, where))
            values.append(_encode(item, path + [str(key)], arrays))
        return {"t": "dict", "k": keys, "v": values}
    if isinstance(value, (list, tuple)):
        items = []
        for index, item in enumerate(value):
            items.append(_encode(item, path + [str(index)], arrays))
        if isinstance(value, tuple):
            return {"t": "tuple", "v": items}
        return {"t": "list", "v": items}
    if isinstance(value, np.ndarray):
        # type() and not isinstance(): a subclass (np.matrix, a masked
        # array) would come back as a plain array and change behavior
        if type(value) is not np.ndarray or value.dtype.hasobject:
            raise TypeError(f"cache entry {where}: a {type(value).__name__} of dtype "
                            f"{value.dtype} cannot be stored without a pickle; store "
                            "a plain numpy array of numbers or strings")
        key = _array_key(path, arrays)
        arrays[key] = value
        return {"t": "array", "key": key}
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise TypeError(f"cache entry {where}: a numpy object scalar cannot be "
                            "stored")
        # a numpy scalar is stored as a 0-d array of its dtype, so it reads
        # back as the same numpy type and bits ([()] takes the scalar out)
        key = _array_key(path, arrays)
        arrays[key] = np.asarray(value)
        return {"t": "numpy_scalar", "key": key}
    if value is None:
        return {"t": "none"}
    # bool before int: True and False are ints to isinstance
    if isinstance(value, bool):
        return {"t": "bool", "v": value}
    if isinstance(value, int):
        return {"t": "int", "v": value}
    if isinstance(value, float):
        # repr is the shortest text that reads back as the same float
        # (nan, inf and -0.0 included), which JSON numbers do not promise
        return {"t": "float", "v": repr(value)}
    if isinstance(value, str):
        return {"t": "str", "v": value}
    raise TypeError(f"cache entry {where} has the type {type(value).__name__}, which "
                    "a cache cannot store (dicts, lists, tuples, numpy arrays and "
                    "scalars, None, bool, int, float and str)")


def _decode(node, arrays):
    """Return the structure a node of _encode describes.

    Arguments:
      node   = a JSON node of the header's structure tree.
      arrays = {flat key: numpy array} read from the file.
    """
    kind = node["t"]
    if kind == "dict":
        out = {}
        for pair, item in zip(node["k"], node["v"]):
            out[_decode_key(pair)] = _decode(item, arrays)
        return out
    if kind == "list":
        items = []
        for item in node["v"]:
            items.append(_decode(item, arrays))
        return items
    if kind == "tuple":
        items = []
        for item in node["v"]:
            items.append(_decode(item, arrays))
        return tuple(items)
    if kind == "array":
        return arrays[node["key"]]
    if kind == "numpy_scalar":
        return arrays[node["key"]][()]
    if kind == "none":
        return None
    if kind == "bool":
        return bool(node["v"])
    if kind == "int":
        return int(node["v"])
    if kind == "float":
        return float(node["v"])
    if kind == "str":
        return str(node["v"])
    raise ValueError(f"the cache header has a structure node of unknown kind {kind!r}")


def _collect(value, path, wanted, found):
    """Collect the non-array entries of value whose dict key is in wanted.

    Arguments:
      value  = a cached structure.
      path   = list of str, its path.
      wanted = the dict keys to collect.
      found  = {path joined by "/": entry}, filled in place.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            here = path + [str(key)]
            if key in wanted and not isinstance(item, np.ndarray):
                found["/".join(here)] = item
            else:
                _collect(item, here, wanted, found)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect(item, path + [str(index)], wanted, found)


def _partial_path(folder, stem):
    """Return a new, unused temporary path in folder for a file being written.

    The name starts with a dot and ends with .partial, and holds the
    process id and a random part, so two writers never share it. The file
    is then created with open(..., "xb") (which fails if the file exists);
    a file created that way gets the user's default permissions (the
    umask), so a cache is as readable as any other file of the project.
    """
    return os.path.join(folder, f".{stem}.{os.getpid()}.{uuid.uuid4().hex}.partial")


def _write_npz(path, name, result, settings, record):
    """Write one cache file: the arrays and the JSON header.

    The file is written to a temporary name in the same folder and then
    renamed onto path (os.replace, one step on one file system), so a
    failure part way leaves no half-written file under a cache name.

    Arguments:
      path     = the target .npz file.
      name     = the cache name, stored in the header.
      result   = the computed structure.
      settings = the settings structure it was computed with.
      record   = the provenance record (_provenance_record).
    """
    arrays = {}
    structure = _encode({"result": result, "settings": settings}, [], arrays)
    marks = {}
    _collect(settings, ["settings"], HEADER_MARK_KEYS, marks)
    names = {}
    _collect(result, ["result"], HEADER_NAME_KEYS, names)
    shapes = {}
    for key, array in arrays.items():
        shapes[key] = {"shape": list(array.shape), "dtype": array.dtype.str}
    header = {
        "format": CACHE_FORMAT,
        "name": name,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tiers_and_fingerprints": marks,
        "parameter_names": names,
        "arrays": shapes,
        "structure": structure,
        "provenance": record,
    }
    # default=str writes a numpy value inside the listed marks or names as
    # text; the structure tree itself holds no numpy values
    entries = dict(arrays)
    entries[CACHE_HEADER_KEY] = np.array(json.dumps(header, default=str))
    partial = _partial_path(os.path.dirname(path), name)
    try:
        with open(partial, "xb") as stream:
            # a file object keeps savez_compressed from appending .npz
            np.savez_compressed(stream, **entries)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def _read_npz(path, header_only=False):
    """Read one cache file.

    np.load with allow_pickle=False reads plain arrays only: a file that
    would need a pickle raises instead of running code.

    Arguments:
      path        = the .npz file.
      header_only = True to read the header alone (the lazy .npz reader
                    then decompresses no array).

    Returns:
      header, or (result, settings, header) when header_only is False.

    Raises:
      ValueError when the file is a git-lfs pointer, is not a cache file
      of CACHE_FORMAT, or its arrays differ from the header's list.
    """
    with open(path, "rb") as stream:
        start = stream.read(len(LFS_POINTER_START))
    if start == LFS_POINTER_START:
        raise ValueError(f"{path} is a git-lfs pointer, not the cache: download the "
                         "file with git lfs pull (git-lfs on PATH), then copy it "
                         "again with load_fisher_from_cache(..., overwrite=True)")
    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{path} is not a readable .npz cache file ({error})") from error
    # the with block closes the zip file when it ends, also after an error
    with data:
        if CACHE_HEADER_KEY not in data.files:
            raise ValueError(f"{path} has no {CACHE_HEADER_KEY} entry: it is not a "
                             f"{CACHE_FORMAT} file")
        header = json.loads(data[CACHE_HEADER_KEY].item())
        if header.get("format") != CACHE_FORMAT:
            raise ValueError(f"{path} has the format {header.get('format')!r}; this "
                             f"module reads {CACHE_FORMAT!r} only")
        if header_only:
            return header
        arrays = {}
        for key in data.files:
            if key != CACHE_HEADER_KEY:
                arrays[key] = data[key]
    listed = header["arrays"]
    if sorted(arrays) != sorted(listed):
        raise ValueError(f"{path}: the file holds {len(arrays)} arrays, its header "
                         f"lists {len(listed)}; the file is damaged")
    for key, array in arrays.items():
        if list(array.shape) != listed[key]["shape"] or \
                array.dtype.str != listed[key]["dtype"]:
            raise ValueError(f"{path}: the array {key} is {array.dtype.str} "
                             f"{list(array.shape)}, its header says "
                             f"{listed[key]['dtype']} {listed[key]['shape']}")
    content = _decode(header["structure"], arrays)
    return content["result"], content["settings"], header


def _sha256(path):
    """Return the sha256 of a file's bytes (64 hexadecimal digits)."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        block = stream.read(1 << 20)
        while block:
            digest.update(block)
            block = stream.read(1 << 20)
    return digest.hexdigest()


def _describe(header):
    """Return one line on a cache: its date, tiers and fingerprints."""
    marks = header.get("tiers_and_fingerprints", {})
    tiers = []
    prints = []
    for key, value in marks.items():
        if key.endswith("/tier") and str(value) not in tiers:
            tiers.append(str(value))
        elif key.endswith("/fingerprint"):
            prints.append(str(value)[:12])
        elif key.endswith("/fingerprints") and isinstance(value, dict):
            for item in value.values():
                prints.append(str(item)[:12])
    tier_text = ", ".join(tiers) if tiers else "none recorded"
    print_text = ", ".join(prints) if prints else "none recorded"
    return (f"written {header.get('created')}; tier {tier_text}; "
            f"fingerprints {print_text}")


def _differences(stored, current, path):
    """Return every path where two settings structures differ.

    Dicts compare key by key (the order does not matter, as for ==),
    lists and tuples item by item, arrays with np.array_equal and their
    dtype, everything else with ==.
    """
    out = []
    if isinstance(stored, dict) and isinstance(current, dict):
        for key in stored:
            if key not in current:
                out.append(f"{path}/{key}: only in the cache")
            else:
                out.extend(_differences(stored[key], current[key], f"{path}/{key}"))
        for key in current:
            if key not in stored:
                out.append(f"{path}/{key}: only in the notebook")
        return out
    both_sequences = (isinstance(stored, (list, tuple))
                      and isinstance(current, (list, tuple)))
    if both_sequences and type(stored) is type(current) and len(stored) == len(current):
        for index in range(len(stored)):
            out.extend(_differences(stored[index], current[index], f"{path}/{index}"))
        return out
    if isinstance(stored, np.ndarray) or isinstance(current, np.ndarray):
        same = (isinstance(stored, np.ndarray) and isinstance(current, np.ndarray)
                and stored.dtype == current.dtype
                and np.array_equal(stored, current))
    else:
        # plain ==, the comparison the notebooks made: 1 == 1.0, and a list
        # never equals a tuple
        same = stored == current
    if not same:
        out.append(f"{path}: cache {str(stored)[:70]!r}, notebook {str(current)[:70]!r}")
    return out


def _model_specs(settings):
    """Return the models a settings structure names with their fingerprints.

    A settings dict that holds a model's inputs and its fingerprint -
    "yaml" and "fingerprint", or a "fingerprints" dict keyed by the
    settings fields that name yamls ("yaml", "model_a_yaml") - names one
    model per fingerprint, with its own likelihood overrides
    ("likelihood_overrides" or "overrides"), CAMB overrides
    ("camb_overrides" or "camb"), overall boost and tier. A fingerprint
    stored without the yaml it was made from (a "fingerprints" dict keyed by
    layout names) names no model here.

    Arguments:
      settings = a cache's stored settings.

    Returns:
      {path of the fingerprint entry: {"yaml", "likelihood_overrides",
      "camb_overrides", "accuracy_boost", "tier", "fingerprint"}}.
    """
    specs = {}

    def spec_of(entry, yaml_file, stored):
        return {"yaml": yaml_file,
                "likelihood_overrides": entry.get("likelihood_overrides",
                                                  entry.get("overrides")),
                "camb_overrides": entry.get("camb_overrides", entry.get("camb")),
                "accuracy_boost": entry.get("accuracy_boost", 1.0),
                "tier": entry.get("tier"),
                "fingerprint": stored}

    def walk(entry, path):
        if isinstance(entry, dict):
            stored = entry.get("fingerprint")
            if isinstance(stored, str) and isinstance(entry.get("yaml"), str):
                specs[f"{path}/fingerprint"] = spec_of(entry, entry["yaml"], stored)
            prints = entry.get("fingerprints")
            if isinstance(prints, dict):
                for field, value in prints.items():
                    if isinstance(entry.get(field), str) and isinstance(value, str):
                        here = f"{path}/fingerprints/{field}"
                        specs[here] = spec_of(entry, entry[field], value)
            for key, item in entry.items():
                walk(item, f"{path}/{key}")
        elif isinstance(entry, (list, tuple)):
            for index, item in enumerate(entry):
                walk(item, f"{path}/{index}")

    walk(settings, "settings")
    return specs


def _absolute_yaml(yaml_file):
    """Return a yaml path, read from ROOTDIR when it is relative (as stored)."""
    if os.path.isabs(yaml_file):
        return yaml_file
    return os.path.join(os.environ["ROOTDIR"], yaml_file)


def _fingerprint_changes(settings):
    """Return the stored fingerprints that the current files no longer give.

    Every model of _model_specs is fingerprinted again with
    settings_fingerprint from the current yaml, parameter and dataset
    files; a file edited since the cache was written gives another value.
    A fingerprint stored without its yaml is not recomputed here: the
    notebook's settings, which hold the current values, catch it.

    Arguments:
      settings = a cache's stored settings.

    Returns:
      list of str, one line per changed fingerprint (empty when none did).
    """
    changed = []
    for path, spec in _model_specs(settings).items():
        now = settings_fingerprint(_absolute_yaml(spec["yaml"]),
                                   likelihood_overrides=spec["likelihood_overrides"],
                                   accuracy_boost=spec["accuracy_boost"],
                                   camb_overrides=spec["camb_overrides"],
                                   tier=spec["tier"])
        if now != spec["fingerprint"]:
            changed.append(f"{path}: cache {spec['fingerprint'][:12]}..., files now "
                           f"{now[:12]}... ({spec['yaml']})")
    return changed


def _tier_entries(settings):
    """Return {path: value} of every "tier" entry of a settings structure."""
    found = {}
    _collect(settings, ["settings"], ("tier",), found)
    return found


def _json_ready(value):
    """Return value in plain JSON types, for the provenance record.

    Dict keys become text (a float key by repr), tuples lists, numpy
    scalars Python numbers, an array its dtype and shape, and any other
    object its text. A non-finite float stays a float: Python's json writes
    it as Infinity or NaN (the truncated-normal priors have an infinite
    bound), as settings_fingerprint's text does, and reads it back.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(key, str):
                text_key = key
            elif isinstance(key, float):
                text_key = repr(key)
            else:
                text_key = str(key)
            out[text_key] = _json_ready(item)
        return out
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            items.append(_json_ready(item))
        return items
    if isinstance(value, np.ndarray):
        return {"array": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _git_state(folder):
    """Return the git commit and dirty flag of the repository holding folder.

    "dirty" is True when a tracked file differs from the commit (untracked
    files do not count); "changed_files" lists up to 50 of them. A git
    failure is recorded as "error", never raised: the record says what it
    could not read.
    """
    import subprocess
    out = {"folder": folder}
    commands = {
        "commit": ["git", "-C", folder, "rev-parse", "HEAD"],
        "describe": ["git", "-C", folder, "describe", "--tags", "--always", "--dirty"],
        "status": ["git", "-C", folder, "status", "--porcelain", "--untracked-files=no"],
    }
    replies = {}
    for label, command in commands.items():
        try:
            replies[label] = subprocess.run(command, capture_output=True, text=True,
                                            timeout=GIT_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired) as error:
            out["error"] = f"{' '.join(command)}: {error}"
            return out
    if replies["commit"].returncode != 0:
        out["error"] = replies["commit"].stderr.strip()[:300]
        return out
    out["commit"] = replies["commit"].stdout.strip()
    out["describe"] = replies["describe"].stdout.strip()
    if replies["status"].returncode != 0:
        out["dirty"] = None
        out["error"] = replies["status"].stderr.strip()[:300]
        return out
    changed = []
    for line in replies["status"].stdout.splitlines():
        if line.strip():
            # a porcelain line is two status letters, a blank, the path
            changed.append(line[3:])
    out["dirty"] = len(changed) > 0
    out["changed_count"] = len(changed)
    out["changed_files"] = changed[:50]
    return out


def _code_provenance():
    """Return the state of the code a cache is written with.

    The git commit and dirty flag of this project, of cosmolike_core and of
    the Cocoa repository; the CAMB source folder the example yamls build
    from (its commit, its version, its library's date and whether its
    halofit holds the 1e-7 tolerance patch that check_halofit_patch
    requires); the PATCH_CAMB_HALOFIT_TOLERANCE environment variable; the
    Python, numpy and cobaya versions.
    """
    import platform
    import cobaya
    rootdir = os.environ["ROOTDIR"]
    camb_dir = os.path.join(rootdir, "external_modules", "code", "CAMB")
    halofit = os.path.join(camb_dir, "fortran", "halofit.f90")
    library = os.path.join(camb_dir, "camb", "camblib.so")
    camb = _git_state(camb_dir)
    version_file = os.path.join(camb_dir, "camb", "__init__.py")
    camb["version"] = None
    if os.path.isfile(version_file):
        with open(version_file) as handle:
            found = re.search(r'__version__\s*=\s*["\']([^"\']+)', handle.read())
        if found is not None:
            camb["version"] = found.group(1)
    patched = False
    if os.path.isfile(halofit):
        with open(halofit) as handle:
            patched = "abs(diff).le.1.d-7" in handle.read()
    camb["halofit_tolerance_patch_in_source"] = patched
    camb["library"] = None
    if os.path.isfile(library) and os.path.isfile(halofit):
        stamp = datetime.datetime.fromtimestamp(os.path.getmtime(library))
        camb["library"] = {"path": library,
                           "modified": stamp.astimezone().isoformat(timespec="seconds"),
                           "newer_than_halofit": (os.path.getmtime(library)
                                                  >= os.path.getmtime(halofit))}
    return {
        "lssty1xdesi": _git_state(PROJECT_DIR),
        "cosmolike_core": _git_state(os.path.join(rootdir, "external_modules", "code",
                                                  "cosmolike_core")),
        "cocoa": _git_state(rootdir),
        "camb": camb,
        "PATCH_CAMB_HALOFIT_TOLERANCE": os.environ.get("PATCH_CAMB_HALOFIT_TOLERANCE"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cobaya": getattr(cobaya, "__version__", None),
    }


def _provenance_record(name, settings, models=None, notes=None):
    """Return the provenance record of a cache about to be written.

    One entry per model: every model the settings name with a fingerprint
    (_model_specs), plus the extra `models` (models a settings structure
    describes without their yaml, such as the alternative vectors of the
    linear notebook). Each entry holds the model's inputs, its fully
    resolved cobaya input (_resolve_model: what settings_fingerprint
    hashes, so the sha256 of its sorted JSON text is the fingerprint), the
    fingerprint of the current files and whether it equals the stored one,
    and every dataset file with its sha256 (dataset_files).

    Arguments:
      name     = the cache name.
      settings = the settings stored with the result.
      models   = None, or {label: {"yaml", "likelihood_overrides",
                 "camb_overrides", "accuracy_boost", "tier", and optionally
                 "fingerprint"}}.
      notes    = None for PROVENANCE_NOTES, or {"computed", "resolved_input",
                 "code": text} for a cache written from an older file (whose
                 code_at_write, the state at the write, is not the code that
                 computed it).

    Returns:
      dict of plain JSON types.
    """
    specs = _model_specs(settings)
    for label, spec in (models or {}).items():
        full = {"yaml": spec["yaml"],
                "likelihood_overrides": spec.get("likelihood_overrides"),
                "camb_overrides": spec.get("camb_overrides"),
                "accuracy_boost": spec.get("accuracy_boost", 1.0),
                "tier": spec.get("tier"),
                "fingerprint": spec.get("fingerprint")}
        specs[f"models/{label}"] = full
    model_records = {}
    for path, spec in specs.items():
        yaml_file = _absolute_yaml(spec["yaml"])
        content, datasets = _resolve_model(
            yaml_file,
            likelihood_overrides=spec["likelihood_overrides"],
            accuracy_boost=spec["accuracy_boost"],
            camb_overrides=spec["camb_overrides"],
            tier=spec["tier"])
        now = _content_sha256(content)
        if spec["fingerprint"] is None:
            check = "no fingerprint stored"
        elif spec["fingerprint"] == now:
            check = "equal"
        else:
            check = "differs"
        files = {}
        for likelihood, where in datasets.items():
            files[likelihood] = {"path": where["path"],
                                 **dataset_files(where["path"], where["data_file"])}
        model_records[path] = {
            "yaml": spec["yaml"],
            "yaml_sha256": _sha256(yaml_file),
            "likelihood_overrides": spec["likelihood_overrides"],
            "camb_overrides": spec["camb_overrides"],
            "accuracy_boost": spec["accuracy_boost"],
            "tier": spec["tier"],
            "fingerprint_stored": spec["fingerprint"],
            "fingerprint_of_the_files": now,
            "fingerprint_check": check,
            # the resolved input as the JSON text settings_fingerprint hashes
            # reads it back (numpy values as text, every key a string), so
            # the sha256 of json.dumps(resolved_input, sort_keys=True) is the
            # fingerprint: a reader can check it from the record alone
            "resolved_input": json.loads(json.dumps(content, sort_keys=True,
                                                    default=str)),
            "dataset_files": files,
        }
    marks = {}
    _collect(settings, ["settings"], HEADER_MARK_KEYS, marks)
    if notes is None:
        notes = PROVENANCE_NOTES
    record = {
        "format": PROVENANCE_FORMAT,
        "name": name,
        "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "notes": dict(notes),
        "tiers": _tier_entries(settings),
        "fingerprints": marks,
        "settings": settings,
        "models": model_records,
        "code_at_write": _code_provenance(),
    }
    return _json_ready(record)


def read_fisher_cache_header(name, folder=None):
    """Return the JSON header of the working copy <folder>/<name>.npz.

    The header alone is read (no array): the cache's name, the date it was
    written, its tier and fingerprint entries, its parameter names and the
    shape and dtype of every array. Reading it decides nothing; it serves
    messages, such as the refusal of a script to replace an existing cache.

    Arguments:
      name   = the cache name.
      folder = the folder of the file; None for chains/.

    Returns:
      dict, the header.

    Raises:
      FileNotFoundError when the file does not exist; ValueError when it is
      not a cache file.
    """
    if folder is None:
        folder = CACHE_WORKING
    path = _cache_path(folder, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist")
    return _read_npz(path, header_only=True)


def write_fisher_cache(result, name, settings, overwrite=False, folder=None,
                       models=None, provenance_notes=None):
    """Write a freshly computed result to its working copy chains/<name>.npz.

    The settings are stored with the result, so read_fisher_cache can
    compare them with the notebook's; they must hold every setting the
    result depends on, fingerprints included, fully resolved. The header
    also gets the provenance record (_provenance_record), built from the
    files as they are now: every stored fingerprint must equal the one the
    current files give, or nothing is written (a cache whose files changed
    between the computation and the write would not read back).

    Arguments:
      result    = what the notebook computed: dicts, lists, tuples, numpy
                  arrays and scalars, None, bool, int, float, str.
      name      = the cache name, e.g. 'model_derivatives_dz0p05_final'.
      settings  = the settings it was computed with (the same kinds).
      overwrite = True to replace an existing file.
      folder    = the folder of the file; None for chains/ (a script may
                  write elsewhere, e.g. a scratch folder on a cluster).
      models    = None, or {label: model inputs} of models the settings
                  describe without their yaml (_provenance_record), for the
                  record.
      provenance_notes = None for a cache computed in this session, or
                  {"computed", "resolved_input", "code": text} for a cache
                  written from an older file (the conversion of the pickles).

    Returns:
      the path written.

    Raises:
      FileExistsError when the file exists and overwrite is False (the
      message describes the existing file); TypeError naming an entry that
      cannot be stored without a pickle; ValueError when a stored
      fingerprint differs from the current files' (nothing is written in
      any of these cases).

    Side effects: writes <folder>/<name>.npz (creating the folder).
    """
    if folder is None:
        folder = CACHE_WORKING
    path = _cache_path(folder, name)
    if os.path.exists(path) and not overwrite:
        try:
            existing = _describe(_read_npz(path, header_only=True))
        except ValueError as error:
            existing = f"not a readable cache: {error}"
        raise FileExistsError(
            f"{path} exists ({existing}); nothing was written. Pass overwrite=True "
            "to replace it, or write under another name")
    # encode once before any file exists, so an entry that cannot be stored
    # stops the call with nothing written
    _encode({"result": result, "settings": settings}, [], {})
    record = _provenance_record(name, settings, models=models, notes=provenance_notes)
    stale = []
    for model_path, model in record["models"].items():
        if model["fingerprint_check"] == "differs":
            stored_print = str(model["fingerprint_stored"])[:12]
            stale.append(f"{model_path}: stored {stored_print}..., files "
                         f"{model['fingerprint_of_the_files'][:12]}... ({model['yaml']})")
    if stale:
        raise ValueError(
            f"{name}: the yaml, parameter or dataset files changed after the "
            "fingerprints of the settings were computed: " + "; ".join(stale)
            + ". Nothing was written: compute the result again with the current files")
    os.makedirs(folder, exist_ok=True)
    _write_npz(path, name, result, settings, record)
    print(f"wrote {_short_path(path)}", flush=True)
    return path


def read_fisher_cache(name, settings=None, comparable=None, tier=None, folder=None):
    """Read the working copy chains/<name>.npz after checking it.

    Three checks, each of which raises; nothing is ever recomputed here:

      1. the stored settings equal the notebook's (settings), compared
         after `comparable` when given (a function that drops what may
         differ, such as the file names whose content the fingerprints
         cover);
      2. every tier entry of the stored settings is `tier` (None entries,
         settings outside the tier system, pass);
      3. every fingerprint the stored settings hold with its yaml is
         recomputed from the current yaml, parameter and dataset files and
         must equal the stored one (_fingerprint_changes).

    Arguments:
      name       = the cache name.
      settings   = the notebook's current settings for this cache, or None
                   for a cache another notebook makes (then checks 2 and 3
                   only).
      comparable = None, or function(settings) -> the part of the settings
                   that must match; applied to the stored and the current
                   settings alike.
      tier       = the tier the notebook runs ('final', 'exploration'), or
                   None to skip check 2.
      folder     = the folder of the file; None for chains/.

    Returns:
      the cached result, as write_fisher_cache received it.

    Raises:
      FileNotFoundError when there is no working copy; ValueError naming
      the differing settings, the other tier or the changed fingerprints.
    """
    if folder is None:
        folder = CACHE_WORKING
    path = _cache_path(folder, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist. Copy the archived cache with "
            f"fm.load_fisher_from_cache({name!r}) (the archive holds the final tier "
            "only), or compute the result and write it with fm.write_fisher_cache "
            "(the commented compute and write cells)")
    result, stored, header = _read_npz(path)
    if header.get("name") != name:
        raise ValueError(f"{path} holds the cache {header.get('name')!r}, not {name!r}: "
                         "a renamed file; copy or write the cache under its own name")
    if settings is not None:
        stored_part = stored
        current_part = settings
        if comparable is not None:
            stored_part = comparable(stored)
            current_part = comparable(settings)
        differences = _differences(stored_part, current_part, "settings")
        if differences:
            shown = "; ".join(differences[:10])
            more = f" (and {len(differences) - 10} more)" if len(differences) > 10 else ""
            raise ValueError(
                f"{_short_path(path)} was made with other settings than this "
                f"notebook's ({_describe(header)}). They differ at: {shown}{more}. "
                "Nothing was recomputed: copy the matching archived cache, or compute "
                "the result with the notebook's settings and write it")
    if tier is not None:
        entries = _tier_entries(stored)
        other = {}
        for key, value in entries.items():
            if value is not None and value != tier:
                other[key] = value
        if not entries:
            raise ValueError(
                f"{_short_path(path)} records no tier; this notebook runs the "
                f"{tier!r} tier and reads caches of that tier only. Compute the cache "
                "at that tier and write it")
        if other:
            raise ValueError(
                f"{_short_path(path)} holds the tier entries {other}; this notebook "
                f"runs the {tier!r} tier. Read the cache of that tier, or compute and "
                "write one")
    changed = _fingerprint_changes(stored)
    if changed:
        raise ValueError(
            f"{_short_path(path)}: the yaml, parameter or dataset files changed since "
            f"the cache was written ({_describe(header)}); these fingerprints no "
            "longer match: " + "; ".join(changed) + ". Nothing was recomputed: "
            "compute the result again and write it with overwrite=True")
    print(f"read {_short_path(path)} ({_describe(header)})", flush=True)
    return result


def _archive_checksums():
    """Return {name: sha256} of fisher_caches/checksums.json ({} without it)."""
    if not os.path.exists(CACHE_CHECKSUMS):
        return {}
    with open(CACHE_CHECKSUMS) as stream:
        return json.load(stream)


def _copy_file(source, target):
    """Copy source to target through a temporary file and a rename."""
    partial = _partial_path(os.path.dirname(target), os.path.basename(target))
    try:
        with open(source, "rb") as reader, open(partial, "xb") as writer:
            shutil.copyfileobj(reader, writer)
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def load_fisher_from_cache(name, overwrite=False):
    """Copy the archived fisher_caches/<name>.npz to chains/<name>.npz.

    The archived file is verified first: it must not be a git-lfs pointer
    (a large file git never downloaded), its sha256 must equal the one
    fisher_caches/checksums.json records, and its header must carry this
    format and this name. A working copy with the same bytes is left as it
    is; a different one is replaced only with overwrite=True. Reading the
    copy (read_fisher_cache) is a separate call.

    Arguments:
      name      = the cache name.
      overwrite = True to replace a different working copy.

    Returns:
      the working copy's path.

    Raises:
      FileNotFoundError (no such archived cache), ValueError (a git-lfs
      pointer, no recorded checksum, another checksum or header),
      FileExistsError (a different working copy and overwrite False).

    Side effects: writes chains/<name>.npz (creating chains/).
    """
    source = _cache_path(CACHE_ARCHIVE, name)
    if not os.path.isfile(source):
        archived = []
        if os.path.isdir(CACHE_ARCHIVE):
            for entry in sorted(os.listdir(CACHE_ARCHIVE)):
                if entry.endswith(".npz"):
                    archived.append(entry[:-len(".npz")])
        raise FileNotFoundError(
            f"{source} is not archived. The archive holds the final tier only: "
            f"{archived}. Compute this cache with the notebook's compute cells and "
            "write it with fm.write_fisher_cache")
    with open(source, "rb") as stream:
        start = stream.read(len(LFS_POINTER_START))
    if start == LFS_POINTER_START:
        raise ValueError(
            f"{_short_path(source)} is a git-lfs pointer, not the cache: git-lfs was "
            "not on PATH when the repository was cloned or pulled. Run "
            f"git lfs pull --include fisher_caches/{name}.npz (or git lfs pull) in "
            "the project folder, then call this again")
    checksums = _archive_checksums()
    if name not in checksums:
        raise ValueError(
            f"{_short_path(CACHE_CHECKSUMS)} records no checksum for {name}: an "
            "archived file is added by fm.save_fisher_to_cache only. Archive it with "
            "that call, or restore checksums.json from git")
    found = _sha256(source)
    if found != checksums[name]:
        raise ValueError(
            f"{_short_path(source)} has the sha256 {found[:12]}..., "
            f"checksums.json records {checksums[name][:12]}...: the archived file "
            "changed since it was saved. Restore it from git (git checkout -- "
            f"fisher_caches/{name}.npz, then git lfs pull), or archive it again "
            "with fm.save_fisher_to_cache(..., overwrite=True)")
    header = _read_npz(source, header_only=True)
    if header.get("name") != name:
        raise ValueError(f"{_short_path(source)} holds the cache "
                         f"{header.get('name')!r}, not {name!r}: a renamed file")
    target = _cache_path(CACHE_WORKING, name)
    if os.path.exists(target):
        if _sha256(target) == found:
            print(f"{_short_path(target)} is already the archived copy: nothing copied",
                  flush=True)
            return target
        if not overwrite:
            try:
                existing = _describe(_read_npz(target, header_only=True))
            except ValueError as error:
                existing = f"not a readable cache: {error}"
            raise FileExistsError(
                f"{_short_path(target)} exists and differs from the archived copy "
                f"(working copy: {existing}; archive: {_describe(header)}). Nothing "
                "was copied: pass overwrite=True to replace the working copy")
    os.makedirs(CACHE_WORKING, exist_ok=True)
    _copy_file(source, target)
    print(f"copied {_short_path(source)} to {_short_path(target)} ({_describe(header)})",
          flush=True)
    return target


def save_fisher_to_cache(name, overwrite=False):
    """Archive the working copy chains/<name>.npz in fisher_caches/.

    The only call that changes the archive. The working copy must be a
    final-tier cache - at least one tier entry, and every tier entry that
    is not None equal to 'final' - with a provenance record, whose stored
    fingerprints the current files still give, so an out-of-date cache is
    never archived. The file is copied, its sha256 recorded in
    fisher_caches/checksums.json and its header's provenance record written
    as fisher_caches/<name>.json; git then stages all three (the .npz
    through git-lfs, by the rule fisher_caches/*.npz of .gitattributes;
    the two .json files as plain text).

    Arguments:
      name      = the cache name.
      overwrite = True to replace a different archived file.

    Returns:
      the archived path.

    Raises:
      FileNotFoundError (no working copy), ValueError (another tier, no
      tier, no provenance record, changed fingerprints, a renamed file),
      FileExistsError (a different archived file and overwrite False).

    Side effects: writes fisher_caches/<name>.npz and its provenance record
    fisher_caches/<name>.json, and rewrites fisher_caches/checksums.json.
    """
    source = _cache_path(CACHE_WORKING, name)
    if not os.path.isfile(source):
        raise FileNotFoundError(f"{source} does not exist: compute the cache and write "
                                "it with fm.write_fisher_cache first")
    # the full read also checks every array against the header's list
    _result, stored, header = _read_npz(source)
    if header.get("name") != name:
        raise ValueError(f"{_short_path(source)} holds the cache "
                         f"{header.get('name')!r}, not {name!r}: a renamed file")
    record = header.get("provenance")
    if not isinstance(record, dict) or record.get("format") != PROVENANCE_FORMAT:
        raise ValueError(f"{_short_path(source)} has no {PROVENANCE_FORMAT} record: "
                         "write it again with fm.write_fisher_cache before archiving")
    entries = _tier_entries(stored)
    final = 0
    other = {}
    for key, value in entries.items():
        if value == "final":
            final += 1
        elif value is not None:
            other[key] = value
    if not entries:
        raise ValueError(
            f"{_short_path(source)} records no tier: the archive keeps final-tier "
            "caches only (at least one tier entry, every entry 'final' or None). "
            "Compute the cache at the final tier to archive it")
    if final == 0 or other:
        raise ValueError(
            f"{_short_path(source)} holds the tier entries {entries}: the archive keeps "
            "final-tier caches only (at least one tier entry, every entry 'final' or "
            "None). Compute the cache at the final tier to archive it")
    changed = _fingerprint_changes(stored)
    if changed:
        raise ValueError(
            f"{_short_path(source)}: the files changed since the cache was written: "
            + "; ".join(changed) + ". It is not archived: compute it again first")
    target = _cache_path(CACHE_ARCHIVE, name)
    record_path = os.path.join(CACHE_ARCHIVE, name + ".json")
    # indent and sorted keys keep the record readable and its git diffs short
    record_text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    found = _sha256(source)
    checksums = _archive_checksums()
    if os.path.exists(target):
        same = _sha256(target) == found
        record_same = False
        if os.path.isfile(record_path):
            with open(record_path) as handle:
                record_same = handle.read() == record_text
        if same and checksums.get(name) == found and record_same:
            print(f"{_short_path(target)} is already this file: nothing archived",
                  flush=True)
            return target
        if not same and not overwrite:
            try:
                existing = _describe(_read_npz(target, header_only=True))
            except ValueError as error:
                existing = f"not a readable cache: {error}"
            raise FileExistsError(
                f"{_short_path(target)} is archived already ({existing}); the working "
                f"copy is another file ({_describe(header)}). Nothing was archived: "
                "pass overwrite=True to replace the archived cache")
    os.makedirs(CACHE_ARCHIVE, exist_ok=True)
    _copy_file(source, target)
    # the provenance record next to the archived file: the header's record
    # as indented JSON (plain git: the npz rule of .gitattributes leaves it)
    partial = _partial_path(CACHE_ARCHIVE, name + "_record")
    with open(partial, "x") as stream:
        stream.write(record_text)
    os.replace(partial, record_path)
    checksums[name] = found
    # sorted names and one entry per line keep the git diff of the table
    # to the lines that change
    ordered = {}
    for key in sorted(checksums):
        ordered[key] = checksums[key]
    partial = _partial_path(CACHE_ARCHIVE, "checksums")
    with open(partial, "x") as stream:
        json.dump(ordered, stream, indent=1)
        stream.write("\n")
    os.replace(partial, CACHE_CHECKSUMS)
    print(f"archived {_short_path(source)} as {_short_path(target)} "
          f"({_describe(header)}, sha256 {found[:12]}...) with its record "
          f"{_short_path(record_path)}; stage them and "
          f"{_short_path(CACHE_CHECKSUMS)} with git", flush=True)
    return target
