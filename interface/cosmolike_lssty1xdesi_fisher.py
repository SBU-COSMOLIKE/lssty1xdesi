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
    yaml, and stays as set.

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


class ModelFisher:
    """The Fisher machinery bound to one cobaya model of this project."""

    def __init__(self, yaml_file, params, extra_prefixes=(),
                 likelihood_overrides=None, invcov_scale=1.0,
                 accuracy_boost=1.0):
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
        """
        info = yaml_load_file(yaml_file)
        point = dict(info["sampler"]["evaluate"]["override"])
        for key in ("sampler", "output", "force", "resume", "debug"):
            info.pop(key, None)
        (self.likelihood_name,) = info["likelihood"]
        block = info["likelihood"][self.likelihood_name]
        boost_accuracy(info, block, float(accuracy_boost))
        block.update(likelihood_overrides or {})
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
