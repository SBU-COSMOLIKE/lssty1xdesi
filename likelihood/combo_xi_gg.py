"""Defines the Cobaya likelihood lssty1xdesi.combo_xi_gg (shear and galaxy clustering).

The entry `lssty1xdesi.combo_xi_gg` in the likelihood block of a yaml file makes
Cobaya import this module and build the class of the same name, with the
options and defaults of combo_xi_gg.yaml (next to this file). The base class
_cosmolike_prototype_base does all the work (its module docstring explains
the data vector, the probes and the order of the calls); this subclass
selects the probe "xi_gg": the xi_+- and w(theta) blocks of the data vector
(cosmic shear and galaxy clustering).
"""
# Cocoa links this likelihood directory into Cobaya as the package
# cobaya.likelihoods.lssty1xdesi, hence the import path below.
from cobaya.likelihoods.lssty1xdesi._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_lssty1xdesi_interface as ci
import numpy as np

class combo_xi_gg(_cosmolike_prototype_base):
  """Evaluates the shear plus clustering likelihood (probe "xi_gg")."""
  def initialize(self):
    """Configure the base likelihood for probe "xi_gg".

    Cobaya calls initialize once, when it builds the likelihood;
    super(...) reaches the method of the base class, which reads the
    .dataset file and sets up cosmolike.
    """
    super(combo_xi_gg,self).initialize(probe="xi_gg")