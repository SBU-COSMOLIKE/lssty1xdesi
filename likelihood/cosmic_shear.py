"""Defines the Cobaya likelihood lssty1xdesi.cosmic_shear (LSST-Y1 cosmic shear).

The entry `lssty1xdesi.cosmic_shear` in the likelihood block of a yaml file
makes Cobaya import this module and build the class of the same name, with
the options and defaults of cosmic_shear.yaml (next to this file). The base
class _cosmolike_prototype_base does all the work (its module docstring
explains the data vector, the probes and the order of the calls); this
subclass selects the probe "xi": the xi_+ and xi_- blocks of the data
vector.
"""
# Cocoa links this likelihood directory into Cobaya as the package
# cobaya.likelihoods.lssty1xdesi, hence the import path below.
from cobaya.likelihoods.lssty1xdesi._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_lssty1xdesi_interface as ci
import numpy as np

class cosmic_shear(_cosmolike_prototype_base):
  """Evaluates the cosmic-shear likelihood (probe "xi")."""
  def initialize(self):
    """Configure the base likelihood for probe "xi".

    Cobaya calls initialize once, when it builds the likelihood;
    super(...) reaches the method of the base class, which reads the
    .dataset file and sets up cosmolike.
    """
    super(cosmic_shear,self).initialize(probe="xi")
