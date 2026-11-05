"""combo_3x2pt in Fourier band powers on the LSST Y1 samples.

The LSST Y1 lens and source samples on the DESI layouts' bands, built by
scripts/make_lsst_3x2pt_case.py. Everything is combo_3x2pt's; the class
exists so its yaml can declare the Fourier binning and name the case's
dataset (cobaya merges combo_3x2pt.yaml underneath).
"""
from cobaya.likelihoods.lssty1xdesi.combo_3x2pt import combo_3x2pt


class lsst_y1_3x2pt(combo_3x2pt):
  """combo_3x2pt on the Fourier LSST Y1 layout (see the module docstring)."""
