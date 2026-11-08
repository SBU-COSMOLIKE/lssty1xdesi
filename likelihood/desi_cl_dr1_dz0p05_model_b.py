"""desi_cl_dr1_dz0p05 with the model B bias of notes/higher_order_bias.

On the DESI DR1 fixed-width Delta z = 0.05 bins, with higher_order_bias:
2: model A's one-loop real-space bias (b2 and bs2 tied to each bin's b1
plus per-chunk offsets), k^2 counterterm alpha_0 and stochastic amplitude
SN0, plus the radial counterterm alpha_2 mu^2 k^2 and the line-of-sight
stochastic amplitude SN2 k^2 mu^2, with mu the cosine to the line of sight
(see desi_cl.py). Everything else is desi_cl_dr1_dz0p05's; the class
exists so its yaml can switch the model on and name the model B parameter
fragment next to the linear one (a class's params block replaces the
parent's).
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl_dr1_dz0p05 import desi_cl_dr1_dz0p05


class desi_cl_dr1_dz0p05_model_b(desi_cl_dr1_dz0p05):
  """desi_cl_dr1_dz0p05 with model B higher-order bias (see the module docstring)."""
