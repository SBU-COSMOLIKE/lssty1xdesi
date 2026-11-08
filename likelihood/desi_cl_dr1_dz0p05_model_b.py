"""desi_cl_dr1_dz0p05 with the model B parameters of notes/higher_order_bias.

On the DESI DR1 fixed-width Delta z = 0.05 bins: the one-loop model
(higher_order_bias: 1, see desi_cl.py) with the model B parameter
fragment, alpha_2 free and the DESI DR1 prior widths of the shipped scale
cut. Everything else is desi_cl_dr1_dz0p05's; the class exists so its yaml
can switch the one-loop model on and name the model B parameter fragment
next to the linear one (a class's params block replaces the parent's).
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl_dr1_dz0p05 import desi_cl_dr1_dz0p05


class desi_cl_dr1_dz0p05_model_b(desi_cl_dr1_dz0p05):
  """desi_cl_dr1_dz0p05 with model B higher-order bias (see the module docstring)."""
