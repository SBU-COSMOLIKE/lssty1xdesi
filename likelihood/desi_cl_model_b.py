"""desi_cl with the model B parameters of notes/higher_order_bias.

On the Delta d_com = 62 Mpc bins (76 lens bins): the one-loop model
(higher_order_bias: 1, see desi_cl.py) with the model B parameter
fragment, alpha_2 free and the DESI DR1 prior widths of the shipped scale
cut. Everything else is desi_cl's; the class exists so its yaml can switch
the one-loop model on and name the model B parameter fragment next to the
linear one (a class's params block replaces the parent's).
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl import desi_cl


class desi_cl_model_b(desi_cl):
  """desi_cl with model B higher-order bias (see the module docstring)."""
