"""desi_cl with model B on the growth-tied linear bias.

An exploration class. On the Delta d_com = 62 Mpc bins (76 lens bins): the
one-loop model (higher_order_bias: 1, see desi_cl.py) with the model B
fragment of notes/higher_order_bias (params_desi_model_b.yaml in this
folder: one offset per group for Delta_2, Delta_s, alpha_0, alpha_2, SN0 and
SN2, the tilts fixed at zero) on the growth-tied linear fragment
(../params_desi.yaml: b_{1,i} = B_g / D(zbar_i), every bin of a group tied
to one amplitude). Everything else is desi_cl's; the class exists so its
yaml can switch the one-loop model on and name the two fragments (a class's
params block replaces the parent's). The paper model is
lssty1xdesi.desi_cl_model_b.
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl import desi_cl


class desi_cl_model_b_growth_tied(desi_cl):
  """Model B on the growth-tied linear bias (see the module docstring)."""
