"""desi_cl with the model A parameters of notes/higher_order_bias.

On the Delta d_com = 62 Mpc bins (76 lens bins): the one-loop model
(higher_order_bias: 1, see desi_cl.py) with the model A parameter
fragment, the radial counterterm alpha_2 fixed to 0 and the prior widths
of the k_max = 0.075 h/Mpc analysis. Everything else is desi_cl's; the
class exists so its yaml can switch the one-loop model on and name the
model A parameter fragment next to the linear one (a class's params block
replaces the parent's).
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl import desi_cl


class desi_cl_model_a(desi_cl):
  """desi_cl with model A higher-order bias (see the module docstring)."""
