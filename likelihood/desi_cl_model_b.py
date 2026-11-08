"""desi_cl with the model B bias of notes/higher_order_bias.

On the Delta d_com = 62 Mpc bins (76 lens bins), with higher_order_bias:
2: model A's one-loop real-space bias (b2 and bs2 tied to each bin's b1
plus per-chunk offsets), k^2 counterterm alpha_0 and stochastic amplitude
SN0, plus the radial counterterm alpha_2 mu^2 k^2 and the line-of-sight
stochastic amplitude SN2 k^2 mu^2, with mu the cosine to the line of sight
(see desi_cl.py). Everything else is desi_cl's; the class exists so its
yaml can switch the model on and name the model B parameter fragment next
to the linear one (a class's params block replaces the parent's).
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl import desi_cl


class desi_cl_model_b(desi_cl):
  """desi_cl with model B higher-order bias (see the module docstring)."""
