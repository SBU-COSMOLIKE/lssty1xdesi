"""desi_cl_dr1_dz0p20 with the model A bias of notes/higher_order_bias.

the DESI DR1 fixed-width Delta z = 0.2 bins, with higher_order_bias: 1 (one-loop real-space bias b2 and bs2
tied to each bin's b1 plus per-chunk offsets, the k^2 counterterm
alpha_0 and the stochastic amplitude SN0; see desi_cl.py). Everything
else is desi_cl_dr1_dz0p20's; the class exists so its yaml can switch the model
on and name the model A parameter fragment next to the linear one
(a class's params block replaces the parent's).
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl_dr1_dz0p20 import desi_cl_dr1_dz0p20


class desi_cl_dr1_dz0p20_model_a(desi_cl_dr1_dz0p20):
  """desi_cl_dr1_dz0p20 with model A higher-order bias (see the module docstring)."""
