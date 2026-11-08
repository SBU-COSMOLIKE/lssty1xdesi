"""desi_cl_dr1_dz0p02 with the model A parameters of notes/higher_order_bias.

On the DESI DR1 fixed-width Delta z = 0.02 bins: the one-loop model
(higher_order_bias: 1, see desi_cl.py) with the model A parameter
fragment, the radial counterterm alpha_2 fixed to 0 and the prior widths
of the k_max = 0.075 h/Mpc analysis. Everything else is
desi_cl_dr1_dz0p02's; the class exists so its yaml can switch the one-loop
model on and name the model A parameter fragment next to the linear one (a
class's params block replaces the parent's).
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl_dr1_dz0p02 import desi_cl_dr1_dz0p02


class desi_cl_dr1_dz0p02_model_a(desi_cl_dr1_dz0p02):
  """desi_cl_dr1_dz0p02 with model A higher-order bias (see the module docstring)."""
