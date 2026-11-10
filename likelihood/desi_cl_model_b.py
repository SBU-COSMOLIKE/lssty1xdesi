"""desi_cl with the paper model (model B + tilts, alpha_2 = 0).

On the Delta d_com = 62 Mpc bins (76 lens bins). The bias is the one-loop
model of desi_cl.py (higher_order_bias: 1) with two parameter fragments,
both written by scripts/make_desi_params.py:

  params_desi_bias_pessimistic.yaml = the linear bias under the pessimistic
      prior of notes/correlated_bias_prior_v2: bin i of group g (BGS,
      LRG1-3, ELG1-2) has b_{1,i} = B_g (1 + delta_i) / D(zbar_i), with
      delta_i = s_g (zbar_i - zbar_g) + eps_i, B_g the group amplitude, D
      the linear growth factor, zbar_i the bin's mean redshift, zbar_g the
      mean of the group's bin redshifts, s_g a free tilt and eps_i a
      Gaussian residual of width 3% whose correlation between two bins of
      the group falls as exp(-|zbar_i - zbar_j| / 0.05); and the velocity
      dispersions of the Fingers of God, on Gaussian priors 6x wider
      than the exploration fragments' (they absorb the alpha_2 the
      model lacks).
  params_desi_model_b_tilts.yaml = model B + tilts of
      notes/higher_order_bias without the radial counterterm: per group,
      the offsets Delta_2 and Delta_s of b2 and bs2 from their calibrations
      and the counterterm alpha_0, each with a free straight line in
      redshift, and the stochastic amplitudes SN0 and SN2, at the DESI DR1
      prior widths of the shipped scale cut (k = 0.1185/Mpc at every
      redshift, about 0.176 h/Mpc); alpha_2 is fixed to 0 in every bin.

Everything else is desi_cl's; the class exists so its yaml can switch the
one-loop model on and name the two fragments (a class's params block
replaces the parent's).
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl import desi_cl


class desi_cl_model_b(desi_cl):
  """The paper model on this layout (see the module docstring)."""
