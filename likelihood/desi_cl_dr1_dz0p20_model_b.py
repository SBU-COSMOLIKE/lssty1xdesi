"""desi_cl_dr1_dz0p20 with the paper model: model B + tilts, pessimistic prior.

On the DESI DR1 fixed-width Delta z = 0.2 bins (10 lens bins). The bias is
the one-loop model of desi_cl.py (higher_order_bias: 1) with two parameter
fragments, both written by scripts/make_desi_params.py:

  params_desi_dr1_dz0p20_bias_pessimistic.yaml = the linear bias under the
      pessimistic prior of notes/correlated_bias_prior_v2: bin i of group g
      (BGS, LRG1-3, ELG1-2) has b_{1,i} = B_g (1 + delta_i) / D(zbar_i),
      with delta_i = s_g (zbar_i - zbar_g) + eps_i, B_g the group amplitude,
      D the linear growth factor, zbar_i the bin's mean redshift, zbar_g the
      mean of the group's bin redshifts, s_g a free tilt and eps_i a
      Gaussian residual of width 3% whose correlation between two bins of
      the group falls as exp(-|zbar_i - zbar_j| / 0.05); and the velocity
      dispersions of the Fingers of God. Every group of this layout has one
      or two bins, so the pessimistic prior leaves every bin's bias free.
  params_desi_dr1_dz0p20_model_b_tilts.yaml = model B + tilts of
      notes/higher_order_bias: per group, the offsets Delta_2 and Delta_s of
      b2 and bs2 from their calibrations and the counterterms alpha_0 and
      alpha_2, each with a free straight line in redshift, and the
      stochastic amplitudes SN0 and SN2, at the DESI DR1 prior widths of the
      shipped scale cut (k = 0.1185/Mpc at every redshift, about 0.176
      h/Mpc).

Everything else is desi_cl_dr1_dz0p20's; the class exists so its yaml can
switch the one-loop model on and name the two fragments (a class's params
block replaces the parent's).
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl_dr1_dz0p20 import desi_cl_dr1_dz0p20


class desi_cl_dr1_dz0p20_model_b(desi_cl_dr1_dz0p20):
  """The paper model on this layout (see the module docstring)."""
