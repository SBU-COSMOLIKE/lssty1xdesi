"""desi_cl on the DESI DR1 fixed-width Delta z = 0.05 bins.

36 lens bins (BGS 6, LRG 14, ELG 16), built by scripts/make_desi_dz_case.py.
Everything is desi_cl's; the class exists so its yaml can name the
case's dataset and its own per-bin parameter fragment (cobaya merges
desi_cl.yaml underneath, and a class's params block replaces the
parent's, so no parameter of another binning survives).
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl import desi_cl


class desi_cl_dr1_dz0p05(desi_cl):
  """desi_cl on the Delta z = 0.05 layout (see the module docstring)."""
