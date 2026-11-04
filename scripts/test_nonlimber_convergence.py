"""Non-Limber chi-grid convergence test for the DESI thin bins.

The FFTLog chi grid of the non-Limber C_ell (nchi = Ntable.NL_Nchi,
log-spaced over z = 0.002..4.0) must resolve the thin spectroscopic
kernels: at the cosmolike default nchi = 512 the comb spacing near
z = 1 (about 45 Mpc) is comparable to the 60-80 Mpc bin widths, and the
pure-CAMB data vector moves by chi2 = 1.5e5 between that grid and a
converged one. Fiducial self-consistency cannot catch this (a data
vector generated on a given comb always matches itself), so this test
checks two things that can:

  1. the example evaluated at the project default nonlimber_accuracyboost
     reproduces the stored fiducial data vector (chi2 < 0.01) - catches
     a fiducial regenerated on the wrong grid;
  2. doubling the boost moves the prediction by less than CONV_TOL in
     cross chi2 (masked inverse covariance) - certifies the default is
     converged. Measured 2026-11: boost 8 vs 16 = 8.6, 16 vs 32 = 1.6.

The doubled-boost evaluation is the expensive step (about a minute), so
this belongs with the accuracy suite, not a per-commit hook.

Usage (from cocoa/Cocoa, cocoapy311 active, start_cocoa.sh sourced):
    python projects/lssty1xdesi/scripts/test_nonlimber_convergence.py
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file

import cosmolike_lssty1xdesi_interface as ci

PRJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIKE = "lssty1xdesi.desi_cl"
FID_TOL = 0.01   # chi2 of the default-boost run against the stored fiducial
CONV_TOL = 5.0   # cross chi2 allowed between the default and doubled boost


def run(nl_boost):
    """Data vector and fiducial chi2 at one nonlimber boost.

    Arguments:
      nl_boost = nonlimber_accuracyboost for this evaluation, or None to
                 take the project default from desi_cl.yaml.

    Returns:
      (datavector, chi2 against the stored fiducial, boost used)
    """
    info = yaml_load_file(os.path.join(PRJ, "EXAMPLE_EVALUATE6.yaml"))
    point = dict(info["sampler"]["evaluate"]["override"])
    for key in ("sampler", "output", "timing"):
        info.pop(key, None)
    dv_file = os.path.join(PRJ, "chains", "test_nonlimber_conv.modelvector")
    if nl_boost is not None:
        info["likelihood"][LIKE]["nonlimber_accuracyboost"] = nl_boost
    info["likelihood"][LIKE]["print_datavector"] = True
    info["likelihood"][LIKE]["print_datavector_file"] = dv_file
    model = get_model(info)
    post = model.logposterior(point, cached=False)
    chi2 = -2.0 * post.loglikes[0]
    boost = float(model.likelihood[LIKE].nonlimber_accuracyboost)
    return np.loadtxt(dv_file)[:, 1], chi2, boost


def main():
    dv_def, chi2_fid, boost = run(None)
    print(f"default boost {boost:g}: chi2 vs stored fiducial = "
          f"{chi2_fid:.6f}", flush=True)

    dv_2x, _, _ = run(2.0 * boost)
    inv = np.asarray(ci.get_inv_cov_masked())
    delta = dv_2x - dv_def
    conv = float(delta @ inv @ delta)
    print(f"cross chi2, boost {boost:g} vs {2*boost:g} = {conv:.4f}",
          flush=True)

    ok = chi2_fid < FID_TOL and conv < CONV_TOL
    print("PASS" if ok else
          f"FAIL (fiducial tol {FID_TOL}, convergence tol {CONV_TOL})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
