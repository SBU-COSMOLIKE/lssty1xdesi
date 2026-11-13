"""Write plots/hod_weights/data.json: the LRG occupation and the host
masses of its centrals and satellites.

Uses the DESI One-Percent LRG fit at 0.4 < z < 0.6 (arXiv:2306.06314, the
xi(r_p, r_pi) fit: log M_cut = 12.79, log M_1 = 13.88, sigma = 0.21,
alpha = 1.07, kappa = 1.4; masses in Msun/h) in the Zheng et al. (2007)
form, and a Sheth-Tormen mass function built from the same CAMB linear
P(k) as the other generators (sigma(M) with a real-space top hat; A =
0.3222, a = 0.707, p = 0.3).  Writes:
  - <N_c>(M) and <N_s>(M);
  - the host-mass distributions of centrals and satellites,
    p(ln M) proportional to dn/dlnM <N>(M), each normalized to unit area;
  - the satellite fraction, the mean host masses, and the pieces of the
    full sigma_HOD^2 integral with alpha_c = 0.33 and alpha_s = 0.80.

Run from the plots folder with the cocoa python:
  python make_hod_weights_data.py
"""
import json
import os
import sys

import numpy as np
from scipy.special import erfc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sigv_common as sc

HERE = os.path.dirname(os.path.abspath(__file__))
Z_SNAP = 0.5
LOG_MCUT, SIGMA_DEX, LOG_M1, KAPPA, ALPHA = 12.79, 0.21, 13.88, 1.4, 1.07
ALPHA_C, ALPHA_S = 0.33, 0.80
RHO_MEAN = 2.775e11*sc.OMEGA_M  # Msun/h per (Mpc/h)^3
DELTA_C = 1.686

lv = sc.LinearVelocity()


def sigma_M(log_m_h, z):
    """rms linear overdensity in a top hat holding mass M on average."""
    R = (3.0*10.0**log_m_h/(4.0*np.pi*RHO_MEAN))**(1.0/3.0)/sc.h  # Mpc
    x = lv.k*R
    W = 3.0*(np.sin(x) - x*np.cos(x))/x**3
    integrand = lv.k**2*lv.pk.P(z, lv.k)*W**2*lv.k/(2.0*np.pi**2)
    return float(np.sqrt(np.trapz(integrand, lv.lnk)))


def st_mass_function(log_m, z):
    """Sheth-Tormen dn/dlnM [h^3/Mpc^3]: f(nu) = A (1 + (a nu^2)^-p)
    sqrt(2 a/pi) nu exp(-a nu^2/2), A = 0.3222, a = 0.707, p = 0.3."""
    sig = np.array([sigma_M(lm, z) for lm in log_m])
    nu = DELTA_C/sig
    A, a, p = 0.3222, 0.707, 0.3
    f = A*(1.0 + (a*nu**2)**(-p))*np.sqrt(2.0*a/np.pi)*nu*np.exp(-a*nu**2/2.0)
    lnM = np.log(10.0)*log_m
    dlnsig_dlnM = np.gradient(np.log(sig), lnM)
    return f*RHO_MEAN/10.0**log_m*np.abs(dlnsig_dlnM)


def zheng07(log_m):
    """Zheng et al. (2007) in the AbacusHOD writing (arXiv:2110.11412):
    <N_c> = (1/2) erfc[log10(M_cut/M)/(sqrt(2) sigma)],
    <N_s> = ((M - kappa M_cut)/M_1)^alpha <N_c>."""
    ncen = 0.5*erfc((LOG_MCUT - log_m)/(np.sqrt(2.0)*SIGMA_DEX))
    ms = 10.0**log_m - KAPPA*10.0**LOG_MCUT
    nsat = np.where(ms > 0.0, (np.clip(ms, 0.0, None)/10.0**LOG_M1)**ALPHA,
                    0.0)*ncen
    return ncen, nsat


log_m = np.linspace(11.5, 15.2, 200)
lnM = np.log(10.0)*log_m
dndlnM = st_mass_function(log_m, Z_SNAP)
ncen, nsat = zheng07(log_m)
sh = sc.sigma_halo(log_m, Z_SNAP)

n_c = np.trapz(dndlnM*ncen, lnM)
n_s = np.trapz(dndlnM*nsat, lnM)
p_cen = dndlnM*ncen/n_c
p_sat = dndlnM*nsat/n_s
var_c = np.trapz(dndlnM*ncen*(ALPHA_C*sh)**2, lnM)/(n_c + n_s)
var_s = np.trapz(dndlnM*nsat*(ALPHA_S*sh)**2, lnM)/(n_c + n_s)

data = {
    "about": "make_hod_weights_data.py: Zheng07 occupation of the DESI "
             "One-Percent LRG xi fit at 0.4 < z < 0.6 (arXiv:2306.06314: "
             "log M_cut 12.79, log M_1 13.88, sigma 0.21, kappa 1.4, alpha "
             "1.07, alpha_c 0.33, alpha_s 0.80) on a Sheth-Tormen mass "
             "function from the note's fiducial CAMB linear P(k) at z = 0.5; "
             "host-mass distributions p(lnM) ~ dn/dlnM <N>, unit area",
    "z": Z_SNAP,
    "log_m_h": log_m.tolist(),
    "ncen": ncen.tolist(),
    "nsat": nsat.tolist(),
    "p_cen": p_cen.tolist(),
    "p_sat": p_sat.tolist(),
    "f_sat": float(n_s/(n_c + n_s)),
    "mean_log_m_cen": float(np.trapz(p_cen*log_m, lnM)),
    "mean_log_m_sat": float(np.trapz(p_sat*log_m, lnM)),
    "sigma_hod_full": float(np.sqrt(var_c + var_s)),
    "sigma_cen_part": float(np.sqrt(var_c)),
    "sigma_sat_part": float(np.sqrt(var_s)),
}
out = os.path.join(HERE, "hod_weights", "data.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(data, f, indent=1)
print("wrote", out)
print({k: v for k, v in data.items()
       if isinstance(v, float) or isinstance(v, int)})
