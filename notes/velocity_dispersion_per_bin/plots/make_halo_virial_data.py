"""Write plots/halo_virial/data.json: the virial velocity sigma_halo(M, z).

sigma_halo is the one-dimensional velocity dispersion of the dark matter in
a halo of mass M_200c, the Evrard et al. (2008, astro-ph/0702241) fit
    sigma_halo = 1082.9 km/s [h(z) M_200c / 10^15 Msun]^0.3361,
with h(z) = H(z)/(100 km/s/Mpc), evaluated here with
h(z) M[Msun] = E(z) M[Msun/h] and E(z) of flat LambdaCDM with
Omega_m = 0.3 (the note's fiducial; the massive neutrinos change E(z) by
less than 0.1 percent at these redshifts).  Curves at the mean redshifts
of four dispersion groups, the singular-isothermal-sphere estimate
sigma^2 = G M_200c / (2 R_200c) at z = 0.5 for comparison, and the mean
host masses of the DESI LRG and ELG fits (arXiv:2306.06314, the
xi(r_p, r_pi) fits at 0.4 < z < 0.6; arXiv:2306.06319, zbar = 0.95;
satellite-host means from the Sheth-Tormen integral of
make_hod_weights_data.py and the two-mass toy of the note).

Run from the plots folder: python make_halo_virial_data.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OMEGA_M = 0.3
G = 4.300917270e-9  # Mpc (km/s)^2 / Msun


def efunc(z):
    return np.sqrt(OMEGA_M*(1.0 + z)**3 + 1.0 - OMEGA_M)


def sigma_halo(log_m_h, z):
    return 1082.9*(efunc(z)*10.0**np.asarray(log_m_h)/1e15)**0.3361


def sigma_sis(log_m_h, z):
    """SIS at the same M_200c: M = 100 H^2 R^3/G, sigma^2 = G M/(2 R)."""
    m_sun = 10.0**np.asarray(log_m_h)/0.6732  # Msun at h = 0.6732
    H = 67.32*efunc(z)
    R = (G*m_sun/(100.0*H**2))**(1.0/3.0)
    return np.sqrt(G*m_sun/(2.0*R))


log_m = np.linspace(11.0, 15.0, 81)
curves = []
for name, z in [("BGS", 0.25), ("LRG1", 0.5), ("LRG3 / ELG1", 0.95),
                ("ELG2", 1.35)]:
    curves.append({"name": name, "z": z,
                   "sigma_halo": sigma_halo(log_m, z).tolist()})

markers = [
    {"label": "LRG centrals", "log_m": 13.04, "z": 0.5,
     "sigma_halo": float(sigma_halo(13.04, 0.5))},
    {"label": "LRG satellites", "log_m": 13.82, "z": 0.5,
     "sigma_halo": float(sigma_halo(13.82, 0.5))},
    {"label": "ELG centrals", "log_m": 11.78, "z": 0.95,
     "sigma_halo": float(sigma_halo(11.78, 0.95))},
    {"label": "ELG satellites", "log_m": 13.0, "z": 0.95,
     "sigma_halo": float(sigma_halo(13.0, 0.95))},
]

data = {
    "about": "make_halo_virial_data.py: Evrard et al. (2008) "
             "sigma_halo(M_200c, z) = 1082.9 km/s (E(z) M[Msun/h]/1e15 Msun)"
             "^0.3361 at the mean redshifts of four dispersion groups, the "
             "SIS estimate at z = 0.5, and the mean host masses of the DESI "
             "One-Percent LRG (arXiv:2306.06314, xi fit 0.4 < z < 0.6; "
             "satellite-host mean from the Sheth-Tormen integral) and ELG "
             "(arXiv:2306.06319) fits; E(z) of flat LambdaCDM Omega_m = 0.3",
    "log_m_h": log_m.tolist(),
    "curves": curves,
    "sis_z0p5": sigma_sis(log_m, 0.5).tolist(),
    "markers": markers,
}
out = os.path.join(HERE, "halo_virial", "data.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(data, f, indent=1)
print("wrote", out)
