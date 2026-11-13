"""Write plots/lin_decomposition/data.json: the split of each group's
velocity dispersion sigma_v,g into the linear flow sigma_lin(zbar_i) and
the halo part sigma_HOD,i = sqrt(sigma_v,g^2 - sigma_lin,i^2), for every
bin of the Delta z = 0.02 layout at the note's fiducial cosmology (one
CAMB linear P(k); sigv_common.py holds the cosmology and the machinery).

Also stores, per dispersion group: the group mean redshift zbar_g, the
halo share w_i = sigma_HOD,i^2/sigma_v,g^2, and the T_lin truth
delta sigma_v,i = (sigma_lin,i^2 - sigma_lin(zbar_g)^2)/(2 sigma_v,g).

Run from the plots folder with the cocoa python:
  python make_lin_decomposition_data.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sigv_common as sc

HERE = os.path.dirname(os.path.abspath(__file__))

lv = sc.LinearVelocity()
zbar = sc.read_zbar_dz0p02()

groups = []
for g, (i0, i1) in sc.GROUP_BINS_DZ0P02.items():
    zb = zbar[i0 - 1:i1]
    zg = float(np.mean(zb))
    slin = np.array([lv.sigma_lin(z) for z in zb])
    sv = sc.SIGV[g]
    shod = np.sqrt(sv**2 - slin**2)
    slin_g = lv.sigma_lin(zg)
    groups.append({
        "name": g, "bins": [int(i0), int(i1)], "sigma_v": sv,
        "zbar": zb.tolist(), "zbar_g": zg,
        "sigma_lin": slin.tolist(),
        "sigma_lin_at_zg": slin_g,
        "sigma_hod": shod.tolist(),
        "sigma_hod_at_zg": float(np.sqrt(sv**2 - slin_g**2)),
        "w": (shod**2/sv**2).tolist(),
        "w_at_zg": float(1.0 - (slin_g/sv)**2),
        "t_lin": ((slin**2 - slin_g**2)/(2.0*sv)).tolist(),
    })

data = {
    "about": "make_lin_decomposition_data.py: sigma_lin(zbar_i) from one "
             "CAMB linear P(k) at the paper fiducial (As_1e9 2.1, ns "
             "0.96605, H0 67.32, omegab 0.04, omegam 0.3, mnu 0.06 eV, "
             "w -1), sigma_lin = a H f sigma_d with sigma_d^2 = "
             "(1/6 pi^2) int P_lin dk; sigma_HOD,i = sqrt(sigma_v,g^2 - "
             "sigma_lin,i^2) on the Delta z = 0.02 bins; sigma_v,g of the "
             "seven dispersion groups from the parameter files",
    "layout": "Delta z = 0.02",
    "sigma_d0_Mpc": lv.sigma_d(0.0),
    "sigma_d0_Mpc_over_h": lv.sigma_d(0.0)*sc.h,
    "groups": groups,
}
out = os.path.join(HERE, "lin_decomposition", "data.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(data, f, indent=1)
print("wrote", out)
