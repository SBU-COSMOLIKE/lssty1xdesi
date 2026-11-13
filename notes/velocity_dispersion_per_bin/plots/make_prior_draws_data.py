"""Write plots/prior_draws/data.json: random truths of the per-bin model,
delta sigma_v,i = w_i sigma_v,g [s_g (zbar_i - zbar_g) + eps_i], on the
Delta z = 0.02 layout.

Reads w_i, zbar_i and zbar_g from plots/lin_decomposition/data.json (run
make_lin_decomposition_data.py first).  For each of the two residual
priors of the study,
  A (population drift): sigma = 10 percent of sigma_HOD, L = 0.2;
  B (selection edges): sigma = 20 percent (BGS groups 30), L = 0.05,
draws per group: a slope s_g from N(0, 1) per unit z (the baseline width)
and a residual eps from the exponential kernel sigma^2 exp(-|dz|/L), then
removes the kernel-weighted straight line in z (the generalized
least-squares fit with the inverse kernel as weights, the projection of
the correlated-prior note), so the draw charges neither the level nor the
slope.  Stores three draws per prior as percent of sigma_v,g.  Draw d of
both priors uses the seed [SEED, d], so the slopes repeat between the
two priors and only the residual changes.

Run from the plots folder: python make_prior_draws_data.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SLOPE_WIDTH = 1.0   # the baseline slope prior, per unit z
PRIORS = {"A": {"sigma": {"default": 0.10}, "L": 0.2},
          "B": {"sigma": {"default": 0.20, "BGS0": 0.30, "BGS1": 0.30},
                "L": 0.05}}
N_DRAWS = 3
SEED = 20261112

with open(os.path.join(HERE, "lin_decomposition", "data.json")) as f:
    dec = json.load(f)


def projected_draw(rng, zb, sigma, L):
    """One residual draw from N(0, sigma^2 exp(-|dz|/L)), with the
    kernel-weighted line removed: eps -> eps - U (U^T Q U)^-1 U^T Q eps,
    U = [1, z - zbar], Q the inverse kernel."""
    n = len(zb)
    K = sigma**2*np.exp(-np.abs(zb[:, None] - zb[None, :])/L)
    eps = np.linalg.cholesky(K + 1e-14*np.eye(n)) @ rng.standard_normal(n)
    if n < 3:
        return np.zeros(n)
    Q = np.linalg.inv(K)
    U = np.column_stack([np.ones(n), zb - zb.mean()])
    coef = np.linalg.solve(U.T @ Q @ U, U.T @ Q @ eps)
    return eps - U @ coef


out_priors = {}
for pname, p in PRIORS.items():
    draws = []
    for d in range(N_DRAWS):
        rng = np.random.default_rng([SEED, d])
        per_group = []
        for g in dec["groups"]:
            zb = np.array(g["zbar"])
            w = np.array(g["w"])
            sig = p["sigma"].get(g["name"], p["sigma"]["default"])
            s_g = SLOPE_WIDTH*rng.standard_normal()
            eps = projected_draw(rng, zb, sig, p["L"])
            delta_pct = 100.0*w*(s_g*(zb - g["zbar_g"]) + eps)
            per_group.append({"name": g["name"], "zbar": zb.tolist(),
                              "delta_pct": delta_pct.tolist()})
        draws.append(per_group)
    out_priors[pname] = {"L": p["L"], "sigma": p["sigma"], "draws": draws}

data = {
    "about": "make_prior_draws_data.py: random truths delta sigma_v,i = "
             "w_i sigma_v,g [s_g (zbar_i - zbar_g) + eps_i] on Delta z = "
             "0.02, in percent of sigma_v,g; slope width 1.0 per unit z, "
             "residual A (10 percent of sigma_HOD, L = 0.2) and B (20 "
             "percent, BGS 30, L = 0.05) with the kernel-weighted line "
             "removed per group; w_i and zbar from "
             "lin_decomposition/data.json; draw d of both priors uses "
             "seed [20261112, d], so the slopes repeat between priors",
    "layout": "Delta z = 0.02",
    "slope_width": SLOPE_WIDTH,
    "priors": out_priors,
}
out = os.path.join(HERE, "prior_draws", "data.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(data, f, indent=1)
print("wrote", out)
