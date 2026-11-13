"""Write the data.json of the three result figures of the per-bin sigma_v
study, from the result files of scripts/compute_sigv_bins_fits.py.

The fits script writes one file results_<tag>.json per bin layout (tag
dcom62, dz0p02, dz0p05 or dz0p20). Each holds, at every scale cut k_max
(the z = 0 value; each bin is cut at k_max/D(zbar_i)), the FoMs of the
variants V0-V4 and the accuracy scores eta under each truth. This script
copies the numbers the figures draw into the plot folders:

  tlin_shift_kmax/data.json
      the deterministic shift of today's model V0 (one sigma_v per group)
      under the truth T_lin (the linear flow's own redshift dependence),
      for (w0, wa) in units of its error ellipse, sqrt(dp^T C^-1 dp), and
      for Omega_m and A_s in units of their own sigma;
  sigv_eta_kmax/data.json
      eta = sqrt(<chi^2>/n_dof) of V0 and of V1 (the halo-part slope at
      the baseline width 1.0) under the truth T_lin + B, for (w0, wa),
      Omega_m and A_s; the other truths are stored too;
  sigv_fom_loss_kmax/data.json
      the FoM loss 100 (1 - FoM/FoM_V0), in percent, of every variant, for
      FoM(w0, wa) and FoM(Omega_m, 10^9 A_s); the figure draws V1 at the
      baseline width and V4 (every bin free).

A cut where a layout keeps no band is stored as None (a gap in the line).

Run from the plots folder:
    python make_sigv_results_data.py --results <folder of results_<tag>.json>
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# The four bin layouts in the project's order, with the names, labels and
# colors of the paper-figures skill (the same as every other note).
LAYOUTS = [
    {"tag": "dcom62", "name": "Delta d_com = 62 Mpc",
     "label": r"$\Delta d_\mathrm{com} = 62$ Mpc", "color": "#0072B2"},
    {"tag": "dz0p02", "name": "Delta z = 0.02",
     "label": r"$\Delta z = 0.02$", "color": "#E69F00"},
    {"tag": "dz0p05", "name": "Delta z = 0.05",
     "label": r"$\Delta z = 0.05$", "color": "#CC79A7"},
    {"tag": "dz0p20", "name": "Delta z = 0.2",
     "label": r"$\Delta z = 0.2$", "color": "#56B4E9"},
]
# the scored parameters of the accuracy test: the result-file key and the
# key this script writes
SCORES = [("w0wa", "w0wa"), ("omegam", "omegam"), ("As_1e9", "As")]
# the variants and the truths the figures draw
SHIFT_VARIANT = "V0"
SHIFT_TRUTH = "T_lin"
ETA_VARIANTS = ["V0", "V1 1.0"]
ETA_TRUTH = "T_lin + B"
LOSS_VARIANTS = ["V1 0.5", "V1 1.0", "V1 2.0", "V2", "V3", "V3 white", "V4"]
FOMS = ["fom", "fom_om_as"]
# the cuts of the study [h/Mpc at z = 0]
KMAX = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]


def parse_arguments():
    """The command line: the folder of the result files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", required=True,
                        help="folder of results_<tag>.json (compute_sigv_bins_fits.py --out)")
    return parser.parse_args()


def cut_label(kmax):
    """The key of a cut in the result files: 0.05 -> '0.05', 0.2 -> '0.2'."""
    return f"{kmax:g}"


def per_cut(result, pick):
    """One value per cut of KMAX: pick(entry) for a fitted cut, None where
    the layout keeps no band."""
    values = []
    for kmax in KMAX:
        entry = result["cuts"][cut_label(kmax)]
        if "variants" not in entry:
            values.append(None)
        else:
            values.append(float(pick(entry)))
    return values


def write(folder, data):
    """Write data to <folder>/data.json under the plots folder."""
    path = os.path.join(HERE, folder, "data.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=1)
    print("wrote", path)


def main():
    args = parse_arguments()
    results = {}
    for layout in LAYOUTS:
        path = os.path.join(args.results, f"results_{layout['tag']}.json")
        with open(path) as handle:
            results[layout["tag"]] = json.load(handle)

    about_common = ("scripts/compute_sigv_bins_fits.py (results_<tag>.json), copied by "
                    "make_sigv_results_data.py; k_max is the z = 0 value of the per-bin "
                    "cut k_max/D(zbar_i); None marks a cut that keeps no band")

    # 1. the deterministic shift of V0 under T_lin
    layouts = []
    for layout in LAYOUTS:
        result = results[layout["tag"]]
        shift = {}
        for key, out in SCORES:
            shift[out] = per_cut(
                result, lambda e, k=key: e["eta"][SHIFT_VARIANT][SHIFT_TRUTH][k + "_shift_only"])
        layouts.append({"name": layout["name"], "label": layout["label"],
                        "color": layout["color"], "nbins": result["nbins"],
                        "shift": shift})
    write("tlin_shift_kmax", {
        "about": about_common + "; the shift of today's model V0 under the truth T_lin, "
                 "(w0, wa) as sqrt(dp^T C^-1 dp) of the 2x2 marginalized covariance, "
                 "Omega_m and A_s as |dp|/sigma",
        "variant": SHIFT_VARIANT, "truth": SHIFT_TRUTH,
        "kmax_h_per_Mpc": KMAX, "layouts": layouts})

    # 2. eta of V0 and V1 (baseline slope) under T_lin + B, and every truth
    layouts = []
    for layout in LAYOUTS:
        result = results[layout["tag"]]
        truths = list(result["cuts"]["0.2"]["eta"]["V0"])
        eta = {}
        for variant in ETA_VARIANTS:
            eta[variant] = {}
            for truth in truths:
                eta[variant][truth] = {}
                for key, out in SCORES:
                    eta[variant][truth][out] = per_cut(
                        result, lambda e, v=variant, t=truth, k=key: e["eta"][v][t][k])
        layouts.append({"name": layout["name"], "label": layout["label"],
                        "color": layout["color"], "nbins": result["nbins"], "eta": eta})
    write("sigv_eta_kmax", {
        "about": about_common + "; eta = sqrt(<chi^2>/n_dof) with chi^2 = dp^T C_p^-1 dp, "
                 "n_dof = 2 for (w0, wa) and 1 for Omega_m or A_s, over the linear-bias "
                 "truths of the paper prior and the sigma_v truth named",
        "variants": ETA_VARIANTS, "truth": ETA_TRUTH,
        "kmax_h_per_Mpc": KMAX, "layouts": layouts})

    # 3. the FoM loss of every variant against V0, in percent
    layouts = []
    for layout in LAYOUTS:
        result = results[layout["tag"]]
        loss = {}
        for variant in LOSS_VARIANTS:
            loss[variant] = {}
            for fom in FOMS:
                loss[variant][fom] = per_cut(
                    result, lambda e, v=variant, f=fom:
                    100.0*(1.0 - e["variants"][v][f]/e["variants"]["V0"][f]))
        v0 = {}
        for fom in FOMS:
            v0[fom] = per_cut(result, lambda e, f=fom: e["variants"]["V0"][f])
        layouts.append({"name": layout["name"], "label": layout["label"],
                        "color": layout["color"], "nbins": result["nbins"],
                        "loss_percent": loss, "fom_v0": v0})
    write("sigv_fom_loss_kmax", {
        "about": about_common + "; the FoM loss 100 (1 - FoM/FoM_V0) of each variant, "
                 "for FoM(w0, wa) and FoM(Omega_m, 10^9 A_s); fom_v0 holds today's "
                 "model's FoMs",
        "drawn": ["V1 1.0", "V4"],
        "kmax_h_per_Mpc": KMAX, "layouts": layouts})


if __name__ == "__main__":
    main()
