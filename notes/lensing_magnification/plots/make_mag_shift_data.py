"""Write plots/mag_shift_kmax/data.json from the magnification fits.

Reads the outputs of scripts/compute_magnification_fits.py in the project's
chains/ folder:
  magnification_shifts_today.json          (the paper pair lists, every layout)
  magnification_shifts_pairmax_<tag>.json  (one layout on its maximal list)
and keeps, per layout, per pair list and per coefficient set, the absolute
linear-response shifts of the paper model with every nuisance free: (w0, wa)
in units of its error ellipse, Omega_m and A_s in units of their marginalized
errors, at every cut of the notebook's KMAX_CUTS. The pairmax list is the
largest gap point of the layout (gap<=3, or gap<=2 on Delta z = 0.02 and 0.2).

Run (cocoa environment): python make_mag_shift_data.py <project folder>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# per layout: (chains tag, plain name, panel label, colorblind-safe color)
LAYOUTS = [
    ("dcom62", "Delta d_com = 62 Mpc", r"$\Delta d_\mathrm{com} = 62$ Mpc", "#0072B2"),
    ("dz0p02", "Delta z = 0.02", r"$\Delta z = 0.02$", "#E69F00"),
    ("dz0p05", "Delta z = 0.05", r"$\Delta z = 0.05$", "#CC79A7"),
    ("dz0p20", "Delta z = 0.2", r"$\Delta z = 0.2$", "#56B4E9"),
]
# the coefficient sets kept for the figure (the fits' "linear" and
# "quadratic" diagnostic columns are not drawn)
SETS = ["fiducial", "elg_plus", "elg_minus"]
# per scored parameter: (data.json key, field name in the fits' "free" dict)
KEYS = [("w0wa", "w0wa_ellipse"), ("omegam", "omegam"), ("As", "As_1e9")]


def series(cuts, labels, column):
    """The absolute shifts of one column at every cut label (None without a fit).

    Arguments:
      cuts   = the "cuts" dict of one fit (cut label -> entries).
      labels = the cut labels to read, in plot order.
      column = the coefficient set to read (an entry of SETS).

    Returns:
      dict {data.json key: list of |shift| per cut label, None where that
      cut has no fit} for the three KEYS, every shift with the nuisances
      free.
    """
    out = {}
    for key, field in KEYS:
        values = []
        for label in labels:
            entry = cuts.get(label, {})
            if column not in entry:
                values.append(None)
            else:
                values.append(abs(float(entry[column]["free"][field])))
        out[key] = values
    return out


def main():
    """Collect the shifts of every layout and write data.json.

    Arguments (command line): the project folder, whose chains/ holds the
    magnification_shifts_*.json files of compute_magnification_fits.py.

    Side effects: writes mag_shift_kmax/data.json next to this script.

    Raises:
      IndexError without the project-folder argument; FileNotFoundError
      when a fits file is missing (run both lists of the fits script
      first); KeyError when a file lacks a layout or a coefficient set.
    """
    project = sys.argv[1]
    chains = os.path.join(project, "chains")
    with open(os.path.join(chains, "magnification_shifts_today.json")) as f:
        today = json.load(f)
    labels = None
    layouts = []
    for tag, name, label, color in LAYOUTS:
        cuts_today = today["fits"][tag]["cuts"]
        with open(os.path.join(chains, f"magnification_shifts_pairmax_{tag}.json")) as f:
            pm = json.load(f)["fits"]["points"]
        # list(pm) is the point names in file order; the fits script
        # writes the gap points smallest to largest, so the last entry is
        # the layout's largest gap point (the maximal list drawn)
        largest = list(pm)[-1]
        cuts_pm = pm[largest]["cuts"]
        if labels is None:
            labels = sorted(set(cuts_today) | set(cuts_pm), key=float)
        entry = {"name": name, "label": label, "color": color, "pairmax_point": largest,
                 "n_pairs": {"paper": int(pm["today"]["n_pairs"]),
                             "pairmax": int(pm[largest]["n_pairs"])},
                 "paper": {}, "pairmax": {}}
        for s in SETS:
            entry["paper"][s] = series(cuts_today, labels, s)
            entry["pairmax"][s] = series(cuts_pm, labels, s)
        layouts.append(entry)
    data = {
        "about": ("Absolute linear-response shifts of the paper model (one-loop, alpha_2 = 0, "
                  "sigma_v widths x6, pessimistic linear-bias prior; every nuisance free) "
                  "when the data carry lensing magnification at C = 5s - 2 per bias group "
                  "(fiducial: BGS 2, LRG 3, ELG 0; elg_plus / elg_minus: ELG +2 / -2). "
                  "(w0, wa) in units of its error ellipse, Omega_m and A_s in units of "
                  "their marginalized errors. Written by plots/make_mag_shift_data.py from "
                  "chains/magnification_shifts_*.json (scripts/compute_magnification_fits.py)."),
        "kmax_h_per_Mpc": [float(x) for x in labels],
        "sets": SETS,
        "layouts": layouts,
    }
    os.makedirs(os.path.join(HERE, "mag_shift_kmax"), exist_ok=True)
    with open(os.path.join(HERE, "mag_shift_kmax", "data.json"), "w") as f:
        json.dump(data, f, indent=1)
    print("wrote", os.path.join(HERE, "mag_shift_kmax", "data.json"))


if __name__ == "__main__":
    main()
