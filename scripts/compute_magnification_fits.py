"""Fit the magnification shift test of notes/lensing_magnification.

The question
------------
The paper forecast sets the magnification coefficient C = 5 s - 2 to zero in
every lens bin. If nature carries magnification at the measured DESI
coefficients and the model leaves it out, where does the fit land, in units
of its own errors? A Fisher forecast's synthetic data are the fiducial model's
own prediction, so the forecast itself has no shift; this is a what-if
question, and its answer is an estimate of how safe the model is for the
future chains.

To first order the answer is linear response. With m(theta) the model, D its
derivative columns at the fiducial point, C_d the data covariance and
F = D^T C_d^-1 D + priors the Fisher matrix, data m(theta_0) + delta d move
the best fit by

    delta theta = F^-1 D^T C_d^-1 delta d,

delta d = m(C) - m(0) the magnification part of the vector
(compute_magnification_vectors.py, the caches model_magnification_<tag>_final).
Nothing else is new: D, the priors, the scale cuts and the covariance are the
paper notebook's (EXAMPLE_EVALUATE_FISHER_MODEL.ipynb), run through its own
cells in one namespace (compute_pair_axes_fits.run_cells).

How delta d enters
------------------
delta d is appended to the notebook's base columns B as one more column per
coefficient set, B_aug = [B | delta d_1 | ...], and the Gram matrices
G = B_aug^T C_d^-1 B_aug (times the area ratio INVCOV_SCALE) are formed on the
rows each cut keeps. In these coordinates delta d is a unit vector e, so

    delta theta = F^-1 A^T G e,        delta chi^2 = e^T G e = G[e, e],

with A the paper model's map from the base columns to its parameters
(padded with zero rows for the appended columns). The Fisher system is that
of the paper fit (compute_sigv_bins_fits.fit: the paper model on the
pessimistic linear-bias prior, the no-data rule, the pseudo-inverse).

What is reported, per cut of the notebook's KMAX_CUTS (h/Mpc at z = 0, a per-bin
cut at k_max/D(zbar_i)) and per coefficient set:
  - "delta_chi2": e^T G e, the size of the neglected signal on the kept rows;
  - "free": the shifts of w0, wa, Omega_m and 10^9 A_s in units of their
    marginalized errors, and "w0wa_ellipse" = sqrt(d^T C_w^-1 d), the shift
    of (w0, wa) in units of its own error ellipse (C_w the marginalized 2 x 2
    covariance), with every nuisance parameter free;
  - "frozen": the same with every nuisance parameter held at its fiducial
    (the cosmology-only Fisher block), in units of that fit's own errors; the
    difference to "free" shows how much of delta d the free nuisances absorb.
No threshold is applied.

The coefficient sets are those of the vectors script ("fiducial", "elg_plus",
"elg_minus"), plus "linear" and "quadratic", the two parts of the fiducial
response separated by the sign-flipped set: (d(C) - d(-C))/2 and
(d(C) + d(-C))/2 - d(0).

Two pair lists
--------------
--list today:   the paper lists, in the notebook's own namespace (all four
                layouts; the derivative base model_derivatives_<tag>_final).
--list pairmax: one layout on its maximal list, in the namespace of the
                pair-axes study (compute_pair_axes_fits.notebook_namespace),
                with every axis point of the gap axis fitted by row selection
                (the point "today" again, then "gap<=g", where the gap g of
                a pair is the separation of its two bins' raw n(z) ranges
                counted in bin widths): the same machinery as the pair-axes
                FoMs, so the shifts and those FoMs belong to one system.

Output: chains/magnification_shifts_<list>[_<tag>].json and a log on stdout.

Run (cocoa environment, start_cocoa.sh sourced, from any folder):
    python compute_magnification_fits.py --list today
    python compute_magnification_fits.py --list pairmax --layout dz0p20
"""
import argparse
import datetime
import json
import os
import sys
import time

# the notebook's BLAS thread count, before numpy loads (compute_pair_axes_fits
# explains it)
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import compute_pair_axes_fits as pa  # noqa: E402
import compute_sigv_bins_fits as sv  # noqa: E402

TIER = pa.TIER
PAPER_VARIANT = pa.PAPER_VARIANT
PRIOR = "pessimistic"
CACHE_NAME = "model_magnification_{tag}_{tier}"
# the coefficient sets read from the cache, in column order
SETS = ("fiducial", "elg_plus", "elg_minus", "fiducial_negative")
# the derived columns built from them: name -> (weights on SETS)
DERIVED = {
    "linear": (0.5, 0.0, 0.0, -0.5),
    "quadratic": (0.5, 0.0, 0.0, 0.5),
}
# what the output reports
REPORTED = ("fiducial", "elg_plus", "elg_minus", "linear", "quadratic")


def parse_arguments():
    """Return the command-line arguments (--list, --layout, --overwrite)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", required=True, choices=("today", "pairmax"))
    parser.add_argument("--layout", choices=tuple(pa.LAYOUTS),
                        help="the layout of --list pairmax")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_delta(ns, key, cache_tag):
    """Read one layout's delta d columns and check they belong to the model.

    Arguments:
      ns        = the notebook namespace.
      key       = the notebook's layout key.
      cache_tag = the cache's layout tag (e.g. 'dz0p20' or 'dz0p20_pairmax').

    Returns:
      (n, len(REPORTED)) array: the columns of REPORTED, in that order.

    Raises:
      RuntimeError when the cache's fingerprint is not the notebook's for
      this layout (the cache belongs to another model); the cache reader
      raises on a missing or mismatched cache.
    """
    fm = ns["fm"]
    name = CACHE_NAME.format(tag=cache_tag, tier=TIER)
    cache = fm.read_fisher_cache(name, tier=TIER)
    header = fm.read_fisher_cache_header(name)
    if header["tiers_and_fingerprints"]["settings/fingerprint"] != ns["FINGERPRINT"][key]:
        raise RuntimeError(f"{cache_tag}: the magnification cache belongs to another "
                           "model than the notebook's derivative base")
    raw = []
    for s in SETS:
        raw.append(np.asarray(cache["delta"][s], dtype=float))
    columns = {}
    for s, vector in zip(SETS, raw):
        columns[s] = vector
    for d, weights in DERIVED.items():
        # a weighted sum of the four raw columns: zip pairs each weight
        # with its column, and sum adds the weighted columns one by one
        columns[d] = sum(w*v for w, v in zip(weights, raw))
    out = []
    for r in REPORTED:
        out.append(columns[r])
    return np.column_stack(out)


def scores(system, response):
    """Turn one parameter response into the reported shifts.

    Arguments:
      system   = a fit (compute_sigv_bins_fits.fit's dict, or frozen_fit's).
      response = delta theta over the system's live parameters.

    Returns:
      dict: "w0", "wa" and each single parameter in units of its
      marginalized error, and "w0wa_ellipse" = sqrt(d^T C_w^-1 d).
    """
    iw = system["iw"]
    d_w = response[iw]
    sigma_w = np.sqrt(np.diag(system["Cw"]))
    out = {"w0": float(d_w[0]/sigma_w[0]), "wa": float(d_w[1]/sigma_w[1]),
           "w0wa_ellipse": float(np.sqrt(d_w @ system["Ci"] @ d_w))}
    for p, i in system["ip"].items():
        out[p] = float(response[i]/system["sigma"][p])
    return out


def frozen_fit(ns, key, gram, M, names):
    """The cosmology-only fit: every nuisance parameter held at its fiducial.

    Holding a parameter fixed deletes its row and column from the Fisher
    matrix (exact conditioning for a Gaussian likelihood), so the system is
    F_c = A_c^T G A_c, A_c the cosmology columns of the paper map; the
    cosmology priors are flat.

    Arguments:
      ns    = the notebook namespace (COSMO and SINGLE_PARAMS are read).
      key   = the notebook's layout key (unused; kept for the fit_cut
              call pattern of the free fit).
      gram  = G of the augmented base at the cut [naug, naug].
      M     = the paper map padded to the augmented base [naug, nparams].
      names = the map's parameter names, in column order.

    Returns:
      dict with "A", "F_inv", "iw", "ip", "Cw", "Ci" and "sigma" (the keys
      scores reads, and "A" and "F_inv" for the response).
    """
    cosmo = []
    for p in ns["COSMO"]:
        cosmo.append(names.index(p))
    A = M[:, cosmo]
    F_inv = np.linalg.inv(A.T @ gram @ A)
    used = list(ns["COSMO"])
    iw = [used.index("w0"), used.index("wa")]
    ip = {}
    sigma = {}
    for p in ns["SINGLE_PARAMS"]:
        ip[p] = used.index(p)
        sigma[p] = float(np.sqrt(F_inv[ip[p], ip[p]]))
    Cw = F_inv[np.ix_(iw, iw)]
    return {"A": A, "F_inv": F_inv, "iw": iw, "ip": ip, "Cw": Cw,
            "Ci": np.linalg.inv(Cw), "sigma": sigma}


def fit_cut(ns, key, m, gram, gram_shipped, M, tags, names, nb):
    """Fit one cut: the shifts of every reported column, nuisances free and frozen.

    Arguments:
      ns, key       = the notebook namespace and the layout key.
      m             = the cut index in KMAX_CUTS.
      gram          = G of the augmented base at the cut [naug, naug].
      gram_shipped  = G on the shipped mask (the no-data reference).
      M, tags, names = the paper map padded to the augmented base, its prior
                      tags and parameter names.
      nb            = the number of base columns (the reported columns follow).

    Returns:
      dict {column: {"delta_chi2", "free", "frozen"}} or {"status": ...} when
      the paper fit does not exist at this cut.
    """
    naug = gram.shape[0]
    system, status = sv.fit(ns, key, m, gram, gram_shipped, M, tags,
                            np.zeros((naug, 0)), np.zeros((0, 0)))
    if system is None:
        return {"status": status}
    frozen = frozen_fit(ns, key, gram, M, names)
    out = {"status": "fit", "fom": system["fom"], "fom_om_as": system["fom_om_as"]}
    # enumerate yields (position, element): j counts the reported columns
    for j, r in enumerate(REPORTED):
        col = nb + j
        free_response = system["F_inv"] @ (system["A"].T @ gram[:, col])
        frozen_response = frozen["F_inv"] @ (frozen["A"].T @ gram[:, col])
        out[r] = {"delta_chi2": float(gram[col, col]),
                  "free": scores(system, free_response),
                  "frozen": scores(frozen, frozen_response)}
    return out


def padded_map(ns, key, naug):
    """The paper map with zero rows for the appended columns, its tags and names.

    The map M sends the model's parameters to the base columns; the
    appended delta d columns belong to no parameter, so their rows are
    zero: a parameter step never moves along delta d.

    Arguments:
      ns   = the notebook namespace.
      key  = the notebook's layout key.
      naug = the number of augmented base columns (base + appended).

    Returns:
      (M, tags, names): the padded map [naug, nparams], the prior tags
      and the parameter names of its columns.
    """
    M0, names, tags = ns["COLUMNS"][(key, PAPER_VARIANT)]
    M = np.vstack([M0, np.zeros((naug - M0.shape[0], M0.shape[1]))])
    return M, tags, names


def fit_today():
    """The paper lists of all four layouts in the notebook's namespace.

    Per layout: read the delta d columns, append them to the base columns
    B, form the Gram matrices of the augmented base on the rows each cut
    keeps, and fit every cut (fit_cut).

    Returns:
      dict {layout tag: {"key", "delta_chi2_shipped", "cuts"}}, the
      "fits" entry of the output JSON.
    """
    ns = sv.build_namespace()
    result = {}
    for tag, spec in pa.LAYOUTS.items():
        key = spec["key"]
        start = time.time()
        delta = read_delta(ns, key, tag)
        B = ns["base_matrix"](key)[0]
        if B.shape[0] != delta.shape[0]:
            raise RuntimeError(f"{tag}: the base has {B.shape[0]} rows, delta d "
                               f"{delta.shape[0]}")
        nb = B.shape[1]
        grams, _ = ns["layout_grams"](key, np.hstack([B, delta]))
        del B
        M, tags, names = padded_map(ns, key, nb + delta.shape[1])
        cuts = {}
        for m, kmax in enumerate(ns["KMAX_CUTS"]):
            label = f"{float(kmax):g}"
            if grams[m] is None:
                cuts[label] = {"status": "no band kept"}
                continue
            cuts[label] = fit_cut(ns, key, m, grams[m], grams["shipped"], M, tags,
                                  names, nb)
        shipped_chi2 = {}
        for j, r in enumerate(REPORTED):
            shipped_chi2[r] = float(grams["shipped"][nb + j, nb + j])
        result[tag] = {"key": key, "delta_chi2_shipped": shipped_chi2, "cuts": cuts}
        log(tag, "today", cuts, time.time() - start)
    return result


def fit_pairmax(tag):
    """One layout on its maximal list: every gap point by row selection.

    The maximal list's base and delta d hold every pair once; an axis
    point (a pair list) is fitted by keeping the rows of its pairs
    (keep_pair below) when the Gram matrix is formed, so every point
    shares one base - the pair-axes machinery.

    Arguments:
      tag = the layout tag (a compute_pair_axes_fits.LAYOUTS key).

    Returns:
      dict {"key", "points": {point: {"n_pairs", "delta_chi2_shipped",
      "cuts"}}}, the "fits" entry of the output JSON.
    """
    key = pa.LAYOUTS[tag]["key"]
    layout = pa.read_layout(tag)
    points = pa.axis_points(tag=tag, layout=layout)
    ns = pa.notebook_namespace(tag)
    start = time.time()
    delta = read_delta(ns, key, f"{tag}_pairmax")
    B = ns["base_matrix"](key)[0]
    if B.shape[0] != delta.shape[0]:
        raise RuntimeError(f"{tag}: the base has {B.shape[0]} rows, delta d "
                           f"{delta.shape[0]}")
    nb = B.shape[1]
    B_aug = np.hstack([B, delta])
    del B
    C = ns["layout_covariance"](key, B_aug.shape[0])
    pairs = []
    for pair in ns["TABLES"][key]["pairs"]:
        pairs.append(tuple(pair))
    M, tags, names = padded_map(ns, key, B_aug.shape[1])
    result = {"key": key, "points": {}}
    for point, plist in points.items():
        # the gap axis only: the sigma_v axis points answer another question
        if not (point == "today" or point.startswith("gap<=")):
            continue
        members = set(plist)
        keep_pair = np.zeros(len(pairs), dtype=bool)
        for p in range(len(pairs)):
            keep_pair[p] = pairs[p] in members
        shipped, _ = pa.point_gram(namespace=ns, key=key, B=B_aug, C=C, m="shipped",
                                   keep_pair=keep_pair)
        cuts = {}
        for m, kmax in enumerate(ns["KMAX_CUTS"]):
            label = f"{float(kmax):g}"
            # a cut that keeps no entry has no fit (the notebook's
            # layout_grams returns None there; point_gram cannot factor an
            # empty block)
            if ns["kept_rows"](key, m).size == 0:
                cuts[label] = {"status": "no band kept"}
                continue
            gram, n_rows = pa.point_gram(namespace=ns, key=key, B=B_aug, C=C, m=m,
                                         keep_pair=keep_pair)
            cuts[label] = fit_cut(ns, key, m, gram, shipped, M, tags, names, nb)
        shipped_chi2 = {}
        for j, r in enumerate(REPORTED):
            shipped_chi2[r] = float(shipped[nb + j, nb + j])
        result["points"][point] = {"n_pairs": len(plist),
                                   "delta_chi2_shipped": shipped_chi2, "cuts": cuts}
        log(tag, point, cuts, time.time() - start)
    return result


def log(tag, point, cuts, seconds):
    """Print the fiducial set's (w0, wa) ellipse shift and delta chi^2 per cut."""
    fields = []
    for label, entry in cuts.items():
        if "fiducial" not in entry:
            continue
        e = entry["fiducial"]
        fields.append(f"{label}: dchi2 {e['delta_chi2']:.3g}, r {e['free']['w0wa_ellipse']:.3f}"
                      f" (frozen {e['frozen']['w0wa_ellipse']:.3f})")
    print(f"{tag} {point} [{seconds:.0f} s]  " + "; ".join(fields), flush=True)


def main():
    """Fit one pair list and write the JSON file."""
    args = parse_arguments()
    if args.list == "pairmax" and args.layout is None:
        raise SystemExit("--list pairmax needs --layout")
    # one simple ternary: the pairmax output carries its layout
    suffix = "today" if args.list == "today" else f"pairmax_{args.layout}"
    output = os.path.abspath(os.path.join(pa.CHAINS, f"magnification_shifts_{suffix}.json"))
    if os.path.exists(output) and not args.overwrite:
        raise FileExistsError(f"{output} exists: pass --overwrite to replace it")
    if args.list == "today":
        fits = fit_today()
    else:
        pa.check_prerequisites(args.layout)
        fits = fit_pairmax(args.layout)
    result = {
        "about": ("Linear-response shifts of the paper model when the data carry lensing "
                  "magnification; written by scripts/compute_magnification_fits.py (its "
                  "docstring defines every entry)."),
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "list": args.list,
        "prior": PRIOR,
        "variant": PAPER_VARIANT,
        "kmax_unit": "h/Mpc at z = 0 (per-bin cut at k_max/D(zbar_i))",
        "columns": list(REPORTED),
        "fits": fits,
    }
    with open(output, "w") as handle:
        json.dump(result, handle, indent=1)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
