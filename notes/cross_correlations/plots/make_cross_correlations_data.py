"""Write the data.json files of the two figures of notes/cross_correlations.

What the note shows. The lssty1xdesi forecast cuts the DESI galaxies into
thin redshift bins and measures, besides the angular power spectrum C_ii
of every bin i (its auto spectrum), the cross spectrum C_ij of some pairs
of bins (i, j). Which pairs are measured is decided by the redshift
distributions n_i(z): the histogram of the measured redshifts of the
galaxies of bin i. Two bins are paired when their n(z) overlap or touch.
The model never uses these histograms directly: it first blurs each one
with the Fingers-of-God (FoG) smearing, a Gaussian in redshift that
stands for the random velocities of galaxies inside their halos. The two
figures draw the pairs (nz_overlap_layouts) and the blur
(fog_smearing).

Inputs, read from the project's data/ and likelihood/ folders for each of
the four bin layouts (Delta d_com = 62 Mpc, the comoving layout whose
files are named dr3, and Delta z = 0.02, 0.05 and 0.2):

  data/<stem>_bins.txt  = one row per bin: index, sample (bgs, lrg, elg),
                          group index, group name (BGS, LRG1-3, ELG1-2)
                          and mean redshift zbar;
  data/<stem>.nz        = the n(z) table: column 0 the redshift grid,
                          column i + 1 the histogram of bin i. The grid is
                          the DESI grid refined five-fold by linear
                          interpolation (scripts/make_desi_nz.py), so
                          every fifth row is an original DESI node;
  data/<stem>_pairs.txt = the measured pairs (i, j), i < j, 0-based;
  likelihood/params_desi[_<case>].yaml = the velocity dispersions sigma_v
                          of the seven FoG groups: the prior centre of
                          each group leader, the value every evaluation
                          example starts from.

Outputs, next to this script:

  nz_overlap_layouts/data.json = per layout, every bin's raw n(z) on the
      DESI nodes (scaled to unit peak), the pairs, whether each pair's
      raw ranges overlap or touch, the redshift range shared by LRG3 and
      ELG1, and the counts the note quotes;
  nz_overlap_zoom/data.json = the same for the corner where LRG2 ends
      and LRG3 and ELG1 begin, on the two thin layouts, with the bins
      numbered from 1 in data-vector order;
  fog_smearing/data.json = the raw and FoG-convolved n(z) of the bins in
      one narrow redshift window, sigma_z and the width of every bin,
      the overlap integral of every pair of neighbouring bins with and
      without FoG, and of every pair outside the data vector that the
      blur makes overlap.

The FoG convolution is the one of likelihood/desi_cl.py
(set_lens_related): scipy's gaussian_filter1d on the refined grid, with
the width sigma_z = (1 + zbar) sigma_v / c expressed in grid steps and
mode="constant" (zero beyond the grid ends). When the covariance
adapter's convolved tables covariance/<stem>_fog.nz exist (they are
derived files, outside git), the script compares its result with them.

Run from any folder with the cocoa environment's Python:

    python -I make_cross_correlations_data.py
    python -I make_cross_correlations_data.py --project /path/to/lssty1xdesi
"""
import argparse
import json
import os

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
# this file sits in <project>/notes/cross_correlations/plots/
DEFAULT_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

# speed of light [km/s]: a line-of-sight velocity v shifts a measured
# redshift by (1 + z) v / c
C_KMS = 299792.458

# The four layouts, top to bottom as the figures draw them: the file stem
# of each, the case suffix of its parameter file, the label the figures
# print, and the house color and marker of the layout
# (.claude/skills/paper-figures/SKILL.md).
LAYOUTS = [
    {"key": "dr1_dz0p20",
     "params": "params_desi_dr1_dz0p20.yaml",
     "label": "Delta z = 0.2",
     "color": "#56B4E9",
     "marker": "^"},
    {"key": "dr1_dz0p05",
     "params": "params_desi_dr1_dz0p05.yaml",
     "label": "Delta z = 0.05",
     "color": "#CC79A7",
     "marker": "D"},
    {"key": "dr1_dz0p02",
     "params": "params_desi_dr1_dz0p02.yaml",
     "label": "Delta z = 0.02",
     "color": "#E69F00",
     "marker": "s"},
    {"key": "dr3",
     "params": "params_desi.yaml",
     "label": "Delta d_com = 62 Mpc",
     "color": "#0072B2",
     "marker": "o"},
]

# The n(z) table refines the DESI grid five-fold (REFINE of
# scripts/make_desi_nz.py): rows 0, 5, 10, ... are the DESI nodes.
REFINE = 5

# The pairing rule of scripts/make_desi_dz_case.py (cross_pairs): two
# bins pair when their raw ranges overlap or come within 1.5 DESI grid
# steps. Neighbouring bins either share a node or sit one node apart, so
# 1.5 steps catches both and nothing farther.
PAIR_SLACK_STEPS = 1.5

# The redshift window of the FoG close-up: inside LRG1 (one sigma_v
# group), away from the BGS-LRG boundary at z = 0.4, wide enough for
# three to five thin bins and one full Delta z = 0.05 bin.
ZOOM_WINDOW = (0.435, 0.535)
# the layouts of the close-up, top to bottom: the two thin layouts and
# Delta z = 0.05 for contrast
ZOOM_LAYOUTS = ["dr1_dz0p02", "dr3", "dr1_dz0p05"]
# keep every second refined row of the convolved curves: the FoG edge
# (sigma_z ~ 1.8e-3) still gets about 13 points per sigma
ZOOM_STRIDE = 2

# The close-up of the LRG-ELG overlap: the corner where LRG2 ends and
# LRG3 and ELG1 begin, on the two thin layouts (top to bottom). The window
# shows the one LRG2-ELG1 pair and the first LRG3-ELG1 pairs.
OVERLAP_ZOOM_WINDOW = (0.763, 0.887)
OVERLAP_ZOOM_LAYOUTS = ["dr1_dz0p02", "dr3"]

# An unmeasured pair counts as overlapping after the blur when its
# convolved overlap integral exceeds this value: far above the 1e-14 that
# the 4-sigma-truncated Gaussian tails of two next-but-one bins can reach,
# far below the 1e-2 of any measured neighbour pair.
NEAR_THRESHOLD = 1.0e-6

# decimals kept in data.json: n(z) at unit peak to 1e-5, redshifts to
# 1e-7 (the refined step is 7e-5)
N_DECIMALS = 5
Z_DECIMALS = 7


def read_bins(path):
    """Read one bin table into lists of samples, groups and mean redshifts.

    Arguments:
      path = data/<stem>_bins.txt (columns: index, sample, group index,
             group name, mean redshift; lines starting with # skipped).

    Returns:
      (samples, groups, zbar): two lists of strings and a float array,
      one entry per bin in data-vector order.
    """
    samples = []
    groups = []
    zbar = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            samples.append(fields[1])
            groups.append(fields[3])
            zbar.append(float(fields[4]))
    return samples, groups, np.array(zbar)


def read_sigma_v(path, nbin):
    """Read the default velocity dispersion of every bin from a params yaml.

    Each FoG group has one leader parameter DESI_SIGV_<n> carrying the
    prior; every other member is tied to it by a cobaya value function,
    the string 'lambda DESI_SIGV_<m>: DESI_SIGV_<m>'. The default of a
    leader is its prior centre (loc), the value the evaluation examples
    set; a member takes its leader's.

    Arguments:
      path = likelihood/params_desi[_<case>].yaml.
      nbin = the number of bins of the layout.

    Returns:
      float array [nbin] of sigma_v in km/s, and the dict
      {leader name: (group name, sigma_v)}, the group name read off the
      leader's latex label \\sigma_v^\\mathrm{<group>}.

    Raises:
      ValueError when a bin's entry is missing or neither a leader nor a
      plain tie to one.
    """
    with open(path) as f:
        params = yaml.safe_load(f)
    sigma_v = np.zeros(nbin)
    leaders = {}
    for n in range(1, nbin + 1):
        name = f"DESI_SIGV_{n}"
        if name not in params:
            raise ValueError(f"{path}: {name} is missing; every bin needs one")
        entry = params[name]
        if isinstance(entry, dict) and "prior" in entry:
            value = float(entry["prior"]["loc"])
            # the group name is the text inside \mathrm{...} of the label
            latex = entry.get("latex", "")
            if "\\mathrm{" not in latex:
                raise ValueError(f"{path}: {name} has the label {latex!r}; "
                                 "expected \\sigma_v^\\mathrm{<group>}")
            start = latex.index("\\mathrm{") + len("\\mathrm{")
            group = latex[start:latex.index("}", start)]
            leaders[name] = (group, value)
            sigma_v[n - 1] = value
            continue
        # a tied member: the value string names its leader between
        # "lambda " and ":"
        text = entry["value"] if isinstance(entry, dict) else ""
        if not text.startswith("lambda ") or ":" not in text:
            raise ValueError(f"{path}: {name} = {entry!r} is neither a leader "
                             "nor a tie 'lambda LEADER: LEADER'")
        leader = text[len("lambda "):text.index(":")].strip()
        if leader not in leaders:
            raise ValueError(f"{path}: {name} is tied to {leader}, which is not "
                             "a leader listed before it")
        sigma_v[n - 1] = leaders[leader][1]
    return sigma_v, leaders


def raw_ranges(z_raw, n_raw):
    """Find the first and last DESI node where each bin's n(z) is nonzero.

    Arguments:
      z_raw = the DESI redshift nodes, shape [nz].
      n_raw = the raw n(z) on those nodes, shape [nz, nbin].

    Returns:
      int array [nbin, 2] of node indices (first, last).
    """
    nbin = n_raw.shape[1]
    ranges = np.zeros((nbin, 2), dtype=int)
    for i in range(nbin):
        live = np.nonzero(n_raw[:, i] > 0.0)[0]
        if live.size == 0:
            raise ValueError(f"bin {i} has an empty n(z)")
        ranges[i, 0] = live[0]
        ranges[i, 1] = live[-1]
    return ranges


def overlap(n_a, n_b, z):
    """Return the normalized overlap integral of two distributions.

    O = integral n_a n_b dz / sqrt(integral n_a^2 dz integral n_b^2 dz):
    1 for identical shapes, 0 for disjoint ranges, and independent of
    how each distribution is normalized. For two thin bins it is close
    to the Limber correlation coefficient C_ab / sqrt(C_aa C_bb), since
    in that approximation every spectrum integrates the product of its
    two kernels against a power that barely changes across a thin bin.

    Arguments:
      n_a, n_b = the two distributions on the grid z, shape [nz].
      z        = the redshift grid, shape [nz].

    Returns:
      the overlap, a float in [0, 1].
    """
    cross = np.trapz(n_a*n_b, z)
    auto_a = np.trapz(n_a*n_a, z)
    auto_b = np.trapz(n_b*n_b, z)
    return float(cross/np.sqrt(auto_a*auto_b))


def rounded(values, decimals):
    """Return a numpy array as a list of floats rounded for data.json."""
    return [round(float(v), decimals) for v in values]


def build_layout(project, layout):
    """Read one layout and compute everything both figures draw from it.

    Arguments:
      project = the lssty1xdesi project folder.
      layout  = one entry of LAYOUTS.

    Returns:
      a dict with the layout's bins, raw and convolved n(z), pairs,
      sigma_z, widths and overlaps (see main for the keys used).

    Raises:
      ValueError when the n(z) grid is not uniform, when its rows at
      multiples of REFINE are not the nodes the others interpolate, or
      when the bin table and the n(z) table disagree on the bin count.
    """
    stem = os.path.join(project, "data", f"lssty1xdesi_desi_{layout['key']}")
    samples, groups, zbar = read_bins(stem + "_bins.txt")
    pairs = np.loadtxt(stem + "_pairs.txt", dtype=int, ndmin=2)
    table = np.loadtxt(stem + ".nz")
    z = table[:, 0]
    n_fine = table[:, 1:]
    nbin = len(zbar)
    if n_fine.shape[1] != nbin:
        raise ValueError(f"{stem}.nz holds {n_fine.shape[1]} bins and the bin "
                         f"table {nbin}")
    step = np.diff(z)
    if np.max(np.abs(step - step[0])) > 1.0e-6*step[0]:
        raise ValueError(f"{stem}.nz: the redshift grid is not uniform")
    dz = float(step[0])

    # the DESI nodes are every REFINE-th row; the rows between must be the
    # straight lines joining them, or the table is not the refined DESI file
    z_raw = z[::REFINE]
    n_raw = n_fine[::REFINE, :]
    check = np.interp(z, z_raw, n_raw[:, 0])
    if np.max(np.abs(check - n_fine[:, 0])) > 1.0e-12*np.max(n_fine[:, 0]):
        raise ValueError(f"{stem}.nz: rows between DESI nodes are not linear "
                         "interpolations; the node spacing is not REFINE rows")
    dz_raw = float(z_raw[1] - z_raw[0])
    ranges = raw_ranges(z_raw, n_raw)

    # the pairing rule on the raw ranges, compared with the pair file
    pair_set = set()
    for row in pairs:
        pair_set.add((int(row[0]), int(row[1])))
    touching = []
    for i, j in pairs:
        lo_i = z_raw[ranges[i, 0]]
        hi_i = z_raw[ranges[i, 1]]
        lo_j = z_raw[ranges[j, 0]]
        hi_j = z_raw[ranges[j, 1]]
        slack = PAIR_SLACK_STEPS*dz_raw
        touch = (lo_j <= hi_i + slack) and (lo_i <= hi_j + slack)
        touching.append(bool(touch))

    # FoG: the likelihood's convolution, bin by bin
    sigma_v, leaders = read_sigma_v(
        os.path.join(project, "likelihood", layout["params"]), nbin)
    sigma_z = (1.0 + zbar)*sigma_v/C_KMS
    n_fog = np.empty_like(n_fine)
    for i in range(nbin):
        n_fog[:, i] = gaussian_filter1d(n_fine[:, i], sigma=sigma_z[i]/dz,
                                        mode="constant")
    fog_file = os.path.join(project, "covariance",
                            f"lssty1xdesi_desi_{layout['key']}_fog.nz")
    fog_check = None
    if os.path.exists(fog_file):
        adapter = np.loadtxt(fog_file)[:, 1:]
        fog_check = float(np.max(np.abs(adapter - n_fog))/np.max(np.abs(n_fog)))
        print(f"{layout['key']}: largest difference from {fog_file}, relative "
              f"to the largest n(z) value: {fog_check:.1e}")

    # the width of each bin: its raw range plus one node, the redshift
    # interval its galaxies were selected in
    widths = (z_raw[ranges[:, 1]] - z_raw[ranges[:, 0]]) + dz_raw

    # overlaps of same-sample neighbours (i, i + 1) and of next-but-one
    # bins (i, i + 2), raw and convolved, on the refined grid
    neighbours = []
    next_but_one = []
    for i in range(nbin - 1):
        for gap, store in ((1, neighbours), (2, next_but_one)):
            j = i + gap
            if j >= nbin or samples[j] != samples[i]:
                continue
            store.append({"i": i,
                          "j": j,
                          "zmid": 0.5*(zbar[i] + zbar[j]),
                          "raw": overlap(n_fine[:, i], n_fine[:, j], z),
                          "fog": overlap(n_fog[:, i], n_fog[:, j], z)})

    # every pair of bins whose convolved n(z) share any redshift: the
    # convolved overlap of each measured pair, and the unmeasured pairs the
    # blur makes overlap by more than NEAR_THRESHOLD
    fog_first = np.zeros(nbin)
    fog_last = np.zeros(nbin)
    for i in range(nbin):
        live = np.nonzero(n_fog[:, i] > 0.0)[0]
        fog_first[i] = z[live[0]]
        fog_last[i] = z[live[-1]]
    measured_fog = {}
    unmeasured_near = []
    for i in range(nbin):
        for j in range(i + 1, nbin):
            if fog_first[j] > fog_last[i] or fog_first[i] > fog_last[j]:
                value = 0.0
            else:
                value = overlap(n_fog[:, i], n_fog[:, j], z)
            if (i, j) in pair_set:
                measured_fog[(i, j)] = value
            elif value > NEAR_THRESHOLD:
                unmeasured_near.append({"i": i,
                                        "j": j,
                                        "zmid": 0.5*(zbar[i] + zbar[j]),
                                        "raw": overlap(n_fine[:, i], n_fine[:, j], z),
                                        "fog": value})
    return {"layout": layout, "samples": samples, "groups": groups,
            "zbar": zbar, "pairs": pairs, "pair_set": pair_set,
            "touching": touching, "z": z, "n_fine": n_fine, "n_fog": n_fog,
            "z_raw": z_raw, "n_raw": n_raw, "dz_raw": dz_raw, "ranges": ranges,
            "sigma_v": sigma_v, "sigma_z": sigma_z, "leaders": leaders,
            "widths": widths, "neighbours": neighbours,
            "next_but_one": next_but_one, "measured_fog": measured_fog,
            "unmeasured_near": unmeasured_near, "fog_check": fog_check}


def pair_summary(built):
    """Count the pairs by kind and find the separations the note quotes.

    Arguments:
      built = the dict of build_layout.

    Returns:
      a dict: bin and pair counts, pairs by kind (same sample along the
      redshift chain, or two samples), partners per bin (smallest and
      largest), and the largest centre separation |zbar_i - zbar_j| of a
      measured pair against the smallest of an unmeasured one, each with
      its bins (1-based) and groups.
    """
    samples = built["samples"]
    groups = built["groups"]
    zbar = built["zbar"]
    nbin = len(zbar)
    kinds = {}
    partners = np.zeros(nbin, dtype=int)
    largest = (-1.0, None)
    for i, j in built["pairs"]:
        if samples[i] == samples[j] and j == i + 1:
            kind = "same sample, next bin"
        elif samples[i] == samples[j]:
            kind = f"same sample, {j - i} bins apart"
        else:
            kind = f"{groups[i]} x {groups[j]}"
        kinds[kind] = kinds.get(kind, 0) + 1
        partners[i] += 1
        partners[j] += 1
        separation = abs(zbar[j] - zbar[i])
        if separation > largest[0]:
            largest = (separation, (int(i), int(j)))
    smallest = (np.inf, None)
    smallest_same = (np.inf, None)
    for i in range(nbin):
        for j in range(i + 1, nbin):
            if (i, j) in built["pair_set"]:
                continue
            separation = abs(zbar[j] - zbar[i])
            if separation < smallest[0]:
                smallest = (separation, (i, j))
            if samples[i] == samples[j] and separation < smallest_same[0]:
                smallest_same = (separation, (i, j))

    def describe(entry):
        separation, (i, j) = entry
        return {"separation": round(float(separation), 4),
                "bins_1based": [i + 1, j + 1],
                "groups": [groups[i], groups[j]],
                "zbar": [round(float(zbar[i]), 4), round(float(zbar[j]), 4)]}

    return {"nbins": nbin,
            "npairs": int(len(built["pairs"])),
            "kinds": kinds,
            "partners_min": int(partners.min()),
            "partners_max": int(partners.max()),
            "pairs_not_touching_1based": [
                [int(i) + 1, int(j) + 1]
                for (i, j), t in zip(built["pairs"], built["touching"]) if not t],
            "largest_measured": describe(largest),
            "smallest_unmeasured": describe(smallest),
            "smallest_unmeasured_same_sample": describe(smallest_same)}


def measured_minimum(built):
    """Return the measured pair with the smallest FoG-convolved overlap.

    Arguments:
      built = the dict of build_layout.

    Returns:
      a dict with the pair's bins (1-based), groups and convolved overlap.
    """
    smallest = None
    for (i, j), value in built["measured_fog"].items():
        if smallest is None or value < smallest[0]:
            smallest = (value, i, j)
    value, i, j = smallest
    return {"bins_1based": [i + 1, j + 1],
            "groups": [built["groups"][i], built["groups"][j]],
            "fog": float(f"{value:.3e}")}


def shared_range(built, first, second):
    """Return the redshift range where two groups' raw n(z) both live.

    Arguments:
      built         = the dict of build_layout.
      first, second = two group names, e.g. "LRG3" and "ELG1".

    Returns:
      [z_low, z_high], the intersection of the two groups' raw ranges.
    """
    z_raw = built["z_raw"]
    edges = {}
    for name in (first, second):
        members = [i for i, g in enumerate(built["groups"]) if g == name]
        low = z_raw[built["ranges"][members[0], 0]]
        high = z_raw[built["ranges"][members[-1], 1]]
        edges[name] = (low, high)
    return [round(float(max(edges[first][0], edges[second][0])), 4),
            round(float(min(edges[first][1], edges[second][1])), 4)]


def main():
    parser = argparse.ArgumentParser(
        description="Write the data.json files of notes/cross_correlations.")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="the lssty1xdesi project folder (default: three "
                             "folders above this script)")
    args = parser.parse_args()

    built = {}
    for layout in LAYOUTS:
        built[layout["key"]] = build_layout(args.project, layout)

    # ---- nz_overlap_layouts ----
    layouts_out = []
    for layout in LAYOUTS:
        b = built[layout["key"]]
        bins_out = []
        for i in range(len(b["zbar"])):
            # the raw n(z) on the DESI nodes from one node before the first
            # nonzero value to one after the last, so each curve starts and
            # ends at zero; scaled to unit peak
            first = max(b["ranges"][i, 0] - 1, 0)
            last = min(b["ranges"][i, 1] + 1, len(b["z_raw"]) - 1)
            values = b["n_raw"][first:last + 1, i]
            values = values/np.max(values)
            bins_out.append({"sample": b["samples"][i],
                             "group": b["groups"][i],
                             "zbar": round(float(b["zbar"][i]), 5),
                             "z_first": round(float(b["z_raw"][first]), Z_DECIMALS),
                             "n": rounded(values, N_DECIMALS)})
        layouts_out.append({
            "key": layout["key"],
            "label": layout["label"],
            "color": layout["color"],
            "marker": layout["marker"],
            "dz_node": round(b["dz_raw"], 10),
            "bins": bins_out,
            "pairs": [[int(i), int(j)] for i, j in b["pairs"]],
            "touching": b["touching"],
            "unmeasured_near": [[p["i"], p["j"]] for p in b["unmeasured_near"]],
            "lrg3_elg1_range": shared_range(b, "LRG3", "ELG1"),
            "summary": pair_summary(b)})
    nz_data = {
        "about": ("Written by notes/cross_correlations/plots/"
                  "make_cross_correlations_data.py from the project's "
                  "data/lssty1xdesi_desi_<layout>.nz, _bins.txt and "
                  "_pairs.txt. Each bin's raw n(z) is given on the DESI "
                  "redshift nodes (spacing dz_node) from z_first on, scaled "
                  "to unit peak. Pairs and bins are 0-based in data-vector "
                  "order; 'touching' is the pairing rule of "
                  "scripts/make_desi_dz_case.py (raw ranges overlap or come "
                  "within 1.5 node steps) applied to each listed pair. "
                  "unmeasured_near: the pairs not in the data vector whose "
                  "FoG-convolved n(z) overlap by more than 1e-6 (see "
                  "fog_smearing/data.json). lrg3_elg1_range is where the raw "
                  "LRG3 and ELG1 ranges intersect."),
        "layouts": layouts_out}
    target = os.path.join(HERE, "nz_overlap_layouts", "data.json")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as f:
        json.dump(nz_data, f, indent=1)
    print(f"wrote {target}")

    # ---- nz_overlap_zoom: the same drawing on the LRG2-LRG3-ELG1 corner
    # of the two thin layouts, with the bins numbered ----
    zoom_layouts = []
    for entry in layouts_out:
        if entry["key"] not in OVERLAP_ZOOM_LAYOUTS:
            continue
        # the bins reaching into the window, then every partner of one of
        # them, so that each pair touching the window is drawn (a partner
        # outside the window is clipped by the axis, its segment is not)
        core = []
        for i, b in enumerate(entry["bins"]):
            z_first = b["z_first"]
            z_last = z_first + entry["dz_node"]*(len(b["n"]) - 1)
            if z_last > OVERLAP_ZOOM_WINDOW[0] and z_first < OVERLAP_ZOOM_WINDOW[1]:
                core.append(i)
        keep = list(core)
        for i, j in entry["pairs"]:
            if i in core and j not in keep:
                keep.append(j)
            if j in core and i not in keep:
                keep.append(i)
        keep.sort()
        bins_out = []
        for i in keep:
            b = dict(entry["bins"][i])
            # bins are numbered from 1 in the note, as in the other notes
            b["number"] = i + 1
            bins_out.append(b)
        pairs_out = []
        touching_out = []
        for (i, j), t in zip(entry["pairs"], entry["touching"]):
            if i in keep and j in keep:
                pairs_out.append([keep.index(i), keep.index(j)])
                touching_out.append(t)
        near_out = []
        for i, j in entry["unmeasured_near"]:
            if i in keep and j in keep:
                near_out.append([keep.index(i), keep.index(j)])
        zoom_layouts.append({"key": entry["key"],
                             "label": entry["label"],
                             "color": entry["color"],
                             "marker": entry["marker"],
                             "dz_node": entry["dz_node"],
                             "lrg3_elg1_range": entry["lrg3_elg1_range"],
                             "bins": bins_out,
                             "pairs": pairs_out,
                             "touching": touching_out,
                             "unmeasured_near": near_out})
    zoom_order = []
    for key in OVERLAP_ZOOM_LAYOUTS:
        for entry in zoom_layouts:
            if entry["key"] == key:
                zoom_order.append(entry)
    zoom_data = {
        "about": ("Written by notes/cross_correlations/plots/"
                  "make_cross_correlations_data.py: the bins of "
                  "nz_overlap_layouts/data.json that reach into the window "
                  "on two thin layouts, plus every partner of one of them, "
                  "with their 1-based numbers in data-vector order, the "
                  "measured pairs among them and the unmeasured pairs whose "
                  "FoG-convolved n(z) overlap by more than 1e-6 (indices "
                  "into this file's bin list)."),
        "window": list(OVERLAP_ZOOM_WINDOW),
        "layouts": zoom_order}
    target = os.path.join(HERE, "nz_overlap_zoom", "data.json")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as f:
        json.dump(zoom_data, f, indent=1)
    print(f"wrote {target}")

    # ---- fog_smearing ----
    zoom_out = []
    for key in ZOOM_LAYOUTS:
        b = built[key]
        z = b["z"]
        in_window = (z >= ZOOM_WINDOW[0]) & (z <= ZOOM_WINDOW[1])
        rows = np.nonzero(in_window)[0][::ZOOM_STRIDE]
        raw_rows = np.nonzero((b["z_raw"] >= ZOOM_WINDOW[0])
                              & (b["z_raw"] <= ZOOM_WINDOW[1]))[0]
        bins_out = []
        for i in range(len(b["zbar"])):
            low = b["z_raw"][b["ranges"][i, 0]]
            high = b["z_raw"][b["ranges"][i, 1]]
            # bins reaching into the window, with a margin of five sigma_z
            # so the convolved tails entering the window are kept
            margin = 5.0*b["sigma_z"][i]
            if high < ZOOM_WINDOW[0] - margin or low > ZOOM_WINDOW[1] + margin:
                continue
            # one scale for both curves: the raw peak, so the convolved
            # curve keeps the raw area
            peak = np.max(b["n_fine"][:, i])
            bins_out.append({
                "bin": i,
                "group": b["groups"][i],
                "zbar": round(float(b["zbar"][i]), 5),
                "sigma_v": float(b["sigma_v"][i]),
                "sigma_z": round(float(b["sigma_z"][i]), 7),
                "n_raw": rounded(b["n_raw"][raw_rows, i]/peak, N_DECIMALS),
                "n_fog": rounded(b["n_fog"][rows, i]/peak, N_DECIMALS)})
        zoom_out.append({"key": key,
                         "label": b["layout"]["label"],
                         "color": b["layout"]["color"],
                         "marker": b["layout"]["marker"],
                         "z_raw": rounded(b["z_raw"][raw_rows], Z_DECIMALS),
                         "z_fog": rounded(z[rows], Z_DECIMALS),
                         "bins": bins_out})
    layouts_out = []
    for layout in LAYOUTS:
        b = built[layout["key"]]
        ratio = b["sigma_z"]/b["widths"]
        nb = b["neighbours"]
        nnb = b["next_but_one"]
        layouts_out.append({
            "key": layout["key"],
            "label": layout["label"],
            "color": layout["color"],
            "marker": layout["marker"],
            "samples": b["samples"],
            "zbar": rounded(b["zbar"], 5),
            "sigma_z": rounded(b["sigma_z"], 7),
            "width": rounded(b["widths"], 5),
            "sigma_z_over_width": rounded(ratio, 5),
            "neighbours": {
                "samples": [b["samples"][p["i"]] for p in nb],
                "zmid": [round(p["zmid"], 5) for p in nb],
                "raw": [float(f"{p['raw']:.4e}") for p in nb],
                "fog": [float(f"{p['fog']:.4e}") for p in nb]},
            "next_but_one_max": {
                "raw": float(f"{max(p['raw'] for p in nnb):.3e}"),
                "fog": float(f"{max(p['fog'] for p in nnb):.3e}")},
            "unmeasured_near": {
                "bins_1based": [[p["i"] + 1, p["j"] + 1]
                                for p in b["unmeasured_near"]],
                "groups": [[b["groups"][p["i"]], b["groups"][p["j"]]]
                           for p in b["unmeasured_near"]],
                "zmid": [round(p["zmid"], 5) for p in b["unmeasured_near"]],
                "raw": [float(f"{p['raw']:.4e}") for p in b["unmeasured_near"]],
                "fog": [float(f"{p['fog']:.4e}") for p in b["unmeasured_near"]]},
            "measured_fog_min": measured_minimum(b),
            "fog_check_against_covariance_adapter": b["fog_check"]})
        print(f"{layout['key']}: sigma_z/width {ratio.min():.3f}-{ratio.max():.3f}; "
              f"neighbour overlap raw {min(p['raw'] for p in nb):.2e}-"
              f"{max(p['raw'] for p in nb):.2e}, FoG {min(p['fog'] for p in nb):.3f}-"
              f"{max(p['fog'] for p in nb):.3f}; next-but-one largest raw "
              f"{max(p['raw'] for p in nnb):.1e}, FoG {max(p['fog'] for p in nnb):.1e}; "
              f"{len(b['unmeasured_near'])} unmeasured pairs overlap after the blur")
    leaders = built["dr3"]["leaders"]
    fog_data = {
        "about": ("Written by notes/cross_correlations/plots/"
                  "make_cross_correlations_data.py. The FoG convolution is "
                  "that of likelihood/desi_cl.py (scipy gaussian_filter1d on "
                  "the refined n(z) grid, sigma_z = (1 + zbar) sigma_v / c in "
                  "grid steps, mode constant) at the default sigma_v of "
                  "likelihood/params_desi[_<case>].yaml. zoom: raw n(z) on the "
                  "DESI nodes and convolved n(z) on every second refined row, "
                  "both divided by the raw peak of the bin. width: the bin's "
                  "raw redshift range plus one node. neighbours: the "
                  "normalized overlap integral of each same-sample pair "
                  "(i, i + 1), int n_i n_j dz / sqrt(int n_i^2 int n_j^2), "
                  "on the refined grid, raw and convolved; next_but_one_max: "
                  "the largest such overlap of the pairs (i, i + 2); "
                  "unmeasured_near: every pair outside the data vector whose "
                  "convolved overlap exceeds 1e-6; measured_fog_min: the "
                  "measured pair with the smallest convolved overlap."),
        "c_kms": C_KMS,
        "sigma_v_groups": {
            leaders[name][0]: leaders[name][1] for name in leaders},
        "zoom_window": list(ZOOM_WINDOW),
        "zoom": zoom_out,
        "layouts": layouts_out}
    target = os.path.join(HERE, "fog_smearing", "data.json")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as f:
        json.dump(fog_data, f, indent=1)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
