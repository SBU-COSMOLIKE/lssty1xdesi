"""Redraw the Fisher-derivative convergence figure from the study's results.

Reads results/fisher_derivatives_<layout>.json (run_fisher_derivatives.py,
compare stage) and writes results/fisher_derivatives_<layout>.png: the
figure of merit FoM(w0, wa) of the paper model (model B + tilts on the
pessimistic prior) at k_max = 0.075 and 0.2 h/Mpc (z = 0 values of the
per-bin UV-fraction cut), each relative to the finest configuration of the
study, in three panels:

  (a) against the chi grid, in multiples of the MCMC grid, at the five-point
      rule and step h (the MCMC settings at grid 1 as an open marker);
  (b) against the step, in multiples of h, at the reference grid (4x) and at
      twice it (8x): five-point solid, three-point dashed;
  (c) the change of every other control and of DerivKit against the
      reference, |Delta FoM|/FoM, on a logarithmic axis, with the tolerance.

It also writes the data of the Fisher machinery note's figure
notes/fisher_machinery/plots/fom_convergence/ (data.json, from
results/fisher_derivatives_dz0p05.json and, when the spot check has run,
results/fisher_derivatives_dcom62_vs_candidate_ladder.json) and redraws it
with that folder's plot_fom_convergence.py.

Run from any folder: python generate_fisher_derivatives_figures.py [layout]
"""
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Figure style of the whole notebook: the shared plotting functions never change
# rcParams, so these lines set every font, tick and grid default.
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SYSTEM = "model B + tilts, pessimistic"
# the two cuts: color, line width and marker (never color alone)
CUTS = {"0.075": {"color": "#0072B2", "lw": 2.4, "marker": "o",
                  "label": r"$k_\mathrm{max} = 0.075$"},
        "0.2": {"color": "#E69F00", "lw": 1.8, "marker": "s",
                "label": r"$k_\mathrm{max} = 0.2$"}}
# panel (a): the chi-grid rungs, in multiples of the MCMC grid
GRID_CONFIGS = {1.0: "chi_grid_mcmc", 2.0: "chi_grid_half", 4.0: "reference",
                8.0: "chi_grid_double", 16.0: "chi_grid_quad"}
# panel (b): the grids whose step ladder is drawn, and their line styles
STEP_GRIDS = {"reference": {"label": r"$\chi$ grid $4\times$ MCMC", "filled": False},
              "chi_grid_double": {"label": r"$\chi$ grid $8\times$ MCMC", "filled": True}}
STEP_MULTIPLES = (0.5, 1.0, 2.0, 3.0)
# panel (c): the other controls, in order, with their labels
CONTROLS = [
    ("accuracyboost_2", "accuracyboost 2"),
    ("accuracyboost_3", "accuracyboost 3"),
    ("integration_accuracy_0", "integration accuracy 0"),
    ("rank_5", "rank 5"),
    ("lowrank_lmax_1000", r"multipole cap 1000"),
    ("camb_accuracy_1p05", "CAMB AccuracyBoost 1.05"),
    ("camb_accuracy_1p5", "CAMB AccuracyBoost 1.5"),
    ("camb_accuracy_3", "CAMB AccuracyBoost 3"),
    ("k_per_logint_15", "CAMB k_per_logint 15"),
    ("camb_kmax_40", r"CAMB $k_\mathrm{max} = 40$/Mpc"),
    ("candidate", "final settings"),
    ("DerivKit at candidate", "DerivKit, final settings"),
    ("DerivKit at mcmc", "DerivKit, MCMC settings"),
    ("mcmc", "MCMC settings"),
]
FLOOR = 1.0e-3  # [%]: smaller changes are drawn at the floor


def line_points(ax):
    """Display coordinates of points sampled along every drawn line of ax."""
    pts = [np.zeros((0, 2))]
    t = np.linspace(0.0, 1.0, 60)[:, None]
    for line in ax.get_lines():
        xy = np.column_stack([np.asarray(v, dtype=float) for v in line.get_data()])
        display_xy = line.get_transform().transform(xy)
        finite = np.all(np.isfinite(display_xy), axis=1)
        pts.append(display_xy[finite])
        if line.get_linestyle() in ("None", "none", " ", ""):
            continue
        for i in range(len(display_xy) - 1):
            if finite[i] and finite[i + 1]:
                pts.append(display_xy[i] + t*(display_xy[i + 1] - display_xy[i]))
    return np.vstack(pts)


def check_clear(fig, axes, artists):
    """Raise when a legend or label touches a drawn point or line, with a
    margin of 3 points."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = 3.0*fig.dpi/72.0
    pts = np.vstack([line_points(ax) for ax in axes])
    for artist in artists:
        box = artist.get_window_extent(renderer)
        hit = ((pts[:, 0] > box.x0 - pad) & (pts[:, 0] < box.x1 + pad)
               & (pts[:, 1] > box.y0 - pad) & (pts[:, 1] < box.y1 + pad))
        if np.any(hit):
            raise RuntimeError(f"{type(artist).__name__} {artist} touches the data")


# the corners a panel name may take, in order of preference: (x, y, ha, va)
# in axes coordinates
CORNERS = [(0.98, 0.95, "right", "top"), (0.02, 0.95, "left", "top"),
           (0.02, 0.05, "left", "bottom"), (0.98, 0.05, "right", "bottom")]


def place_panel_name(fig, ax, name):
    """Write a panel name in the first corner of ax where it clears every
    drawn point and line by 3 points; raise when no corner does."""
    pad = 3.0*fig.dpi/72.0
    for x, y, ha, va in CORNERS:
        text = ax.text(x, y, name, transform=ax.transAxes, ha=ha, va=va, fontsize=22)
        fig.canvas.draw()
        box = text.get_window_extent(fig.canvas.get_renderer())
        pts = line_points(ax)
        hit = ((pts[:, 0] > box.x0 - pad) & (pts[:, 0] < box.x1 + pad)
               & (pts[:, 1] > box.y0 - pad) & (pts[:, 1] < box.y1 + pad))
        if not np.any(hit):
            return text
        text.remove()
    raise RuntimeError(f"panel name {name}: no corner of the panel is free of data")


def fom(entry, cut):
    """FoM(w0, wa) of the paper model at one cut, or NaN without a fit."""
    result = entry["systems"][SYSTEM][cut]
    return result.get("fom_w0wa", np.nan)


# The note's figure: per layout, the results file and the configuration of
# each chi grid (the combined factor accuracyboost x nonlimber_accuracyboost).
# Delta z = 0.05 scans the grid at the reference's other settings; the spot
# check of Delta d_com = 62 Mpc runs the step ladder at the final settings
# (grid 256), so its FoMs are divided by its own value at step h.
NOTE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(HERE)), "notes",
                           "fisher_machinery", "plots", "fom_convergence")
NOTE_LAYOUTS = [
    {"key": "dz0p05", "name": "Delta z = 0.05", "file": "fisher_derivatives_dz0p05.json",
     "grids": {32: "chi_grid_mcmc", 64: "chi_grid_half", 128: "reference",
               256: "chi_grid_double", 512: "chi_grid_quad"},
     "settings": "the reference's other settings"},
    {"key": "dcom62", "name": "Delta d_com = 62 Mpc",
     "file": "fisher_derivatives_dcom62_vs_candidate_ladder.json",
     "grids": {256: "candidate_ladder"},
     "settings": "the final settings"},
]
NOTE_STEPS = (0.5, 1.0, 2.0, 3.0)
NOTE_FOMS = ("fom_w0wa", "fom_om_as")
NOTE_CUTS = ("0.075", "0.2")
# the points of Delta z = 0.05 at other settings: the five-point rule at h
# with every control at its MCMC value and at the final settings, and
# DerivKit at the same two settings (grid: where each is drawn)
NOTE_POINTS = {"mcmc": {"config": "mcmc", "grid": 32},
               "final": {"config": "candidate", "grid": 256},
               "derivkit_mcmc": {"config": "DerivKit at mcmc", "grid": 32},
               "derivkit_final": {"config": "DerivKit at candidate", "grid": 256}}


def note_value(configs, label, fom_key, cut):
    """One FoM of the paper model, or None when that configuration is absent."""
    if label not in configs:
        return None
    value = configs[label]["systems"][SYSTEM][cut].get(fom_key)
    return None if value is None else float(value)


def export_note_figure():
    """Write the note's fom_convergence/data.json and redraw the figure.

    Every FoM is divided by its layout's converged value: the five-point
    rule at step h on the finest chi grid of that layout. Absent
    configurations (a stage not run yet) are written as None.
    """
    out = {"about": ("FoM(w0, wa) and FoM(Omega_m, 10^9 A_s) of the paper model (model B + "
                     "tilts, pessimistic linear-bias prior) against the combined chi-grid "
                     "factor and the stencil step, as ratios to the converged value (the "
                     "five-point rule at step h on the finest grid of each layout). Written "
                     "by tests/studies/generate_fisher_derivatives_figures.py from the "
                     "results of tests/studies/run_fisher_derivatives.py."),
           "system": SYSTEM, "kmax_h_per_Mpc": [float(c) for c in NOTE_CUTS],
           "foms": list(NOTE_FOMS), "steps_in_h": list(NOTE_STEPS), "h": 0.02,
           "layouts": [], "points": {}}
    for entry in NOTE_LAYOUTS:
        path = os.path.join(RESULTS, entry["file"])
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            configs = json.load(handle)["configurations"]
        grids = sorted(entry["grids"])
        finest = entry["grids"][grids[-1]]
        lay = {"name": entry["name"], "settings": entry["settings"], "grid": grids,
               "converged": {}, "ratio": {}}
        for fom_key in NOTE_FOMS:
            lay["converged"][fom_key] = {}
            lay["ratio"][fom_key] = {}
            for cut in NOTE_CUTS:
                base = note_value(configs, finest, fom_key, cut)
                lay["converged"][fom_key][cut] = base
                lay["ratio"][fom_key][cut] = {}
                for step in NOTE_STEPS:
                    row = []
                    for grid in grids:
                        config = entry["grids"][grid]
                        label = config if step == 1.0 else f"{config}, five-point, step x{step:g}"
                        value = note_value(configs, label, fom_key, cut)
                        row.append(None if value is None or base is None else value/base)
                    lay["ratio"][fom_key][cut][f"{step:g}"] = row
        out["layouts"].append(lay)
        if entry["key"] == "dz0p05":
            for name, point in NOTE_POINTS.items():
                ratio = {}
                for fom_key in NOTE_FOMS:
                    ratio[fom_key] = {}
                    for cut in NOTE_CUTS:
                        value = note_value(configs, point["config"], fom_key, cut)
                        base = lay["converged"][fom_key][cut]
                        ratio[fom_key][cut] = None if value is None else value/base
                out["points"][name] = {"grid": point["grid"], "ratio": ratio}
    os.makedirs(NOTE_FOLDER, exist_ok=True)
    with open(os.path.join(NOTE_FOLDER, "data.json"), "w") as handle:
        json.dump(out, handle, indent=1)
    script = os.path.join(NOTE_FOLDER, "plot_fom_convergence.py")
    if os.path.exists(script):
        subprocess.run(args=[sys.executable, script], check=True)


layout = sys.argv[1] if len(sys.argv) > 1 else "dz0p05"
with open(os.path.join(RESULTS, f"fisher_derivatives_{layout}.json")) as handle:
    data = json.load(handle)
configs = data["configurations"]
finest = None
for factor in sorted(GRID_CONFIGS, reverse=True):
    if GRID_CONFIGS[factor] in configs:
        finest = GRID_CONFIGS[factor]
        break
tolerance = 100.0*data["tolerance"]["fom"]

fig, axes = plt.subplots(3, 1, figsize=(10.0, 15.0), gridspec_kw={"hspace": 0.42})
# (a) the chi grid
ax = axes[0]
for cut, st in CUTS.items():
    base = fom(configs[finest], cut)
    x = []
    y = []
    for factor, name in GRID_CONFIGS.items():
        if name in configs:
            x.append(factor)
            y.append(100.0*(fom(configs[name], cut)/base - 1.0))
    ax.plot(x, y, color=st["color"], lw=st["lw"], marker=st["marker"], ms=9)
    if "mcmc" in configs:
        ax.plot([1.0], [100.0*(fom(configs["mcmc"], cut)/base - 1.0)], ls="none",
                marker=st["marker"], ms=13, mfc="white", mec=st["color"], mew=2.0)
ax.axhline(0.0, color="0.5", lw=1.0, ls="--")
ax.set_xscale("log", base=2)
ax.set_xticks(list(GRID_CONFIGS))
ax.set_xticklabels([f"{f:g}" for f in GRID_CONFIGS])
ax.set_xlabel(r"$\chi$ grid / MCMC grid", fontsize=24)
ax.set_ylabel("FoM change [%]", fontsize=24)
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=19)
panel_names = [place_panel_name(fig, ax, "(a)")]

# (b) the step
ax = axes[1]
for cut, st in CUTS.items():
    base = fom(configs[finest], cut)
    for grid, gst in STEP_GRIDS.items():
        if grid not in configs:
            continue
        for rule, dash in (("five", "-"), ("three", "--")):
            x = []
            y = []
            for multiple in STEP_MULTIPLES:
                label = grid if (rule == "five" and multiple == 1.0) \
                    else f"{grid}, {rule}-point, step x{multiple:g}"
                if label in configs:
                    x.append(multiple)
                    y.append(100.0*(fom(configs[label], cut)/base - 1.0))
            ax.plot(x, y, color=st["color"], lw=st["lw"], ls=dash, marker=st["marker"],
                    ms=9, mfc=st["color"] if gst["filled"] else "white", mec=st["color"],
                    mew=1.6)
ax.axhline(0.0, color="0.5", lw=1.0, ls="--")
ax.set_xticks(list(STEP_MULTIPLES))
ax.set_xticklabels([f"{m:g}" for m in STEP_MULTIPLES])
ax.set_xlabel(r"step / $h$  ($h = 0.02$)", fontsize=24)
ax.set_ylabel("FoM change [%]", fontsize=24)
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=19)
panel_names.append(place_panel_name(fig, ax, "(b)"))

# (c) the other controls against the reference
ax = axes[2]
names = [(key, label) for key, label in CONTROLS if key in configs]
positions = np.arange(len(names))
reference = configs[data["base"]]
for cut, st in CUTS.items():
    values = []
    for key, label in names:
        change = 100.0*abs(fom(configs[key], cut)/fom(reference, cut) - 1.0)
        values.append(max(change, FLOOR))
    ax.plot(positions, values, ls="none", marker=st["marker"], ms=11, mfc=st["color"],
            mec=st["color"])
ax.axhline(tolerance, color="0.5", lw=1.0, ls="--")
ax.set_yscale("log")
ax.set_xticks(positions)
ax.set_xticklabels([label for key, label in names], rotation=55, ha="right", fontsize=17)
ax.set_ylabel(r"$|\Delta$FoM$|$ [%]", fontsize=24)
ax.tick_params(axis="y", labelsize=19)
panel_names.append(place_panel_name(fig, ax, "(c)"))

handles = []
for cut, st in CUTS.items():
    handles.append(Line2D([], [], color=st["color"], lw=st["lw"], marker=st["marker"], ms=9,
                          label=st["label"]))
handles.append(Line2D([], [], color="0.3", lw=1.8, ls="-", label="five-point"))
handles.append(Line2D([], [], color="0.3", lw=1.8, ls="--", label="three-point"))
for grid, gst in STEP_GRIDS.items():
    handles.append(Line2D([], [], color="0.3", ls="none", marker="o", ms=9,
                          mfc="0.3" if gst["filled"] else "white", mec="0.3", mew=1.6,
                          label=gst["label"]))
legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.895),
                    ncol=2, frameon=False, fontsize=20)
check_clear(fig, list(axes), [legend, *[t for t in panel_names]])
fig.savefig(os.path.join(RESULTS, f"fisher_derivatives_{layout}.png"), dpi=180)
plt.close(fig)
if layout == "dz0p05":
    export_note_figure()
