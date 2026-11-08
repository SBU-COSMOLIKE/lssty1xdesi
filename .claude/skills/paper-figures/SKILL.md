---
name: paper-figures
description: The figure style of this project's notes and papers. Use it whenever a figure, a plot script (notes/*/plots/*/plot_*.py), a notebook plotting cell, a figure caption or a figure-heavy note section is written or changed in projects/lssty1xdesi, including FoM, k_max, accuracy-test, drift, prior-draw and one-loop term figures. It fixes the house rcParams block, the one-column layout, the colorblind-safe palette with redundant encodings, legend and label placement enforced in code, caption wording, the plot-folder contract and the rules that replace tables by figures.
---

# Paper figures

Every figure is made for a two-column article and must carry its main
message at a glance. The rules below come from repeated corrections
and are fixed choices, not suggestions. Apply them when a figure is
first written, not as a later cleanup.

## The plot-folder contract

Each figure lives in `notes/<note>/plots/<name>/`:

- `data.json`: everything the figure draws, plus an `about` string
  naming its producer (a notebook export cell, a generator script in
  the note's `plots/` folder, or a hand transcription from a paper);
- `plot_<name>.py`: redraws the figure from `data.json` alone;
- `<name>.pdf` (included by the note) and `<name>.png` (dpi 300).

Test: copy the folder to an empty directory and run
`.local/bin/python -I -W error::RuntimeWarning plot_<name>.py` with the
cocoa environment's Python. It must redraw without the notebook,
CosmoLike or the repository. Look at every PNG after redrawing.

## The house rcParams block

Every `plot_*.py`, and every notebook cell that draws a paper figure,
carries this block verbatim:

```python
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
```

## Layout

- **Width.** One column (about 3.5 in) by default. A full-width strip
  (about 6.4 in) only for a grid that must sit side by side, and then
  short: a strip of the page, never a full page.
- **Stacked panels.** Panels are stacked vertically, wide and short
  (about 1.1 to 1.3 in each), with a thin gap (`hspace` about 0.06,
  never 0). They share the x axis, with tick labels on the bottom
  panel only. Every panel of a stack uses the exact same y range, and
  there is one shared y label (`fig.supylabel`).
- **Panel names.** Inside the panel (a corner free of data) or rotated
  90 degrees as a right-hand y label (used for parameter names and
  prior names). A title only when the caption cannot carry it.
- **Different natural ranges.** When the panels of a grid have
  different natural y ranges (a column per sample group), divide each
  column by one round constant, written inside its panel (e.g.
  delta_g = 20%), so that every panel shares one y axis, only the first
  column needs tick labels and the panels nearly touch
  (`wspace` about 0.06). Separate per-panel ranges give a poor aspect
  ratio and are not used.
- **Sibling figures.** Two figures with the same axes (two priors, two
  scenarios) become rows of one grid.
- **Crowding.** Draw only the variants the text discusses; values that
  are 1 or 0 by construction go into the caption, not onto the plot.
  When curves stay apart, one panel with redundant encodings beats
  several panels. Use a linear axis unless the values span decades.
- **Narrow x spacing.** Categorical x axes keep cases close; no wide
  figures with large gaps between cases.

## Color and redundant encoding

- **Palette.** Okabe-Ito, colorblind safe, and never red together with
  green (its green is not used). The four bin layouts:
  - Delta d_com = 62 Mpc: `#0072B2` (blue)
  - Delta z = 0.02: `#E69F00` (orange)
  - Delta z = 0.05: `#CC79A7` (reddish purple)
  - Delta z = 0.2: `#56B4E9` (sky blue)
  References (LSST Y1 3x2pt) are gray. Other categorical sets take
  further Okabe-Ito colors or grays.
- **Never color alone.** Every category is encoded redundantly, so the
  eye separates it from several cues at once: color + line width +
  marker for the layouts, line style for models or variants.

  | layout | line width | marker |
  |---|---|---|
  | 62 Mpc | 2.4 | circle `o` |
  | Delta z = 0.02 | 1.8 | square `s` |
  | Delta z = 0.05 | 1.3 | diamond `D` |
  | Delta z = 0.2 | 0.9 | triangle `^` |

  Model A solid, model B dashed, with a dash pattern visible at the
  thinnest width. Check: the PNG converted to grayscale still lets
  every line be identified.

## Legends, labels and references

- Legends `frameon=False`; no `bbox` on text; no `\%` in labels
  without usetex (it prints the backslash).
- A legend, label or text must never touch a data line, a marker or a
  reference line. Enforce it in code: sample every drawn line in
  display coordinates (`line_points`, as in
  `notes/correlated_bias_prior_v2/plots/baseline_ladder/plot_baseline_ladder.py`),
  try candidate corners or grow the y headroom by the least amount,
  and raise an error when nothing clears. Never rely on a visual check
  alone.
- Legend entries show the markers and line widths that are actually
  drawn; for filled-versus-open pairs, use a tuple handle with the
  drawn pair, never a neutral stand-in.
- Stacked legends (line styles above, layouts below) sit close
  together, with no large vertical gap.
- Reference lines carry no legend entry; the caption names them. A
  reference far above the data (more than 1.5 times the highest
  point) is not drawn on a stretched axis: an in-panel text such as
  "LSST Y1 3x2pt: 215" quotes it instead, included in the bounding-box
  check.
- Legend labels name the model plainly ("model A", "model B"); a
  variant is stated once in the caption when the figure draws only
  one, so it does not clash with the prior names in the panels.

## k_max on figures

- k_max axes stop at 0.2 h/Mpc, and figures and text quote no FoM above
  it.
- k_max is the z = 0 value of a per-bin cut at k_max/D(zbar_i)
  (the UV-fraction criterion, intersected with the shipped mask). Axis
  labels say "at z = 0"; captions carry the clause "k_max quoted at
  z = 0; each bin is cut at k_max/D(zbar_i)". A sample with its own cut
  is named.

## Captions

- Short statements, never questions. Say what is drawn, what is
  omitted and the k_max clause.
- No value the reader takes from the plot; no FoM to two digits.
- Symbols defined once in the text; the caption only points to them.
- The accuracy statistic sqrt(<chi^2>/n_dof) has no name. Its value is
  eta; the caption opens with a statement such as "How well each fit's
  quoted error bar covers where the fit actually lands, measured by
  eta", then gives chi^2 = Delta p^T C_p^-1 Delta p compactly
  (n_dof = 2 for (w0, wa), 1 for Omega_m or A_s; 1 = honest error bars).

## Tables and narrative order

- No big numeric tables in any note: a table of many numbers becomes a
  compact figure (stacked panels, one line per entry against the
  natural variable). Only short didactic tables remain. A table that a
  figure duplicates is removed.
- Logic before FoM: a prior or model that fails the accuracy test gets
  no FoM in any figure (its FoM means nothing). The FoM figures show
  only what survives.
- Two scenarios that the author has not ranked (e.g. baseline and
  pessimistic variants) are shown side by side with equal weight; no
  caption calls one the headline.

## Names on figures

The linear-bias priors are growth-tied, baseline, pessimistic and free
(never "rung"); sample groups are "groups" (never "chunk"); the
primordial amplitude is written A_s, never a bare "As".
