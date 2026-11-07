# The bias notes: notebook cells, figures and builds

Machine-local recipes for the three galaxy-bias notes under `notes/`
and for the section of EXAMPLE_EVALUATE_FISHER.ipynb that feeds them.
The physics is in the `.tex` notes (SKILL.md, "The bias notes are the
physics reference"); this file holds how the numbers are computed,
where the figures come from and how the documents are rebuilt.

## 1. How the notes are organized

- `notes/correlated_bias_prior/correlated_bias_prior.tex` opens with
  a Summary of 2-3 pages written for collaborators: the problem, the
  model, the four-rung ladder, why a ladder, the FoM table, Figure 1
  (the FoM ladder), the accuracy paragraph, Figure 2 (the figure of
  bias with random truths), Figures 3-4 (draws of the baseline and
  pessimistic priors) and a higher-order paragraph. `\appendix`
  follows immediately: every other section is an appendix. New
  material goes into an appendix; it reaches the Summary only when it
  changes a headline statement or figure.
- `notes/higher_order_bias/higher_order_bias.tex` builds models A and
  B on the per-bin linear bias of the correlated-prior note and holds
  the implementation inventory.
- `notes/bias_model_summary/bias_model_summary.tex` is a standalone
  2-page summary of models A and B for collaborators. Its parameter
  table is the same table as tab:ladder of the higher-order note; the
  two must change together.
- Naming: "baseline" and "pessimistic" mean the stated ladder rungs.
  The older kernel-scan prose says "the kernel scan's reference point"
  and "the scanned range"; the fom_range and fom_collapse legends
  still carry the old "baseline"/"plausible" labels, and their
  captions give the mapping.

## 2. The notebook cells behind the ladder

All in section 8 of EXAMPLE_EVALUATE_FISHER.ipynb, subsection "How
much bias freedom costs".

- **Per-bin derivatives** (the cell that opens "The derivatives of
  each layout's DESI vector with respect to every bin's own bias"):
  b_i dm/db_i for every bin, from a greedy coloring of the cross-pair
  graph (two evaluations per color), cached in
  `chains/EXAMPLE_EVALUATE_FISHER_BIN_BIAS.pkl` (RECOMPUTE_BIN_BIAS
  False by default). C_ell is at most quadratic in the biases, so the
  central difference is exact. Check: per-bin columns of a group sum
  to the group's cached Fisher column.
- **Export of the scan figures** ("Export the data behind the FoM
  figures"): data.json of fom_heatmap, fom_range, fom_collapse,
  memory_heatmap and memory_lines.
- **Ladder** ("The stated assumptions of the bias-prior note"):
  BIAS_ASSUMPTIONS (the baseline and pessimistic sigma_b per group
  and L), LADDER_RUNGS, SHIFT_THRESHOLD = 1/3, BIN_ZBAR, bias_groups,
  group_bias_precision(key, L, sigma_by_group, free_tilt=True),
  bin_bias_fisher, bias_drift(key, law), ladder_rung. Self-check:
  the ladder FoMs reproduce the kernel and memory scans to 7.1e-8.
- **Roughness** ("The same accuracy test on a continuous scale"):
  ROUGH_SIGMAS = geomspace(1e-3, 0.5, 40), ROUGH_L = [0.05, 0.1, 0.2,
  0.5, 1.0], both models. Self-check: 1.9e-11 against the memory grid.
- **Random truths** ("The accuracy test with random truths"):
  SCATTER_DRAWS = 2000, SCATTER_SEED = 20261107, rung_response ->
  (A, C_w, N_w), residual_covariance, results in
  scatter_bias[layout][truth][rung] with mean, p16, p50, p84,
  drift_only, spread_chi2 and error_inflation. Self-check on the
  matched cases: |E[chi2_w] - drift^2 - 2| = 1.4e-10.
- **Export of the ladder** ("Export the ladder and the roughness
  scan"): writes baseline_ladder, ladder_bias_scatter and
  bias_roughness data.json, runs each folder's plot script by
  subprocess and displays the PNG.

BIAS_ASSUMPTIONS must equal STATED_PRIORS of
`notes/correlated_bias_prior/plots/make_theory_data.py`, which draws
the baseline and pessimistic histories of Figures 3-4.

### Why these cells are fast

No cell after the derivative cache evaluates a data vector. With F
the rung's Fisher matrix (data plus priors), C the data covariance and
dm the data-vector change of a bias drift, the linear response of the
fit is dp = F^-1 D^T C^-1 dm. A drift delta_i of the bin amplitudes
beta_i gives dm = sum_i (dm/dbeta_i) beta_i delta_i, so
D^T C^-1 dm = F_data[:, bias] (beta * delta), columns the Fisher
matrix already holds. Each rung and layout therefore reduces to one
2 x n_bins matrix A with dw = A delta, and 2000 random truths cost one
matrix product. The tied rung is evaluated in the tied parameter
basis; the independent rung absorbs any delta exactly.

The error inflation uses
E[chi2_w] = Delta_drift^2 + tr(C_w^-1 A Sigma A^T) + tr(C_w^-1 N_w):
Delta_drift the shift of the drift alone, Sigma the truth's residual
kernel, C_w the rung's marginalized (w0, wa) covariance and N_w its
share from data noise and the sigma_v priors. When the rung's prior
is the truth's kernel, P Sigma P = P makes the last two terms sum to
exactly 2, which is the self-check above.

## 3. The plot-folder contract

Every figure of `notes/correlated_bias_prior` has a folder
`plots/<name>/` holding:

- `data.json`, written by a notebook export cell or by
  `plots/make_theory_data.py`;
- `plot_<name>.py`, standalone: it reads the `data.json` beside it
  (`os.path.dirname(os.path.abspath(__file__))`), sets the lsst_y1
  rcParams block (STIX fonts, no top or right ticks, faint grid,
  savefig bbox tight) and writes `<name>.pdf`, plus `<name>.png` for
  the figures that go to collaborators (dpi 200-300);
- `<name>.pdf`, the file the note includes.

Test of the contract: copy a folder to an empty directory and run
`python -I plot_<name>.py` with the cocoa environment's Python. It
must redraw without the notebook, CosmoLike or the repository, so a
paper figure can be restyled from data.json alone.

| folder | data written by | figure |
|---|---|---|
| baseline_ladder | ladder export cell | Summary Figure 1: FoM of the four rungs per layout, LSST Y1 3x2pt dashed |
| ladder_bias_scatter | ladder export cell | Summary Figure 2: figure of bias with random truths, top panel baseline truth, bottom pessimistic truth |
| baseline_draws, pessimistic_draws | make_theory_data.stated_prior_draws | Summary Figures 3-4 |
| bias_roughness | ladder export cell | the roughness appendix, kernel versus memory |
| fom_heatmap, fom_range, fom_collapse, memory_heatmap, memory_lines | scan export cell | kernel and memory scans |
| theory_bz, theory_delta, theory_draws | make_theory_data.py | the drift laws and prior draws |

Figure rules: legends `frameon=False` and never over a data line
(above the panel or in an empty corner); no bbox around text; no `\%`
in labels of non-usetex figures; every figure is rendered and looked
at before it goes into a note or to collaborators. The Python style
and plotting conventions of the cosmolike-dev skill
(`references/python.md`) apply.

## 4. Running the notebook

Execute a scratch copy, never the committed file in place, and run it
in the background (from scratch the derivatives take hours; with the
pickles in `chains/` the notebook reruns from its caches):

```bash
source /Users/vivianmiranda/miniforge/etc/profile.d/conda.sh
conda activate cocoapy311
cd /Users/vivianmiranda/data/COCOA/november2026/sbu_cosmolike/cocoa/Cocoa
source start_cocoa.sh
export JUPYTER_PATH=<scratch>/jupyter   # kernels/cocoa/kernel.json
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cocoa --ExecutePreprocessor.timeout=-1 \
  <scratch>/EXAMPLE_EVALUATE_FISHER.ipynb
```

The scratch kernelspec's `kernel.json` runs
`Cocoa/.local/bin/python -m ipykernel_launcher -f {connection_file}`.
New cells are added to the scratch copy by small Python scripts that
edit the notebook JSON. After the run: no cell may carry an error
output, the self-check lines of Section 2 must print, and new
data.json files are compared against any prototype before the
notebook is copied back. The notebook is committed executed, so its
figures show on GitHub.

## 5. Building the notes

Build in a scratch directory, never in the repository:

1. copy `<doc>.tex` and, if present, its `plots/` folder there;
2. `pdflatex -interaction=nonstopmode <doc>.tex` three times;
3. count in `<doc>.log`: lines starting with `!` (errors), `undefined`,
   `Overfull` and `too large` (a float that does not fit); errors,
   undefined references and oversized floats must be zero;
4. copy `<doc>.pdf` back next to the `.tex`;
5. for the correlated-prior note, run `python verify_prior.py` in its
   folder with the cocoa environment: 23 PASS lines.

## 6. Commits

lssty1xdesi commits go directly on `main`, never pushed and never
amended. The repository uses Git LFS, so commits need git-lfs on
PATH: the cocoa environment, or
`export PATH="/Users/vivianmiranda/miniforge/pkgs/git-lfs-3.7.1-h07d3073_0/bin:$PATH"`.
The desktop app's worktree guard blocks the Edit tool inside nested
project repositories: edit a scratch mirror and copy it back.
