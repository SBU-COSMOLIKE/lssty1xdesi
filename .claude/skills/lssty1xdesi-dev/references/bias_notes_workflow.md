# The bias notes: notebook cells, figures and builds

Machine-local recipes for the four galaxy-bias notes under `notes/`
and for the notebooks that feed them. The physics is in the `.tex`
notes (SKILL.md, "The bias notes are the physics reference"); this
file holds how the numbers are computed, where the figures come from
and how the documents are rebuilt.

## 1. How the notes are organized

There are four notes, and no others (no v3: new forecast results go
into `correlated_bias_prior_v2`):

- `notes/correlated_bias_prior_v2/correlated_bias_prior_v2.tex` (with
  `appendix_physics.tex` and `appendix_math.tex`) is the forecast note
  for models A and B. A one-page Summary, then The forecast, The
  linear-bias prior, Model A, The accuracy test (random truths and the
  calibration drift) and The figure of merit; then the appendices: the
  derivations of the linear-bias priors (The models; Which L and
  sigma_g are physical; The memory model; the two step-by-step
  derivations; the accuracy-test and projection algebra; Numerical
  verification) and the model A record (Implementation and the Fisher
  construction; The known drifts; Numerical checks; The counterterm
  requires the low-rank engine; Comparison with linear bias and with
  exact calibrations; Limitations; Open issues). Every result is a
  figure drawn from a `plots/` folder. New material goes into an
  appendix and reaches the Summary only when it changes a headline
  statement or figure. Model B (`fom_kmax_models`, `model_a_break`)
  enters this note.
- `notes/linear_bias_kmax/linear_bias_kmax.tex`: a linear-bias fit to
  model A truths against k_max on the four layouts, and the accuracy
  test of the linear-bias priors scored for Omega_m and A_s (Summary,
  Method, Numerical checks, Limitations, Open issues). Figures
  `fob_kmax`, `fom_kmax`, `ladder_bias_params`.
- `notes/higher_order_bias/higher_order_bias.tex` builds models A and
  B on the per-bin linear bias of the correlated-prior note and holds
  the implementation inventory. Its Recommendation section and
  subsection "Model B and the order of work" predate the
  implementation; their text is kept in
  `higher_order_bias_recommendation.md`, and the note is rewritten to
  the rules below.
- `notes/bias_model_summary/bias_model_summary.tex` is written last,
  from the other three: a presentable summary for collaborators that
  carries their key figures, not a compact digest. The current file
  is the earlier 2-page summary; its parameter table equals tab:ladder
  of the higher-order note, and the two change together until the
  rewrite.

Rules for every note:

- A note describes what was done: the question, the method, the
  result and its limits. It holds no recommendation (recommendations
  and decisions live in the skill) and no list of gaps already
  addressed.
- Open items that matter go into an "Open issues" appendix.
- No large tables: a result is a figure, and the values a caption
  leaves out are in that folder's `data.json`.
- A revision cuts the prose by about 20%.
- Sections are cited by title, never by appendix letter.
- Naming: the linear-bias priors are growth-tied, baseline,
  pessimistic and free (the code keys `'tied'` and `'independent'`
  name the first and last; "rung" stays in code identifiers only).
  "Baseline" and "pessimistic" also name the model A and B variants,
  so the text says "the baseline prior" or "model A baseline". The
  accuracy statistic sqrt(<chi^2>/n_dof) has no name: the documents
  write the symbol and explain it at its first use; the code key is
  `'error_inflation'`.

Figure rules: the project skill `.claude/skills/paper-figures/SKILL.md`
holds every figure rule (layout, palette and redundant encodings,
legend placement enforced in code, captions, k_max on figures, tables
turned into figures). Read it before writing or changing a figure.

## 2. The notebook cells behind the ladder

### The linear notebook

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
- **Ladder** ("The stated assumptions of the bias-prior note"):
  BIAS_ASSUMPTIONS (the baseline and pessimistic sigma_b per group
  and L), LADDER_RUNGS, SHIFT_THRESHOLD = 1/3, BIN_ZBAR, bias_groups,
  group_bias_precision(key, L, sigma_by_group, free_tilt=True),
  bin_bias_fisher, bias_drift(key, law), ladder_rung. Self-check:
  the ladder FoMs reproduce the kernel and memory scans to 9.6e-7.
- **Roughness** ("The same accuracy test on a continuous scale"):
  ROUGH_SIGMAS = geomspace(1e-3, 0.5, 40), ROUGH_L = [0.05, 0.1, 0.2,
  0.5, 1.0], both models. Self-check: 7.5e-12 against the memory grid.
- **Random truths** ("The accuracy test with random truths"):
  SCATTER_DRAWS = 2000, SCATTER_SEED = 20261107, rung_response ->
  (A, C_w, N_w), residual_covariance, results in
  scatter_bias[layout][truth][rung] with mean, p16, p50, p84,
  drift_only, spread_chi2 and error_inflation. Self-check on the
  matched cases: |E[chi2_w] - drift^2 - 2| = 5.3e-11.
- **No exports.** The linear notebook writes no figure folder; it
  imports the drift laws from
  `notes/correlated_bias_prior_v2/plots/make_theory_data.py`. Its
  Fisher matrices are rebuilt from the cached derivatives and the
  current covariance at every run (SKILL.md, the Fisher module's
  "Cache derivatives, never Fisher matrices").

### The model A, model B and k_max notebooks

- EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb reads the linear notebook's
  scale-cut cache (`chains/EXAMPLE_EVALUATE_FISHER_SCALE_CUTS.pkl`) at
  k_max = 0.075 h/Mpc, caches its model A derivatives in
  `chains/EXAMPLE_EVALUATE_FISHER_MODEL_A_DERIVATIVES_<n>.pkl`, imports
  the drift laws from `notes/correlated_bias_prior_v2/plots/make_theory_data.py`
  (THEORY_PLOTS), and takes the linear-bias FoMs at k_max from the
  k_max study's scan, `notes/linear_bias_kmax/plots/fob_kmax/data.json`
  (KMAX_SCAN), which equal the linear notebook's ladder values. Its
  export cell writes baseline_ladder, ladder_bias_scatter,
  goodness_of_fit, drift_shifts and calibration_drift into
  `notes/correlated_bias_prior_v2/plots/`.
- EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb reads the linear
  notebook's derivative, per-bin bias and scale-cut caches and its
  Gaussian priors from the example yamls (cobaya `update_info`),
  imports the same drift laws through THEORY_PLOTS, and writes
  fob_kmax, fom_kmax and ladder_bias_params into
  `notes/linear_bias_kmax/plots/`. Self-check: the matched-prior
  identity E[chi2_w] - drift^2 = 2 to 1.6e-10.
- EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb (executed, not yet committed)
  reads, never writes, the model A derivative caches and the linear
  notebook's scale cuts. It computes the model B columns once and
  caches them in `chains/EXAMPLE_EVALUATE_FISHER_MODEL_B_TEMPLATES_<n>.pkl`:
  the alpha_2 columns by central steps through the compiled setter, on
  the coloring of the cross-pair graph, and the exact SN_2 templates
  (`stochastic_templates(order=2)`). It writes fom_kmax_models and
  model_a_break into `notes/correlated_bias_prior_v2/plots/`.
  Self-checks: the order-0 templates equal the model A caches' SN_0
  columns, the fiducial is unchanged after the alpha_2 steps, forward
  and backward alpha_2 differences agree to <= 3e-10, the FoMs at
  0.075 equal the model A notebook's to 4.4e-10, and the matched
  random-truth statistic is 1 to 2.8e-9. The notebook is generated by
  a builder script outside the repository (session scratch:
  `/private/tmp/claude-501/-Users-vivianmiranda-data-COCOA-november2026-sbu-cosmolike-cocoa--claude-worktrees-cosmolike-fork-testing-68ac58/2c0631f4-e9d7-4351-929c-5bfa353b08fd/scratchpad/modelB/build_model_b_notebook.py`);
  until it is committed, edits go to the builder.

BIAS_ASSUMPTIONS of all four notebooks must equal STATED_PRIORS of
`notes/correlated_bias_prior_v2/plots/make_theory_data.py`, which
draws the baseline and pessimistic histories (one figure,
prior_draws: a row per prior, a column per group).

### Why these cells are fast

No cell after the derivative cache evaluates a data vector. With F
the prior's Fisher matrix (data plus priors), C the data covariance
and dm the data-vector change of a bias drift, the linear response of
the fit is dp = F^-1 D^T C^-1 dm. A drift delta_i of the bin
amplitudes beta_i gives dm = sum_i (dm/dbeta_i) beta_i delta_i, so
D^T C^-1 dm = F_data[:, bias] (beta * delta), columns the Fisher
matrix already holds. Each prior and layout therefore reduces to one
2 x n_bins matrix A with dw = A delta, and 2000 random truths cost one
matrix product. The growth-tied prior is evaluated in its own
parameter basis, one amplitude per group; the free prior absorbs any
delta exactly.

The (w0, wa) value of sqrt(<chi^2>/n_dof), n_dof = 2, uses
<chi^2_w> = Delta_drift^2 + tr(C_w^-1 A Sigma A^T) + tr(C_w^-1 N_w):
Delta_drift the shift of the drift alone, Sigma the truth's residual
kernel, C_w the fit's marginalized (w0, wa) covariance and N_w its
share from data noise and the sigma_v priors. When the fit's prior
is the truth's kernel, P Sigma P = P makes the last two terms sum to
exactly 2, which is the self-check above.

## 3. The plot-folder contract

Every figure of every note has a folder `plots/<name>/` holding:

- `data.json`, written by a notebook export cell, by a generator
  script in the note's `plots/` folder, or transcribed by hand from a
  paper (its `about` says which);
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

The folders of `notes/correlated_bias_prior_v2/plots/` (section that
includes the figure in parentheses):

| folder | data written by | figure |
|---|---|---|
| prior_draws | make_theory_data.stated_prior_draws | draws of the baseline and pessimistic priors, a row per prior and a column per group, each column scaled by a round delta_g (The linear-bias prior) |
| ladder_bias_scatter | model A export cell | the (w0, wa) shifts of the random truths and of the calibration drift (The accuracy test) |
| goodness_of_fit | model A export cell | sqrt(<chi^2>/n_dof) of (w0, wa), Omega_m and A_s under pessimistic scatter (The accuracy test) |
| calibration_drift | model A export cell | the shifts of the calibration drift (The accuracy test) |
| baseline_ladder | model A export cell | two panels: `baseline_ladder.pdf`, FoM(w0, wa), and `baseline_ladder_om_as.pdf`, FoM(Omega_m, 10^9 A_s), per prior and layout, model A baseline and pessimistic (The figure of merit) |
| drift_shifts | model A export cell | the shifts of the two known drifts, one column per drift (appendix The known drifts) |
| zhou_lrg_bias | transcribed from Zhou et al. 2021 (arXiv:2001.06018) | measured LRG bias against the constant-amplitude shape (appendix What the measurements say) |
| theory_bz, theory_delta, theory_draws | make_theory_data.py | the drift laws and prior draws of the derivation appendices |
| fom_kmax_models (pending commit) | model B export cell | FoM(w0, wa) and FoM(Omega_m, 10^9 A_s) of models A and B against k_max, three stacked panels (baseline, pessimistic, free); model A drawn up to its break point |
| model_a_break (pending commit) | model B export cell | the largest shift of model A fits to model B truths against k_max, with the sigma/3 threshold; data.json also holds the random-truth statistics |

`plots/make_theory_data.py` sits in the same folder. The folders of
`notes/linear_bias_kmax/plots/`:

| folder | data written by | figure |
|---|---|---|
| fob_kmax | k_max export cell | the shifts of a linear-bias fit to the model A truths against k_max, per layout and prior |
| fom_kmax | k_max export cell | the FoM(w0, wa) of every linear-bias fit against k_max |
| ladder_bias_params | k_max export cell | the accuracy test of the linear-bias priors scored for (w0, wa), Omega_m and A_s |

The folders of `notes/higher_order_bias/plots/`, all written by
`plots/make_figure_data.py` in the `ccl` conda environment (CAMB
1.6.5, FAST-PT 4.0, Takahashi halofit as in the likelihood):

| folder | figure |
|---|---|
| coevolution_biases | the fiducial b_1, b_2 (Lazeyras fit) and b_{s^2} (co-evolution) across the six chunks |
| draft_bias_check | the draft paper's linear-bias column read as (1 + b_1^L) sigma_8(z_eff), against B_g/D(z_eff) |
| k_neg | the wavenumber where the one-loop P_gg first turns negative, against Delta_2 |
| term_sizes | each one-loop term in percent of b_1^2 P_mm for four tracers |
| poisson_power | the Poisson power 1/nbar of the thin bins in percent of b_1^2 P_mm |

Every figure also follows the rules of Sec. 1, has no bbox around
text and no `\%` in labels of non-usetex figures, and is rendered and
looked at before it goes into a note or to collaborators. The Python
style and plotting conventions of the cosmolike-dev skill
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
5. for the correlated-prior note, run `python verify_prior.py` in
   `notes/correlated_bias_prior_v2/` with the cocoa environment: 23
   PASS lines.

## 6. Commits

lssty1xdesi commits go directly on `main`, never pushed and never
amended. The repository uses Git LFS, so commits need git-lfs on
PATH: the cocoa environment, or
`export PATH="/Users/vivianmiranda/miniforge/pkgs/git-lfs-3.7.1-h07d3073_0/bin:$PATH"`.
The desktop app's worktree guard blocks the Edit tool inside nested
project repositories: edit a scratch mirror and copy it back.
