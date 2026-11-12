# The bias notes: notebook cells, figures and builds

Machine-local recipes for the four galaxy-bias notes under `notes/`
and for the notebooks that feed them. The physics is in the `.tex`
notes (SKILL.md, "The bias notes are the physics reference"); this
file holds how the notes are organized, how the numbers are computed,
where the figures come from and how the documents are rebuilt. The
figure rules are in `.claude/skills/paper-figures/SKILL.md`; the
documentation rules (names, symbols, the statistic eta) are in
SKILL.md, "Decisions and their reasons".

## 1. How the notes are organized

There are four galaxy-bias notes; new bias-model work goes into them,
not
into new note folders (no v3: forecast results go into
`correlated_bias_prior_v2`):

- `notes/correlated_bias_prior_v2/correlated_bias_prior_v2.tex` (with
  `appendix_physics.tex` and `appendix_math.tex`), the forecast note:
  Summary; Fiducial model and parameter values; The forecast; The
  linear-bias prior; The bias model; The accuracy
  test; The fiducial model against k_max.
  Appendices: the physics of the linear-bias priors (The
  models; Which L and sigma_g are physical; The memory model; The bias
  continuity equation, step by step; The BGS flux-limit bias model,
  step by step), the mathematics (The projected
  precision for two bins; The projection as a marginalization; Many
  bins: the tridiagonal inverse; The memory projection, step by step;
  How the prior enters the Fisher matrix; The accuracy test; Numerical
  verification), and
  the model record (Implementation and the Fisher construction;
  Numerical checks; The counterterm requires the low-rank
  engine;
  Limitations; Open issues). Every result is a figure drawn from a
  `plots/` folder. New material goes into an appendix and reaches the
  Summary only when it changes a headline statement or figure.
- `notes/linear_bias_kmax/linear_bias_kmax.tex`: a linear-bias fit to
  the fiducial model's one-loop truths against k_max on the four
  layouts, and the accuracy
  test of the linear-bias priors scored for Omega_m and A_s (Summary;
  Fiducial model and parameter values;
  appendices Method, Numerical checks, Limitations, Open issues).
  Figures `fob_kmax`, `fom_kmax`, `ladder_bias_params`.
- `notes/higher_order_bias/higher_order_bias.tex`: the one-loop model
  as implemented (Scope and conventions; The one-loop galaxy power
  spectrum; The DESI DR1 full-shape model and priors; The draft paper's
  bias table: the tidal-bias verdict; The size of every term; The model
  as implemented; Projection to thin-bin angular spectra; Validation;
  appendices Bases and conversions, with every factor; The loop
  templates and their low-k constants; Numerical setup for the figures;
  Open issues). Its pre-implementation Recommendation and "Model B and
  the order of work" are kept verbatim in
  `higher_order_bias_recommendation.md`; the gap lists and the code
  inventory are gone from the note.
- `notes/bias_model_summary/bias_model_summary.tex`, the presentable
  summary for collaborators, rewritten last from the other three, with
  their key figures at a readable size, white space and short
  sections, as few pages as possible without looking compact.  It
  presents one model and one prior, neither named nor compared.
  Sections: Summary; Fiducial model and parameter values; Data and
  scale cut; The cross pairs and their
  overlaps; The bias model; The prior: the freedom of the wide sample;
  Projection, covariance and the Fisher forecast; Forecast; What the
  grouping is worth; Where the details are.

Rules for every note:

- A note describes what was done: the question, the method, the result
  and its limits. No TODO, no Recommendation section, no "should",
  "will" or "needs"; recommendations and decisions live in the skill.
  Whatever a recommendation or gap list asked for and was done becomes
  part of the model's explanation in the main text, written as how the
  model works, not as a closed to-do; a gap list leaves the note once
  addressed.
- Open items that matter (decisions, known problems) go into an "Open
  issues" appendix, one to three sentences each, only where relevant.
- The Summary comes first, short (at most two to three pages for
  collaborators); everything else goes into appendices.
- No large tables: a result is a figure, and the values a caption
  leaves out are in that folder's `data.json`. Only small didactic
  tables survive.
- A revision cuts the prose by about 20%: say each thing once, remove
  repeated caveats first, prefer a figure to a paragraph, short
  paragraphs, every symbol defined once.
- k_max is explained in full once, where it is first used (the z = 0
  value of a per-bin cut at k_max/D(zbar_i), the UV-fraction criterion
  with alpha = 0.05, intersected with the shipped mask); figures and
  captions follow the paper-figures skill.
- Sections are cited by title, never by appendix letter.
- The text describes the covariance as it is (Gaussian, on the one-loop
  fiducial); the history of the earlier wrong covariance stays out.

## 2. The notebook cells behind the ladder

### The linear notebook

All in section 8 of EXAMPLE_EVALUATE_FISHER.ipynb, subsection "How
much bias freedom costs".

- **Per-bin derivatives** (the cell that opens "The derivatives of
  each layout's DESI vector with respect to every bin's own bias"):
  b_i dm/db_i for every bin, from a greedy coloring of the cross-pair
  graph (two evaluations per color), through a wrapper of
  `ci.set_nuisance_bias`, cached in
  `chains/EXAMPLE_EVALUATE_FISHER_BIN_BIAS.pkl` (RECOMPUTE_BIN_BIAS
  False by default). C_ell is at most quadratic in the biases, so the
  central difference is exact. Check: per-bin columns of a group sum to
  the group's cached derivative column.
- **Ladder** ("The stated assumptions of the bias-prior note"):
  BIAS_ASSUMPTIONS (the baseline and pessimistic sigma_b per group and
  L), LADDER_RUNGS, SHIFT_THRESHOLD = 1/3, BIN_ZBAR, bias_groups,
  group_bias_precision(key, L, sigma_by_group, free_tilt=True),
  bin_bias_fisher, bias_drift(key, law), ladder_rung. Self-check: the
  ladder FoMs reproduce the kernel and memory scans to 9.6e-7.
- **Roughness** ("The same accuracy test on a continuous scale"):
  ROUGH_SIGMAS = geomspace(1e-3, 0.5, 40), ROUGH_L = [0.05, 0.1, 0.2,
  0.5, 1.0], both models. Smooth deviations constrain only
  sigma_g/sqrt(L), so scans use x = sigma_g sqrt(0.1/L), the rms change
  of delta between bins Delta z = 0.05 apart. Self-check: 7.5e-12
  against the memory grid.
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
  current covariance at every run.

### The k_max and model notebooks

These two are generated by builder scripts (SKILL.md, "Workflow").

- **EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb** reads the linear
  notebook's derivative, per-bin bias and scale-cut caches and its
  Gaussian priors from the example yamls (`prior_diagonal`, cobaya
  `update_info`; never a cached Fisher matrix), caches the fiducial
  model's one-loop truth vectors (built from the paper yamls
  EXAMPLE_EVALUATE<n>.yaml with the linear vector's own settings -
  `fm.derivative_settings(dataset, 'final')`, which points each layout
  at its paper data set) in
  `chains/linear_kmax_truths_final.npz`,
  imports the drift laws through THEORY_PLOTS, and writes fob_kmax,
  fom_kmax and ladder_bias_params into `notes/linear_bias_kmax/plots/`.
  Self-check: the matched-prior identity E[chi2_w] - drift^2 = 2 to
  5.6e-11.
- **EXAMPLE_EVALUATE_FISHER_MODEL.ipynb** (builder build_model_notebook.py)
  copies, reads and computes its own two kinds of cache, each in explicit
  cells (a copy cell that restores the archived `fisher_caches/` caches
  to their `chains/` working copies, a read cell that stops on another
  settings dict or fingerprint, and commented compute and write cells,
  the write refusing to replace a file without `overwrite=True`):
  - section 3, the UV-fraction cuts on the paper fiducial
    (`chains/model_scale_cuts_final.npz`, the four
    layouts, cuts[key]['kmax'] and cuts[key]['mask'] as in the linear
    notebook's cache, plus settings and a fingerprint), computed as the
    study's run_cuts does but on EXAMPLE_EVALUATE<n>.yaml, with the
    study's k_max list and alpha, at the final derivative tier
    (`CUT_ACCURACY = DERIVATIVE_TIER`, each layout's derivative
    settings);
  - section 4, the derivative base on the paper yamls
    (`chains/model_derivatives_<layout>_final.npz`: stencil
    columns of COSMO, the six B_g and the seven sigma_v leaders, per-bin
    b1, b2, bs2, alpha_0 columns, SN0 templates, the fiducial vector and
    biases; `chains/model_templates_<layout>_final.npz`: per-bin alpha_2
    columns and the SN0 and SN2 templates). The per-bin steps go through the step
    classes of `scripts/compute_fisher_base_mpi_classes.py`, in the
    evaluation order of `scripts/compute_fisher_base_mpi.py` on one
    rank, which writes the same caches on MPI ranks.
  It scans seven cuts to k_max = 0.2 (`KMAX_MAX`) for the reference
  model with and without tilts (data keys 'B tilts', 'B') on three
  priors (168 Fisher systems), runs the break test of the variant
  without tilts (section 7), the fiducial model (section 9: alpha_2
  fixed to 0, every sigma_v prior width times PAPER_SIGV_SCALE = 6;
  data key 'B tilts noA2 sv x6'), the growth-tied and untied comparisons,
  and the alpha_2 / sigma_v study (section 11, whose fit without alpha_2
  is 'B tilts noA2'). Its exports: fom_kmax_models and growth_tied_kmax
  (`notes/correlated_bias_prior_v2/plots/`), summary_fom_kmax,
  summary_fom_untied, summary_prior_draws and summary_pair_threshold
  (`notes/bias_model_summary/plots/`), fog_templates,
  fog_degeneracy_kmax, fog_sigv_width and fog_pair_overlap
  (`notes/fog_parameter_degeneracy_study/plots/`) and pair_threshold
  (`notes/cross_correlations/plots/`). The sigma_v widths of the
  reference model are the exploration ones (`EXPLORATION_PARAMS`, the
  linear parameter files); the paper yaml's are six times wider, which
  section 4 checks. The pair-axes figure data of
  `notes/cross_correlations/plots/` (pair_axes_gap, pair_axes_sigv,
  pair_axes_attribution, pair_axes_priors) is written by the three
  `make_pair_axes_*_data.py` generators from
  `chains/pair_axes_fits_<layout>.json`, the output of
  `scripts/compute_pair_axes_fits.py` on the archived
  `model_*_<layout>_pairmax_final` caches (its docstring holds the
  usage and the prerequisites). Self-checks: the chain-rule columns reproduce the
  B_g stencil column (at most 1.5e-9 of the column); the two caches hold
  one fiducial vector (delta chi2 = 0) and equal SN0 templates; the
  vector after the alpha_2 steps equals the one before bit for bit;
  forward and backward alpha_2 differences agree to 1.6e-10; +- truth
  pairs score identically; the pair scan's "current" rows equal the
  FoMs of sections 6 and 9.
- The former EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb (model A at
  k_max = 0.075: the ladder, the accuracy test, the drifts) was removed
  on 2026-11-10 together with its figure folders; its last version is in
  commit 567751f.

BIAS_ASSUMPTIONS of all three notebooks must equal STATED_PRIORS of
`notes/correlated_bias_prior_v2/plots/make_theory_data.py`, which
draws the baseline and pessimistic histories (figure prior_draws).
The chains/ caches are gitignored, local files: a fresh clone
recomputes the derivatives (hours) and every notebook runs from its
caches afterwards. Derivative caches do not depend on the covariance;
Fisher matrices and FoMs do, and are rebuilt at every run.

### Why these cells are fast

No cell after the derivative cache evaluates a data vector. With F the
prior's Fisher matrix (data plus priors), C the data covariance and dm
the data-vector change of a bias drift, the linear response of the fit
is dp = F^-1 D^T C^-1 dm. A drift delta_i of the bin amplitudes beta_i
gives dm = sum_i (dm/dbeta_i) beta_i delta_i, so D^T C^-1 dm =
F_data[:, bias] (beta * delta), columns the Fisher matrix already
holds. Each prior and layout therefore reduces to one 2 x n_bins
matrix A with dw = A delta, and 2000 random truths cost one matrix
product. The growth-tied prior is evaluated in its own parameter basis,
one amplitude per group; the free prior absorbs any delta exactly.

The (w0, wa) value of eta = sqrt(<chi^2>/n_dof), n_dof = 2, uses
<chi^2_w> = Delta_drift^2 + tr(C_w^-1 A Sigma A^T) + tr(C_w^-1 N_w):
Delta_drift the shift of the drift alone, Sigma the truth's residual
kernel, C_w the fit's marginalized (w0, wa) covariance and N_w its
share from data noise and the sigma_v priors. When the fit's prior is
the truth's kernel, P Sigma P = P makes the last two terms sum to
exactly 2, which is the self-check above; the drift term leaves
sqrt(1 + Delta_drift^2/2) <= 1.03 in the matched cases.

## 3. The plot folders

Every figure of every note has a folder `plots/<name>/` under the
contract of the paper-figures skill (data.json with its `about`,
a standalone `plot_<name>.py` with the house rcParams block, the PDF
and a PNG, and the standalone `python -I` redraw test).

The folders of `notes/correlated_bias_prior_v2/plots/` (the section
that includes the figure in parentheses):

| folder | data written by | figure |
|---|---|---|
| prior_draws | make_theory_data.stated_prior_draws | draws of the baseline and pessimistic priors, a row per prior and a column per group, each column scaled by a round delta_g (The linear-bias prior) |
| zhou_lrg_bias | transcribed from Zhou et al. 2021 (arXiv:2001.06018) | measured LRG bias against the constant-amplitude shape (appendix What the measurements say) |
| theory_bz, theory_delta, theory_draws | make_theory_data.py | the drift laws and prior draws of the derivation appendices |
| fom_kmax_models | EXAMPLE_EVALUATE_FISHER_MODEL.ipynb, section 8 | one full-width 2x3 grid: a row per scenario (the baseline prior with the variant without tilts; the pessimistic prior with the reference model), columns FoM(w0, wa), FoM(Omega_m, 10^9 A_s) and the largest tested shift with the sigma/3 line (the pessimistic row carries a text: its tilts absorb the drift exactly). data.json holds every FoM, shift and break point (with the truth that sets it) and the free prior (The fiducial model against k_max) |
| growth_tied_kmax | EXAMPLE_EVALUATE_FISHER_MODEL.ipynb, section 9 | eta and coverage of the reference model on the growth-tied prior against the pessimistic prior |

`plots/make_theory_data.py` sits in the same folder. The folders of
`notes/linear_bias_kmax/plots/`:

| folder | data written by | figure |
|---|---|---|
| fob_kmax | k_max export cell | the shifts of a linear-bias fit to the one-loop truths against k_max, per layout and prior |
| fom_kmax | k_max export cell | the FoM(w0, wa) of every linear-bias fit against k_max |
| ladder_bias_params | k_max export cell | the accuracy test of the linear-bias priors scored for (w0, wa), Omega_m and A_s |

The folders of `notes/higher_order_bias/plots/`, all written by
`plots/make_figure_data.py` in the `ccl` conda environment (CAMB 1.6.5,
FAST-PT 4.0, Takahashi halofit as in the likelihood yamls, m_nu = 0.06
and n_s = 0.96605 as the likelihood fiducial, which gives
sigma4 = 4391 (Mpc/h)^3):

| folder | figure |
|---|---|
| coevolution_biases | the fiducial b_1, b_2 (Lazeyras fit) and b_{s^2} (co-evolution) across the six groups |
| draft_bias_check | the draft paper's linear-bias column read as (1 + b_1^L) sigma_8(z_eff), against B_g/D(z_eff) |
| k_neg | the wavenumber where the one-loop P_gg first turns negative, against Delta_2 |
| term_sizes | each one-loop term in percent of b_1^2 P_mm for four tracers |
| poisson_power | the Poisson power 1/nbar of the thin bins in percent of b_1^2 P_mm |

The Python style and plotting conventions of the cosmolike-dev skill
(`references/python.md`) apply to every script.

## 4. Running a notebook

Execute a scratch copy, never the committed file in place, and run it
in the background (from scratch the derivatives take hours; with the
pickles in `chains/` a notebook reruns from its caches):

```bash
source /Users/vivianmiranda/miniforge/etc/profile.d/conda.sh
conda activate cocoapy311
cd /Users/vivianmiranda/data/COCOA/november2026/sbu_cosmolike/cocoa/Cocoa
source start_cocoa.sh
export JUPYTER_PATH=<scratch>/jupyter   # kernels/cocoa/kernel.json
export OPENBLAS_NUM_THREADS=1
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=cocoa --ExecutePreprocessor.timeout=-1 \
  <scratch>/EXAMPLE_EVALUATE_FISHER.ipynb
```

The scratch kernelspec's `kernel.json` runs
`Cocoa/.local/bin/python -m ipykernel_launcher -f {connection_file}`;
ipykernel, nbclient and nbconvert are in `Cocoa/.local`, and the
user's own Jupyter configuration is never touched. After the run,
restore the notebook's original `metadata.kernelspec`.

- **The linear notebook** (EXAMPLE_EVALUATE_FISHER.ipynb) has no
  builder: new cells are added to the scratch copy by small Python
  scripts that edit the notebook JSON.
- **The two generated notebooks** are rebuilt from their builders
  (`python -I build_<x>_notebook.py <out.ipynb>`) and executed by the
  `run_*_nb.sh` wrappers of the builder folder; the wrappers name a
  session scratchpad (`SP=...`), which must be set to an existing
  scratch folder holding the builder and the `jupyter/` kernelspec.
  Edit markdown through the builder's `md(...)` blocks and code through
  its `code(...)` blocks; a hand edit of the generated notebook is lost
  at the next build.
- **Validate before copying back**: `python -I validate_nb.py
  <executed.ipynb>` prints the cell count, error outputs, warning
  streams, DRAFT cells and scratch paths (all must be zero) and the
  self-check lines (`check`, `systems with`, `dropped`, `matched`,
  `safe k_max`), which must show the values of Sec. 2. New data.json
  files are compared against any prototype, and every redrawn PNG is
  looked at.
- The notebook is committed executed, with its outputs and figures, so
  they render on GitHub.

## 5. Building the notes

Build in a scratch directory, never in the repository:

1. copy `<doc>.tex` (with its appendix files) and its `plots/` folder
   there;
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
amended or forced. The repository uses Git LFS, so commits need
git-lfs on PATH: the cocoa environment, or
`export PATH="/Users/vivianmiranda/miniforge/pkgs/git-lfs-3.7.1-h07d3073_0/bin:$PATH"`.
The desktop app's worktree guard blocks Edit and Write inside the
nested project repository: edit a scratch mirror and copy it back
(SKILL.md, "Workflow").
