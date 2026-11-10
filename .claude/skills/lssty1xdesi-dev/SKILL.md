---
name: lssty1xdesi-dev
description: Development on the lssty1xdesi project - DESI thin-bin spectroscopic angular clustering C_ell^gg (non-Limber band powers) on four bin layouts (Delta d_com = 62 Mpc, 76 bins of equal comoving width; fixed-width Delta z = 0.02, 0.05, 0.2), with LSST Y1 cosmic shear and an LSST Y1 3x2pt reference case. Use this skill for ANY work in projects/lssty1xdesi - the likelihood desi_cl (linear bias, and the one-loop option whose parameter files are models A and B), the Fingers-of-God model and sigma_v, the n(z) files, the low-rank non-Limber engine, the gg-cross block, the covariance adapter, the test suite, the data scripts, the Fisher module and the Fisher notebooks (EXAMPLE_EVALUATE_FISHER.ipynb, _LINEAR_KMAX, _MODEL_A, _MODEL_B) with their builders and caches, and the four galaxy-bias notes under notes/ (correlated_bias_prior_v2, linear_bias_kmax, higher_order_bias, bias_model_summary) with their plots/ folders. Also trigger on mentions of DESI thin bins, nonlimber_accuracyboost, NL_Nchi, DESI_SIGV, DESI_DB2/DBS/A0/A2/SN0/SN2, gg cross pairs, the dndz FoG convolution, the correlated bias prior and its priors (growth-tied, baseline, pessimistic, free), the memory model, the statistic sqrt(<chi^2>/n_dof), higher-order bias models A and B, alpha_2, SN_2, where models A and B break, model A + tilts, the UV-fraction scale cut, the one-loop covariance, or the amplitude cut. This survey is the numerical extreme of the cosmolike projects, and defaults that are safe elsewhere fail here.
---

# lssty1xdesi: the thin-bin extreme

This project forecasts DESI spectroscopic galaxy clustering with LSST
Y1 cosmic shear: angular C_ell in thin bins, not a 3D power-spectrum
analysis. The reference layout slices the galaxy sample into 76 bins of
equal comoving width, about 62.3 Mpc (Delta d_com = 62 Mpc, d_com the
comoving distance; mean redshifts 0.108 to 1.549; BGS bins 0-17, LRG
18-48, ELG 49-75, named by `data/lssty1xdesi_desi_dr3_bins.txt`). The
data vector is Fourier band powers, 53 bands of width Delta ell = 10
over ell = 15 to 544: 15 shear pairs + 76 clustering autos + 97
measured cross-bin pairs = 188 spectra, 9,964 entries, 5,309 unmasked.
No galaxy-galaxy lensing rows are measured (`ggl_exclude: all`). Three
fixed-width layouts of the DESI DR1 distributions (Delta z = 0.02, 0.05,
0.2: 90, 36 and 10 lens bins, `scripts/make_desi_dz_case.py`) and an
LSST Y1 3x2pt case complete the examples:

| example | layout | covariance case |
|---|---|---|
| EXAMPLE_EVALUATE1 (MODEL_A1, MODEL_B1) | Delta d_com = 62 Mpc | `dr3` |
| EXAMPLE_EVALUATE2 (A2, B2) | Delta z = 0.02 | `dr1_dz0p02` |
| EXAMPLE_EVALUATE3 (A3, B3) | Delta z = 0.05 | `dr1_dz0p05` |
| EXAMPLE_EVALUATE4 (A4, B4) | Delta z = 0.2 | `dr1_dz0p20` |
| EXAMPLE_EVALUATE5 | LSST Y1 3x2pt alone | `lsst_y1_3x2pt` |

File stems and case names keep `dr3` and `dr1`; reader-facing text
writes Delta d_com = 62 Mpc and Delta z = ... ("Decisions and their
reasons").
The survey inputs come from the source forecast's sacc file and DESI
text files under the repository-external `to_reproduce/` folder;
`scripts/make_desi_nz.py` and `scripts/make_desi_dataset.py` convert
them.

The thin bins are the defining numerical challenge: every setting below
exists because a default that photometric surveys never notice fails
here by hundreds to tens of thousands of chi2. Validate any change
against the frozen suite; never transfer a "converged elsewhere"
setting without measuring it on this vector.

## Where things live

| topic | file |
|---|---|
| current design, decisions, pitfalls, workflow, open work | this file |
| numerical settings and their evidence: non-Limber requirement, chi grid, accuracy boosts, Boltzmann k range, FoG and sigma_v priors, n(z) grid, amplitude cut and N-random-models check | `references/numerical_settings.md` |
| the one-loop model in detail: option, parameters, units, conventions, projections and checks, FAST-PT validation, white-noise constant and positivity, priors not adopted, parameter budget | `references/one_loop_model.md` |
| the covariance: one-loop Gaussian term, non-Limber grid, pitfalls, how to build the files, validation, the G + SSC plan | `references/covariance.md` |
| the Fisher module's rules, the LSST Y1 3x2pt case, engine Fisher biases, the UV-fraction scale cut, linear-bias FoM tables | `references/fisher_and_scale_cuts.md` |
| every result table with its notebook section and data.json | `references/results.md` |
| the 2026-11-06 review: the thin/wide FoM ratio, what sets the absolute FoM, the draft paper's bias model against CCL, why model A carries SN_2 | `references/fom_and_bias_review.md` |
| how the notes are organized and built, the notebook cells and caches, the plot folders, running and validating notebooks | `references/bias_notes_workflow.md` |
| the pre-implementation Recommendation of the higher-order note, verbatim | `references/higher_order_bias_recommendation.md` |
| every figure rule (layout, palette, legends, captions, k_max on axes, the plot-folder contract) | `.claude/skills/paper-figures/SKILL.md` |
| the physics: derivations, assumptions, literature | `notes/*/*.tex` (the source, not the PDF) |

## Workflow

- **Repository and commits.** `projects/lssty1xdesi` is its own git
  repository (git@github.com:SBU-COSMOLIKE/lssty1xdesi.git). Commit
  directly on `main`, no topic branches: the project repository plays
  the role of the branch. Never push, amend or force: the PI pushes.
  The repository uses Git LFS (the two 92 MB n(z) copies, the frozen
  covariance, the window tables), so commits need git-lfs on PATH: run
  git from the cocoa environment (`conda activate cocoapy311`) or
  export the git-lfs path of `references/bias_notes_workflow.md`,
  Sec. 6. Core changes go to cosmolike_core on its branch
  `lowrank_fftlog`, under the same never-push rule.
- **The nested-repository guard.** Desktop-app sessions in a cocoa
  worktree cannot Edit or Write files under the main checkout's
  `Cocoa/` (a PreToolUse hook), and the nested repositories
  (`Cocoa/projects/lssty1xdesi`, `Cocoa/external_modules/code/cosmolike_core`)
  exist only there. Copy the file to the session scratchpad, edit the
  copy, copy it back with `cp`, and verify with `cmp` and `git diff`.
  Before each copy back, check that the target has not changed since
  it was read (another session may be editing the same tree).
- **The notebook builders.** EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb,
  EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb and
  EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb are generated by builder scripts
  that are in no repository. Their persistent copy is
  `/Users/vivianmiranda/Library/Application Support/Claude/side-session-notes/local_6b327622-bba3-4f4a-bf11-769622bccf03/notebook_builders/`
  (build_linear_kmax_notebook.py, build_model_a_notebook.py,
  build_model_b_notebook.py, validate_nb.py, the run_*_nb.sh nbconvert
  wrappers, plot_scripts/ and a README); session scratchpads are
  cleaned, so a builder edited in a scratchpad is copied back there.
  Edit markdown through the `md(...)` blocks and code through the
  `code(...)` blocks, never the generated notebook. Whether the
  builders belong in the project repository is undecided: ask the PI
  before adding them. The run, validation and copy-back recipe is
  `references/bias_notes_workflow.md`, Sec. 4.
- **Notebooks are committed executed**, with outputs and figures, so
  the plots render on GitHub; stale outputs from older code are wrong.
- **Figures.** Every figure follows `.claude/skills/paper-figures/SKILL.md`
  (the plot-folder contract `notes/<note>/plots/<name>/` with data.json,
  a standalone plot script, PDF and PNG; the house rcParams block; one
  column; Okabe-Ito colors with redundant encodings; legends checked off
  the data in code; captions; k_max <= 0.2). Read it before writing or
  changing any figure. It is not repeated here.
- **Notes** are built in a scratch directory with pdflatex three times
  (zero errors, undefined references and oversized floats), and
  `notes/correlated_bias_prior_v2/verify_prior.py` prints 23 PASS lines
  (`references/bias_notes_workflow.md`, Sec. 5).
- **Documentation prose** (READMEs, test guides, notebook markdown,
  note prose, skill reference text) is delegated to a Fable 5 subagent
  (`subagent_type: fable5`); until 2026-11-09 a general-purpose Opus
  subagent takes the same brief. The brief carries the facts and numbers
  with their sources (the subagent cannot see the conversation), the
  files, the nested-repository rule and the documentation rules of
  "Decisions": say each thing once, avoid repeated caveats, prefer a
  figure to a paragraph. Code and code comments stay with the main
  session or a general-purpose agent.
- **Timings** are quoted only from runs on AC power on a quiet machine,
  one job at a time (`pmset -g batt` before measuring). The published
  runtimes are the 2026-11-05 Apple M2 Pro AC runs. Battery runs
  throttle the CPU, and the "derivatives in N s" lines printed by the
  executed Fisher notebooks come from runs that shared the machine, so
  neither is usable.

## Numerical settings

The adopted values (evidence in `references/numerical_settings.md`):

- **Non-Limber clustering on every layout**: `adopt_limber_gg: 0` with
  the low-rank engine (`nonlimber_lowrank_gg: 1`, rank 5, multipole cap
  1000). Pure Limber is off by delta chi2 = 21,158, FKEM at its
  defaults by 4,243; the one-loop counterterm also needs the low-rank
  engine.
- **The chi grid**: `accuracyboost: 2` everywhere, with
  `nonlimber_accuracyboost` 32 / 32 / 16 / 4 (combined factor 64 / 64 /
  32 / 8) on 62 Mpc / Delta z = 0.02 / 0.05 / 0.2. The default grid
  (NL_Nchi = 512) aliases thin kernels by delta chi2 ~ 1.5e5, and
  between factors 32 and 64 there is a one-bin cliff. Fisher
  derivatives use `nonlimber_accuracyboost: 64` and the patched CAMB
  halofit (`PATCH_CAMB_HALOFIT_TOLERANCE`).
- **CAMB**: AccuracyBoost 1.05, `kmax_boltzmann: 20` (40/Mpc at boost
  2, for the low-Omega_m prior corner).
- **n(z)**: five-fold refined grid (49,996 rows), `photoz_zmid_convention:
  1` (bin midpoints), `external_nz_modeling: 1` for the per-point FoG
  convolution with seven sampled sigma_v groups under measured Gaussian
  priors.
- **Priors**: Omega_m in [0.2, 0.5], w0 in [-3, -1/3], w0 + wa in
  [-3, -0.01], H0 in [55, 91]; DESI bias boxes [0.8, 4] (kept); the
  cobaya `amplitude_cut` (A_s/A_s,fid)(b_i/b_i,fid)^2 <= 8 per group.

## The bias model

### The linear bias and its four priors

Bin i of group g (BGS, LRG1, LRG2, LRG3, ELG1, ELG2) has

    b_{1,i} = B_g (1 + delta_i) / D(zbar_i),   delta_i = s_g (zbar_i - zbar_g) + eps_i,
    <eps_i eps_j> = sigma_g^2 exp(-|zbar_i - zbar_j| / L)   (same group; 0 across),

with B_g the group amplitude (DESI_B1 of the group leader, the bias at
z = 0 under a constant clustering amplitude b D), D the linear growth
factor (D(0) = 1) read off the linear P(k) ratio at k = 5e-3/Mpc, s_g a
free tilt, zbar_g the group's mean bin redshift, eps_i the residual
around the group's line, sigma_g its per-bin width and L its
correlation length in redshift. The Fisher prior per group is the
projected precision P = Q - Q U (U^T Q U)^-1 U^T Q (Q the inverse
kernel, U = [1, z - zbar]): it charges only the residual, never the
level or the line, and is zero for a group of one or two bins, so on
Delta z = 0.2 the baseline, pessimistic and free priors coincide.

| prior | definition | stated source |
|---|---|---|
| growth-tied | b_i = B_g/D(zbar_i) (the shipped yamls) | the upper FoM limit |
| baseline | sigma_g = 3% BGS, 1% other groups; L = 0.2 | BGS: the r < 19.5 flux limit drifts the bias 22% across the group and a line leaves 2.4% rms; 1% hedges unmodeled selection curvature, at least six times the 0.16% a conserved population leaves |
| pessimistic | sigma_g = 3% every group; L = 0.05 | local features of the largest derived size on the redshift scale of the 4000 A break crossing a filter edge |
| free | one flat bias per bin | the lower limit |

A conserved population obeys b - 1 proportional to 1/D (Fry 1996), not
b D constant; that is why the tilt is free. The derivations are in
`notes/correlated_bias_prior_v2` (appendices on the models and the
memory model).

### One likelihood option; models A and B as parameter files

`likelihood/desi_cl.py` option `higher_order_bias`: 0 is linear bias;
1 is the one-loop model with every term per bin. Models A and B are
parameter files of option 1 (`scripts/make_desi_params.py --model-a`
or `--model-b`), each in two variants: **model A** (one value per group
for every higher-order family) and **model A + tilts**, in which
Delta_2, Delta_s and alpha_0 (and alpha_2 in model B + tilts) also get
a free straight line in redshift per group, like the linear bias's
s_g, with flat priors on the tilts; likewise model B and model B +
tilts. Either model runs under any of the four linear-bias priors.

The real-space power of bin i is b_{1,i}^2 P_nl + P_loop(b_{1,i},
b_{2,i}, b_{s^2,i}) + alpha_{0,i} k^2 P_nl, with
b_{2,i} = b2_Lazeyras(b_{1,i}) + Delta_{2,g} (arXiv:1511.01096) and
b_{s^2,i} = -(4/7)(b_{1,i} - 1) + Delta_{s,g} (co-evolution), b3 = 0.
Redshift space is linear Kaiser on each bin's FoG-convolved n(z), plus
the counterterm alpha_2 and the stochastic terms:

| per group | model A | model B |
|---|---|---|
| offsets Delta_2 / Delta_s | width 1.0 / 0.5 | width 2.0 / 1.0 |
| alpha_0 [h^-2 Mpc^2] | 0.5/k_max^2 = 88.9 at k_max = 0.075 | 0.5/k_max^2 (16.1 in the yamls, the shipped mask's 0.176 h/Mpc; per cut in the Fisher scan) |
| alpha_2 [h^-2 Mpc^2] | fixed to 0 | 0.5/k_max^2 |
| SN0 [P_Pois,i = 1/nbar_i] | width 2 | width 2 |
| SN2 [f_sat sigma_sat^2 P_Pois,i] | width 5 | width 5 |
| tilted in the + tilts variant | Delta_2, Delta_s, alpha_0 | Delta_2, Delta_s, alpha_0, alpha_2 |
| parameters, without / with tilts | 5 / 8 (30 / 48 in all) | 6 / 10 (36 / 60) |

All Gaussian priors are centered on zero, so both fiducials sit on the
calibrations and coincide. SN0 and SN2 never tilt: their Poisson part
is known bin by bin.

Conventions that matter:

- **alpha_0** rides cosmolike's slot 2 b1 bK k^2 P_nl with k in units of
  H0/c, so bK = alpha_0/[2 b1 (c/H0)^2], c/H0 = 2997.92 Mpc/h. It
  multiplies P_nl, not DESI's (b1 + f mu^2) b1 P_lin. Its width is a
  correction of 0.5/B_g^2 of the linear galaxy power at each bin's own
  cut (BGS 28%, LRG 17%, ELG 71% at model A's cut).
- **alpha_2** follows the DESI DR1 field convention (Maus et al.,
  arXiv:2404.07312): P_CT = (b1 + f mu^2) f alpha_2 mu^2 k^2 P_lin, mu
  the cosine to the line of sight, f the linear growth rate. alpha_2 k^2
  is then a fractional correction of the velocity share of the Kaiser
  power whatever b1 and f, so 0.5/k_max^2 means a 50% correction at the
  cut in every bin. Set per bin with `ci.set_nuisance_bias_alpha2(A2=...)`
  (numpy array, h^-2 Mpc^2); `galpha2` in bias.c applies b1 and f.
- **SN0** is the deviation from Poisson (the covariance carries the
  Poisson part), nbar_i fixed by the bin's surface density
  (`<stem>_ngal.txt`) so that 1/nbar_i projects to the angular shot
  noise.
- **SN2** is SN2 k^2 mu^2 in units of f_sat sigma_sat^2 P_Pois,i
  (`SN2_UNIT`, from Maus et al.'s stochastic-prior table): LRG
  (f_sat, sigma_sat) = (0.1, 7.8 Mpc/h), ELG (0.1, 2.9 Mpc/h);
  sigma_sat is the satellites' dispersion, not the FoG sigma_v (the code
  writes sigma_v). BGS has no row there and borrows the LRG values; that
  cannot matter, because SN2 in BGS alone moves no parameter (0.00 sigma
  at 0.075, with the borrowed unit or one from BGS's own FoG
  dispersion).
- **Stochastic cross terms** exist only within one tracer sample (the
  mean of the two bins' units); LRG x ELG pairs at the same redshift get
  none.
- A zero amplitude switches a term off at no cost (alpha_2 = 0 skips the
  C work; an all-zero stochastic family is not computed). Option 1
  needs `bias_model: [0,0,0,0,0,0]`.

### Projections and their checks

Every term projects without the Limber approximation where it matters
(details and numbers: `references/one_loop_model.md`, Sec. 4):

- **One-loop autos**: the low-rank engine factorizes P_g - P_inf, the
  spectrum with the unsubtracted templates, P_inf = -(sigma4/2) D^4
  (b2 + 2 bs2/3)^2 (sigma4 = 4391 (Mpc/h)^3); P_inf reaches C_ell
  through Limber, exact for a constant. Linear-bias and FKEM vectors are
  bitwise unchanged; 1 and 8 threads identical.
- **One-loop cross pairs**: the Limber pair batch carries the bilinear
  one-loop pair spectrum, and its density-only output is the geometric
  mean sqrt(P_g,i P_g,j), the high-ell limit of the low-rank pair term.
  A same-n(z) pair equals the polarization of the autos to 1e-14.
- **alpha_0**: the low-rank response is 7.3x Limber at ell = 20 and
  ~1.05x at ell 200-400 on Delta z = 0.05; FKEM moves it through Limber,
  so the model requires the low-rank engine.
- **alpha_2** reuses the Kaiser transfer and its RSD row in every gg
  engine: zero is bitwise the old build, linear to 1e-12, a brute-force
  j_l / j_l'' integral agrees to <= 6e-4 (thin auto and adjacent pair);
  at 12.5 h^-2 Mpc^2 it is a median 0.7-1.1% of the autos, at most 2.9%.
- **SN0**: C_ij = P_ij int dchi W_i W_j/chi^2, exact for white noise.
- **SN2**: a k_par^2 white noise is the radial derivative of white
  noise, so C_ij = U_ij int dchi W_i' W_j'/chi^2
  (`stochastic_templates(order=2)`; order 0 reproduces SN0 bitwise;
  linear to 3e-14). Adjacent thin bins share an edge where their kernel
  derivatives have opposite signs: the normalized neighbour term has a
  median of -0.50 over the 73 adjacent pairs on 62 Mpc, against +0.05
  for SN0, which no SN0 can mimic.
- **C-FASTPT against Python FAST-PT** (b3 = 0): delta chi2 <= 3.5e-5 on
  the 62 Mpc vector, pairs included.

## The covariance

Gaussian, built at 14,000 deg^2 (the forecasts rescale its inverse to
12,300 deg^2) on the one-loop fiducial: b_{1,i} = B_g/D(zbar_i) with D
computed as the likelihood does, b2 and bs2 on the calibrations, every
other term zero. One covariance therefore serves models A and B. The
one-loop terms come from cosmolike's own C-FASTPT tables, through the
data vector's bilinear pair formula, in Limber; the low-ell non-Limber
recompute is linear (an error of at most 1.3% of C_ell in the lowest
bands of thin bins). `covariance/default.yaml` runs
`nonlimber_accuracyboost: 2`, which converges the adjacent thin-bin
cross spectra to 2% and the diagonal to 0.4%. The covariance takes
massless neutrinos, so its signal sits 3% above the data vector
(m_nu = 0.06 eV) at every redshift; that flat 1.03 ratio is the standing
check of its bias. Against a linear-bias covariance the model A FoMs at
k_max = 0.075 move by -0.4% to +1.2%; on the shipped mask, where the
loops are larger, the linear-bias FoMs rise by 3-10%. The LSST Y1
3x2pt case stays linear on its own lens sample. Every case ships the
Gaussian term only (G + SSC is the plan for chains). Building the files
and the evidence: `references/covariance.md`.

## The Fisher machinery and the notebooks

`interface/cosmolike_lssty1xdesi_fisher.py` (`ModelFisher`) builds
Fisher matrices on the cobaya model of an example yaml. Its rules, each
from a failure it prevents (`references/fisher_and_scale_cuts.md`,
Sec. 2): evaluate with `Model.loglike` and `cached=False`; read the
likelihood's `last_datavector` (the interface vector lacks the
stochastic templates); one active model per process; cache derivatives,
never Fisher matrices; priors from the yamls, never from a cached
Fisher matrix; invert through `stable_inverse`; wrap the setters the
likelihood calls instead of calling them directly.

**Derivative accuracy has two tiers.** Every derivative is a five-point
stencil, or an exact template for a term linear in its amplitude (SN0,
SN2, alpha_2); central differences are never used, because their noise
enters the Fisher matrix at full strength. Exploration runs may use the
fast settings the exploration notebooks used. Every figure and number of
a paper, and of a note's final version, uses the final tier: the
settings converged by the derivative convergence study in
`tests/studies/`, above the MCMC settings (which only need
|Delta chi2| < 0.2) and including CAMB's accuracy and k sampling. The
criterion is the stability of the FoMs, the marginalized errors and the
bias shifts under further refinement of the stencil step, the
non-Limber chi grid, cosmolike's accuracy boosts and CAMB's settings.
Runtime is not a criterion: a final recompute may take many hours.
Noisy derivative columns add spurious information and bias the FoM high.
At the MCMC settings the paper model's FoM(w0, wa) on Delta z = 0.05
moved by 57% at k_max = 0.075 and 10% at 0.2, although the fiducial
vector moved by only Delta chi2 = 0.015. Each cache records its tier and
settings in its fingerprint. DerivKit (Cocoa's `DERIVKIT_*` keys;
`get_ddv_dkit` in `cosmolike_notebook_utils/fisher.py`, as used in
roman_real's EXAMPLE_EVALUATE1.ipynb) is the cross-check of the stencil.

The scale cut is the non-Limber UV-fraction criterion (arXiv:2603.13031):
bin i is cut at k_max/D(zbar_i), a band kept while
|C_UV|/(|C_UV| + |C_IR|) < 0.05, intersected with the shipped mask
(k = 0.1185/Mpc = 0.176 h/Mpc in every bin, no growth scaling). k_max is
always the z = 0 value. The cut is built on the linear-bias spectra.

| notebook | does | caches in `chains/` (gitignored) | writes |
|---|---|---|---|
| EXAMPLE_EVALUATE_FISHER.ipynb | linear-bias Fisher, engine Fisher biases, amplitude cut against the contours, the UV-fraction cuts, the linear-bias ladder and accuracy test | `EXAMPLE_EVALUATE<n>_FISHER_AB<b>.pkl`, `..._BIN_BIAS.pkl`, `..._SCALE_CUTS.pkl`, `..._FOM_SCALE_CUTS.pkl`, `..._ALTERNATIVES.pkl` | no figure folder |
| EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb (builder) | linear-bias fits to model A truths against k_max; the accuracy test for Omega_m and A_s | `..._LINEAR_KMAX_TRUTHS.pkl` | `notes/linear_bias_kmax/plots/` |
| EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb (builder) | model A and model A + tilts at k_max = 0.075: the ladder, the accuracy test, the drifts | `..._MODEL_A_DERIVATIVES_<n>.pkl`, `..._MODEL_A_SN2_<n>.pkl` | v2 plots: baseline_ladder, ladder_bias_scatter, goodness_of_fit, drift_shifts, calibration_drift |
| EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb (builder) | the alpha_2 and SN2 templates, models A and B against k_max to 0.2, where each model breaks | `..._MODEL_B_TEMPLATES_<n>.pkl` | v2 plot: fom_kmax_models (one 2x3 grid) |

The cached names start with `EXAMPLE_EVALUATE_FISHER` where abbreviated.
Derivative caches do not depend on the covariance; every notebook
rebuilds its Fisher matrices from them and the current covariance file
at every run. The cells, self-checks and the fast linear-response
algebra are in `references/bias_notes_workflow.md`, Sec. 2.

## Results

Main values; full tables, checks and the data.json keys are in
`references/results.md`. Layout order: 62 Mpc / Delta z = 0.02 / 0.05 /
0.2; DESI C_ell^gg alone; k_max in h/Mpc at z = 0; the reference is LSST
Y1 3x2pt alone, FoM(w0, wa) = 17.26 and FoM(Omega_m, 10^9 A_s) = 214.9
(the lsst_y1 project's SRD Fisher forecast).

- **Linear bias holds only at k_max = 0.05**, and for every prior and
  truth only on 62 Mpc, keeping 1-6% of the FoM; at 0.075 every prior
  fails on every layout (62 Mpc: Omega_m 1.50-1.79 sigma). Source:
  EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb, section 4;
  `notes/linear_bias_kmax/plots/fob_kmax/data.json`.
- **Model A at k_max = 0.075**, FoM(w0, wa): model A on the baseline
  prior 4.33 / 4.00 / 0.522 / 0.0884; model A + tilts on the
  pessimistic prior 2.59 / 2.70 / 0.268 / 0.0536. FoM(Omega_m,
  10^9 A_s): 10.1 / 8.57 / 3.10 / 0.716 and 8.09 / 6.58 / 1.77 / 0.457.
  Source: EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb, section 6;
  `notes/correlated_bias_prior_v2/plots/baseline_ladder/data.json`.
- **The accuracy test** (2000 random bias histories per scatter family
  under model A; eta = sqrt(<chi^2>/n_dof)): growth-tied fails (eta of
  (w0, wa) 3.4-6.6 under model A); baseline is honest under its own
  scatter (1.00-1.02) but 1.24-1.70 under pessimistic scatter, where its
  Omega_m bar covers the truth in 40-63% of universes; pessimistic is
  honest under both (eta <= 1.05); free is exactly 1. A linear drift of
  the calibration offsets moves model A by 0.40-0.41 sigma in (w0, wa)
  on the wide layouts; model A + tilts absorbs it exactly. Source: model
  A notebook, section 7; `goodness_of_fit/data.json`,
  `calibration_drift/data.json`.
- **Where each model breaks.** Four fixed truths, each at one prior
  width: (a) alpha_2 = +-12.5 h^-2 Mpc^2 (model A only; model B fits
  it), (b) SN2 = +-5, (c) the offsets at +-1 sigma of the variant's own
  widths, (d) the calibration drift (Delta_2 linear across each group,
  changing by 1, Delta_s by 0.5, zero mean). A model holds at a cut
  while every shift stays below sigma/3 there and at every smaller
  fitted cut.
  - Baseline scenario (the baseline prior with model A and model B):
    both break at 0.075 on 62 Mpc and Delta z = 0.02 and at 0.05 on
    Delta z = 0.05 and 0.2, through the calibration drift in every fit
    but model A on Delta z = 0.02 (alpha_2: (w0, wa) 1.03 sigma at 0.1).
    At 0.1 the drift moves (w0, wa) by 0.92 sigma (model A) and 0.68
    (model B) on 62 Mpc; from 0.125 on it sets the largest shift
    everywhere, up to 2.75 sigma.
  - Pessimistic scenario (the pessimistic prior with model A + tilts and
    model B + tilts, whose tilts absorb the drift exactly): model A +
    tilts holds to 0.075 on the thin layouts and 0.15 on Delta z = 0.05,
    each time ended by alpha_2 (A_s 0.40 sigma at 0.1 on 62 Mpc,
    (w0, wa) 1.10 at 0.1 on Delta z = 0.02, A_s 0.53 at 0.175 on
    Delta z = 0.05); model B + tilts holds to 0.075 on those three
    layouts, ended by its own offsets (Omega_m 0.35-0.43 sigma at 0.1),
    and beyond 0.1 its largest shift stays at 0.25-0.34 sigma, near the
    threshold. Both pass every cut to 0.2 on Delta z = 0.2.
  - SN2 sets no break point (at most 0.30 sigma at any cut).
  Source: EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb, section 7;
  `fom_kmax_models/data.json` (`shift`, `worst`, `break`).
- **Models A and B to k_max = 0.2**, FoM(w0, wa): model B + tilts on the
  pessimistic prior 1.29 / 1.34 / 0.227 / 0.0495 at 0.075 and 10.9 /
  13.9 / 2.64 / 0.289 at 0.2; model B on the baseline prior 17.5 / 24.6
  / 3.33 / 0.458 at 0.2, above LSST Y1 on the thin layouts
  (Delta z = 0.02 from 0.15) but beyond that scenario's break point.
  FoM(Omega_m, 10^9 A_s) of model B stays below 214.9 (at most 116).
  Source: model B notebook, section 6; `fom_kmax_models/data.json`.
- **Coverage**: one parameter with eta = s and its drift absorbed covers
  the truth in a fraction 2 Phi(1/s) - 1 of universes (s = 1.3 -> 56%,
  1.75 -> 43%, 2.5 -> 31%); exact to 0.1% for the baseline, pessimistic
  and free priors, an upper bound for growth-tied.

## Decisions and their reasons

These are settled; reopen one only with the PI.

**The bias model**

- **No mocks; a theory paper.** No single prior is defensible without
  mock catalogs, so the forecast reports a reasonable and a pessimistic
  configuration between the two absolute limits (growth-tied, free); a
  referee who rejects the baseline moves one step down and nothing is
  retracted.
- **Exactly four linear-bias priors**: growth-tied, baseline,
  pessimistic, free. There is no growth-tied-plus-tilt prior.
- **The accuracy test before any FoM.** The truth is a known drift plus
  a random residual, 2000 draws per family; the scatter figure uses the
  baseline draws (top) and the pessimistic draws (bottom). A prior or
  model that fails the test (growth-tied) gets no FoM in any figure. The
  test always scores Omega_m and A_s next to (w0, wa).
- **Model A at k_max = 0.075, model B scanned in k_max**, both one-loop.
  b2 and bs2 inherit the full b_{1,i}, plus a group offset (model A,
  model B) or an offset and a tilt (model A + tilts, model B + tilts).
  Only b2, bs2, alpha_0 and alpha_2 tilt (they follow the host-halo
  population); SN0 and SN2 never do. Free b2 and bs2 per bin (152
  parameters) is a bound, not a model.
- **SN2 is in model A.** Without it, an SN2 truth at one prior width
  moved Omega_m by 1.3-1.7 sigma at 0.075 through the anticorrelated
  adjacent cross spectra, and model A broke before its own cut on the
  thin layouts; no white noise, per group or per bin, repairs that, and
  dropping the cross pairs keeps only 18-46% of the thin-layout FoM.
  With SN2, model A holds to 0.075 at a cost of 4-8% of FoM(w0, wa)
  (`references/fom_and_bias_review.md`, Sec. 6).
- **Model A fixes alpha_2 = 0**: at DESI's width it changes the kept
  autos by about 1%, and a fit without it holds to 0.075 against
  alpha_2 = +-12.5.
- **Calibration-informed offset priors** (Delta_2 width 1, Delta_s 0.5
  for model A) instead of DESI DR1's agnostic N(0, 5^2) on b2 sigma8^2
  and bs sigma8^2: at 0.075 the data barely measure b2, and for the
  dense BGS a low b2 drives P_gg negative. Fiducials come from the
  calibrations, never from DESI chain values.
- **No FoM rests on exact calibrations.** The linear-bias and
  exact-calibration FoMs at the same cut are diagnostics in the
  notebooks' printed tables and data.json, never on a figure.
- **One likelihood option**, models A and B as parameter files; the
  former `higher_order_bias: 2` is gone.
- **The FAST-PT reference comparison** runs through TATT + IA_code 1 +
  the fastpt theory block; no reset knob was added for it.
- **The DESI bias boxes [0.8, 4] stay**; the N-models figures argue for
  narrowing them with collaborators. The amplitude cut is 8, the common
  cut of every cosmolike project.

**The covariance**

- **One covariance on the one-loop fiducial**, with the one-loop Gaussian
  term computed in C (the clean option, chosen over rescaling the
  linear signal by the model A to linear ratio). Models A and B share
  the fiducial, so they share the covariance.

**Figures, scenarios and notes**

- **k_max <= 0.2 h/Mpc** on every figure axis and in every quoted FoM,
  the PI's rule (0.2 h/Mpc is also the reach of DESI DR1's one-loop
  full-shape fits). The model B scan stops there (`KMAX_MAX`); the
  scale-cut cache's larger cuts are not used.
- **Two scenarios, and the headline is the PI's call.** The baseline
  scenario is the baseline prior with model A and model B; the
  pessimistic scenario is the pessimistic prior with model A + tilts and
  model B + tilts. The paper figure `fom_kmax_models` is one 2x3 grid:
  a row per scenario, columns FoM(w0, wa), FoM(Omega_m, 10^9 A_s) and
  the largest shift (model A drawn up to its break in the FoM columns).
  Which scenario is the paper's headline is decided by the PI after
  seeing both side by side; until then both carry equal weight, and
  neither is labeled headline, conservative or defensible.
- **Four notes, the summary last**: correlated_bias_prior_v2 (with
  model B; no v3), linear_bias_kmax, higher_order_bias, and
  bias_model_summary, rewritten after the other three as a presentable
  summary with their key figures (not compact). Every paper figure has
  a plot folder. In correlated_bias_prior_v2 the accuracy test precedes
  "Model B and the reach of model A", and the mathematics appendix
  places the accuracy-test method after "How the prior enters the
  Fisher matrix"; in bias_model_summary the model and the prior precede
  the section "Projection, covariance and the Fisher forecast", which
  precedes the results.
- **Notes describe what was done**; recommendations and decisions live
  in this skill (the higher-order note's original Recommendation is in
  `references/higher_order_bias_recommendation.md`); open items go to
  each note's "Open issues" appendix (`references/bias_notes_workflow.md`,
  Sec. 1).
- **The paper must explain didactically** the white-noise constant and
  its two conventions (CosmoLike subtracts it; DESI's velocileptors keeps
  it in SN0), why DESI never sees a negative P_gg and what the exact
  split does, and the two remedies for high k_max (a loop taper, a
  positivity prior for chains).

**Documentation rules** (notes, notebook markdown, READMEs, code
comments, captions):

- Define every symbol where it is first used; write A_s and b_i, never
  a bare "As" (A_s the primordial amplitude, sampled as As_1e9 =
  10^9 A_s; b_i a bin's linear bias, DESI_B1_<n>/D(zbar_n) here).
- Never name a method by another paper's equation number: give a
  descriptive name, the small equation, and the arXiv ID once (for
  example, the non-Limber UV-fraction scale cut, arXiv:2603.13031).
- Names: the priors are growth-tied / baseline / pessimistic / free,
  never "rung" (code keeps `'tied'`, `'independent'`, `LADDER_RUNGS`,
  `rung_columns`). The one-loop variants are "model A" and "model A +
  tilts" (likewise model B), where "+ tilts" means Delta_2, Delta_s and
  alpha_0 (and alpha_2 in model B) get a free straight line in redshift
  per group, like the linear bias's s_g; the parameter files are
  `params_desi[_<case>]_model_{a,b}.yaml` and `..._model_{a,b}_tilts.yaml`,
  and the data keys `'A'`, `'A tilts'`, `'B'`, `'B tilts'`. "Baseline"
  and "pessimistic" name only linear-bias priors and the scenarios
  built on them, never a model variant. A sample group is a "group",
  never "chunk". The layouts are Delta d_com = 62 Mpc (never "DR3",
  which reads as DESI's DR3 P(k), nor
  "Delta chi", which clashes with Delta chi^2) and Delta z = ...; no
  "passive evolution", no "6x2pt"; "DR1" never labels the density,
  which is the completed survey (~28M tracers, not DR1's ~6M).
- The statistic sqrt(<chi^2>/n_dof) has no name (not "goodness of fit",
  not "error inflation"); its value is eta. Explain it once with a
  statement, never a question: "How well each fit's quoted error bar
  covers where the fit actually lands", with chi^2 = Delta p^T C_p^-1
  Delta p (recovered minus true, the fit's marginalized covariance;
  n_dof = 2 for (w0, wa), 1 for Omega_m or A_s; 1 = honest error bars).
  The code key stays `'error_inflation'` and the folder
  `plots/goodness_of_fit`.
- An accuracy-setting comment opens with its purpose in plain words,
  then the evidence.

## Pitfalls and how they were caught

- **carma zero-fills long lists.** A plain Python list of more than 16
  entries crosses the pybind/carma boundary as zeros (Armadillo's
  inline preallocation; 5-bin donors never see it). It struck the
  likelihood's nuisance arrays and the covariance forecast's per-bin
  bias (caught when the Gaussian diagonal equaled 2 N^2/n_modes
  exactly). Every per-bin array crossing the compiled interface is
  `np.asarray(..., dtype=float)`. When a bias reads zero in C, check
  list lengths first.
- **Thin-bin aliasing of the chi grid.** At the default NL_Nchi = 512 a
  thin kernel gets one or two samples; any tiny change of
  chi(z_bin)/chi(z = 4) swings C_ell by tens of percent (boost 1 against
  16: delta chi2 = 154,439). Caught through the hybrid emulator's
  chi2 = 54,951.6; a chi2 = 0 self-consistency test cannot see it, so
  convergence tests must vary the grid.
- **The covariance's own non-Limber grid.** At
  nonlimber_accuracyboost 1 the covariance's adjacent cross spectra
  were off by up to 15-30% and its diagonal by 10%; it runs 2.
- **The z = 0 bias in the covariance.** The covariance once gave every
  bin its group's z = 0 amplitude, inflating every DESI FoM 2.5-3.4x.
  Caught by a counterfactual covariance built on the data vector; the
  flat 1.03 signal ratio is the check (`references/covariance.md`,
  Sec. 3).
- **Stale cached Fisher matrices.** The linear caches store F built with
  the old covariance (rebuild check 0.76 for DESI); an earlier notebook
  differenced it, diag(F_cached - F_data), to get the priors, which
  produced a fake prior once the covariance changed. Priors come from
  the yamls; F is always rebuilt.
- **Singular Fisher from dead columns.** A bias whose bins keep no band
  leaves a column of derivative round-off (it once made a 62 Mpc FoM
  infinite at k_max = 0.05), and flat BGS tilts with only 7 of 18 BGS
  bins kept made a null direction that `np.linalg.inv` turned into a
  spurious 2 sigma shift. `stable_inverse` drops sub-1e-12 directions
  and prints their component on the scored parameters.
- **The likelihood resets alpha_2 at every point** (zeros under linear
  bias and model A), and every other nuisance likewise, so a value left
  by another likelihood cannot survive, and a derivative step written
  into the compiled setter is overwritten. The notebooks wrap the
  setter, add the step, restore it, and check the vector returns
  bitwise.
- **Two models in one process.** The gg cross-pair list is a static of
  cosmo2D.c; before `initial_setup` cleared it, a model built after a
  DESI one inherited 13 pairs and died in `gbmag`. Any new static setter
  needs the same reset. The interface holds one dataset, so build each
  probe's derivatives right after its model.
- **`logposterior` and caching in Fisher steps** return a stale vector
  outside a flat prior or at a cached point; use `Model.loglike` and
  `cached=False`.
- **The dual-space interface.** `compute_data_vector_masked` keeps the
  donor's real-space layout and indexes out of bounds on the Fourier
  dataset; the likelihood scores
  `compute_data_vector_fourier_gg_cross_masked` ([EE | gs | gg |
  gg-cross]; the plain Fourier getter leaves the cross block zero).
  Tests declare `BINNING_SPACE = "fourier"` in
  `tests/cocoa_test_utils.py`, and the shared suites resolve the getter
  at call time because the pair list is installed only by the first
  model build.
- **The gg-cross stash contract.** Both engines (low-rank and the FKEM
  twin) compute the cross corrections in the auto-gg pass and stash
  them for the same evaluation (`ggx_stash_store`, consumed once). The
  correction applies only when the autos are non-Limber; with
  `adopt_limber_gg: 1` the crosses are Limber too and the stash guard
  must not fire.
- **`lmax_nonlimber` is the FKEM cap** (`limits.LMAX_NOLIMBER`, set by
  `init_lmax_nonlimber` after `init_ntable_lmax`); it does nothing with
  the low-rank engine on. No early exit exists on the Fourier/cross path
  (the engine refuses cross pairs with tol > 0).
- **Other conventions**: `ggl_exclude` is the string "all" here, not a
  pair list (`_ggl_excluded`); this project carries `band_ells`, not
  `ncl` (`_fourier_nlen`); `data/*cov` is gitignored; the M1-M6
  mask/dataset files under data/ are donor leftovers.
- **The frozen test covariances** under `tests/frozen/data/` predate the
  per-bin covariance bias (the dr3 copy is the sacc covariance), so every
  pinned delta chi2 scores that state, not the shipped `data/*_cov`.
- **Bias code 1 under the one-loop model** silently replaces the
  likelihood's per-bin b3 (b1 - 1 on the wrong `tab_d1d3` table) or bs2;
  option 1 refuses any `bias_model` other than all zeros.
- **`internal_accuracyboost: 4`** breaks the C-FASTPT bias tables
  (delta chi2 ~ 1e12); keep 1 or 2.
- **Timings on battery or a busy machine** are not usable (Workflow).
- **The session scratchpad is cleaned**: builders, diagnostics and
  scratch kernelspecs that matter go to the persistent builder folder.

## Test suite contract

`tests/` follows the shared machinery (guide:
`tests/data_vector/README.md`). The frozen fiducial reproduces chi2 = 0
exactly; the emulator configuration is advisory (4.44 against exact
physics at the adopted state: the thin bins resolve emulator error that
photometric surveys blur). After any refreeze, re-measure and re-pin the
non-Limber drift values (`DCHI2_MEASURED` of example1 and the
`DCHI2_MEASURED_CASES` of examples 2-4, in both engine test files) and
re-run the accuracy scan, whose project-specific knobs are the chi-grid
boost (with a 64 -> 128 convergence row), the SVD rank and the engine
cap. The N-random-models check fails its 0.2 budget at the extreme prior
corners by a documented judgment (`references/numerical_settings.md`,
Sec. 6). No source nuisance parameters are declared yet (IA amplitudes,
shear m, source dz), so the TATT variants, the FASTPT comparison and the
source-sector cache rungs skip. README numbers that predate a model
change are replaced the moment the frozen state moves, never appended.

## The bias notes are the physics reference

For any physics question (a derivation, an equation, an assumption, a
citation, a literature value) read the `.tex` source, not the PDF and
not this skill; find the part with `grep -n '\\section\|\\subsection'
<note>.tex` (line numbers drift). Cite a note's section by its title,
never by an appendix letter. When the skill and a note disagree on
physics, the note is right and the skill is corrected.

| question | note: section |
|---|---|
| the forecast on a page | correlated_bias_prior_v2: Summary |
| the setup, the scale cut, the covariance | correlated_bias_prior_v2: The forecast |
| the linear-bias prior; physical L and sigma_g | correlated_bias_prior_v2: The linear-bias prior; appendices The models, Which L and sigma_g are physical (subsection What the measurements say) |
| model A; model B and where model A breaks | correlated_bias_prior_v2: Model A; Model B and the reach of model A |
| random truths, the statistic, coverage; the calibration drift | correlated_bias_prior_v2: The accuracy test; appendix The accuracy test |
| the FoM at 0.075 | correlated_bias_prior_v2: The figure of merit at k_max = 0.075 |
| memory model; projection algebra | correlated_bias_prior_v2: The memory model; The projected precision for two bins; Many bins: the tridiagonal inverse; The memory projection, step by step |
| b - 1 proportional to 1/D; the BGS flux limit | correlated_bias_prior_v2: The bias continuity equation, step by step; The BGS flux-limit bias model, step by step |
| derivatives, Fisher construction, checks, the counterterm engine | correlated_bias_prior_v2: Implementation and the Fisher construction; Numerical checks; The counterterm requires the low-rank engine |
| linear bias against k_max | linear_bias_kmax: Summary; Method |
| one-loop P_gg, bases and conversions, DESI DR1 priors | higher_order_bias: The one-loop galaxy power spectrum; The DESI DR1 full-shape model and priors; appendix Bases and conversions, with every factor |
| the draft paper's bias table | higher_order_bias: The draft paper's bias table: the tidal-bias verdict |
| term sizes | higher_order_bias: The size of every term |
| the implemented model, its widths, the covariance | higher_order_bias: The model as implemented |
| projections, stochastic terms, the white-noise constant | higher_order_bias: Projection to thin-bin angular spectra (subsections Stochastic terms in projection; The white-noise limit of the renormalized loops) |
| validation of the pieces | higher_order_bias: Validation |
| open issues | each note: appendix Open issues |

## Open issues and open work

Open issues of the physics and numerics (each note's "Open issues"
appendix carries the relevant ones):

- **The covariance's one-loop terms are Limber**; its low-ell non-Limber
  recompute is linear (at most 1.3% of C_ell in the lowest thin-bin
  bands).
- **The scale cut was built on the linear model.** On Delta z = 0.05 at
  0.075 the alpha_0 template's UV fraction is a median 0.076 (max 0.14)
  on the kept bands, above alpha = 0.05 (0.017 and 0.052 for the full
  model A vector), and about half of each bin's SN2 kernel comes from
  k_par above its own cut. Rebuilding the cut on the one-loop model is
  undecided.
- **Positivity far from the fiducial.** Negative alpha_0 or b2 a unit
  below its calibration makes the low-rank engine floor P_gg - P_inf
  (alpha_0 = -88.9: above k ~ 0.14 h/Mpc, 47% of the grid; the vector
  then curves in alpha_0), and on the dense BGS at model B's cuts
  P_gg turns negative near 0.15-0.2 h/Mpc. Harmless for the Fisher
  forecast at the fiducial; chains need a loop taper above the trusted
  scale (near 0.15 for BGS) or a positivity prior on P_gg. Neither is
  implemented; a tighter b2 prior or SN0 does not fix the sign.
- **The C-FASTPT b3 template** (`tab_d1d3`) is cosmology-independent and
  wrong, so b3 = 0; `internal_accuracyboost: 4` breaks the bias tables.
  Both are core bugs.
- **No redshift-space one-loop terms**: model B relies on the sampled
  sigma_v and on alpha_2.
- **Adjacent cross spectra see the shared edge**: any small-scale radial
  physics at a common edge reaches the cross spectra the way SN2 does.
- **N-random-models corners**: delta chi2 of order 10 and above at the
  most extreme prior corners (ratio >= 5 or Omega_m at the box edge with
  large neutrino masses); raising accuracyboost on the thin layouts is
  the available lever and is not decided.
- **The joint LSST + DESI lens sample** for chains (and the Z_LOW / Z_MID
  convention mix it brings) is not built; Fisher matrices are summed.
- **G + SSC covariance files** for the final chains are planned, not made
  (`references/covariance.md`, Sec. 6).

Open work:

- **Finish and commit the scenario rework** (in the working tree,
  uncommitted, as of 2026-11-08): the renamed variants (parameter files
  `*_pessimistic.yaml` -> `*_tilts.yaml`, data keys `'A'` / `'A tilts'`),
  the model B notebook's four-truth break test of both models with its
  scan stopped at 0.2, the 2x3 `fom_kmax_models` grid with
  `model_a_break` removed, and the reruns of the four Fisher notebooks
  on the current covariance. Then correlated_bias_prior_v2 (its Summary,
  "Model B and the reach of model A", the figure captions) and the other
  notes are brought to them: the committed text still names model A
  baseline/pessimistic, draws two FoM figures and model_a_break, runs the
  break test with three truths on model A alone, says model B
  pessimistic holds at every cut and quotes FoMs at 0.25.
- **Rewrite bias_model_summary** last, from the other three notes.
- **EXAMPLE_EVALUATE_FISHER.ipynb's markdown** still quotes the
  linear-bias-covariance values (41.2 / 50.2 / 12.6 / 2.22 on the
  shipped mask; the rerun prints 43.4 / 51.8 / 13.2 / 2.43) and predates
  the naming rules ("independent" and "tied" for the free and
  growth-tied priors, a "defensible" k_max window).
- **Stale text outside the skill** (through the docs subagent): the
  project README's Fisher section still quotes the pre-fix FoMs
  (135 / 126 / 32.0 / 5.67) and says the covariance fix "will be made";
  its amplitude-cut contour ratios (1.33-2.11) predate the corrected
  covariance (now 1.5-3.1). `likelihood/desi_cl.py`'s module docstring
  still says cross-bin spectra have no slot, cites "the paper's eq. 20",
  says "passive evolution", and points to the higher-order note's
  removed recommendation section, as do `scripts/make_desi_params.py`
  (model A and B blocks) and the code comments of the model A and model
  B builders. `covariance/lssty1xdesi_covariance.py`'s docstring still
  calls the 62 Mpc likelihood covariance the sacc import. The model A
  notebook's export markdown names the old layout colors.
- **The comparison with the sacc covariance** is re-measured after the
  per-bin bias and the one-loop term.
- **Optional**: a nonlinear spot-check of the linear-response shifts (a
  few data-vector evaluations at drawn truths); an AB 5 run to confirm
  the AB 3 Fisher convergence; a rerun of the n(z) reconstruction study.
- **Declare the source nuisances** (IA, shear m, source dz) and port
  their tests.
- **After lssty1xdesi is finished**, roll its decisions out to the other
  projects: the cosmological prior boxes above, the amplitude cut at 8,
  the `omegam_cap` top-hat in CMB-parameterized examples, the N-models
  check constants with a `--nmodels` generation, the wrappers' mirroring
  of the likelihood's projection options, and a Fisher figure showing
  each cut far outside the 2 sigma contours (lsst_y1 is done).
