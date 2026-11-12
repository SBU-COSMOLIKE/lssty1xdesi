---
name: lssty1xdesi-dev
description: Development on the lssty1xdesi project - DESI thin-bin spectroscopic angular clustering C_ell^gg (non-Limber band powers) on four bin layouts (Delta d_com = 62 Mpc, 76 bins of equal comoving width; fixed-width Delta z = 0.02, 0.05, 0.2), with LSST Y1 cosmic shear and an LSST Y1 3x2pt reference case. Use this skill for ANY work in projects/lssty1xdesi - the likelihood desi_cl (linear bias, and the one-loop option of the fiducial model), the Fingers-of-God model and sigma_v, the n(z) files, the low-rank non-Limber engine, the gg-cross block, the covariance adapter, the test suite, the data scripts, the Fisher module and the Fisher notebooks (EXAMPLE_EVALUATE_FISHER.ipynb, _LINEAR_KMAX, _MODEL) with their builders and caches, the MPI script of the derivative base, the pair-axes script, and the galaxy-bias notes under notes/ (correlated_bias_prior_v2, linear_bias_kmax, higher_order_bias, bias_model_summary) with their plots/ folders. Also trigger on mentions of DESI thin bins, nonlimber_accuracyboost, NL_Nchi, DESI_SIGV, DESI_DB2/DBS/A0/A2/SN0/SN2, gg cross pairs, the dndz FoG convolution, the correlated bias prior and its priors (growth-tied, baseline, pessimistic, free), the memory model, the statistic sqrt(<chi^2>/n_dof), the fiducial one-loop model and its reference variant with alpha_2 free, SN_2, the UV-fraction scale cut, the one-loop covariance, or the amplitude cut. This survey is the numerical extreme of the cosmolike projects, and defaults that are safe elsewhere fail here.
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
| EXAMPLE_EVALUATE1 | Delta d_com = 62 Mpc | `dr3` |
| EXAMPLE_EVALUATE2 | Delta z = 0.02 | `dr1_dz0p02` |
| EXAMPLE_EVALUATE3 | Delta z = 0.05 | `dr1_dz0p05` |
| EXAMPLE_EVALUATE4 | Delta z = 0.2 | `dr1_dz0p20` |
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
- **The notebook builders.** EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb
  and EXAMPLE_EVALUATE_FISHER_MODEL.ipynb are generated by builder
  scripts that are in no repository. Their persistent copy is
  `/Users/vivianmiranda/Library/Application Support/Claude/side-session-notes/local_6b327622-bba3-4f4a-bf11-769622bccf03/notebook_builders/`
  (build_linear_kmax_notebook.py, build_model_notebook.py, validate_nb.py,
  the run_*_nb.sh nbconvert wrappers, plot_scripts/ and a README; the
  builders of the removed model A notebook and of the model B notebook,
  build_model_a_notebook.py and build_model_b_notebook.py, stay there as
  records); session scratchpads are
  cleaned, so a builder edited in a scratchpad is copied back there.
  Edit markdown through the `md(...)` blocks and code through the
  `code(...)` blocks, never the generated notebook. Whether the
  builders belong in the project repository is undecided: ask the PI
  before adding them. The run, validation and copy-back recipe is
  `references/bias_notes_workflow.md`, Sec. 4.
- **Notebooks are committed executed**, with outputs and figures, so
  the plots render on GitHub; out-of-date outputs from older code are wrong.
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

### One likelihood option; the model naming

`likelihood/desi_cl.py` option `higher_order_bias`: 0 is linear bias;
1 is the one-loop model with every term per bin. **The fiducial
model**, the model the paper forecasts, is a parameter file of
option 1 (`scripts/make_desi_params.py --model-b`, a flag that keeps
its historical name): per-group offsets Delta_2, Delta_s and alpha_0
at the DESI DR1 widths, each with a free straight line in redshift per
group (the tilts, like the linear bias's s_g, flat priors), the
stochastic amplitudes SN0 and SN2 without tilts, alpha_2 fixed to 0 in
every bin, and every sigma_v prior width multiplied by 6. **The
reference model** of the companion notes is the fiducial model with
alpha_2 free (width 0.5/k_max^2 per cut) at the exploration sigma_v
widths; the k_max scans also fit it without its tilts. The historical
names stay in the code and say what they mean once: data keys 'B' (the
reference model without tilts), 'B tilts' (the reference model) and
'B tilts noA2 sv x6' (the fiducial model); likelihood classes
`desi_cl*_model_b`; parameter files `*_model_b_tilts.yaml`. "Model A"
(a tighter one-loop variant for k_max = 0.075) and "model B" were the
development names; model A is retired, and its examples, classes and
parameter files are deleted from the repository. Either model runs
under any of the four linear-bias priors.

The real-space power of bin i is b_{1,i}^2 P_nl + P_loop(b_{1,i},
b_{2,i}, b_{s^2,i}) + alpha_{0,i} k^2 P_nl, with
b_{2,i} = b2_Lazeyras(b_{1,i}) + Delta_{2,g} (arXiv:1511.01096) and
b_{s^2,i} = -(4/7)(b_{1,i} - 1) + Delta_{s,g} (co-evolution), b3 = 0.
Redshift space is linear Kaiser on each bin's FoG-convolved n(z), plus
the counterterm alpha_2 and the stochastic terms:

| per group | the fiducial model | the reference model |
|---|---|---|
| offsets Delta_2 / Delta_s | width 2.0 / 1.0 | the same |
| alpha_0 [h^-2 Mpc^2] | 0.5/k_max^2 (16.1 in the yamls, the shipped mask's 0.176 h/Mpc; per cut in the Fisher scan) | the same |
| alpha_2 [h^-2 Mpc^2] | fixed to 0 | 0.5/k_max^2 |
| SN0 [P_Pois,i = 1/nbar_i] | width 2 | width 2 |
| SN2 [f_sat sigma_sat^2 P_Pois,i] | width 5 | width 5 |
| sigma_v prior widths | the measured widths x6 | the measured widths |
| tilted | Delta_2, Delta_s, alpha_0 | Delta_2, Delta_s, alpha_0, alpha_2 (none in 'B') |
| parameters | 8 (48 in all) | 10 (60); without tilts 6 (36) |

All Gaussian priors are centered on zero, so the fiducials sit on the
calibrations and coincide. SN0 and SN2 never tilt: their Poisson part
is known bin by bin.

Conventions that matter:

- **alpha_0** rides cosmolike's slot 2 b1 bK k^2 P_nl with k in units of
  H0/c, so bK = alpha_0/[2 b1 (c/H0)^2], c/H0 = 2997.92 Mpc/h. It
  multiplies P_nl, not DESI's (b1 + f mu^2) b1 P_lin. Its width is a
  correction of 0.5/B_g^2 of the linear galaxy power at each bin's own
  cut (BGS 28%, LRG 17%, ELG 71% at k_max = 0.075).
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
other term zero. One covariance therefore serves every one-loop
variant. The
one-loop terms come from cosmolike's own C-FASTPT tables, through the
data vector's bilinear pair formula, in Limber; the low-ell non-Limber
recompute is linear (an error of at most 1.3% of C_ell in the lowest
bands of thin bins). `covariance/default.yaml` runs
`nonlimber_accuracyboost: 2`, which converges the adjacent thin-bin
cross spectra to 2% and the diagonal to 0.4%. The covariance takes
massless neutrinos, so its signal sits 3% above the data vector
(m_nu = 0.06 eV) at every redshift; that flat 1.03 ratio is the standing
check of its bias. Against a linear-bias covariance the one-loop FoMs
at k_max = 0.075 move by -0.4% to +1.2%; on the shipped mask, where the
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
At the MCMC settings the fiducial model's FoM(w0, wa) on Delta z = 0.05
moved by 57% at k_max = 0.075 and 10% at 0.2, although the fiducial
vector moved by only Delta chi2 = 0.015. Each cache records its tier and
settings in its fingerprint. DerivKit (Cocoa's `DERIVKIT_*` keys;
`get_ddv_dkit` in `cosmolike_notebook_utils/fisher.py`, as used in
roman_real's EXAMPLE_EVALUATE1.ipynb) is the cross-check of the stencil.

The scale cut is the non-Limber UV-fraction criterion (arXiv:2603.13031):
bin i is cut at k_max/D(zbar_i), a band kept while
|C_UV|/(|C_UV| + |C_IR|) < 0.05, intersected with the shipped mask
(k = 0.1185/Mpc = 0.176 h/Mpc in every bin, no growth scaling). k_max is
always the z = 0 value. EXAMPLE_EVALUATE_FISHER_MODEL.ipynb builds the
cut on the paper fiducial (EXAMPLE_EVALUATE<n>.yaml at its evaluate
point, b2 and bs2 on their calibrations; run at the final derivative
tier, the settings of the derivative base); the linear notebook's cut, on the
linear-bias spectra, serves the linear-bias experiments. The two differ
by a few percent in FoM(w0, wa) on the main layouts, up to about 6% at
single points (`notes/scale_cut_fiducial_dependence`).

| notebook | does | caches (working copies in `chains/`, gitignored; final tier archived in `fisher_caches/`) | writes |
|---|---|---|---|
| EXAMPLE_EVALUATE_FISHER.ipynb | linear-bias Fisher, engine Fisher biases, amplitude cut against the contours, the UV-fraction cuts, the linear-bias ladder and accuracy test | `linear_fisher_<layout>_final.npz`, `linear_probes_dz0p20_lsst_ab3_final.npz`, `linear_bin_bias_final.npz`, `linear_scale_cuts_final.npz`, `linear_fom_scale_cuts_final.npz`, `linear_alternatives_final.npz` | no figure folder |
| EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb (builder) | linear-bias fits to the fiducial model's one-loop truths against k_max (truths built from the paper yamls EXAMPLE_EVALUATE<n>.yaml with the linear vector's own settings, which point each layout at its paper data set); the accuracy test for Omega_m and A_s | `linear_kmax_truths_final.npz` | `notes/linear_bias_kmax/plots/` |
| EXAMPLE_EVALUATE_FISHER_MODEL.ipynb (builder) | the scale cuts on the paper fiducial and the derivative base on the paper yamls (explicit copy, read, compute, write and archive cells), the reference model with and without tilts against k_max to 0.2 and where the variant without tilts breaks, the fiducial model with its error-bar test, untied, growth-tied and pair-threshold comparisons, and the alpha_2 / sigma_v study | `model_scale_cuts_final.npz`, `model_derivatives_<layout>_final.npz`, `model_templates_<layout>_final.npz` | v2: fom_kmax_models, growth_tied_kmax; summary: summary_fom_kmax, summary_fom_untied, summary_prior_draws, summary_pair_threshold; fog_parameter_degeneracy_study: fog_templates, fog_degeneracy_kmax, fog_sigv_width, fog_pair_overlap; cross_correlations: pair_threshold |

A cache name says what it holds, the layout (`<layout>` = `dcom62`,
`dz0p02`, `dz0p05`, `dz0p20`) and the tier. Caches are `.npz` files of
plain arrays with a JSON header, no pickles; the archive
(`fisher_caches/`, `.npz` through git-lfs) also keeps each cache's
provenance record as `fisher_caches/<name>.json` and the checksums in
`checksums.json` (guide: `fisher_caches/README.md`). Four module
functions move them, each called in its own notebook cell:
`load_fisher_from_cache` copies an archived final-tier cache to
`chains/` after the checksum test; `read_fisher_cache` checks a
working copy's settings, tier and fingerprints against the notebook's
and raises on a mismatch, never recomputing; `write_fisher_cache`
writes a freshly computed working copy; `save_fisher_to_cache`
archives a final-tier working copy, the only call that changes the
archive. Nothing checks for a file to decide what runs, and nothing
replaces a file without `overwrite=True`.
`scripts/compute_fisher_base_mpi.py` (with
`compute_fisher_base_mpi_classes.py`) writes the same derivative caches
on MPI ranks; one rank and the notebook evaluate in one order, and 1 x 8,
2 x 2 and 2 x 4 ranks x threads reproduce a serial run bit for bit.
`scripts/compute_pair_axes_fits.py` runs the pair-axes study of the
cross-correlation note on the archived
`model_{derivatives,templates,scale_cuts}_<layout>_pairmax_final`
caches (the derivative base, templates and final-tier scale cuts on
each layout's maximal pair list, `data/<stem>_pairmax_pairs.txt`) and
writes `chains/pair_axes_fits_<layout>.json`, which the
`make_pair_axes_*_data.py` generators of
`notes/cross_correlations/plots/` read; its docstring holds the usage
and the prerequisites (explicit `fm.load_fisher_from_cache` calls, the
`data/<stem>_pairmax_cov` covariance built with `compute_covariance.py`
on `EXAMPLE_EVALUATE_COVARIANCE<n>_PAIRMAX.yaml` then
`make_desi_covariance.py --case <case>_pairmax`, and
`covariance/forecast_camb.npz` for the 62 Mpc gap axis). The
base caches of the former model A and model B notebooks
(`..._MODEL_A_DERIVATIVES_<n>.pkl`, `..._MODEL_B_TEMPLATES_<n>.pkl`,
`..._MODEL_A_SN2_<n>.pkl`) stay in `chains/`: they hold the same arrays
as the new caches under leader-bin names (DESI_B1_<n> instead of
DESI_B1_<g>), and the convergence study of `tests/studies/` reads them.
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
  truth only on 62 Mpc, keeping about 1% of the FoM; at 0.075 every
  prior fails on every layout (62 Mpc: Omega_m 1.72-1.94 sigma for the
  calibrated truth). Source:
  EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb, section 4;
  `notes/linear_bias_kmax/plots/fob_kmax/data.json`.
- **Model A at k_max = 0.075** (a record: EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb
  and its five v2 figure folders, baseline_ladder, ladder_bias_scatter,
  goodness_of_fit, drift_shifts and calibration_drift, were removed on
  2026-11-10; their last final-tier run is commit 567751f, on the
  linear-bias cuts). FoM(w0, wa): model A on the baseline prior 1.09 /
  0.866 / 0.289 / 0.0884; model A + tilts on the pessimistic prior 1.01 /
  0.675 / 0.197 / 0.0538. The accuracy test (2000 random bias histories
  per scatter family; eta = sqrt(<chi^2>/n_dof)): growth-tied fails (eta
  of (w0, wa) 4.0-6.8 under model A, 1.5-2.4 under model A + tilts); the
  baseline prior is honest under its own scatter (1.00-1.01; 1.04-1.11
  on Delta z = 0.05) and up to 1.19 under pessimistic scatter; the
  pessimistic prior is honest under both (at most 1.08); free is exactly
  1. The calibration drift moves model A by 0.39-0.46 sigma on the wide
  layouts; model A + tilts absorbs it exactly.
- **Where the reference model without tilts breaks** (paper-fiducial
  cuts). Four fixed truths,
  each at one prior width: (a) alpha_2 at one reference-model width,
  0.5/k_max^2, (b) SN2 = +-5, (c) the offsets at +-1 sigma of the
  widths, (d) the calibration drift (Delta_2 linear across each group,
  changing by 1, Delta_s by 0.5, zero mean). (a)-(c) lie inside the
  prior and are reported; (d) is tested: a model holds at a cut
  while every tested shift stays below sigma/3 there and at every
  smaller fitted cut.
  - Baseline scenario (the baseline prior with the variant without
    tilts): it breaks at
    0.075 on 62 Mpc and Delta z = 0.02 (the drift moves (w0, wa) by
    0.82 and 0.86 sigma at 0.1) and at 0.05 on Delta z = 0.05 and 0.2
    (0.35 and 0.36 sigma at 0.075); from 0.1 on the largest tested shift
    is 0.53-2.79 sigma.
  - Pessimistic scenario (the pessimistic prior with the reference
    model, whose tilts absorb the drift exactly): passes every cut, a
    pass with
    no information on its own; its in-prior shifts reach 0.98 sigma at
    0.075 on 62 Mpc and 1.10 at 0.1 on Delta z = 0.02 (alpha_2), and
    0.33-0.73 sigma from 0.125 on on the three thinner layouts.
  - SN2 sets no break point (at most 0.26 sigma at any cut).
  Source: EXAMPLE_EVALUATE_FISHER_MODEL.ipynb, section 7;
  `fom_kmax_models/data.json` (`shift`, `worst`, `worst_in_prior`,
  `break`).
- **The reference model to k_max = 0.2**, FoM(w0, wa): with its tilts on
  the pessimistic prior 0.407 / 0.237 / 0.155 / 0.0426 at 0.075 and 7.53 /
  8.42 / 2.45 / 0.300 at 0.2; without tilts on the baseline prior 8.46 /
  9.91 / 2.77 / 0.478 at 0.2, below LSST Y1. Its FoM(Omega_m, 10^9 A_s)
  stays below 214.9 (at most 67.5). Source: EXAMPLE_EVALUATE_FISHER_MODEL.ipynb,
  section 6; `fom_kmax_models/data.json`.
- **The fiducial model** (alpha_2 fixed to 0 in every bin, every sigma_v
  prior width x6, pessimistic prior; the yamls EXAMPLE_EVALUATE1-4;
  data key 'B tilts noA2 sv x6'),
  FoM(w0, wa): 0.564 / 0.352 / 0.173 / 0.0447 at 0.075 and 6.70 / 7.95 /
  2.34 / 0.307 at 0.2; FoM(Omega_m, 10^9 A_s) 50.3 / 53.2 / 19.9 / 3.20
  at 0.2. eta <= 1.030 at every cut; alpha_2 = +-12.5 moves no score by
  more than 0.327 sigma. On the paper-fiducial cuts these FoMs are
  0.85-1.04 times those on the linear-bias cuts (0.97-1.00 on 62 Mpc and
  Delta z = 0.02). Source: EXAMPLE_EVALUATE_FISHER_MODEL.ipynb, section
  9; `notes/bias_model_summary/plots/summary_fom_kmax/data.json`.
- **Pairs beyond the measured list** (the pair-axes study:
  `notes/cross_correlations`, sections "What would pairs beyond the
  list add?" and "Why the added pairs help: the free bias levels and
  slopes"; summary in `notes/bias_model_summary`): on the three thinner
  layouts the added pairs raise the fiducial model's FoM(w0, wa) by up
  to 55-59%, on Delta z = 0.2 by at most 2%. Attribution fits trace the
  gain to the free level and slope of each group's bias amplitude
  beta_i = b_{1,i} D(zbar_i); with every nuisance parameter held fixed
  the added pairs are worth only 2-8%. Source:
  `scripts/compute_pair_axes_fits.py` ->
  `chains/pair_axes_fits_<layout>.json`;
  `notes/cross_correlations/plots/pair_axes_*/data.json`.
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
- **Model A is retired.** The Fisher notebook
  EXAMPLE_EVALUATE_FISHER_MODEL.ipynb carries the reference model and
  the fiducial model only, builds its derivative base on the paper
  yamls and its scale cuts on the paper fiducial. The model A notebook
  and its five v2 figure folders are removed (record: commit 567751f),
  and model A's exploration yamls, likelihood classes and parameter
  files are deleted; `likelihood/exploration/` keeps only
  `params_desi_dr3_forecast.yaml`.
- **Only b2, bs2, alpha_0 and alpha_2 tilt** (they follow the host-halo
  population); SN0 and SN2 never do. b2 and bs2 inherit the full
  b_{1,i}, plus a group offset and a tilt. Free b2 and bs2 per bin (152
  parameters) is a bound, not a model.
- **SN2 is in the model.** Without it, an SN2 truth at one prior width
  moved Omega_m by 1.3-1.7 sigma at 0.075 through the anticorrelated
  adjacent cross spectra, and the k_max = 0.075 variant of the
  development (model A) broke before its own cut on the
  thin layouts; no white noise, per group or per bin, repairs that, and
  dropping the cross pairs keeps only 18-46% of the thin-layout FoM.
  With SN2, that variant held to 0.075 at a cost of 4-8% of FoM(w0, wa)
  (`references/fom_and_bias_review.md`, Sec. 6).
- **The fiducial model fixes alpha_2 = 0 in every bin**: alpha_2 is
  about 99% degenerate with the sampled sigma_v (the FoG degeneracy
  note), every sigma_v prior width is multiplied by 6 in exchange, and
  alpha_2 = +-12.5 h^-2 Mpc^2 moves no score by more than 0.327 sigma.
- **Calibration-informed offset priors** (Delta_2 width 2, Delta_s 1)
  instead of DESI DR1's agnostic N(0, 5^2) on b2 sigma8^2
  and bs sigma8^2: at low cuts the data barely measure b2, and for the
  dense BGS a low b2 drives P_gg negative. Fiducials come from the
  calibrations, never from DESI chain values.
- **No FoM rests on exact calibrations.** The linear-bias and
  exact-calibration FoMs at the same cut are diagnostics in the
  notebooks' printed tables and data.json, never on a figure.
- **One likelihood option**, the model variants as parameter files; the
  former `higher_order_bias: 2` is gone.
- **The paper keeps the standard data vector.** The cross pairs are
  those with blurred-kernel overlap O_ij >= 1e-6 at the fiducial
  sigma_v (`scripts/make_desi_pairs.py`); the pair-axes study's longer
  lists are a comparison, not an open choice.
- **The FAST-PT reference comparison** runs through TATT + IA_code 1 +
  the fastpt theory block; no reset knob was added for it.
- **The DESI bias boxes [0.8, 4] stay**; the N-models figures argue for
  narrowing them with collaborators. The amplitude cut is 8, the common
  cut of every cosmolike project.

**The covariance**

- **One covariance on the one-loop fiducial**, with the one-loop Gaussian
  term computed in C (the clean option, chosen over rescaling the
  linear signal by the one-loop to linear ratio). Every one-loop
  variant shares the fiducial, so one covariance serves them all.

**Figures, scenarios and notes**

- **k_max <= 0.2 h/Mpc** on every figure axis and in every quoted FoM,
  the PI's rule (0.2 h/Mpc is also the reach of DESI DR1's one-loop
  full-shape fits). The k_max scan stops there (`KMAX_MAX`); the
  scale-cut cache's larger cuts are not used.
- **Two scenarios, and the headline is the PI's call.** The baseline
  scenario is the baseline prior with the reference model without
  tilts; the pessimistic scenario is the pessimistic prior with the
  reference model. The paper figure `fom_kmax_models` is one 2x3 grid:
  a row per scenario, columns FoM(w0, wa), FoM(Omega_m, 10^9 A_s) and
  the largest shift.
  Which scenario is the paper's headline is decided by the PI after
  seeing both side by side; until then both carry equal weight, and
  neither is labeled headline, conservative or defensible.
- **Four galaxy-bias notes, the summary last**: correlated_bias_prior_v2
  (the forecast note; no v3), linear_bias_kmax, higher_order_bias, and
  bias_model_summary, rewritten after the other three as a presentable
  summary with their key figures (not compact). Every paper figure has
  a plot folder. In correlated_bias_prior_v2 the accuracy test precedes
  "The fiducial model against k_max", and the mathematics appendix
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
  `rung_columns`). The paper's model is **the fiducial model**
  (one-loop, alpha_2 = 0 in every bin, sigma_v prior widths x6, the
  pessimistic linear-bias prior); **the fiducial model with alpha_2
  free**, at the exploration sigma_v widths, is the reference model of
  the companion notes, and **the fiducial model without tilts** is its
  variant with no redshift tilts (the old model B). "Model A" and
  "model B" are retired development names and appear only in records.
  The code keeps the historical names, each stated once where used:
  data keys `'B'`, `'B tilts'`, `'B tilts noA2 sv x6'`; likelihood
  classes `desi_cl*_model_b`; parameter files `*_model_b_tilts.yaml`;
  the `--model-b` flag of `make_desi_params.py`. The tilts are a free
  straight line in redshift per group on Delta_2, Delta_s and alpha_0
  (and alpha_2 when free), like the linear bias's s_g. "Baseline"
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
- **Out-of-date cached Fisher matrices.** The linear caches store F built with
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
  bias and under the fiducial model, whose alpha_2 is fixed), and every
  other nuisance likewise, so a value left
  by another likelihood cannot survive, and a derivative step written
  into the compiled setter is overwritten. The notebooks wrap the
  setter, add the step, restore it, and check the vector returns
  bitwise.
- **Two models in one process.** The gg cross-pair list is a static of
  cosmo2D.c; before `initial_setup` cleared it, a model built after a
  DESI one inherited 13 pairs and died in `gbmag`. Any new static setter
  needs the same reset. The interface holds one dataset, so build each
  probe's derivatives right after its model.
- **`logposterior` and caching in Fisher steps** return an out-of-date vector
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
| the bias model in the forecast note; the fiducial model against the scale cut and where the variant without tilts breaks | correlated_bias_prior_v2: The bias model; The fiducial model against k_max |
| random truths, the statistic, coverage; the calibration drift | correlated_bias_prior_v2: The accuracy test; appendix The accuracy test |
| each note's fiducial point and parameter values | every note: Fiducial model and parameter values |
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
  one-loop vector), and about half of each bin's SN2 kernel comes from
  k_par above its own cut. Rebuilding the cut on the one-loop model is
  undecided.
- **Positivity far from the fiducial.** Negative alpha_0 or b2 a unit
  below its calibration makes the low-rank engine floor P_gg - P_inf
  (alpha_0 = -88.9: above k ~ 0.14 h/Mpc, 47% of the grid; the vector
  then curves in alpha_0), and on the dense BGS P_gg turns negative
  near 0.15-0.2 h/Mpc, inside the scanned cuts. Harmless for the Fisher
  forecast at the fiducial; chains need a loop taper above the trusted
  scale (near 0.15 for BGS) or a positivity prior on P_gg. Neither is
  implemented; a tighter b2 prior or SN0 does not fix the sign.
- **The C-FASTPT b3 template** (`tab_d1d3`) is cosmology-independent and
  wrong, so b3 = 0; `internal_accuracyboost: 4` breaks the bias tables.
  Both are core bugs.
- **No redshift-space one-loop terms**: the model relies on the sampled
  sigma_v (and, in the reference model, on alpha_2).
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

- **EXAMPLE_EVALUATE_FISHER.ipynb's markdown** still quotes the
  linear-bias-covariance values (41.2 / 50.2 / 12.6 / 2.22 on the
  shipped mask; the rerun prints 43.4 / 51.8 / 13.2 / 2.43) and predates
  the naming rules ("independent" and "tied" for the free and
  growth-tied priors, a "defensible" k_max window).
- **Out-of-date text** (through the docs subagent):
  `likelihood/desi_cl.py`'s module docstring
  still says cross-bin spectra have no slot, cites "the paper's eq. 20",
  says "passive evolution", and points to the higher-order note's
  removed recommendation section, as does a comment of
  `scripts/make_desi_params.py`.
  `covariance/lssty1xdesi_covariance.py`'s docstring still
  calls the 62 Mpc likelihood covariance the sacc import.
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
