# The Fisher module, the LSST Y1 3x2pt case and the scale cuts

Detail behind SKILL.md, "The Fisher machinery and the notebooks". Unless
stated, the numbers below are linear bias (the growth-tied shipped
yamls) and were measured with the corrected linear-bias covariance of
2026-11-07. The covariance on the one-loop fiducial that replaced it
moves FoMs at k_max <= 0.1 by about 1% and raises the shipped-mask
linear-bias FoMs by 3-10% (`covariance.md`, Sec. 1); the scale-cut FoM
table of Sec. 5 is the rerun on the current covariance. These numbers
are diagnostics, not forecasts: the forecast numbers are those of the
one-loop models (`results.md`).

## 1. The LSST Y1 3x2pt case

`EXAMPLE_EVALUATE5.yaml` (case `lsst_y1_3x2pt`, built by
`scripts/make_lsst_3x2pt_case.py`) is LSST Y1 alone in Fourier band
powers on the 53 DESI bands, so its Fisher matrix sums with a DESI
case's. Layout EE 795 + gs 1,325 (all 25 pairs) + gg 265 = 2,385
entries, 1,629 unmasked: EE keeps every band, gs and gg the lens bin's
k_max cut. The n(z) are lsst_y1's, read Z_LOW
(`photoz_zmid_convention: 0`) although their first z value is a bin
midpoint: the donor's convention, kept so the case is the original
lsst_y1 n(z). The DESI cases read their files Z_MID; mixing the two
conventions in one joint lens sample needs a decision. The TATT
amplitudes are fixed parameters of params_source.yaml and must stay out
of the evaluate override (cobaya rejects a fixed parameter there).

Two LSST references appear in the results. This project's band-power
case gives FoM(w0, wa) = 17.4 (17.36 at AB 3), consistent with the
DESC SRD (arXiv:1809.01669: Y1 WL+LSS at 22 without priors). The
figures and notes draw the lsst_y1 project's real-space SRD forecast:
FoM(w0, wa) = 17.26 and FoM(Omega_m, 10^9 A_s) = 214.9, from the
Fisher matrix of lsst_y1's EXAMPLE_EVALUATE2.ipynb
(`chains/EXAMPLE_EVALUATE2_FISHER_3x2pt.pkl` of that project; its
fiducial has w0 = -0.9 against -1 here).

## 2. Design rules of `ModelFisher`

`interface/cosmolike_lssty1xdesi_fisher.py` (`ModelFisher`) builds the
Fisher matrix on the cobaya model of an EXAMPLE yaml. Each rule comes
from a failure it prevents.

- **Evaluate with `Model.loglike`, never `logposterior`.** Outside a
  flat prior `logposterior` returns -inf without running the
  likelihood, and the interface keeps the previous point's vector: a
  silently wrong derivative. The ELG bias fiducial (0.84) sits next to
  its prior edge (0.8), so the h = 0.05 stencil leaves the prior. A
  non-finite loglike raises.
- **`cached=False`.** A cached point returns without recomputing, and
  the interface vector would again be stale. The price is a CAMB run
  for every nuisance step.
- **Read the likelihood's vector.** The compiled interface's vector
  lacks the stochastic templates the one-loop likelihood adds in
  Python; the module reads the likelihood's `last_datavector` when it
  exists.
- **Zero-fiducial steps.** cnu.get_ddv falls back to an absolute step h
  for a zero fiducial: 10 sigma for a source photo-z shift at h = 0.02.
  The module steps such a parameter by h sigma_prior/0.02 (the prior
  width at h = 0.02) and by h when its prior is flat (wa).
- **One active model.** The interface holds one dataset; `dv` and
  `invcov` refuse once another ModelFisher was built. Compute each
  probe's derivatives, Fisher matrices and bias numerators right after
  building its reference model, alternatives first.
- **Area.** DESI case covariances are built at 14,000 deg^2; the
  common-area forecast multiplies their inverse by 12,300/14,000.
- **Cache derivatives, never Fisher matrices.** A covariance change
  must not need new derivatives. Every notebook rebuilds F, the bias
  numerators and the delta chi2 from the current covariance file at
  every run (`refresh_fisher` in EXAMPLE_EVALUATE_FISHER.ipynb), with
  the Gaussian priors read from the example yaml through cobaya
  `update_info` (`prior_diagonal` in the k_max notebook). The F stored
  in the linear caches (`chains/EXAMPLE_EVALUATE<n>_FISHER_AB<b>.pkl`)
  predates the covariance fix and is unused: the printed rebuild check
  reads max |dF_ij|/sqrt(F_ii F_jj) = 0.76 for DESI and 1.8e-14 for
  LSST. The scale-cut FoM cache stores each layout's covariance sha256
  and recomputes on a change.
- **Never difference a cached Fisher matrix.** An earlier k_max
  notebook obtained the Gaussian prior as diag(F_cached - F_data); once
  F_data used the new covariance and F_cached the old one, the
  difference was a fake prior on every parameter. The priors now come
  from the yamls only.
- **Pseudo-inverse with an information test.** `stable_inverse` scales
  each Fisher matrix to unit diagonal, diagonalizes it, drops the
  directions below 1e-12 of the largest eigenvalue (PINV_RTOL) and
  prints each dropped direction with its largest component on
  (w0, wa), Omega_m and A_s, which must vanish (7.2e-12 and 7.0e-10
  measured). A bias parameter whose information at a cut is below a
  small fraction of its shipped-mask information (1e-10 in the linear
  study) leaves the fit; a cut without data for the cosmology gives no
  fit and an FoM of 0. Two failures set this rule: derivative
  round-off on a group bias with no kept band once made the
  Delta d_com = 62 Mpc FoM at k_max = 0.05 infinite, and under model A
  + tilts on the free prior the flat BGS tilts (only 7 of 18 BGS
  bins keep a band at 0.075 on that layout) left one null direction
  along which `np.linalg.inv` produced a spurious 2 sigma shift.
- **Wrap setters, do not call them directly.** The likelihood sets
  every nuisance at every point (alpha_2 included, with zeros under
  linear bias and model A), so a derivative step written into the compiled
  interface before an evaluation is overwritten. The notebooks wrap
  the setter the likelihood calls (`ci.set_nuisance_bias` for the
  linear per-bin columns, `set_nuisance_bias_tidal` for the model A
  operators, `set_nuisance_bias_alpha2` for the alpha_2 columns), add
  the step to the values the likelihood passes, restore the original
  setter afterwards, and check that the vector after the steps equals
  the one before bit for bit.
- **Per-bin columns by graph coloring.** A greedy coloring of the
  cross-pair graph (two bins joined by a measured pair never share a
  color) shifts every bin of one color together, so each spectrum
  involves at most one shifted bin; four colors per layout give every
  bin's column in eight evaluations instead of two per bin. C_ell is at
  most quadratic in the biases, so the central difference is exact for
  b1; the chain-rule columns must sum, with weights 1/D(zbar_i), to the
  stencil derivative of B_g.
- **LSST Y1 3x2pt + DESI C_gg = block-diagonal sum.** `combine` adds
  Fisher matrices by parameter name; no prior is shared, so none is
  double-counted. The joint likelihood for chains still needs the
  concatenated LSST+DESI lens sample.
- **Smooth vectors first.** A stencil derivative is only as good as the
  smoothness of the vector in the parameters. Diagnostic: scan one
  parameter on 17 fine points, whiten by the Cholesky factor of C^-1,
  fit a quartic, read the residual per point in sqrt(chi2) units
  (2026-11-04, pre-fix DESI covariance, Delta z = 0.2 and the LSST
  case):

  | setting | DESI omegam / w0 / wa | LSST omegam / w0 / wa |
  |---|---|---|
  | stock halofit, NL 8, AB 1 | 4.2e-2 / 3.0e-2 / 2.8e-2 | 2.3e-2 / 3.5e-2 / 4.8e-2 |
  | stock halofit, AB 3 | 2.0e-2 / 2.4e-2 / 2.6e-2 | 2.4e-2 / 3.4e-2 / 4.9e-2 |
  | EE2 (no halofit), NL 8 | 3.4e-2 / 1.6e-2 / 6.3e-3 | 3.5e-4 / 4.9e-3 / 3.0e-4 |
  | EE2, NL 32 / 64 / 128 (omegam) | 1.3e-2 / 1.1e-3 / 6.9e-4 | - |
  | patched halofit, NL 8 | 3.4e-2 / 1.6e-2 / 6.3e-3 | 2.7e-4 / 1.3e-3 / 2.2e-4 |
  | patched halofit, NL 64 | 9.8e-4 / 3.3e-3 / 1.4e-3 | (no chi grid dependence) |

  Two sources, neither touched by AccuracyBoost: stock CAMB halofit's
  bisection tolerance (1e-3; the jumps repeat at the same parameter
  values at AB 1 and 3) and, for the thin DESI bins, the chi grid at
  the likelihood's boost (8 converges one chi2; derivatives need 64).
  With stock halofit at h = 0.01-0.05 the FoM(w0, wa) moved 22.7 / 7.3
  / 5.8 / 5.1 (DESI) and 52 / 14 / 24 / 19 (LSST). CAMB is installed
  with `PATCH_CAMB_HALOFIT_TOLERANCE` (1e-7), and the notebooks set
  `nonlimber_accuracyboost: 64` for every DESI Fisher evaluation.
- **Step and accuracy stability with both fixes** (derivatives
  2026-11-05; FoM(w0, wa) at h = 0.01 / 0.02 / 0.03 / 0.05,
  Delta z = 0.2):

  | probe | AB 1 | AB 3 |
  |---|---|---|
  | DESI C_gg | 2.53 / 2.20 / 2.18 / 2.24 | 2.23 / 2.22 / 2.22 / 2.22 |
  | LSST 3x2pt | 17.68 / 17.67 / 17.65 / 17.68 | 17.36 at every step |
  | LSST Y1 3x2pt + DESI C_gg | 24.9 / 24.2 / 24.2 / 24.3 | 24.09 / 24.08 / 24.08 / 24.08 |

  Cost (M2 Pro, AC, 8 threads, quiet, 2026-11-05): one evaluation is
  3.8 s / 14.3 s (DESI at NL 64, AB 1 / 3) and 0.8 s / 2.4 s (LSST); a
  step costs 4 evaluations per parameter (20 DESI, 29 LSST), so the
  linear notebook's 4 steps x 2 boosts take about two hours from
  scratch. At AB 3 every step agrees to 0.6%. At AB 1 the DESI
  h = 0.01 derivative is still noise-limited. AB 1 -> 3 at fixed h
  moves DESI by +1% and LSST by -1.8%, an accuracy shift, not noise;
  whether AB 3 is converged needs an AB 5 run.
- **Thin-layout steps.** The thin-layout Fisher matrices use a single
  step (h = 0.02) at AB 1, with no step scan of their own.

## 3. Fisher biases of the projection engines

`bias_vector` (D^T C^-1 delta) and `fisher_bias` (F^-1 B, block
numerators summed). Alternatives are likelihood overrides on the same
yaml: FKEM = `nonlimber_lowrank_gg: 0` (+ `lmax_nonlimber: 550` for
the fair cap), Limber = `adopt_limber_gg: 1` (+ `adopt_limber_gs: 1`
for LSST). Measured at AB 3, h = 0.02, Delta z = 0.2 (max
|delta p/sigma| over the seven cosmological parameters; stable to ~1%
over h; delta chi2 at 12,300 deg^2 and `nonlimber_accuracyboost: 64`):

| alternative | probe | max abs(delta p/sigma) | Delta chi2 |
|---|---|---:|---:|
| FKEM, cap 550 | DESI C_gg | 0.009 (Omega_m) | 0.039 |
| Limber | DESI C_gg | 3.1 (n_s) | 85.0 |
| SVD engine | LSST 3x2pt | 0.0024 | 1.4e-4 |
| Limber | LSST 3x2pt | 0.40 (A_s) | 11.2 |
| DESI FKEM, cap 550 | LSST Y1 3x2pt + DESI C_gg | 0.008 | 0.039 |
| DESI Limber | LSST Y1 3x2pt + DESI C_gg | 0.98 (w_a) | 85.0 |
| Limber in both | LSST Y1 3x2pt + DESI C_gg | 1.01 (w_a) | 96.2 |

The test pins (`DCHI2_MEASURED_CASES`, 0.132 for Delta z = 0.2) score
the frozen pre-fix covariance at the likelihood's chi grid, so the two
are not comparable.

Per layout (AB 1, h = 0.02, alternatives cached in
`chains/EXAMPLE_EVALUATE_FISHER_ALTERNATIVES.pkl`; DESI C_gg alone /
LSST Y1 3x2pt + DESI C_gg; delta chi2 FKEM / Limber):

| layout | FKEM, cap 550 | Limber | Delta chi2 |
|---|---|---|---|
| Delta d_com = 62 Mpc | 0.16 (w_a) / 0.49 (H_0) | 11 (Omega_b) / 12 (Omega_b) | 19.3 / 6548 |
| Delta z = 0.02 | 0.30 (Omega_b) / 0.63 (H_0) | 27 (Omega_b) / 17 (Omega_b) | 31.7 / 1.49e4 |
| Delta z = 0.05 | 0.08 / 0.14 | 7.2 (Omega_b) / 3.0 (A_s) | 1.51 / 1305 |
| Delta z = 0.2 | 0.009 / 0.008 | 3.1 (n_s) / 0.98 (w_a) | 0.039 / 85.0 |

FKEM's Limber treatment of the nonlinear correction is harmless on the
wide bins but biases the thin layouts by a sizable fraction of sigma;
Limber is excluded for DESI clustering on every layout.

## 4. The non-Limber UV-fraction scale cut

The scale cuts are the non-Limber UV-fraction criterion of
arXiv:2603.13031 (following arXiv:2510.18981); they replace the Limber
relation ell <= k_max chi(zbar) - 1/2. Each C_ell^gg band power splits
into C_IR + C_UV: C_IR is computed with the 3D power spectrum
multiplied by a step that keeps k <= k_max/D(z), with D the linear
growth factor (D(0) = 1) and k_max quoted in h/Mpc at z = 0, so the
cut grows with redshift as the nonlinear scale does; C_UV is the rest.
A band is kept while |C_UV| / (|C_UV| + |C_IR|) < alpha = 0.05 (the
paper's baseline; it also uses 0.01). Kept bands form a prefix in ell,
every band up to the first failure; where the fraction is
non-monotonic in k_max, each prefix is shortened to that of the next
larger k_max, so the cut only tightens as k_max falls. The fit uses
(shipped mask) AND (UV-fraction cut): the covariance files have
entries only inside the shipped masks, which stop every bin at the
fixed k = 0.1185/Mpc (~0.176 h/Mpc) with no growth scaling, so they cap
every band count and every FoM curve. The cut was built on the
linear-bias spectra and is reused unchanged for the one-loop models
(an open issue, SKILL.md).

The step multiplies P(k) inside every gg engine of the compiled code
(cosmolike_core `cosmo2D_scuts.c` drives it), so the criterion never
assumes the Limber relation:

- Limber sums: per line-of-sight node, at that node's own k_max/D(a);
- non-Limber sums (the low-rank engine and FKEM): in the final sum over
  the FFTLog wavenumbers, one threshold per bin (per pair for the cross
  spectra), with D at the bin's mean redshift.

The interface exposes `gg_scale_cuts_tomo(kmax, alpha)` - a dict with
`lmax`, `mask`, `uv_fraction`, `full` and `ir`, rows the autos then the
cross pairs in data-vector order - and `C_gg_kwindow_tomo(kmax,
keep_ir)` for the IR or UV spectra directly. The window is off by
default and reset in initial_setup. The linear notebook caches the cuts
in `chains/EXAMPLE_EVALUATE_FISHER_SCALE_CUTS.pkl` (every other
notebook reads them) and the FoM scan in
`chains/EXAMPLE_EVALUATE_FISHER_FOM_SCALE_CUTS.pkl`.

Validation (measured):

- window off: the data vector is bitwise identical to the pre-window
  code, and the driver's full spectra equal the data vector exactly;
- IR + UV = full to 1e-14 (low-rank and FKEM) and 1e-15 (Limber);
  k_max = 1e3 gives IR = full exactly, and k_max = 1e-4 gives IR = 0;
- Delta z = 0.05 autos kept at k_max = 0.05 / 0.075 / 0.1 / 0.125 /
  0.15 h/Mpc, against independent references:

| engine | kept bands | independent reference |
|---|---|---|
| Limber hook | 518 / 808 / 1067 / 1171 / 1198 | numpy Limber integral: 519 / 808 / 1068 / 1171 / 1198 |
| low-rank non-Limber | 367 / 663 / 945 / 1148 / 1195 | flat-sky non-Limber estimate: 376 / 683 / 973 / 1154 / 1198 |
| FKEM with pairs | 370 / 674 / 975 / 1154 / 1196 | - |

Band powers kept (shipped mask AND UV-fraction cut; autos / cross
pairs):

| k_max | Delta d_com = 62 Mpc | Delta z = 0.02 | Delta z = 0.05 | Delta z = 0.2 |
|---|---|---|---|---|
| 0.05 | 167 / 74 | 0 / 0 | 367 / 70 | 122 / 44 |
| 0.075 | 903 / 750 | 904 / 952 | 663 / 184 | 198 / 82 |
| 0.1 | 1585 / 1124 | 1928 / 2471 | 945 / 269 | 267 / 111 |
| 0.125 | 2049 / 1432 | 2755 / 3062 | 1148 / 522 | 307 / 138 |
| 0.15 | 2220 / 2038 | 2982 / 3118 | 1195 / 623 | 319 / 156 |
| 0.2 | 2276 / 2889 | 3016 / 4496 | 1205 / 1240 | 325 / 306 |
| shipped | 2276 / 3033 | 3016 / 4605 | 1205 / 1777 | 329 / 415 |

Conclusions:

- A thin bin's C_ell collects radial modes up to k_parallel ~
  2 pi/Delta chi, so its UV part is large even at the lowest bands. The
  criterion sees those modes because it is a step on P(k); a cut built
  on the Limber relation counts only transverse modes and is badly
  optimistic on the thin layouts.
- The cross spectra of adjacent thin bins come mostly from close pairs
  across the shared edge, small radial scales, and are strongly
  UV-dominated.
- At k_max = 0.05, Delta z = 0.02 keeps nothing, and Delta d_com =
  62 Mpc keeps only the bins above z ~ 0.9, where 1/D(z) raises the cut
  the most (7 of 18 BGS bins keep a band at 0.075).

## 5. Linear-bias FoMs (diagnostics)

FoM(w0, wa) of DESI C_ell^gg alone, growth-tied linear bias, on the
kept block (EXAMPLE_EVALUATE_FISHER.ipynb, section 8, "The figure of
merit against the scale cut"; rerun on the current covariance,
2026-11-08):

| k_max | Delta d_com = 62 Mpc | Delta z = 0.02 | Delta z = 0.05 | Delta z = 0.2 |
|---|---:|---:|---:|---:|
| 0.05 | 0.358 | 0 (no band left) | 0.592 | 0.205 |
| 0.075 | 7.63 | 7.24 | 2.77 | 0.652 |
| 0.1 | 15.1 | 23.2 | 5.32 | 1.17 |
| 0.125 | 24.5 | 35.4 | 8.73 | 1.54 |
| 0.15 | 34.1 | 44.5 | 11.1 | 1.90 |
| 0.175 | 39.2 | 48.4 | 12.3 | 2.12 |
| 0.2 | 41.3 | 51.1 | 12.7 | 2.14 |
| shipped mask | 43.4 | 51.8 | 13.2 | 2.43 |

The linear-bias ladder at k_max = 0.075 (section 8, "How much bias
freedom costs"; the same values on the current covariance are the
`v1_linear` key of `baseline_ladder/data.json`, read from the k_max
study; FoM(w0, wa); on Delta z = 0.2 every group has one or two bins,
so the lower three priors coincide):

| layout | growth-tied | baseline | pessimistic | free |
|---|---:|---:|---:|---:|
| Delta d_com = 62 Mpc | 7.63 | 5.10 | 3.37 | 2.78 |
| Delta z = 0.02 | 7.24 | 4.65 | 3.56 | 1.65 |
| Delta z = 0.05 | 2.77 | 0.712 | 0.492 | 0.410 |
| Delta z = 0.2 | 0.652 | 0.177 | 0.177 | 0.177 |

A linear-bias fit is biased at this cut (`results.md`, Sec. 1), so
these numbers built the mechanics of the ladder and are not results.
Under linear bias the accuracy test gave the same ordering as under
model A: growth-tied excluded (4.7-10x too optimistic), baseline honest
under its own scatter and 1.2-1.7x too small under pessimistic
scatter, pessimistic honest or at most 10% conservative, free exactly
honest; a conserved-population drift moves the growth-tied fit by
8.0 / 7.2 / 11.6 / 5.0 sigma (layout order).
