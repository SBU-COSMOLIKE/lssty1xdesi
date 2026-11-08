# The FoM scale and the galaxy bias model: review conclusions

Conclusions of an adversarial physics and code review (2026-11-06) of
the DESI C_ell^gg Fisher forecast - the EXAMPLE_EVALUATE1-4 layouts at
h = 0.02, AccuracyBoost 1 - and of the draft paper's higher-order bias
model, with the later finding that SN_2 breaks model A (Sec. 6). The
review's Fisher numbers were reproduced from the cached derivative
matrices with the covariance of the time (the rebuilt F matched the
pickles to better than 7e-16) and from counterfactual covariances
built on the fiducial data vector. Its covariance finding is fixed
(Sec. 3); numbers marked "pre-fix" are relative statements.

## 1. The model, and what it leaves out

The DESI C_ell^gg model predicts galaxy clustering as linear bias
times the nonlinear matter power, P_gg = (b1 + f mu^2)^2 P_halofit(k),
with a Fingers-of-God smearing of each bin's n(z). Real galaxy
clustering has more terms at one loop:

- second-order and tidal bias (b2, b_G2, b3nl);
- counterterms proportional to k^2 P_lin, which also carry the
  small-scale velocity damping (k^2 mu^2 terms);
- stochastic terms, such as shot-noise offsets.

These corrections are a few percent at k ~ 0.1 h/Mpc and 10-20% at
0.2-0.3 h/Mpc. A Fisher forecast or likelihood that sets them to zero
converts their amplitude and shape into cosmological information - at
exactly the sub-percent amplitude precision a 76-90 bin forecast
reaches. The sigma_v FoG smearing is not a substitute for them: it is
a mu-independent radial damping of n(z), while the dangerous terms are
scale- and mu-dependent.

## 2. EFT-grade scale cuts

An EFT-grade scale cut is the k_max at which the model in use is
accurate to better than the survey's statistical error; going further
requires freeing the nonlinear nuisances. The two honest choices:

- (a) linear bias with k_max ~ 0.08-0.12 h/Mpc (~0.055-0.08 /Mpc),
  the literature estimate; on this vector a linear-bias fit is
  unbiased only at 0.05 h/Mpc (notes/linear_bias_kmax);
- (b) k_max ~ 0.2 h/Mpc (the DESI DR1 full-shape choice,
  ~0.135 /Mpc) with the one-loop EFT model, marginalizing about 7-12
  nuisances per tracer/redshift chunk: b2, b_G2, counterterms c0, c2,
  c4, stochastic terms.

The shipped masks cut each bin at k = 0.1185 /Mpc (~0.18 h/Mpc):
choice (b)'s reach with choice (a)'s rigidity.

In angular C_ell a per-bin cut is ell_max = k_max chi(z_bin) - 1/2.
Thin bins also resolve k_parallel up to ~pi/Delta chi ~ 0.05 /Mpc, so
a cut on the total k is slightly stricter than the transverse-only
rule. The implemented criterion is the non-Limber UV-fraction scale
cut, following arXiv:2603.13031: a step on P(k) keeps
k <= k_max/D(z), with D the linear growth factor (D(0) = 1) and
k_max quoted in h/Mpc at z = 0, and a band is kept while its UV
fraction |C_UV| / (|C_UV| + |C_IR|) stays below alpha = 0.05. The
skill's scale-cut section holds the implementation, validation
and results. The draft paper (paper_latex/main.tex) instead uses a
scale-leakage criterion with threshold alpha = 0.3 (its equation for
ell_max^ii), and finds that the Limber relation ell_max = k_max chi is
optimistic at low redshift (BGS).

## 3. The thin-bin FoM: physical ratio, covariance-set normalization

FoM(w0, wa), DESI C_gg alone, h = 0.02, shipped mask;
Delta d_com = 62 Mpc names the 76-bin layout of equal
comoving width, d_com the comoving distance. "Pre-fix" is the
covariance the review measured (raw chunk bias, item 1 below);
"corrected" is the shipped one (EXAMPLE_EVALUATE_FISHER.ipynb):

| layout | pre-fix | corrected |
|---|---:|---:|
| Delta z 0.2 (10 bins) | 5.67 | 2.22 |
| Delta z 0.05 (36 bins) | 32.0 | 12.6 |
| Delta z 0.02 (90 bins) | 125.6 | 50.2 |
| Delta d_com = 62 Mpc (76 bins) | 135.2 | 41.2 |

Decomposition from the cached derivatives (pre-fix covariance, same
layout order):

- **free bias per bin**: 1.94 / 11.2 / 56.4 / 66.8 - the thin/wide
  ratio survives (corrected covariance, k_max = 0.25: 0.70 / 4.26 /
  23.2 / 21.0);
- **autos only**: 5.0 / 24.1 / 58.5 / 72.6;
- **everything but w0, wa fixed**: 991 / 3402 / 7265 / 9105, a 7.3x
  thin/wide ratio, close to radial mode counting;
- **ell <= 100 only**: a ratio of 17x, so the gain comes from linear
  scales;
- **per-bin Limber-relation k_max cut at 0.05 /Mpc**: DESI FoM
  1.5 / 8.5 / 33.7 / 38.0, against LSST Y1 3x2pt alone at 17.4.
  Superseded by the non-Limber UV-fraction scale cuts (Sec. 2),
  which count the thin bins' radial modes: at the matching
  k_max = 0.075 h/Mpc (~0.05 /Mpc) they give 1.91 / 8.29 / 19.9 /
  32.9 pre-fix (0.646 / 2.77 / 7.26 / 7.61 corrected), and at
  k_max = 0.05 h/Mpc the Delta z = 0.02 layout keeps
  no band at all. The full tables are in the skill's scale-cut
  section.

Interpretation (thin/wide ratio 24x pre-fix, 19-23x corrected):
about 7x from radial modes (k_parallel resolved to
~0.006 /Mpc at Delta z = 0.2 against ~0.06 /Mpc at Delta z = 0.02),
and about 3x from degeneracy breaking by Kaiser RSD and radial BAO,
which wide bins wash out.

The literature says the same (Asorey et al. 2012, arXiv:1207.6487;
Di Dio et al. 2014, arXiv:1308.6186; Camera et al. 2018,
arXiv:1803.10773): many thin bins recover the 3D P(k, mu) information
and saturate when Delta chi ~ 0.6-0.8 lambda_min. The information is
bounded by the 3D Fisher of the same survey at the same cuts, so it
does not grow without limit as Delta z -> 0. Delta d_com = 62 Mpc
(76 bins) against Delta z 0.02 (90 bins) shows the plateau: 135
against 126 pre-fix, 41 against 50 corrected.

### What sets the absolute FoM

1. **The covariance bias** (fixed). The likelihood uses
   b_i = DESI_B1 / D(zbar_i), and the review found
   `covariance/lssty1xdesi_covariance.py` passing the raw CHUNK_BIAS
   (1.34 / 1.70 / 0.84) through `_lens_bias`, which left the
   covariance signal low by D(z)^2 at high z. Its counterfactual, a
   self-consistent covariance, predicted FoM 2.3 / 13.1 / 51.7 / 42.2.
   `_lens_bias` now divides by D(zbar_i), computed as the likelihood
   does, and the regenerated covariances give 2.22 / 12.6 / 50.2 /
   41.2 (the skill's section "The covariance").
2. **Weak covariance cross spectra at high ell**: the covariance's
   non-Limber correction is FKEM-linear at the covariance's own
   boost, against the low-rank engine in the data vector; worth
   about -28% on Delta z 0.02.
3. **Gaussian-only covariances shipped**: SSC and cNG are computed
   but not included in the likelihood files.
4. **Thin-layout Fisher matrices at a single step** (h = 0.02), with
   no convergence scan.
5. **The density is the completed DESI survey**: 0.62 /arcmin^2 over
   f_sky ~ 0.3, about 28M tracers, not DR1 (about 6M;
   arXiv:2404.03002). The "DR1" label is wrong.
6. **The model rigidity of Secs. 1-2**: linear bias + halofit to the
   shipped cut, with six constant amplitudes and seven sigma_v groups
   as the entire ignorance budget. Measured since: a linear-bias fit
   is unbiased only at k_max = 0.05 h/Mpc (notes/linear_bias_kmax),
   and the forecast is one-loop (models A and B, Sec. 5).

### The LSST comparison

LSST Y1 3x2pt alone has FoM 17.4, consistent with the DESC SRD
(arXiv:1809.01669: Y1 WL+LSS at 22 without priors). The forecast's
"LSST barely helps" conclusion was an artifact of the inflated DESI
side, and does not survive the corrections above.

### Benchmarks

- DESI FDR (arXiv:1611.00036, Table 2.9, all with Planck): galaxy
  BAO 133; BAO + broadband k < 0.1 h/Mpc 332; k < 0.2 h/Mpc 704.
- The real DESI DR1 FS+BAO alone does not quote w0-wa at all
  (arXiv:2411.12022, Sec. 4.2: projection effects).

## 4. The draft paper's bias model against CCL

The draft paper's EPT auto-spectrum (paper_latex/main.tex) was checked
term by term against CCL's EulerianPTCalculator source
(pyccl/nl_pt/ept.py, _get_pgg).

- **The six terms match CCL exactly.** The draft's
  P_gg^EPT = b1^2 P_dd + b1 b2 P_dd2 + (1/4) b2^2 P_d2d2
  + b1 b_s P_ds2 + (1/2) b2 b_s P_d2s2 + (1/4) b_s^2 P_s2s2
  is CCL's auto-spectrum expression.
- **Terms CCL has that the draft drops**: b1 b3nl P_d1p3 (the draft
  sets b3nl to zero following DESI DR1), and b1 b_k2 k^2 P (the
  draft replaces it with its own counterterm alpha0 k^2 P_lin).
- **The low-k white-noise subtraction** (sub_lowk, default off in
  CCL) removes constants that are degenerate with shot noise; the
  draft should state which setting it used.
- **P_dd in the b1^2 term is halofit** (the hybrid EPT of Nicola et
  al. 2023, arXiv:2307.03226), while the loop templates are one-loop
  SPT.
- **Missing pieces**: stochastic and shot-noise offset terms (the
  DESI full-shape analyses carry them), and redshift-space one-loop
  terms - RSD enters only through the linear Kaiser window plus the
  n(z) FoG convolution, with no mu-dependent counterterms. The draft
  defers these.
- **The tidal-bias column is not an Eulerian b_s** (settled in
  notes/higher_order_bias/higher_order_bias.tex, section "The draft
  paper's bias table: the tidal-bias verdict"). The draft's table
  holds the DESI DR1 sampling combinations (1 + b_1^L) sigma_8(z_eff),
  b_2^L sigma_8^2 and b_s^L sigma_8^2 (L for the Lagrangian basis),
  inserted raw as CCL's Eulerian (b1, b2, b_s). Evidence: the linear
  column divided by sigma_8(z_eff) matches the project's
  B_g/D(z_eff) to 1-5% for LRG1-3 and ELG2; the empty ELG1 row is the
  chunk DESI DR1 excluded from full shape. The BGS linear entry is
  27% above the project's value, pending a read of the public DESI
  DR1 chains. The note gives the conversion recipe, but forecast
  fiducials come from the calibrations (co-evolution
  b_s = -(4/7)(b1 - 1), the Lazeyras b2(b1) fit); DESI posteriors set
  prior widths only, because the converted chain values
  (b_s ~ +4 to +6) are prior- and degeneracy-dominated.
- **The linear-bias model the paper must state** is the per-bin bias
  b_{1,i} = B_g (1 + delta_i)/D(z_i) of the correlated-prior note
  (notes/correlated_bias_prior_v2; the skill's ladder section), not
  one constant set per chunk; the shipped yamls' tied
  b_i = DESI_B1/D(zbar_i) is the ladder's growth-tied prior.

## 5. Higher-order bias

The requirements of this review became models A and B of
notes/higher_order_bias/higher_order_bias.tex, summarized in the
skill's section "Higher-order bias: models A and B" with its
parameter budget. The note's pre-implementation Recommendation and
order of work are kept in `higher_order_bias_recommendation.md`.

## 6. SN_2 breaks model A on the thin layouts

**Result.** A stochastic term SN_2 k^2 mu^2 in the truth, which model
A lacks, biases model A at its own cut. With SN_2 = +-5 units
(model B's prior width; unit f_sat sigma_sat^2/nbar, the skill's
model B section) in every bin, k_max = 0.075 h/Mpc, a model A fit
moves Omega_m by 1.25-1.50 sigma on Delta d_com = 62 Mpc (both
variants; baseline, pessimistic and free priors) and by
1.43-1.68 sigma on Delta z = 0.02 (baseline and pessimistic priors;
on the free prior (w0, wa) moves by 1.16-1.41). Model A's break
point is therefore 0.05 on Delta d_com = 62 Mpc and no fitted cut on
Delta z = 0.02 (EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb, section 7).

**Mechanism: the adjacent-bin cross spectra.** Neighbouring thin
bins share an edge, where the radial derivatives of their kernels
have opposite sign (the falling edge of one bin, the rising edge of
the next). The projected SN_2 template is the integral of
W_i' W_j'/chi^2, so it is anticorrelated between neighbours: on
Delta d_com = 62 Mpc the normalized adjacent term
K2_ij/sqrt(K2_ii K2_jj) has median -0.50 in every sample (range
-0.54 to -0.46), against +0.05 to +0.06 for SN_0. No positive white
noise reproduces that pattern:

- the data-space split of the SN_2 = +5 response puts it in the
  crosses: model A pessimistic on the pessimistic prior, Omega_m
  -1.37 sigma in total, -0.13 from the autos and -1.24 from the
  crosses (62 Mpc); -1.58, -0.22 and -1.36 on Delta z = 0.02;
- one free SN_0 per bin (Gaussian width 100) instead of per group
  does not repair it: on 62 Mpc the shift grows (Omega_m -1.37 ->
  -1.45 sigma, (w0, wa) 1.00 -> 1.15); on Delta z = 0.02 Omega_m
  drops to -0.75 sigma but (w0, wa) is still 1.12.

**Where the signal sits.** About half of each bin's SN_2 auto kernel
comes from k_par above the bin's own cut k_max/D(zbar_i) at 0.075:
0.58 (BGS bin 10), 0.46 (LRG bin 34), 0.51 (ELG bin 63) on
Delta d_com = 62 Mpc; at k_max = 0.2 the fractions are 0.18 / 0.09 /
0.12. The UV-fraction cut is evaluated on the linear-bias spectra,
so it does not see this k_par^2-weighted content.

**By sample.** The LRG term carries the break (alone: Omega_m -1.24
to -1.38 sigma, matched configurations). ELG alone moves Omega_m by
at most 0.23 sigma (A_s by up to 0.31); its unit, 0.84 (Mpc/h)^2, is
a seventh of LRG's 6.08. BGS contributes nothing at this cut (0.00),
with the borrowed LRG unit or with f_sat sigma_chi^2, sigma_chi the
BGS bins' own FoG dispersion in Mpc/h (1.5-1.7 (Mpc/h)^2), so the BGS
unit choice does not matter here.

**The alternatives** (largest score over the fixed truths, sigma/3
threshold):

- without SN_2 in the truth (alpha_2 = +-12.5 and the offsets only):
  model A holds to 0.075 on both thin layouts (Delta z = 0.02 model A
  baseline on the baseline prior excepted: alpha_2 moves A_s by
  0.34 sigma at 0.075), and breaks at 0.1 (62 Mpc: alpha_2 moves A_s
  by 0.37-0.42 sigma, or the offsets Omega_m by 0.37; Delta z = 0.02:
  alpha_2 moves (w0, wa) by 1.48-1.59);
- autos only (all three truths): model A holds to 0.075-0.1, but
  dropping the cross pairs keeps only 18-46% of the FoM(w0, wa) on
  the thin layouts (pessimistic prior: model A 0.18-0.19 at 0.075 and
  0.29-0.36 at 0.15; model B 0.35-0.38 and 0.38-0.46), against
  62-88% on the wide ones;
- model B, which fits SN_2, absorbs it: shifts at most 0.22 sigma for
  SN_2 = +5 (62 Mpc and Delta z = 0.02, at 0.075 and 0.15, both
  variants on their matched priors).

**Open.** How the break is quoted in the notes and the paper (with
SN_2 in the truth, without it, or on autos only) is undecided.

**Provenance.** Measured 2026-11-08 on the caches of the executed
model B notebook (break tables, section 7) and with scratch
diagnostics on the same namespace (the sample split, the BGS unit,
the adjacent cross terms, the per-bin SN_0 control, the auto/cross
split, the k_par content and the variants; not in the repository).
