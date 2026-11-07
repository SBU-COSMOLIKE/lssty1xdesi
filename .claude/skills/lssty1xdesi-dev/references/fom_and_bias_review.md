# The FoM scale and the galaxy bias model: review conclusions

Conclusions of an adversarial physics and code review (2026-11-06) of
the DESI C_ell^gg Fisher forecast - the EXAMPLE_EVALUATE1-4 layouts at
h = 0.02, AccuracyBoost 1 - and of the draft paper's higher-order bias
model. The Fisher numbers below were reproduced from the cached
derivative matrices (the rebuilt F matches the pickles to better than
7e-16) and from counterfactual covariances built on the fiducial data
vector.

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

- (a) linear bias with k_max ~ 0.08-0.12 h/Mpc (~0.055-0.08 /Mpc);
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

## 3. The thin-bin FoM: physical ratio, inflated normalization

Measured FoM(w0, wa), DESI C_gg alone, h = 0.02 (2026-11-06);
Delta d_com = 62 Mpc names the 76-bin layout of equal
comoving width, d_com the comoving distance:

| layout | FoM |
|---|---:|
| Delta z 0.2 (10 bins) | 5.67 |
| Delta z 0.05 (36 bins) | 32.0 |
| Delta z 0.02 (90 bins) | 125.6 |
| Delta d_com = 62 Mpc (76 bins) | 135.2 |

Decomposition from the cached derivatives (same layout order):

- **free bias per bin**: 1.94 / 11.2 / 56.4 / 66.8 - the thin/wide
  ratio survives;
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
  32.9, and at k_max = 0.05 h/Mpc the Delta z = 0.02 layout keeps
  no band at all. The full tables are in the skill's scale-cut
  section.

Interpretation: about 7x from radial modes (k_parallel resolved to
~0.006 /Mpc at Delta z = 0.2 against ~0.06 /Mpc at Delta z = 0.02),
and about 3x from degeneracy breaking by Kaiser RSD and radial BAO,
which wide bins wash out.

The literature says the same (Asorey et al. 2012, arXiv:1207.6487;
Di Dio et al. 2014, arXiv:1308.6186; Camera et al. 2018,
arXiv:1803.10773): many thin bins recover the 3D P(k, mu) information
and saturate when Delta chi ~ 0.6-0.8 lambda_min. The information is
bounded by the 3D Fisher of the same survey at the same cuts, so it
does not grow without limit as Delta z -> 0. Delta d_com = 62 Mpc
(76 bins, FoM 135)
against Delta z 0.02 (90 bins, FoM 126) shows the plateau.

### Why the absolute FoM is inflated

1. **The covariance bias bug** (confirmed in the code, not yet
   fixed). The likelihood uses b_i = DESI_B1 / D(zbar_i)
   (likelihood/desi_cl.py, around lines 166-180), while
   covariance/lssty1xdesi_covariance.py passes the raw CHUNK_BIAS
   (1.34 / 1.70 / 0.84) through `_lens_bias` (around lines 104-111
   and 274-291). The covariance signal is therefore low by D(z)^2 at
   high z, and the FoM is inflated by 2.4-3.2x in every layout. A
   self-consistent covariance gives FoM 2.3 / 13.1 / 51.7 / 42.2.
   The fix: divide by D(zbar_i) in `_lens_bias` and regenerate the
   DESI covariances.
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
   as the entire ignorance budget.

The honest range for DESI C_gg alone is FoM ~ 10-40.

### The LSST comparison

LSST Y1 3x2pt alone has FoM 17.4, consistent with the DESC SRD
(arXiv:1809.01669: Y1 WL+LSS at 22 without priors). The forecast's
"LSST barely helps" conclusion is an artifact of the inflated DESI
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
- **The tidal-bias values look off by convention - needs
  verification, not an established error.** The draft's table (BGS
  b1 1.375, b2 -1.087, b_s +1.423; LRG b_s ~ +0.86 to +1.03; ELG2
  b_s +0.64) has b_s positive and large. In the CCL/FAST-PT
  normalization (delta_g contains (1/2) b_s s^2), the co-evolution
  expectation is b_s = -(4/7)(b1 - 1): about -0.21 for BGS, -0.26
  for LRG1, +0.17 for ELG2. A sign or normalization mismatch in
  converting DESI's full-shape best fits (velocileptors, Lagrangian
  basis) to CCL's Eulerian b_s is likely: Lagrangian to Eulerian is
  b_s^E = b_s^L - (2/7) b1^L in the delta_g-contains-b_s-s^2
  normalization, and a factor of 2 more in CCL's (1/2) b_s s^2. The
  b2 values are plausible (the Lazeyras et al. fit gives about -0.76
  at b1 = 1.375).
- **Paper-pipeline mismatch.** The draft describes one constant bias
  set per DESI chunk; the cosmolike pipeline uses
  b_i = DESI_B1 / D(zbar_i), one amplitude per group evolving as
  1/D(z). The two models must be stated consistently.

## 5. Requirements for implementing higher-order bias

- The b2, b_s (or b_G2), b3nl, counterterm and stochastic parameters
  per tracer chunk, with priors.
- One-loop templates: cosmolike already has C-FAST-PT bias templates
  used by the real-space 3x2pt projects (bias_model
  [b1, b2, bs2, b3, bmag]). The DESI low-rank non-Limber engine
  projects the full P_g through a per-bin SVD of sqrt(P_g), so it
  can carry the one-loop galaxy power.
- A covariance consistent with the extended model.
- EFT-grade scale cuts (Sec. 2).
- The matching Fisher nuisance set.
