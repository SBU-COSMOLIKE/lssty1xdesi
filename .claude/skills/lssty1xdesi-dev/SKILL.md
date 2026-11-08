---
name: lssty1xdesi-dev
description: Development on the lssty1xdesi project - DESI thin-bin spectroscopic angular clustering - the Delta d_com = 62 Mpc layout, 76 bins of equal comoving width, from a DESI DR3-like source forecast - cross LSST-Y1 shear in Fourier band powers. Use this skill for ANY work in projects/lssty1xdesi - the likelihood (desi_cl), the Fingers-of-God model and sigma_v parameters, the n(z) files and their sampling, the low-rank non-Limber engine settings, the gg-cross block, the covariance adapter, the test suite, the data-generation scripts, the DESI DR1 fixed-width Delta z cases (EXAMPLE_EVALUATE2-4), the LSST Y1 3x2pt case (EXAMPLE_EVALUATE5), or the Fisher module and notebooks (EXAMPLE_EVALUATE_FISHER.ipynb, EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb, EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb, EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb), and the galaxy-bias notes under notes/ (correlated_bias_prior_v2, linear_bias_kmax, higher_order_bias, bias_model_summary) with their plots/ folders. Also trigger on mentions of DESI thin bins, nonlimber_accuracyboost, NL_Nchi, DESI_SIGV, gg cross pairs, the dndz FoG convolution, the correlated bias prior, the bias ladder (growth-tied, baseline, pessimistic, free), the memory model, the figure of bias, the statistic sqrt(<chi^2>/n_dof) of the accuracy test, higher-order bias models A and B, the counterterm alpha_2, the SN_2 stochastic term, or where model A breaks. This survey is the numerical extreme of the cosmolike projects, and defaults that are safe elsewhere fail here.
---

# lssty1xdesi: the thin-bin extreme

This project forecasts DESI spectroscopic galaxy clustering cross
LSST Y1 cosmic shear: angular C_ell in thin bins, not a 3D
power-spectrum analysis. The galaxy sample is sliced into 76 thin
bins of equal comoving width, about 62.3 Mpc - the Delta d_com = 62 Mpc
layout, d_com the comoving distance (mean redshifts 0.108 to 1.549; BGS
bins 0-17, LRG 18-48, ELG 49-75, named by
`data/lssty1xdesi_desi_dr3_bins.txt`). The data vector is Fourier
band powers, 53 bands of width Delta ell = 10 over ell = 15 to 544:
15 shear pairs + 76 clustering autos + 97 measured cross-bin pairs =
188 spectra, 9,964 entries, 5,309 unmasked. No galaxy-galaxy lensing
rows are measured (`ggl_exclude: all`). The survey inputs come from
the source forecast's sacc file and DESI text files under the
repository-external `to_reproduce/` folder; `scripts/make_desi_nz.py`
and `scripts/make_desi_dataset.py` convert them.

The bins are the project's defining numerical challenge. Every
setting below exists because a default that photometric surveys never
notice fails here by hundreds to tens of thousands of chi2. Validate
any change against the frozen suite; never transfer a "converged
elsewhere" setting without measuring it on this vector.

## The non-Limber requirement (measured 2026-11-04)

Thin radial kernels keep real non-Limber power far past the ell = 150
hand-off photometric surveys use. The likelihood runs
`adopt_limber_gg: 0` with the low-rank engine
(`nonlimber_lowrank_gg: 1`, rank 5, multipole cap 1000): a per-bin
SVD of sqrt(P_gg) whose rank-one profiles are FFTLog-projected with
no internal Limber hand-off. Against the converged vector, scored
with the project covariance:

- pure Limber everywhere: delta chi2 = 21,158;
- FKEM at its defaults (cap `lmax_nonlimber` 150, cross pairs with
  the FKEM split): 4,243;
- rank 5 -> 8: 0.000; engine cap 1000 -> 2000: 0.002.

The two non-Limber tests in `tests/data_vector/` pin the first two as
drift checks (10% relative), re-measured at every refreeze.

The fair method comparison (paper-grade, 2026-11-04, same chi grid)
puts FKEM at cap 550 - every band - with its cross pairs: delta
chi2 = 74.0 against the engine, the genuine cost of FKEM keeping the
nonlinear density term in Limber. The unfair configurations it
replaces: cap 150 with Limber pairs, 17,388; cap 150 with pairs,
4,243. Never quote the first two as a method comparison.
`tests/data_vector/test_nonlimber_fkem_fair.py` pins the fair value
(through the shared lowrank_gg_suite's `overrides` argument); the
17,388 configuration no longer exists in the code (pre-twin commit
0851ce4).

FKEM with cross pairs is a separate twin in cosmo2D.c
(C_cl_tomo_fkem_pairs_core and C_gg_tomo_fkem_pairs_ells, in the
CROSS-BIN section), dispatched from C_gg_tomo_ells only when a pair
list is installed: every project without pairs runs the original
C_cl_tomo_core, bitwise. The twin pools its storage by 16-multipole
blocks, so a cap of 550 at NL_Nchi = 32,705 fits in about 1 GB (the
original core's layout would need about 55 GB).

## The chi grid: nested refinement and the one-bin cliff

The non-Limber chi grid is endpoint-inclusive log-spaced
(`dlnchi = log(chi_max/chi_min)/(NL_Nchi - 1)`, z = 0.002 to 4). The
shared core refines it by nested interval doubling:
`NL_Nchi - 1 = 511*m`, with m = accuracyboost *
nonlimber_accuracyboost rounded UP to a power of two (so the knob is
effectively dyadic; 3 buys the same grid as 4). Non-nested
ceil-scaling was replaced because consecutive doublings measured
delta chi2 = 1.3, 1.1 of pure node-placement difference.

The Delta d_com = 62 Mpc and Delta z = 0.02 references use
the combined factor 64
(NL_Nchi = 32,705) as `accuracyboost: 2` with
`nonlimber_accuracyboost: 32`. The log spacing makes the
HIGHEST-redshift thin bins the narrowest in ln(chi), and their
FoG-smoothed edges resolve between combined factors 32 and 64: below
that threshold the z = 1.32 ELG bin (bin 68) carries a coherent
+0.18 sigma broadband amplitude bias across all its bands
(delta chi2 = 1.9 with its crosses), constant from 16 to 32, gone at
64. This is a cliff, not smooth convergence - do not infer safety at
intermediate factors from a two-point scan. Factor 64 vs 128 on
Delta d_com = 62 Mpc:
0.0033 at accuracyboost 1, 0.0045 at 2.

The global accuracyboost (cosmolike's tables: the a grid, the P(k) z
nodes, the ell and n(z) sampling) is 2 on all four layouts; CAMB's
own AccuracyBoost stays at the examples' 1.05. Against accuracyboost
5 (Delta d_com = 62 Mpc / Delta z = 0.02, 2026-11-05,
patched CAMB halofit):

| accuracyboost, combined chi-grid factor | Delta d_com = 62 Mpc | dz 0.02 | Delta d_com = 62 Mpc evaluation |
|---|---:|---:|---:|
| 1, 64 | 0.133 | 0.101 | 12.7 s |
| 1, 512 | 0.129 | 0.096 | - |
| 2, 64 (the reference) | 0.017 | 0.033 | 13.7 s |
| 2, 128 | 0.010 | 0.011 | 24.5 s |

The tables carry the error; a finer chi grid alone does not remove
it, and doubling the chi grid doubles the evaluation (4 threads, Apple
laptop on AC power; x86 production timings are a separate
measurement). At the fiducial the wide layouts move by at most 0.053
(Delta z = 0.05) and 0.013 (Delta z = 0.2) at accuracyboost 6, but at
amplitude-cut prior corners boost 1 lets the pushed-settings delta
chi2 of the N-random-models check reach 0.7 (Delta z = 0.05) and 0.9
(Delta z = 0.2, measured with a 10/Mpc power table), so they run
boost 2 as well, with nonlimber_accuracyboost 16 and 4 keeping their
combined chi-grid factors at 32 and 8.

## The Boltzmann k range (measured 2026-11-05)

Every DESI layout sets kmax_boltzmann 20 (`likelihood/desi_cl.yaml`;
the Delta z = 0.02 case inherits it), so CAMB's P(k) table reaches
kmax_boltzmann x accuracyboost = 40/Mpc at the reference boost 2. The
range protects the prior corners, not the fiducial: at a low-Omega_m
point (Omega_m 0.237, w0 -2.39, w0 + wa -2.87, H0 78.6) a 10/Mpc
table moved chi2 by 2.09 (Delta d_com = 62 Mpc) and
2.06 (Delta z = 0.02) through the
extrapolated high-k tail; 40/Mpc brings that to 0.12 and 0.10. The
Delta z = 0.05 and 0.2 layouts carry 20 for the same reason.

EXAMPLE_MCMC1.yaml runs EXAMPLE_EVALUATE1's settings (accuracyboost
2, kmax_boltzmann 20), so chains sample with the settings that
generated the data vector. The EMUL2 examples keep accuracyboost 1
and kmax_boltzmann 5.

## The Fingers-of-God model

A line-of-sight velocity dispersion sigma_v smears observed redshifts
by sigma_z = (1 + zbar) sigma_v / c, and the bins are far narrower
than the smearing's effect, so the convolution is the dominant radial
systematic. `likelihood/desi_cl.py` convolves a COPY of each raw
thin-bin histogram with that Gaussian at EVERY sampled point
(scipy gaussian_filter1d in grid steps, mode "constant") and sends
the result through the external-n(z) path - `external_nz_modeling: 1`
is mandatory, and the per-point n(z) rebuild (which invalidates the
kernel caches each evaluation) is part of the 7.8 s budget. The
source forecast's reference implementation
(dndz_stretch_fog_gaussian) is the same operation; the only
differences are scipy's +-4 sigma kernel truncation (a <= 6e-5 tail
reshaping; pass truncate=8 for exact agreement) and its explicit
post-renormalization, a no-op here because no bin approaches the
grid ends.

The sigma_v are SAMPLED, in seven groups - one more than the six
Table-2 bias chunks, because the simulation-measured sigma_v(zbar)
steps inside BGS at z = 0.255 (between bin means 0.247 and 0.263):
BGS0 = 371 (bins 1-10), BGS1 = 358 (bins 11-18), LRG1-3 =
357/348/343, ELG1-2 = 304/289 km/s. Grouping is pure yaml: the
likelihood declares DESI_B1_<n> and DESI_SIGV_<n> per bin, and
`scripts/make_desi_params.py` writes the fragments that tie group
members to leaders (SIGV leaders 1/11/19/30/39/50/62) with cobaya
value functions. Changing a grouping is a regeneration of
`likelihood/params_desi.yaml`, never code.

Priors are Gaussian, width five times the per-bin error bars of the
simulations' measured sigma_v(zbar) figure (digitized 2026-11-04 with
the y scale calibrated on the seven group means, residuals < 5 km/s):
145/105/70/50/50/40/30 km/s, truncated at zero only (the BGS0
fiducial sits 2.6 prior sigma above zero). A flat [100, 1000] prior
was rejected: the measurement excludes dispersions anywhere near the
flat bounds, and the numerical cost of covering sigma_v = 100
(near-top-hat kernels again) buys nothing physical.

## n(z) sampling: features, not bin counts

`data/lssty1xdesi_desi_dr3.nz` holds 49,996 uniform rows
(dz = 7.0e-5): a nested five-fold linear refinement of the DESI
10^4-point grid (REFINE in `scripts/make_desi_nz.py`; original
samples keep their exact values). The per-bin sample counts of the
raw grid (43-100) already matched a broad photometric bin (lsst_y1:
47-102 lens, 63-237 source), but per-bin counts are the wrong
measure: the sharpest feature here is the FoG-smoothed edge,
sigma_z = 1.3e-3 to 2.4e-3, which had only 4-7 raw samples per sigma
and now has 20-34. Judge any future grid change by samples per
feature. Every bin now holds 219-505 nonzero samples.

The z column holds bin midpoints: `photoz_zmid_convention: 1` is the
project default, and the frozen references carry it - a Z_LOW
reading shifts chi2 by 3.47. The n(z) reconstruction study
(`tests/studies/`) showed the cubic interpolant ringing (negative
n(z)) on the raw top-hat edges; its stored numbers predate the FoG
fiducials and the five-fold grid and await a rerun.

## The covariance

`covariance/lssty1xdesi_covariance.py` is the survey adapter to the
shared covariance package. The Gaussian term runs redshift
distortions in both the Limber window and the non-Limber recompute's
j_l'' transfer (gaussian rsd = nonlimber = true); SSC and cNG stay
Limber at mnu = 0. The adapter bakes the fiducial FoG convolution
into a derived lens n(z) at ingestion (`_fog_convolved_lens_file`,
CHUNK_SIGMA_V with the same seven groups and the BGS split) - the
smearing is this project's systematic, so it lives in the adapter and
the shared covariance code never models FoG. The derived file is
regenerated on every configuration() call and gitignored.

The covariance signal must carry the data vector's bias. `_lens_bias`
gives each bin the likelihood's fiducial b_i = B_g/D(zbar_i), with B_g
the chunk amplitude (CHUNK_BIAS, the bias at z = 0) and D the linear
growth read off the linear P(k) ratio at k = 5e-3/Mpc exactly as
`likelihood/desi_cl.py` does (`_lens_growth`; GROWTH_KREF must equal
the likelihood's KREF). Passing the raw CHUNK_BIAS instead left the
signal low by D(zbar_i)^2 at high redshift and inflated every DESI FoM
by a factor 2.5-3.3 at the shipped mask (2.7-4.3 for linear bias at
k_max = 0.075 h/Mpc). With the per-bin bias the ratio of the
covariance's mean signal to each fiducial data vector is 1.03 on the
kept autos at every redshift (the remaining 3% is the covariance's
massless neutrinos; the raw bias let it fall to ~0.25), and
`_lens_growth` agrees with an independent LCDM growth integral to 2e-4
(measured at the fix, commit e663480). The one-loop terms of model A,
about 2% of the kept spectra at k_max = 0.075 h/Mpc, stay out of the
covariance.

All four DESI cases score the project's own Gaussian covariance:
`scripts/make_desi_covariance.py --case <case>` writes
`data/<stem>_cov` from the Gaussian term of the case's archive
(`covariance/production_covariance[_<case>].npz`). The source
forecast's sacc covariance is kept as
`data/lssty1xdesi_desi_dr3_sacc_cov`, for comparison only.

Validation against that supplied covariance: the Knox formula
with the project's own band values closes the bookkeeping exactly
(first clustering rows 1.007/1.018), and the pre-FoG-kernel
comparison measured sigma ratios with medians 1.218/1.228 (both
totals positive definite); both predate the per-bin bias, and the
comparison is re-measured after any kernel or fiducial change. One
qualitative difference is physical, not a bug: the source forecast's
adjacent-bin cross bandpowers turn NEGATIVE at high multipole (FoG
smearing anticorrelates neighboring thin bins),
while the project's geometric-mean cross spectra are nonnegative by
construction. `covariance/default.yaml` still runs its own
nonlimber_accuracyboost at 1 and has not been retuned to the adopted
data-vector state.

## Covariance files: Gaussian only, and the G + SSC plan

The DESI DR1 fixed-width Delta z layouts (cases dr1_dz0p20,
dr1_dz0p05, dr1_dz0p02: 10, 36 and 90 lens bins, built by
scripts/make_desi_dz_case.py) and the Delta d_com = 62 Mpc case
(dr3) all ship the **Gaussian term only** of the project's own
forecast (compute_covariance.py with `case:` in the covariance yaml,
then scripts/make_desi_covariance.py). The CLI
also computes SSC and cNG and keeps them in the archive
(covariance/production_covariance_<case>.npz).

Measured 2026-11-04, before the per-bin bias fix, on the unmasked
entries (error-bar inflation on the diagonal):

| case | SSC over G: median / max | cNG over G+SSC: max |
|---|---|---|
| dr1_dz0p20 | 1.0000 / 1.0013 | 1.016 |
| dr1_dz0p05 | 1.0000 / 1.0020 | 1.076 |

SSC moves almost no error bar; what it adds is a coherent,
amplitude-like correlation across bands that the free per-chunk
biases largely absorb. cNG is the larger term and the least reliable
one for spectroscopic tracers (a halo-model trispectrum), so it stays
out of every likelihood covariance.

Plan for the final chains (before the paper): switch the case
covariance files to **G + SSC, never cNG** - a one-line change in the
builder's covariance stage (`gaussian` -> `gaussian + ssc`). SSC
makes the matrix dense, and the cosmolike reader takes text, so the
files grow:

| case | entries, G -> G+SSC | text | xz -9 |
|---|---|---:|---:|
| dr1_dz0p05 | 106,816 -> 1,419,284 | 42 MB | 10.9 MB |
| dr1_dz0p02 | 686,774 -> about 29 M | about 865 MB | about 225 MB |

(dr1_dz0p02 scaled by its entry count; xz compresses this text about
3.9x, decompresses in about 15 s, compresses once in about 10 min.)

The storage design for that switch:

- ship `<stem>_cov.xz` in the repository (Git LFS) and gitignore the
  decompressed `<stem>_cov`;
- decompress in the project start script (start_lssty1xdesi.sh,
  sourced by start_cocoa.sh), only when the plain file is missing or
  older than the archive: about 15 s once, nothing afterwards;
- never decompress inside the likelihood: every MPI rank of a chain
  would race to write the same file. When the plain file is missing,
  the likelihood fails with a message naming the fix.

A smaller alternative if size ever matters more: SSC is low rank (in
dr1_dz0p05, 83 of 2,982 eigenmodes exceed 1e-6 of the largest), so it
could ship as a few hundred eigenvectors - a few MB even for
dr1_dz0p02 - but that needs the covariance reader to accept a
low-rank term, a change of its own.

## The LSST Y1 3x2pt case and the Fisher module

`EXAMPLE_EVALUATE5.yaml` (case `lsst_y1_3x2pt`, built by
`scripts/make_lsst_3x2pt_case.py`) is LSST Y1 alone in Fourier band
powers on the 53 DESI bands, so its Fisher matrix sums with a DESI
case's. Layout EE 795 + gs 1,325 (all 25 pairs) + gg 265 = 2,385
entries, 1,629 unmasked: EE keeps every band, gs and gg the lens
bin's k_max cut. The n(z) are lsst_y1's, read Z_LOW
(`photoz_zmid_convention: 0`) although their first z value is a bin
midpoint - the donor's convention, kept so the case is "the original
lsst_y1 n(z)". The DESI cases read their files Z_MID; mixing the two
conventions in one joint lens sample needs a decision.

Its Gaussian covariance (12,300 deg^2) carries NLA. The covariance
takes one constant A1 per source bin, so the builder evaluates the
fiducial power law A1 ((1+z)/1.62)^eta1 at each source bin's mean
redshift (1 + z0 = `nuisance.oneplusz0_ia` = 1.62). The TATT amplitudes
are fixed parameters of params_source.yaml and must stay out of the
evaluate override (cobaya rejects a fixed parameter there).

`interface/cosmolike_lssty1xdesi_fisher.py` (`ModelFisher`) builds the
Fisher matrix on the cobaya model of an EXAMPLE yaml. Design rules,
each from a failure it prevents:

- **Evaluate with `Model.loglike`, never `logposterior`.** Outside a
  flat prior `logposterior` returns -inf without running the
  likelihood, and the interface keeps the previous point's vector: a
  silently wrong derivative. The ELG bias fiducial (0.84) sits next to
  its prior edge (0.8), so the h = 0.05 stencil leaves the prior. A
  non-finite loglike raises.
- **`cached=False`.** A cached point returns without recomputing, and
  the interface vector would again be stale. The price is a CAMB run
  for every nuisance step.
- **Zero-fiducial steps.** cnu.get_ddv falls back to an absolute step h
  for a zero fiducial: 10 sigma for a source photo-z shift at
  h = 0.02. The module steps such a parameter by h sigma_prior/0.02
  (equal to the prior width at h = 0.02) and by h when its prior is
  flat (wa).
- **One active model.** The interface holds one dataset; `dv` and
  `invcov` refuse once another ModelFisher was built. Compute each
  probe's derivatives, Fisher matrices and bias numerators right after
  building its reference model, alternatives first.
- **Area.** DESI case covariances are built at 14,000 deg^2; the
  common-area forecast multiplies their inverse by 12,300/14,000 (exact
  for a Gaussian covariance).
- **Cache derivatives, never Fisher matrices.** A covariance change
  must not need new derivatives. EXAMPLE_EVALUATE_FISHER.ipynb keeps
  only the derivatives of its caches (`chains/EXAMPLE_EVALUATE<n>_FISHER_AB<b>.pkl`)
  and rebuilds F, the bias numerators and the Delta chi2 from the
  covariance file at every run (`refresh_fisher`), with the Gaussian
  priors read from the example yaml (cobaya `update_info`); the F
  stored in those caches predates the covariance fix and is unused
  (the printed check reads 0.76 for DESI, 1.8e-14 for LSST). The
  scale-cut FoM cache stores each layout's covariance sha256 and
  recomputes on a change. EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb
  reads its priors from the yamls the same way.
- **LSST Y1 3x2pt + DESI C_gg = block-diagonal sum.** `combine` adds Fisher matrices by
  parameter name; no prior is shared between the probes, so none is
  double-counted. The joint likelihood for chains still needs the
  concatenated LSST+DESI lens sample.
- **Smooth vectors first.** A stencil derivative is only as good as the
  smoothness of the vector in the parameters. Diagnostic: scan one
  parameter on 17 fine points, whiten by the Cholesky factor of C^-1,
  fit a quartic, read the residual per point in sqrt(chi2) units
  (measured 2026-11-04 with the pre-fix DESI covariance, Delta z = 0.2
  DESI and the LSST case):

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
  With stock halofit at h = 0.01-0.05 the FoM(w0, wa) moved 22.7 /
  7.3 / 5.8 / 5.1 (DESI, pre-fix covariance) and 52 / 14 / 24 / 19
  (LSST). CAMB is now
  installed with `PATCH_CAMB_HALOFIT_TOLERANCE` (1e-7), and the
  notebook sets `nonlimber_accuracyboost: 64` for every DESI Fisher
  evaluation.
- **Step and accuracy stability with both fixes** (derivatives
  measured 2026-11-05, Fisher matrices rebuilt with the corrected
  covariance; FoM(w0, wa) at h = 0.01 / 0.02 / 0.03 / 0.05, Delta z = 0.2):

  | probe | AB 1 | AB 3 |
  |---|---|---|
  | DESI C_gg | 2.53 / 2.20 / 2.18 / 2.24 | 2.23 / 2.22 / 2.22 / 2.22 |
  | LSST 3x2pt | 17.68 / 17.67 / 17.65 / 17.68 | 17.36 at every step |
  | LSST Y1 3x2pt + DESI C_gg | 24.9 / 24.2 / 24.2 / 24.3 | 24.09 / 24.08 / 24.08 / 24.08 |

  Cost (M2 Pro, AC, 8 threads, quiet, 2026-11-05): one evaluation is
  3.8 s / 14.3 s (DESI at NL 64, AB 1 / 3) and 0.8 s / 2.4 s (LSST);
  a step costs 4 evaluations per parameter (20 DESI, 29 LSST), so the
  notebook's 4 steps x 2 boosts take about two hours from scratch.
  At AB 3 every step agrees to 0.6% (marginalized sigmas likewise).
  At AB 1 the DESI h = 0.01 derivative is still noise-limited. AB 1
  -> 3 at fixed h moves DESI by +1% and LSST by -1.8%: an accuracy
  shift, not noise. Whether AB 3 is converged needs an AB 5 run.
- **Fisher bias.** `bias_vector` (D^T C^-1 delta) and `fisher_bias`
  (F^-1 B, block numerators summed). Alternatives are likelihood
  overrides on the same yaml: FKEM = `nonlimber_lowrank_gg: 0`
  (+ `lmax_nonlimber: 550` for the fair cap), Limber =
  `adopt_limber_gg: 1` (+ `adopt_limber_gs: 1` for LSST).
  Measured at AB 3, h = 0.02 (max |delta p/sigma| over the seven
  cosmological parameters; stable to ~1% over h):

  | alternative | probe | max abs(delta p/sigma) | Delta chi2 |
  |---|---|---:|---:|
  | FKEM, cap 550 | DESI C_gg | 0.009 (Omega_m) | 0.039 |
  | Limber | DESI C_gg | 3.1 (n_s) | 85.0 |
  | SVD engine | LSST 3x2pt | 0.0024 | 1.4e-4 |
  | Limber | LSST 3x2pt | 0.40 (A_s) | 11.2 |
  | DESI FKEM, cap 550 | LSST Y1 3x2pt + DESI C_gg | 0.008 | 0.039 |
  | DESI Limber | LSST Y1 3x2pt + DESI C_gg | 0.98 (w_a) | 85.0 |
  | Limber in both | LSST Y1 3x2pt + DESI C_gg | 1.01 (w_a) | 96.2 |

  These Delta chi2 use the corrected covariance rescaled to
  12,300 deg^2 (x 0.879), at `nonlimber_accuracyboost: 64` and AB 3.
  The test pins (`DCHI2_MEASURED_CASES`, 0.132 for Delta z = 0.2)
  score the frozen pre-fix covariance at the likelihood's chi grid, so
  the two are not comparable.
- **Per-layout Fisher biases.** The notebook repeats the two DESI
  alternatives on every layout: each alternative vector is evaluated
  at the layout's own overrides and AB 1 and cached in
  `chains/EXAMPLE_EVALUATE_FISHER_ALTERNATIVES.pkl`, and one bar
  figure draws the shifts of the four layouts side by side. Measured
  at h = 0.02 (max |delta p/sigma| over the seven cosmological
  parameters, DESI C_gg alone / LSST Y1 3x2pt + DESI C_gg; Delta chi2
  at 12,300 deg^2, FKEM / Limber):

  | layout | FKEM, cap 550 | Limber | Delta chi2 |
  |---|---|---|---|
  | Delta d_com = 62 Mpc | 0.16 (w_a) / 0.49 (H_0) | 11 (Omega_b) / 12 (Omega_b) | 19.3 / 6548 |
  | Delta z = 0.02 | 0.30 (Omega_b) / 0.63 (H_0) | 27 (Omega_b) / 17 (Omega_b) | 31.7 / 1.49e4 |
  | Delta z = 0.05 | 0.08 / 0.14 | 7.2 (Omega_b) / 3.0 (A_s) | 1.51 / 1305 |
  | Delta z = 0.2 | 0.009 / 0.008 | 3.1 (n_s) / 0.98 (w_a) | 0.039 / 85.0 |

  FKEM's Limber treatment of the nonlinear correction is harmless on
  the wide bins but biases the thin layouts by a sizable fraction of
  sigma; Limber is excluded for DESI clustering on every layout.

## The FoM scale and the bias model (review conclusions, 2026-11-06)

An adversarial physics and code review of the DESI C_ell^gg Fisher
forecast fixed conclusions that bind every future headline number and
the higher-order bias work. Full detail - measurement tables, the
CCL term-by-term check and the arXiv citations - lives in
`references/fom_and_bias_review.md`. The load-bearing conclusions:

- **The thin/wide FoM ratio is physical** (19-23x at the shipped
  mask with the corrected covariance). In the review's decomposition
  (pre-fix covariance, ratio 24x) about 7x is radial
  mode counting (k_parallel resolved to ~0.006 /Mpc at Delta z = 0.2
  against ~0.06 /Mpc at 0.02) and about 3x is degeneracy breaking by
  Kaiser RSD and radial BAO, which wide bins wash out. The ratio
  survives a free bias per bin, an autos-only vector, a
  self-consistent covariance and an ell <= 100 cut, and it saturates
  near Delta chi ~ 0.6-0.8 lambda_min (Asorey et al. 1207.6487),
  where the Delta d_com = 62 Mpc layout already sits.
- **The absolute FoM is set by the covariance and the bias model.**
  The covariance now carries the likelihood's per-bin bias (section
  "The covariance"): FoM(w0, wa) of DESI C_gg alone at the shipped
  mask is 41.2 / 50.2 / 12.6 / 2.22 (Delta d_com = 62 Mpc, Delta z =
  0.02, 0.05, 0.2), as the review's self-consistent counterfactual
  predicted (42.2 / 51.7 / 13.1 / 2.3). The rest is model rigidity:
  linear bias + halofit to the shipped k = 0.1185 /Mpc (~0.18 h/Mpc)
  cut with six constant amplitudes and seven sigma_v groups as the
  entire ignorance budget. Measured on this vector, linear bias holds
  only to k_max = 0.05 h/Mpc (`notes/linear_bias_kmax`), so the
  forecast is one-loop: model A at 0.075 and model B against k_max
  (sections below). The
  "DR1" label is also wrong: the density is the completed survey
  (~28M tracers), not DR1 (~6M).
- **"LSST barely helps" was an artifact** of the inflated DESI side:
  LSST Y1 3x2pt alone (FoM 17.4) matches the DESC SRD (1809.01669),
  and DESI C_gg alone lies below it under model A at 0.075 and under
  model B pessimistic on the pessimistic prior up to k_max = 0.25.

The bias model that answers this review - a per-bin linear bias
reported as a ladder of stated priors, and the one-loop models A and
B built on it - has its own sections below. The draft paper's EPT
expression matches CCL's EulerianPTCalculator term by term; its
tidal-bias column is the DESI DR1 sampling combination, not an
Eulerian b_s, and is never imported as a fiducial (reference file,
Sec. 4).

## The non-Limber UV-fraction scale cuts

The Fisher notebook's scale cuts (section 8) are the non-Limber
UV-fraction criterion, following arXiv:2603.13031; they replace the
Limber relation ell <= k_max chi(zbar) - 1/2. Each C_ell^gg band
power splits into C_IR + C_UV: C_IR is computed with the 3D power
spectrum multiplied by a step that keeps k <= k_max/D(z), with D the
linear growth factor (D(0) = 1) and k_max quoted in h/Mpc at z = 0,
so the cut grows with redshift as the nonlinear scale does; C_UV is
the rest. A band is kept while its UV fraction
|C_UV| / (|C_UV| + |C_IR|) stays below alpha = 0.05 (the paper's
baseline; it also uses 0.01). Kept bands form a prefix in ell, every
band up to the first failure; where the fraction is non-monotonic in
k_max, each prefix is shortened to that of the next larger k_max, so
the cut only tightens as k_max falls. The fit uses (shipped mask) AND
(UV-fraction cut): the covariance files have entries only inside the
shipped masks, which stop every bin at the fixed k = 0.1185/Mpc
(~0.176 h/Mpc) with no growth scaling, so they cap every band count
and every FoM curve.

The step multiplies P(k) inside every gg engine of the compiled code,
so the criterion never assumes the Limber relation:

- Limber sums: per line-of-sight node, at that node's own k_max/D(a);
- non-Limber sums (the low-rank engine and FKEM): in the final sum
  over the FFTLog wavenumbers, one threshold per bin (per pair for
  the cross spectra), with D at the bin's mean redshift.

The interface exposes `gg_scale_cuts_tomo(kmax, alpha)` - a dict with
`lmax`, `mask`, `uv_fraction`, `full` and `ir`, rows the autos then
the cross pairs in data-vector order - and
`C_gg_kwindow_tomo(kmax, keep_ir)` for the IR or UV spectra directly.
The notebook caches the cuts in
`chains/EXAMPLE_EVALUATE_FISHER_SCALE_CUTS.pkl` and the FoM scan in
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

Band powers kept (shipped mask AND UV-fraction cut, alpha = 0.05;
each cell autos / cross pairs; k_max in h/Mpc at z = 0):

| k_max | Delta d_com = 62 Mpc | Delta z = 0.02 | Delta z = 0.05 | Delta z = 0.2 |
|---|---|---|---|---|
| 0.05 | 167 / 74 | 0 / 0 | 367 / 70 | 122 / 44 |
| 0.075 | 903 / 750 | 904 / 952 | 663 / 184 | 198 / 82 |
| 0.1 | 1585 / 1124 | 1928 / 2471 | 945 / 269 | 267 / 111 |
| 0.125 | 2049 / 1432 | 2755 / 3062 | 1148 / 522 | 307 / 138 |
| 0.15 | 2220 / 2038 | 2982 / 3118 | 1195 / 623 | 319 / 156 |
| 0.2 | 2276 / 2889 | 3016 / 4496 | 1205 / 1240 | 325 / 306 |
| 0.25 | 2276 / 3025 | 3016 / 4605 | 1205 / 1560 | 327 / 415 |
| shipped | 2276 / 3033 | 3016 / 4605 | 1205 / 1777 | 329 / 415 |

FoM(w0, wa) of DESI C_ell^gg alone (growth-tied linear bias), the
Fisher matrix rebuilt from the cached derivatives on the kept block of
the corrected covariance. A galaxy-bias parameter whose bins lost
every band has a flat prior and no data, so it is dropped without
touching the cosmology; the test is a relative tolerance, because
derivative round-off on such a parameter once made the
Delta d_com = 62 Mpc FoM at 0.05 infinite. With no band
power left at all there is no w0-wa constraint, and the FoM is
reported as 0:

| k_max | Delta d_com = 62 Mpc | Delta z = 0.02 | Delta z = 0.05 | Delta z = 0.2 |
|---|---:|---:|---:|---:|
| 0.05 | 0.358 | 0 (no band left) | 0.592 | 0.205 |
| 0.075 | 7.61 | 7.26 | 2.77 | 0.646 |
| 0.1 | 15.0 | 23.1 | 5.25 | 1.15 |
| 0.125 | 24.3 | 35.4 | 8.64 | 1.50 |
| 0.15 | 33.4 | 44.0 | 10.8 | 1.82 |
| 0.2 | 39.5 | 49.6 | 12.1 | 2.02 |
| 0.25 | 41.1 | 50.2 | 12.4 | 2.17 |
| shipped mask | 41.2 | 50.2 | 12.6 | 2.22 |

Conclusions:

- A thin bin's C_ell collects radial modes up to k_parallel ~
  2 pi / Delta chi, so its UV part is large even at the lowest
  bands. The criterion sees those modes because it is a step on
  P(k); a cut built on the Limber relation counts only transverse
  modes and is badly optimistic on the thin layouts.
- The cross spectra of adjacent thin bins come mostly from close
  pairs across the shared edge - small radial scales - and are
  strongly UV-dominated.
- At k_max = 0.05 h/Mpc, Delta z = 0.02 keeps nothing, and the
  Delta d_com = 62 Mpc layout keeps only the bins above z ~ 0.9, where 1/D(z) raises the cut
  the most.
- These are linear-bias FoMs. A linear-bias fit is unbiased only at
  k_max = 0.05 (`notes/linear_bias_kmax`), so the forecast numbers
  are those of models A and B below.

## The bias notes are the physics reference

The physics of the galaxy-bias model lives in four LaTeX notes under
`notes/`:

- `correlated_bias_prior_v2`, the forecast note: the per-bin
  linear-bias priors and their derivations, the prior ladder and
  accuracy test under model A, and models A and B against k_max (its
  results are figures, no tables). There is no v3: model B results go
  into this note.
- `linear_bias_kmax`: up to which k_max a linear-bias fit holds, and
  the accuracy test of the linear-bias priors for Omega_m and A_s.
- `higher_order_bias`: builds models A and B. Its Recommendation
  section predates the implementation; that text is kept in
  `references/higher_order_bias_recommendation.md`.
- `bias_model_summary`: written last, from the other three, as a
  presentable summary carrying their key figures (not a compact
  digest).

For any physics question - a
derivation, an equation, an assumption, a citation, a literature
value - read the `.tex` source, not the PDF and not this skill; find
the part with `grep -n '\\section\|\\subsection' <note>.tex` (line
numbers drift). Cite a note's section by its title, never by an
appendix letter. This skill keeps decisions, load-bearing numbers and
code pointers only; when it and a note disagree on physics, the note
is right and the skill is corrected.

| question | note: section |
|---|---|
| the model A forecast on one page | correlated_bias_prior_v2: Summary |
| the forecast setup; the prior; model A; the accuracy test; the FoM | correlated_bias_prior_v2: The forecast; The linear-bias prior; Model A; The accuracy test; The figure of merit |
| the models; physical L and sigma_g (continuity, selection, measurements) | correlated_bias_prior_v2: The models; Which L and sigma_g are physical (subsection What the measurements say) |
| memory model; shift formula, random truths, sqrt(<chi^2>/n_dof) | correlated_bias_prior_v2: The memory model; appendix The accuracy test |
| projection algebra | correlated_bias_prior_v2: The projected precision for two bins; Many bins: the tridiagonal inverse; The memory projection, step by step |
| b - 1 proportional to 1/D, BGS flux limit and its omissions | correlated_bias_prior_v2: The bias continuity equation, step by step; The BGS flux-limit bias model, step by step |
| model A's derivatives, checks, counterterm, open issues | correlated_bias_prior_v2: Implementation and the Fisher construction; Numerical checks; The counterterm requires the low-rank engine; Open issues |
| the identities checked numerically | correlated_bias_prior_v2: Numerical verification |
| linear bias against k_max; the linear-bias ladder for Omega_m and A_s | linear_bias_kmax: Summary; Method |
| one-loop P_gg, bases and conversions, DESI DR1 full-shape priors | higher_order_bias: The one-loop galaxy power spectrum; The DESI DR1 full-shape model and priors; appendix Bases and conversions, with every factor |
| the draft paper's bias table | higher_order_bias: The draft paper's bias table: the tidal-bias verdict |
| projection to thin bins (stochastic terms included), term sizes, the higher-order ladder | higher_order_bias: Projection to thin-bin angular spectra; The size of every term; Bias evolution within a group |
| code inventory against model A | higher_order_bias: Implementation in CoCoA |

`notes/correlated_bias_prior_v2/verify_prior.py` checks the
identities of the correlated-prior note (23 PASS lines). Recipes and
the notebook cells behind the numbers below:
`references/bias_notes_workflow.md`.

Naming in every note, notebook markdown and README: the four
linear-bias priors are growth-tied, baseline, pessimistic and free;
the code keeps `'tied'` and `'independent'` for the first and last,
and calls a prior a rung (`LADDER_RUNGS`, `rung_columns`,
`rung_system`), a word kept out of reader-facing text. The accuracy
statistic sqrt(<chi^2>/n_dof) has no name: write the symbol, and
explain it at its first use in each document (chi^2 the squared
distance of the recovered parameters from the truth in units of the
fit's marginalized covariance, n_dof = 2 for (w0, wa) and 1 for
Omega_m or A_s; 1 means the error bars honestly cover where the fit
lands). The code stores it under the key `'error_inflation'`.

## The linear-bias ladder

The shipped yamls tie the bins of a tracer group g to one amplitude,
b_i = B_g/D(z_i) (B_g the group's DESI_B1, D the linear growth factor,
D(0) = 1). That shape keeps b D constant; a conserved population obeys
b - 1 proportional to 1/D instead (Fry 1996). The forecast states
four assumptions on the per-bin linear bias, all cases of one per-bin
model, the memory model:

    b_{1,i} = B_g (1 + delta_i) / D(z_i),   delta_i = s_g (z_i - zbar_g) + eps_i,
    <eps_i eps_j> = sigma_g^2 exp(-|z_i - z_j| / L)   (same group; 0 across),

with s_g a free tilt per group (flat, like B_g), zbar_g the group's
mean bin redshift, eps_i the residual around the group's line,
sigma_g its per-bin prior width and L its correlation length in
redshift. The Fisher prior per group is the projected precision
P = Q - Q U (U^T Q U)^-1 U^T Q (Q the inverse kernel, U = [1, z - zbar]):
it charges the residual's shape, never the level or the line, and is
zero for groups of one or two bins.

| prior | definition | stated source |
|---|---|---|
| growth-tied | b_i = B_g/D(z_i) | the upper FoM limit |
| baseline | sigma_g = 3% BGS, 1% other groups; L = 0.2 | BGS: the r < 19.5 flux limit drifts the bias 22% across the group and a line leaves 2.4% rms; 1% hedges unmodeled selection curvature, ~10x the <= 0.16% a conserved population leaves |
| pessimistic | sigma_g = 3% every group; L = 0.05 | local features of the largest measured size on the scale of the 4000 A break crossing a filter edge (Delta z ~ 0.03-0.05) |
| free | one flat bias per bin | the lower limit |

The four priors are the whole ladder (no growth-tied-plus-tilt
prior). Without mock catalogs no single prior is defensible, so the
paper shows how the FoM depends on the assumption between the two
absolute limits: a referee who rejects the baseline moves one step
down, and nothing is retracted.

The numbers of this section are linear bias, the reference that
built the mechanics; `notes/linear_bias_kmax` shows that a linear fit
is already biased at this cut (Omega_m moves by 1.5-1.8 sigma on
Delta d_com = 62 Mpc under every prior; the fit is safe only at
k_max = 0.05, and for every prior and truth only on that layout,
keeping 1-6% of the FoM), so the forecast is the
ladder under model A (next section). FoM(w0, wa) with linear bias,
DESI C_gg alone, k_max = 0.075 h/Mpc (UV-fraction cut; LSST Y1 3x2pt
alone: 17.26; on Delta z = 0.2 every group has one or two bins, so
the lower three priors coincide):

| layout | growth-tied | baseline | pessimistic | free |
|---|---:|---:|---:|---:|
| Delta d_com = 62 Mpc | 7.61 | 5.10 | 3.37 | 2.78 |
| Delta z = 0.02 | 7.26 | 4.66 | 3.57 | 1.65 |
| Delta z = 0.05 | 2.77 | 0.712 | 0.492 | 0.411 |
| Delta z = 0.2 | 0.646 | 0.176 | 0.176 | 0.176 |

- **Roughness.** Smooth deviations constrain only sigma_g/sqrt(L),
  so scans use x = sigma_g sqrt(0.1/L), the rms change of delta
  between bins Delta z = 0.05 apart; the memory model collapses in L
  at fixed x. The priors sit at x = 0.71% (baseline, non-BGS), 2.1%
  (baseline BGS) and 4.2% (pessimistic).
- **Figure of bias** Delta_w = sqrt(dw^T C_w^-1 dw): the linear-response
  (w0, wa) shift dw of a fit to biased data, in units of the fit's
  own marginalized ellipse C_w; threshold 1/3. A conserved-population
  drift moves the growth-tied fit by 8.0 / 7.2 / 11.6 / 5.0 sigma
  (table order); baseline and pessimistic stay at or below 0.11 for
  both known drifts, except the BGS flux limit on Delta z = 0.05
  (0.31 baseline, 0.35 pessimistic).
- **Random truths and sqrt(<chi^2>/n_dof).** 2000 truths per case,
  each the known drift (BGS flux limit in BGS, conserved population
  elsewhere) plus a residual drawn from the baseline or the
  pessimistic kernel, fitted under every prior. The verdict is
  sqrt(<chi^2>/n_dof), n_dof = 2 for (w0, wa), chi^2 the squared
  (w0, wa) distance from the truth in ellipse units with data noise
  and sigma_v priors included: 1 for an honest ellipse, above 1 the
  factor the axes are too small by (the FoM is too large by its
  square). With linear bias:

| truth scatter | growth-tied | baseline | pessimistic | free |
|---|---|---|---|---|
| baseline | 4.75-9.86 | 1.00-1.01 | 0.936 / 0.910 / 0.980 / 1.00 | 1.00 |
| pessimistic | 5.18-10.2 | 1.62 / 1.24 / 1.65 / 1.00 | 1.00-1.03 | 1.00 |

  The growth-tied prior is excluded on accuracy (4.7-10x too
  optimistic). The baseline is honest under its own scatter but
  1.2-1.7x too small if local features reach the pessimistic size, so
  the baseline FoM rests on that assumption and the pessimistic FoM is
  the safe statement; the pessimistic prior is honest or at most 10%
  conservative. Matched cases give scatter + noise = 2 exactly
  (P Sigma P = P); the drift term leaves
  sqrt(1 + Delta_drift^2/2) <= 1.03. Scored for Omega_m and A_s (one
  parameter, n_dof = 1) the verdict is the same
  (`notes/linear_bias_kmax`, figure ladder_bias_params).
- **Limits.** First order in delta (a few percent of the shift for
  1-3% residuals, perhaps 10-20% for the 22% BGS drift).

## Higher-order bias: models A and B

Both models are one-loop: the full real-space Eulerian set (b_1, b_2,
b_{s^2}, b_3nl), a k^2 counterterm alpha_0 and a stochastic amplitude
SN_0 per group, on the per-bin b_{1,i} of the ladder. The higher-order
biases follow the full b_{1,i}, deviations included:

    b_{2,i} = b_2^fit(b_{1,i}) + Delta_{2,g},   b_{s^2,i} = -(4/7) (b_{1,i} - 1) + Delta_{s,g},

with b_2^fit the Lazeyras et al. relation (arXiv:1511.01096), the
co-evolution b_{s^2} (CCL/FAST-PT normalization), free offsets
Delta_{2,g} and Delta_{s,g} per group, and b_3nl fixed.

- **Model A, k_max = 0.075 h/Mpc** (matched to the thin-bin radial
  resolution): linear Kaiser RSD and the sampled sigma_v groups, one
  alpha_0 and one SN_0 per group. Gaussian priors at zero: 1.0 on
  Delta_2, 0.5 on Delta_s, 0.5/k_max^2 ~ 89 h^-2 Mpc^2 on alpha_0,
  twice the Poisson power on SN_0.
- **Model B, scanned over k_max = 0.05-0.25 h/Mpc**: model A's
  linear sector with the offset priors widened to 2.0 (Delta_2) and
  1.0 (Delta_s), plus per group the redshift-space counterterm
  alpha_2 and the stochastic term SN_2 k^2 mu^2 (mu the cosine to the
  line of sight). Gaussian priors at zero: 0.5/k_max^2 on alpha_0 and
  alpha_2 (DESI's 12.5 h^-2 Mpc^2 at 0.2), 5 units on SN_2, 2 on SN_0.
  No redshift-space one-loop terms: CosmoLike has none.
- **Ladder.** Baseline: one offset per group. Pessimistic: offset plus
  tilt, Delta_g -> Delta_g + t_g (z_i - zbar_g). Only b_2, b_{s^2},
  alpha_0 and alpha_2 tilt (they follow the host-halo population);
  SN_0 and SN_2 never do (the Poisson 1/nbar, nbar the tracer density,
  is known per bin). Free b_2, b_{s^2} per bin (152 parameters) is a
  bound, not a model: a bin's loop terms are degenerate with its b_1
  and counterterm.

| bias-sector parameters | A baseline | A pessimistic | B baseline | B pessimistic |
|---|---:|---:|---:|---:|
| linear tilts s_g | 6 | 6 | 6 | 6 |
| residuals held by the kernel prior | 64 | 64 | 64 | 64 |
| Delta_2, Delta_s | 12 | 24 | 12 | 24 |
| alpha_0 / alpha_2 | 6 / - | 12 / - | 6 / 6 | 12 / 12 |
| SN_0 / SN_2 | 6 / - | 6 / - | 6 / 6 | 6 / 6 |
| new parameters | 94 | 112 | 106 | 130 |
| without a tight prior, incl. the 6 B_g | 36 | 54 | 48 | 72 |

DESI DR1 full shape uses 7 nuisances per chunk, 42 in total at
k_max = 0.2 h/Mpc; model B baseline is that budget plus one b_1 tilt
per chunk. Fiducials come from the calibrations, never from DESI
chain values.

Model A runs end to end (higher_order_bias, "Gaps against model A"):
`desi_cl` option `higher_order_bias: 1`, the example yamls
EXAMPLE_EVALUATE_MODEL_A1-A4.yaml, and
EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb, which builds the Fisher
matrices of the correlated-prior note on the band powers that
EXAMPLE_EVALUATE_FISHER.ipynb keeps at k_max = 0.075 h/Mpc. The note quotes two
configurations (DESI C_gg alone, layout order
Delta d_com = 62 Mpc, Delta z = 0.02, 0.05, 0.2; LSST Y1 3x2pt alone:
17.26 and 214.9):

| configuration | FoM(w0, wa) | FoM(Omega_m, 10^9 A_s) | stated assumption |
|---|---|---|---|
| model A baseline on the baseline prior | 4.50 / 4.32 / 0.523 / 0.0881 | 11.1 / 9.26 / 3.12 / 0.713 | residuals no larger or rougher than the baseline prior; calibrations exact up to one offset per group |
| model A pessimistic on the pessimistic prior | 2.75 / 2.88 / 0.270 / 0.0535 | 9.07 / 7.02 / 1.78 / 0.456 | honest under both truth families; absorbs a linear calibration drift exactly |

The growth-tied prior is excluded under both variants:
sqrt(<chi^2>/n_dof) of (w0, wa) is 3.3-6.5 under model A baseline and
1.4-2.3 under model A pessimistic. The baseline prior gives 1.00-1.01
under its own scatter and 1.24-1.70 under pessimistic scatter, where
its Omega_m error bar covers the truth in 40-63% of universes instead
of 68% (Delta z = 0.2 excepted, where the priors coincide). A linear
drift of the calibration offsets shifts (w0, wa) under model A
baseline by 0.13-0.27 sigma on Delta d_com = 62 Mpc, at most 0.08 on
Delta z = 0.02 and 0.37-0.40 on Delta z = 0.05 and 0.2 (baseline,
pessimistic and free priors); model A pessimistic absorbs it exactly.
On the pessimistic prior the BGS flux-limit drift moves model A
baseline on Delta z = 0.05 by 0.37 sigma, just above the threshold.

### Model B and where model A breaks

The model B pieces are in the code:

- **alpha_2** (cosmolike_core, `galpha2` in bias.c): the DESI DR1
  convention (Maus et al., arXiv:2404.07312), adding
  P_CT = (b_1 + f mu^2) f alpha_2 mu^2 k^2 P_lin, with f the linear
  growth rate and P_lin the linear matter power. One value per lens
  bin, set with `ci.set_nuisance_bias_alpha2(A2=...)` (a numpy array,
  in h^-2 Mpc^2). It is projected through the Kaiser transfers in
  every gg engine (FFTLog, low-rank, FKEM pairs, Limber); alpha_2 = 0
  is bitwise the previous build, and a brute-force j_l / j_l''
  integral agrees to 6e-4 on a thin auto and an adjacent pair.
- **SN_2** (`desi_cl.stochastic_templates(order=2)`): a k_par^2 white
  noise is the radial derivative of white noise, so it projects
  exactly as integral dchi W_i' W_j' / chi^2 (W_i the FoG-convolved
  kernel of bin i, ' = d/dchi) times the unit U_i = f_sat sigma_sat^2
  P_Pois,i: f_sat the satellite fraction and sigma_sat the satellites'
  dispersion in Mpc/h (DESI DR1 values, Maus et al.; `SN2_UNIT`: LRG
  0.1 x 7.8^2, ELG 0.1 x 2.9^2, and BGS takes the LRG value), not the
  FoG sigma_v. `order=0` reproduces the SN_0 templates bitwise.

EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb (executed, not yet committed)
reads the model A derivative caches and the linear notebook's scale
cuts, adds the alpha_2 columns and SN_2 templates (cached in
`chains/EXAMPLE_EVALUATE_FISHER_MODEL_B_TEMPLATES_<n>.pkl`), and
scans k_max for models A and B; its export writes `fom_kmax_models`
and `model_a_break` (also pending commit). The model B likelihood
option (`higher_order_bias: 2`) is in progress.

Results (pessimistic variant on the pessimistic prior unless stated):

- **Model B FoM(w0, wa)**: 1.29 / 1.34 / 0.228 / 0.0494 at 0.075 and
  11.6 / 14.3 / 2.70 / 0.299 at 0.25 (layout order above).
- **Model A breaks on the thin layouts at its own cut.** The largest
  shift of a model A fit to fixed truths carrying model B freedom
  (alpha_2 = +-12.5 h^-2 Mpc^2, SN_2 = +-5 units, or the offsets at
  model A's 1 sigma) stays below sigma/3 up to 0.05 on
  Delta d_com = 62 Mpc and at no fitted cut on Delta z = 0.02 (both
  variants, every prior); 0.1-0.125 on Delta z = 0.05; every cut on
  Delta z = 0.2. SN_2 sets the break: at 0.075 it moves Omega_m by
  1.25-1.68 sigma on the two thin layouts ((w0, wa) by 1.16-1.41 on
  the free prior of Delta z = 0.02). With alpha_2 and SN_2 drawn from
  model B's prior, model A's error bars there are too small by
  2.4-3.5 (62 Mpc) and 2.5-4.0 (Delta z = 0.02) at 0.075, covering
  the truth in 21-28% and 14-31% of universes. The mechanism and the
  alternatives are in `references/fom_and_bias_review.md`, section 6.

## Open work

- **Model B in the likelihood.** The option `higher_order_bias: 2`
  and its yamls are in progress; the model B notebook and the
  `fom_kmax_models` and `model_a_break` folders await commit; the
  model B results then enter `correlated_bias_prior_v2`, and
  `bias_model_summary` is rewritten last.
- **Open decision: how the model A break is quoted** - with SN_2 in
  the truth, without it, or on autos only
  (`references/fom_and_bias_review.md`, section 6).
- **The notes' open issues** (correlated_bias_prior_v2, Open issues):
  the one-loop terms are outside the covariance; the UV-fraction cut
  is built on the linear model (the alpha_0 template's UV fraction is
  a median 0.076 on the kept bands, above alpha); negative alpha_0 in
  chains makes the low-rank engine floor P_gg; CosmoLike's b_3
  template is broken (b_3 = 0).
- **The G + SSC final covariance and the joint lens sample**, as in
  the sections above.
- **Optional:** a nonlinear spot-check of the linear-response figure
  of bias (a few data-vector evaluations at drawn truths).

## The amplitude cut prior

Every sampling example (EXAMPLE_EVALUATE1-4, EXAMPLE_MCMC1,
EXAMPLE_EMUL2_EVALUATE1, EXAMPLE_EMUL2_MCMC1) carries the cobaya
prior `amplitude_cut`: each DESI bias group keeps
(A_s/A_s,fid)(b_i/b_i,fid)^2 <= 8, with A_s the primordial scalar
amplitude, prior 0 inside and -inf outside, so cobaya never
evaluates an excluded point.
EXAMPLE_EVALUATE5 applies the same cut to the five LSST Y1 lens
biases. 8 is the common cut of the cosmolike project family (lsst_y1
uses 8), not a tuning of this survey.

The galaxy power of a bin scales as A_s b_i^2, and the flat boxes
(1e9 A_s in [0.5, 5], biases in [0.8, 4]) reach a ratio of about 50
for the ELG groups (fiducial bias 0.84). A fixed relative numerical
error of the data vector costs delta chi2 proportional to the ratio
squared against the fiducial covariance: many units of chi2 at ratio
50. The cut is safe for inference because the DESI Fisher 2 sigma
contours reach a ratio of only about 3.2 on the Delta z = 0.2 binning
and 1.5-1.9 on the others (EXAMPLE_EVALUATE_FISHER.ipynb, section 8,
corrected covariance): it removes only prior volume
the data already exclude.

What the cut costs: at the extreme corners a cut of 8 still admits,
the pushed-settings delta chi2 of the N-random-models check can sit
modestly above the 0.2 budget, 0.29 to 0.32 measured on Delta z =
0.05 at ratios of 3.9 to 4.9. The project accepts that at those
data-excluded corners; a delta chi2 of order one would not be
accepted.

## Traps this project springs

- **carma zero-fills long lists.** Any Python list of more than ~16
  entries crosses the pybind/carma boundary zero-filled; the 76-entry
  lens arrays hit this twice (likelihood nuisance, covariance bias).
  Every per-bin vector must be a numpy array. A 5-bin donor never
  sees the bug.
- **The interface is dual-space.** `compute_data_vector_masked` keeps
  the donor's REAL-space layout and indexes out of bounds on the
  fourier dataset. The likelihood scores
  `compute_data_vector_fourier_gg_cross_masked` (the extended
  [EE | gs | gg | gg-cross] layout; the plain fourier getter leaves
  the cross block zero). Tests declare `BINNING_SPACE = "fourier"` in
  `tests/cocoa_test_utils.py`, and the shared suites resolve the
  getter at call time because the cross-pair list is installed only
  by the first model build.
- **The gg-cross stash contract.** Both engines (low-rank and the
  FKEM twin) compute the cross corrections in the auto-gg pass and
  stash them for the same evaluation (consumed once, through the
  shared `ggx_stash_store`). The correction applies only when the
  autos are non-Limber; with `adopt_limber_gg: 1` the crosses are
  Limber too, and the stash guard must not fire.
- **`lmax_nonlimber` is the FKEM cap** (`limits.LMAX_NOLIMBER`, set by
  `init_lmax_nonlimber` after `init_ntable_lmax`, which it validates
  against). It does nothing with the low-rank engine on, which has
  its own cap. The standard gg and gs cores still size their arrays
  by it, so raising it in another project costs memory.
- **No early exit on the Fourier/cross path.** The engine refuses
  cross pairs with tol > 0 (cosmo2D.c): every multipole of every run
  is computed. Do not add per-pair freeze logic casually.
- **Two models in one process.** The gg cross-pair list is a static of
  cosmo2D.c, outside every reset struct. Before the core's
  `initial_setup` cleared it (2026-11-04), a model built after a DESI
  one inherited DESI's 13 pairs and died in `gbmag` ("ni = 6" with
  five LSST lens bins). Any other new static setter needs the same
  reset.
- **`data/*cov` is gitignored.** Covariance files under data/ are built
  locally (the case builders' covariance stage); only the frozen test
  copies are tracked (Git LFS for the large ones).
- **ggl_exclude is the string "all"** here, not a pair list; shared
  code reading it must handle both forms (`_ggl_excluded`).
- **Fourier likelihoods disagree on the band count attribute**:
  roman-style carries `ncl`, this project carries `band_ells`
  (`_fourier_nlen` in the shared suites).
- **No source nuisance parameters are declared yet** (IA amplitudes,
  shear m, source dz): the TATT variants, FASTPT comparison and the
  source-sector cache rungs all skip. Declaring them and porting
  those tests is an open follow-up.
- **Big files ride LFS** (the two 92 MB n(z) copies, the frozen
  covariance, the window tables, the study's inverse): commits and
  pushes need git-lfs on PATH (the cocoa environment provides it).
- **Donor leftovers**: the M1-M6 mask/dataset files under data/
  describe the donor survey's layout, not this one.

## Test suite contract

`tests/` follows the shared machinery (guides:
`tests/data_vector/README.md`). The frozen fiducial reproduces
chi2 = 0 exactly; the emulator configuration is advisory
(4.44 vs exact physics at the adopted state - the thin bins resolve
emulator error photometric surveys blur, so the shipped networks are
not recommended here). After ANY refreeze, re-measure and re-pin the
non-Limber drift values - `DCHI2_MEASURED` of example1 and the
`DCHI2_MEASURED_CASES` of the Delta z examples 2-4, in both engine
test files (the README's refreeze step) - and re-run
the accuracy scan - its project-specific knobs are the chi-grid
boost (with a 64 -> 128 convergence row), the SVD rank and the
engine cap. The frozen covariances under `tests/frozen/data/`
predate the per-bin covariance bias (the dr3 copy is the sacc
covariance), so every pinned Delta chi2 scores that state, not the
shipped `data/*_cov`. Published README numbers that predate a model change are
superseded the moment the frozen state moves: replace them, never
append.
