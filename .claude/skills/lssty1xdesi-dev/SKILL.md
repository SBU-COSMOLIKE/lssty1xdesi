---
name: lssty1xdesi-dev
description: Development on the lssty1xdesi project - DESI DR3 thin-bin spectroscopic clustering (76 comoving bins) cross LSST-Y1 shear in Fourier band powers. Use this skill for ANY work in projects/lssty1xdesi - the likelihood (desi_cl), the Fingers-of-God model and sigma_v parameters, the n(z) files and their sampling, the low-rank non-Limber engine settings, the gg-cross block, the covariance adapter, the test suite, the data-generation scripts, the DESI DR1 fixed-width Delta z cases (EXAMPLE_EVALUATE2-4), the LSST Y1 3x2pt case (EXAMPLE_EVALUATE5), or the Fisher module and notebook (EXAMPLE_FISHER1). Also trigger on mentions of DESI thin bins, nonlimber_accuracyboost, NL_Nchi, DESI_SIGV, gg cross pairs, or the dndz FoG convolution. This survey is the numerical extreme of the cosmolike projects, and defaults that are safe elsewhere fail here.
---

# lssty1xdesi: the thin-bin extreme

This project forecasts DESI DR3 spectroscopic galaxy clustering cross
LSST Y1 cosmic shear. The galaxy sample is sliced into 76 thin
comoving bins of about 62.3 Mpc (mean redshifts 0.108 to 1.549; BGS
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

The project default is `nonlimber_accuracyboost: 64`
(NL_Nchi = 32,705). The log spacing makes the HIGHEST-redshift thin
bins the narrowest in ln(chi), and their FoG-smoothed edges resolve
between boost 32 and 64: below that threshold the z = 1.32 ELG bin
(bin 68) carries a coherent +0.18 sigma broadband amplitude bias
across all its bands (delta chi2 = 1.9 with its crosses), constant
from boost 16 to 32, gone at 64. The ladder at 64 is converged:
64 vs 128 = 7.5e-6. This is a cliff, not smooth convergence - do not
infer safety at intermediate boosts from a two-point scan.

The global accuracyboost stays 1 in production: with the chi grid
held fixed, accuracyboost 1 -> 2 moves only delta chi2 = 0.34 (all in
clustering; shear moves exactly 0), and the P(k) z-grid probe
(`pk_z_refinement` 2) moves 0.0006. The remaining suspect for the
0.34 is `Ntable.nz_fine_sampling_factor`, which has no dedicated
setter yet. Fiducial evaluation: 7.8 s at 8 threads, 43 s serial
(2026-11-04, Apple laptop; x86 production timings are a separate
measurement).

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

Validation against the supplied forecast covariance: the Knox formula
with the project's own band values closes the bookkeeping exactly
(first clustering rows 1.007/1.018), and the pre-FoG-kernel
comparison measured sigma ratios with medians 1.218/1.228 (both
totals positive definite); re-measure after any kernel or fiducial
change. One qualitative difference is physical, not a bug: the
source forecast's adjacent-bin cross bandpowers turn NEGATIVE at
high multipole (FoG smearing anticorrelates neighboring thin bins),
while the project's geometric-mean cross spectra are nonnegative by
construction. `covariance/default.yaml` still runs its own
nonlimber_accuracyboost at 1 and has not been retuned to the adopted
data-vector state.

## Covariance of the fixed-width Delta z cases

The DESI DR1 fixed-width Delta z layouts (cases dr1_dz0p20,
dr1_dz0p05, dr1_dz0p02: 10, 36 and 90 lens bins, built by
scripts/make_desi_dz_case.py) have no supplied covariance. Their
likelihood covariance files hold the **Gaussian term only** of the
project's own forecast (compute_covariance.py with `case:` in the
covariance yaml), the convention of the comoving DR3 sacc. The CLI
also computes SSC and cNG and keeps them in the archive
(covariance/production_covariance_<case>.npz).

Measured 2026-11-04 on the unmasked entries (error-bar inflation on
the diagonal):

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
- **6x2pt = block-diagonal sum.** `combine` adds Fisher matrices by
  parameter name; no prior is shared between the probes, so none is
  double-counted. The joint likelihood for chains still needs the
  concatenated LSST+DESI lens sample.
- **Smooth vectors first.** A stencil derivative is only as good as the
  smoothness of the vector in the parameters. Diagnostic: scan one
  parameter on 17 fine points, whiten by the Cholesky factor of C^-1,
  fit a quartic, read the residual per point in sqrt(chi2) units
  (measured 2026-11-04, Delta z = 0.2 DESI and the LSST case):

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
  7.3 / 5.8 / 5.1 (DESI) and 52 / 14 / 24 / 19 (LSST). CAMB is now
  installed with `PATCH_CAMB_HALOFIT_TOLERANCE` (1e-7), and the
  notebook sets `nonlimber_accuracyboost: 64` for every DESI Fisher
  evaluation.
- **Step and accuracy stability with both fixes** (measured
  2026-11-05, FoM(w0, wa) at h = 0.01 / 0.02 / 0.03 / 0.05):

  | probe | AB 1 | AB 3 |
  |---|---|---|
  | DESI 1x2pt | 6.62 / 5.61 / 5.59 / 5.71 | 5.70 / 5.67 / 5.66 / 5.66 |
  | LSST 3x2pt | 17.68 / 17.67 / 17.65 / 17.68 | 17.36 at every step |
  | 6x2pt | 34.4 / 32.0 / 32.0 / 32.3 | 32.24 / 32.20 / 32.20 / 32.19 |

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
  | FKEM, cap 550 | DESI 1x2pt | 0.006 | 0.115 |
  | Limber | DESI 1x2pt | 5.6 (n_s) | 332 |
  | SVD engine | LSST 3x2pt | 0.0024 | 1.4e-4 |
  | Limber | LSST 3x2pt | 0.40 (A_s) | 11.2 |
  | DESI FKEM, cap 550 | 6x2pt | 0.013 | 0.115 |
  | DESI Limber | 6x2pt | 2.1 (w_a) | 332 |
  | Limber in both | 6x2pt | 2.1 (w_a) | 343 |

  The fair FKEM Delta chi2 is 0.115 here against the test pin 0.147:
  the notebook rescales the DESI inverse covariance to 12,300 deg^2
  (x 0.879, giving 0.129 at the likelihood's chi grid) and runs at
  `nonlimber_accuracyboost: 64` and AB 3.

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
engine cap. Published README numbers that predate a model change are
superseded the moment the frozen state moves: replace them, never
append.
