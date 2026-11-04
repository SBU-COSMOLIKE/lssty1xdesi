---
name: lssty1xdesi-dev
description: Development on the lssty1xdesi project - DESI DR3 thin-bin spectroscopic clustering (76 comoving bins) cross LSST-Y1 shear in Fourier band powers. Use this skill for ANY work in projects/lssty1xdesi - the likelihood (desi_cl), the Fingers-of-God model and sigma_v parameters, the n(z) files and their sampling, the low-rank non-Limber engine settings, the gg-cross block, the covariance adapter, the test suite, or the data-generation scripts. Also trigger on mentions of DESI thin bins, nonlimber_accuracyboost, NL_Nchi, DESI_SIGV, gg cross pairs, or the dndz FoG convolution. This survey is the numerical extreme of the cosmolike projects, and defaults that are safe elsewhere fail here.
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
two non-Limber drift values (the README's refreeze step), and re-run
the accuracy scan - its project-specific knobs are the chi-grid
boost (with a 64 -> 128 convergence row), the SVD rank and the
engine cap. Published README numbers that predate a model change are
superseded the moment the frozen state moves: replace them, never
append.
