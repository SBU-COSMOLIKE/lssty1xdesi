# Numerical settings of the data vector

The thin spectroscopic bins are the project's defining numerical
challenge. Every setting below exists because a default that
photometric surveys never notice fails here by hundreds to tens of
thousands of chi2. Validate any change against the frozen suite and
never transfer a "converged elsewhere" setting without measuring it on
this vector. SKILL.md, "Numerical settings", lists the adopted values;
this file holds the evidence.

## 1. The non-Limber requirement (measured 2026-11-04)

Thin radial kernels keep real non-Limber power far past the
ell = 150 hand-off photometric surveys use. The likelihood runs
`adopt_limber_gg: 0` with the low-rank engine
(`nonlimber_lowrank_gg: 1`, rank 5, multipole cap 1000): a per-bin
SVD of sqrt(P_gg) whose rank-one profiles are FFTLog-projected with no
internal Limber hand-off. Against the converged vector, scored with the
project covariance of the time:

- pure Limber everywhere: delta chi2 = 21,158;
- FKEM at its defaults (cap `lmax_nonlimber` 150, cross pairs with the
  FKEM split): 4,243;
- rank 5 -> 8: 0.000; engine cap 1000 -> 2000: 0.002.

The two non-Limber tests in `tests/data_vector/` pin the first two as
drift checks (10% relative), re-measured at every refreeze.

The fair method comparison (same chi grid) puts FKEM at cap 550 -
every band - with its cross pairs: delta chi2 = 74.0 against the
engine, the genuine cost of FKEM keeping the nonlinear density term in
Limber. The unfair configurations it replaces (cap 150 with Limber
pairs, 17,388; cap 150 with pairs, 4,243) are never quoted as a method
comparison. `tests/data_vector/test_nonlimber_fkem_fair.py` pins the
fair value through the shared lowrank_gg_suite's `overrides` argument;
the 17,388 configuration no longer exists in the code.

FKEM with cross pairs is a separate twin in cosmo2D.c
(C_cl_tomo_fkem_pairs_core and C_gg_tomo_fkem_pairs_ells, in the
CROSS-BIN section), dispatched from C_gg_tomo_ells only when a pair
list is installed: every project without pairs runs the original
C_cl_tomo_core, bitwise. The twin pools its storage by 16-multipole
blocks, so a cap of 550 at NL_Nchi = 32,705 fits in about 1 GB (the
original core's layout would need about 55 GB).

The one-loop model adds a second reason for the low-rank engine: FKEM
moves the counterterm alpha_0 k^2 P through its Limber difference,
while the low-rank engine projects it exactly
(`one_loop_model.md`, Sec. 4).

## 2. The chi grid: thin-kernel aliasing, nested refinement, the one-bin cliff

**Aliasing at the default grid (measured 2026-11-03).** The non-Limber
FFTLog chi grid is log-spaced over the fixed window z = 0.002 to 4,
anchored at chi(z = 4), with NL_Nchi = 512 nodes by default. The thin
bins are 60-80 Mpc wide, with a comb spacing of about 45 Mpc at
z ~ 1, so the default grid gives one or two samples per kernel: CAMB
boost 1 against 16 differed by delta chi2 = 154,439, with per-bin
C_ell errors up to 66%, flat in ell and alternating in sign from bin to
bin. Any 0.1% change of chi(z_bin)/chi(z = 4) - emulator noise or an
ordinary MCMC step - resamples the thin kernels on the comb. The
hybrid emulator's chi2 = 54,951.6 was this aliasing, triggered by a
+7.5 Mpc/h chi error of emulbaosn at z = 4.036 (outside its training
range); at boost 16 the emulator agreed with CAMB to chi2 = 7.6.
A fiducial self-consistency check (chi2 = 0 against a vector computed
with the same settings) cannot detect aliasing: convergence tests must
vary the grid. Broad-bin projects (lsst_y1, roman) are unaffected.

**Nested refinement.** The grid is endpoint-inclusive log-spaced
(`dlnchi = log(chi_max/chi_min)/(NL_Nchi - 1)`). The shared core
refines it by nested interval doubling: `NL_Nchi - 1 = 511*m`, with
m = accuracyboost * nonlimber_accuracyboost rounded up to a power of
two (the knob is effectively dyadic; 3 buys the grid of 4). Non-nested
ceil-scaling was replaced because consecutive doublings measured
delta chi2 = 1.3 and 1.1 of pure node-placement difference. The
decomposition SVDs stay at 512 nodes (NK = ceil(nchi/NL_Nchi_boost)),
and the boosted grid lives in pointer views over block pools, which
keeps the memory of a boosted evaluation near 2.4 GB.

**The one-bin cliff.** The Delta d_com = 62 Mpc and Delta z = 0.02
references use the combined factor 64 (NL_Nchi = 32,705) as
`accuracyboost: 2` with `nonlimber_accuracyboost: 32`. The log spacing
makes the highest-redshift thin bins the narrowest in ln(chi), and
their FoG-smoothed edges resolve between combined factors 32 and 64:
below that threshold the z = 1.32 ELG bin (bin 68) carries a coherent
+0.18 sigma broadband amplitude bias across all its bands (delta chi2
= 1.9 with its crosses), constant from 16 to 32, gone at 64. This is a
cliff, not smooth convergence: safety at intermediate factors cannot
be inferred from a two-point scan. Factor 64 against 128 on
Delta d_com = 62 Mpc: 0.0033 at accuracyboost 1, 0.0045 at 2.

**The global accuracyboost** (cosmolike's tables: the a grid, the
P(k) z nodes, the ell and n(z) sampling) is 2 on all four layouts;
CAMB's own AccuracyBoost stays at the examples' 1.05. Against
accuracyboost 5 (Delta d_com = 62 Mpc and Delta z = 0.02, 2026-11-05,
patched CAMB halofit):

| accuracyboost, combined chi-grid factor | Delta d_com = 62 Mpc | Delta z = 0.02 | 62 Mpc evaluation |
|---|---:|---:|---:|
| 1, 64 | 0.133 | 0.101 | 12.7 s |
| 1, 512 | 0.129 | 0.096 | - |
| 2, 64 (the reference) | 0.017 | 0.033 | 13.7 s |
| 2, 128 | 0.010 | 0.011 | 24.5 s |

The tables carry the error; a finer chi grid alone does not remove it,
and doubling the chi grid doubles the evaluation (4 threads, Apple
laptop on AC power; x86 production timings are a separate
measurement). At the fiducial the wide layouts move by at most 0.053
(Delta z = 0.05) and 0.013 (Delta z = 0.2) at accuracyboost 6, but at
amplitude-cut prior corners boost 1 lets the pushed-settings delta chi2
of the N-random-models check reach 0.7 (Delta z = 0.05) and 0.9
(Delta z = 0.2, with a 10/Mpc power table), so they run boost 2 as
well, with nonlimber_accuracyboost 16 and 4 keeping their combined
chi-grid factors at 32 and 8.

The Fisher notebooks evaluate every DESI vector at
`nonlimber_accuracyboost: 64` (derivatives need a finer grid than one
chi2 does: `fisher_and_scale_cuts.md`, Sec. 2).

## 3. The Boltzmann k range (measured 2026-11-05)

Every DESI layout sets kmax_boltzmann 20 (`likelihood/desi_cl.yaml`;
the Delta z cases inherit it), so CAMB's P(k) table reaches
kmax_boltzmann x accuracyboost = 40/Mpc at boost 2. The range protects
the prior corners, not the fiducial: at a low-Omega_m point
(Omega_m 0.237, w0 -2.39, w0 + wa -2.87, H0 78.6) a 10/Mpc table moved
chi2 by 2.09 (Delta d_com = 62 Mpc) and 2.06 (Delta z = 0.02) through
the extrapolated high-k tail; 40/Mpc brings that to 0.12 and 0.10.

EXAMPLE_MCMC1.yaml runs EXAMPLE_EVALUATE1's settings (accuracyboost 2,
kmax_boltzmann 20), so chains sample with the settings that generated
the data vector. The EMUL2 examples keep accuracyboost 1 and
kmax_boltzmann 5.

## 4. The Fingers-of-God model

A line-of-sight velocity dispersion sigma_v smears observed redshifts
by sigma_z = (1 + zbar) sigma_v / c, and the bins are far narrower than
the smearing's effect, so the convolution is the dominant radial
systematic. `likelihood/desi_cl.py` convolves a copy of each raw
thin-bin histogram with that Gaussian at every sampled point (scipy
gaussian_filter1d in grid steps, mode "constant") and sends the result
through the external-n(z) path: `external_nz_modeling: 1` is mandatory,
and the per-point n(z) rebuild, which invalidates the kernel caches at
each evaluation, is part of every evaluation's cost. The source
forecast's reference implementation (dndz_stretch_fog_gaussian) is the
same operation; the only differences are scipy's +-4 sigma kernel
truncation (a <= 6e-5 tail reshaping; truncate=8 gives exact agreement)
and its explicit post-renormalization, a no-op here because no bin
approaches the grid ends.

The sigma_v are sampled in seven groups, one more than the six bias
groups, because the simulation-measured sigma_v(zbar) steps inside BGS
at z = 0.255 (between bin means 0.247 and 0.263): BGS0 = 371 (bins
1-10), BGS1 = 358 (bins 11-18), LRG1-3 = 357/348/343, ELG1-2 =
304/289 km/s. Grouping is pure yaml: the likelihood declares
DESI_B1_<n> and DESI_SIGV_<n> per bin, and `scripts/make_desi_params.py`
writes the fragments that tie group members to leaders (SIGV leaders
1/11/19/30/39/50/62) with cobaya value functions. Changing a grouping
is a regeneration of `likelihood/params_desi*.yaml`, never code.

Priors are Gaussian, of width five times the per-bin error bars of the
simulations' measured sigma_v(zbar) figure (digitized 2026-11-04 with
the y scale calibrated on the seven group means, residuals < 5 km/s):
145/105/70/50/50/40/30 km/s, truncated at zero only (the BGS0 fiducial
sits 2.6 prior sigma above zero). A flat [100, 1000] prior was
rejected: the measurement excludes dispersions anywhere near the flat
bounds, and covering sigma_v = 100 (near-top-hat kernels again) costs
numerics and buys nothing physical.

## 5. n(z) sampling: features, not bin counts

`data/lssty1xdesi_desi_dr3.nz` holds 49,996 uniform rows
(dz = 7.0e-5): a nested five-fold linear refinement of the DESI
10^4-point grid (REFINE in `scripts/make_desi_nz.py`; original samples
keep their exact values). The per-bin sample counts of the raw grid
(43-100) already matched a broad photometric bin (lsst_y1: 47-102
lens, 63-237 source), but per-bin counts are the wrong measure: the
sharpest feature is the FoG-smoothed edge, sigma_z = 1.3e-3 to 2.4e-3,
which had 4-7 raw samples per sigma and now has 20-34. Judge any grid
change by samples per feature. Every bin holds 219-505 nonzero
samples.

The z column holds bin midpoints: `photoz_zmid_convention: 1` is the
project default, and the frozen references carry it; a Z_LOW reading
shifts chi2 by 3.47. The n(z) reconstruction study (`tests/studies/`)
showed the cubic interpolant ringing (negative n(z)) on the raw top-hat
edges; its stored numbers predate the FoG fiducials and the five-fold
grid and await a rerun.

## 6. The amplitude cut and the N-random-models check

**The cut.** Every sampling example (EXAMPLE_EVALUATE1-4,
EXAMPLE_MCMC1, EXAMPLE_EMUL2_EVALUATE1, EXAMPLE_EMUL2_MCMC1) carries
the cobaya prior `amplitude_cut`: each DESI bias group keeps
(A_s/A_s,fid)(b_i/b_i,fid)^2 <= 8, with A_s the primordial scalar
amplitude (sampled as As_1e9 = 10^9 A_s) and b_i a group leader's
linear bias; the prior is 0 inside and -inf outside, so cobaya never
evaluates an excluded point. EXAMPLE_EVALUATE5 applies the same cut to
the five LSST Y1 lens biases. 8 is the common cut of the cosmolike
project family (lsst_y1 uses 8), not a tuning of this survey; an
interim 3.5 was dropped.

The galaxy power of a bin scales as A_s b_i^2, and the flat boxes
(10^9 A_s in [0.5, 5], biases in [0.8, 4]) reach a ratio of about 50
for the ELG groups (fiducial bias 0.84). A fixed relative numerical
error of the data vector costs delta chi2 proportional to the ratio
squared against the fiducial covariance: many units of chi2 at ratio
50. The bias boxes stay as they are by the PI's decision; the
N-models figures are the argument for narrowing them with
collaborators. The cut is safe for inference because the DESI 2 sigma
contours reach a ratio of only 1.6 (Delta d_com = 62 Mpc), 1.5
(Delta z = 0.02), 1.9 (Delta z = 0.05) and 3.1 (Delta z = 0.2)
(EXAMPLE_EVALUATE_FISHER.ipynb, section 8, growth-tied linear bias on
the shipped mask; the corrected linear-bias and the current covariance
agree to 0.04 here): it removes only prior volume the data already
exclude.

**The check** (core design shared with lsst_y1): always 50 fixed points
(seed 137 + m), no run-time control of their number or seed; the
pushed-settings vectors are stored once in
`tests/frozen/nmodels_<example>.npz` by
`generate_frozen_reference.py --nmodels`, and the routine check builds
one default model and compares. Draw settings live in
`tests/cocoa_test_utils.py` (NMODELS_EXAMPLES, NMODELS_LIMIT 0.2,
NMODELS_BOX with Omega_m in [0.2, 0.5], NMODELS_HOLD resetting
Omega_m, A_s and the DESI biases to the fiducial in every odd model,
NMODELS_AMPLITUDE_CUT 8). The figure per layout is the triangle of
Omega_m against the amplitude ratio.

**Results** (50 models per layout, 2026-11-06; held: median / max;
drawn: median / max; drawn with ratio <= 3.5: max):

| layout | held | drawn | drawn, ratio <= 3.5 |
|---|---:|---:|---:|
| Delta d_com = 62 Mpc (`example1`) | 0.110 / 0.294 | 1.41 / 47.4 | 1.45 |
| Delta z = 0.02 (`example2`) | 0.083 / 0.304 | 1.31 / 37.6 | 1.26 |
| Delta z = 0.05 (`example3`) | 0.067 / 0.158 | 0.46 / 12.1 | 0.21 |
| Delta z = 0.2 (`example4`) | 0.012 / 0.040 | 0.21 / 4.64 | 0.23 |

The large values sit at high Omega_m (0.44-0.50) or a high amplitude
ratio (5 or more), mostly with large neutrino masses (model 29: Omega_m
0.50, ratio 6.2, m_nu 0.60 eV: 47.4 and 37.6 on the thin layouts). The
lever differs by layout: on the thin layouts cosmolike's table
resolution carries the residual (Delta z = 0.02 models 19 and 43:
accuracyboost 2 -> 4 alone gives 7.37 and 3.32 of the full 8.73 and
4.59, CAMB AccuracyBoost 1.05 -> 2 gives 1.37 and 1.44, integration
accuracy, kmax_boltzmann and lmax give 0.000); on Delta z = 0.2 CAMB's
AccuracyBoost alone reproduces the three worst points (1.55, 0.76,
0.26). The policy at data-excluded corners: a delta chi2 of order one is
acceptable (1.53 at Omega_m 0.49 and ratio 6.9 on Delta z = 0.2 was
accepted), one much larger than one is not. The values of order 10 and
above sit at the most extreme corners (ratio 5 or more, or Omega_m at
the box edge with large neutrino masses), where the DESI 2 sigma
contours never reach; the project README records them as accepted there,
and the check reports the failure as a documented judgment. Raising
accuracyboost on the thin layouts is the available lever at a cost per
evaluation, not an adopted setting (SKILL.md, "Open issues and open
work").
