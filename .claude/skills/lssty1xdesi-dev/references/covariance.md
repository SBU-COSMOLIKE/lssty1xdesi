# The covariance

`covariance/lssty1xdesi_covariance.py` is the survey adapter to the
shared covariance package (`cosmolike_notebook_utils.covariance`);
`covariance/compute_covariance.py` and
EXAMPLE_EVALUATE_COVARIANCE.ipynb drive it. SKILL.md, "The
covariance", states the design in brief; this file holds the details,
the evidence and the file plan.

## 1. The model of the Gaussian term

**Linear bias per bin.** `_lens_bias` gives each bin the likelihood's
fiducial b_{1,i} = B_g/D(zbar_i), with B_g the group amplitude
(CHUNK_BIAS, the bias at z = 0 under a constant clustering amplitude
b D: BGS 1.34, LRG 1.70, ELG 0.84) and D the linear growth read off
the linear P(k) ratio at k = 5e-3/Mpc exactly as `likelihood/desi_cl.py`
does (`_lens_growth`; GROWTH_KREF must equal the likelihood's KREF).
`_lens_growth` agrees with an independent LCDM growth integral to
2e-4.

**One-loop bias at the model A fiducial.** `_model_a_biases` hands the
core b_{2,i} = 0.412 - 2.143 b1 + 0.929 b1^2 + 0.008 b1^3 (Lazeyras et
al., arXiv:1511.01096) and b_{s^2,i} = -(4/7)(b_{1,i} - 1)
(co-evolution), with every offset, counterterm and stochastic amplitude
zero, mirroring the likelihood. The core's
`oneloop_bias_spectra_cov` adds, in Limber, the data vector's bilinear
pair form u_i^T Q u_j (u = (b1, b2, bs2) n_l H/H0; Q the C-FASTPT
templates 1/2 P_d1d2, 1/4 P_d2d2, 1/2 P_d1s2, 1/4 P_d2s2, 1/4 P_s2s2
times D^4) to lens-lens pairs, and 1/2 D^4 (b2 P_d1d2 + bs2 P_d1s2) to
lens-source pairs, from the core's own C-FASTPT tables with the sigma4
subtraction of cosmo2D.c. The non-Limber low-ell recompute corrects
the linear part only. SSC and cNG keep their linear-bias halo-model
galaxies. The Python driver takes optional per-bin `bias2` and
`bias_s2` settings; zeros keep the linear call bitwise.

Models A and B share this fiducial (every higher-order fiducial is
zero), so one covariance serves both. The LSST Y1 3x2pt case keeps
linear bias on its own lens sample.

**Checks of the one-loop increment** (2026-11-08): with b2 = bs2 = 0
every covariance is bitwise unchanged; the C increment equals an
independent evaluation of the data-vector formula to 1.5e-13 (gg) and
3.2e-12 (gs); 1 and 8 threads are bitwise identical; an -O0 UBSan
build is clean. The covariance's mean signal divided by the model A
data vector is flat at 1.03 on the kept autos at every redshift, the
offset of the covariance's massless neutrinos (the halo forecast behind
SSC runs at m_nu = 0; the data vector at 0.06 eV). Against the
linear-bias covariance the model A FoMs at k_max = 0.075 move by -0.4%
to +1.2%. The one-loop signal matters more at high k: rerun on this
covariance, the growth-tied linear-bias FoM(w0, wa) on the shipped mask
rises from 41.2 / 50.2 / 12.6 / 2.22 to 43.4 / 51.8 / 13.2 / 2.43
(layout order), while at k_max <= 0.1 it moves by about 1%.

**Redshift-space distortions and FoG.** The Gaussian term runs
redshift distortions in both the Limber window and the non-Limber
recompute's j_l'' transfer (gaussian rsd = nonlimber = true). The
adapter bakes the fiducial FoG convolution into a derived lens n(z) at
ingestion (`_fog_convolved_lens_file`, CHUNK_SIGMA_V with the seven
sigma_v groups and the BGS split): the smearing is this project's
systematic, so it lives in the adapter, and the shared covariance code
never models FoG. The derived files (`covariance/*_fog.nz`) are
regenerated on every configuration() call and gitignored.

## 2. The non-Limber grid of the covariance

`covariance/default.yaml` runs `nonlimber_accuracyboost: 2` (8193
log-distance samples), separate from the likelihood's chi grid. The
cross spectrum of two adjacent thin bins comes from the overlap of
their smeared edges, which the grid must resolve. At 1 (4097 samples)
the adjacent cross spectra were off by up to 15-30% (5-95% range) and
the masked Gaussian diagonal by up to 10%; at 2 the adjacent crosses
agree with the finest tested grid to 2% and the diagonal to 0.4%
(default.yaml quotes 0.4% for the 5-95% range of the kept crosses
against 8x finer grids), for about 5 s more per case. FoM(w0, wa)
moves by less than 0.5% between 1 and 2. Raising the GSL radial rule
(integration_accuracy 1 to 3) moved these quantities by less than 1%.

## 3. The bias and the list pitfalls (how they were caught)

- **The z = 0 bias in every bin.** `_lens_bias` once returned the raw
  CHUNK_BIAS for every bin, so the covariance signal was low by
  D(zbar_i)^2 at high redshift and every DESI FoM was inflated by
  2.5-3.4x (linear bias, shipped mask: FoM(w0, wa) 135 / 126 / 32.0 /
  5.67 instead of 41.2 / 50.2 / 12.6 / 2.22 on Delta d_com = 62 Mpc,
  Delta z = 0.02, 0.05, 0.2). The adversarial review of 2026-11-06
  found it with a counterfactual covariance built on the fiducial data
  vector, which predicted the corrected values (42.2 / 51.7 / 13.1 /
  2.3); the ratio of the covariance's mean signal to the data vector,
  about 0.25 at high redshift before the fix and 1.03 after, is the
  standing check.
- **carma zero-fill.** The forecast's
  `set_nuisance_bias(B1=settings["bias"], ...)` in
  `cosmolike_notebook_utils/covariance/forecast.py` once passed the
  per-bin bias as a Python list: lsst_y1's 5 entries survived, the 76
  thin-bin biases arrived as zeros, and every gg entry was pure shot
  noise. It was diagnosed because the Gaussian diagonal equaled
  2 N^2/n_modes exactly; the fix passes numpy arrays at that boundary.

## 4. Building the files

1. `unset IGNORE_COSMOLIKE_LSSTY1XDESI_COVARIANCE` (it enables the
   covariance build) and recompile with
   `scripts/compile_lssty1xdesi.sh` (`covariance/README.md`, steps 1-5).
2. `covariance/compute_covariance.py` with EXAMPLE_EVALUATE_COVARIANCE
   (case dr3, Delta d_com = 62 Mpc), ...COVARIANCE2 (dr1_dz0p02),
   ...COVARIANCE3 (dr1_dz0p05), ...COVARIANCE4 (dr1_dz0p20) and
   ...COVARIANCE5 (lsst_y1_3x2pt); the key `covariance: case:` selects
   the case. Run in the background: a case takes long.
3. `scripts/make_desi_covariance.py --case <case>` writes
   `data/<stem>_cov` from the Gaussian term of the case's archive
   (`covariance/production_covariance[_<case>].npz`).

The DESI cases are built at 14,000 deg^2; the Fisher forecasts rescale
the inverse to the common 12,300 deg^2 (x 0.879, exact for a Gaussian
covariance). `data/*cov` is gitignored; only the frozen test copies are
tracked. The source forecast's sacc covariance is kept as
`data/lssty1xdesi_desi_dr3_sacc_cov` for comparison only. A covariance
change needs no new derivatives: every Fisher notebook rebuilds its
matrices from the cached derivatives and the current files.

The LSST Y1 3x2pt case (12,300 deg^2) carries NLA. The covariance takes
one constant A1 per source bin, so the builder evaluates the fiducial
power law A1 ((1 + z)/1.62)^eta1 at each source bin's mean redshift
(1 + z0 = `nuisance.oneplusz0_ia` = 1.62).

## 5. Validation against the supplied covariance

The Knox formula with the project's own band values closes the
bookkeeping exactly (first clustering rows 1.007/1.018), and the
pre-FoG-kernel comparison with the source forecast's sacc covariance
measured sigma ratios with medians 1.218/1.228 (both totals positive
definite). Both predate the per-bin bias and the one-loop term; the
comparison is re-measured after any kernel or fiducial change. One
qualitative difference is physical, not a bug: the source forecast's
adjacent-bin cross band powers turn negative at high multipole (FoG
smearing anticorrelates neighbouring thin bins), while the project's
geometric-mean cross spectra are nonnegative by construction.

## 6. Gaussian-only files and the G + SSC plan

Every case (dr3, dr1_dz0p20, dr1_dz0p05, dr1_dz0p02; 76, 10, 36 and
90 lens bins) ships the Gaussian term only. The CLI also computes SSC
and cNG and keeps them in the archive. Measured 2026-11-04, before the
per-bin bias, on the unmasked entries (error-bar inflation on the
diagonal):

| case | SSC over G: median / max | cNG over G+SSC: max |
|---|---|---|
| dr1_dz0p20 | 1.0000 / 1.0013 | 1.016 |
| dr1_dz0p05 | 1.0000 / 1.0020 | 1.076 |

SSC moves almost no error bar; what it adds is a coherent,
amplitude-like correlation across bands that the free group biases
largely absorb. cNG is the larger term and the least reliable one for
spectroscopic tracers (a halo-model trispectrum), so it stays out of
every likelihood covariance.

Plan for the final chains: switch the case covariance files to
G + SSC, never cNG, a one-line change in the builder's covariance stage
(`gaussian` -> `gaussian + ssc`). SSC makes the matrix dense, and the
cosmolike reader takes text, so the files grow:

| case | entries, G -> G+SSC | text | xz -9 |
|---|---|---:|---:|
| dr1_dz0p05 | 106,816 -> 1,419,284 | 42 MB | 10.9 MB |
| dr1_dz0p02 | 686,774 -> about 29 M | about 865 MB | about 225 MB |

(dr1_dz0p02 scaled by its entry count; xz compresses this text about
3.9x, decompresses in about 15 s, compresses once in about 10 min.)
The storage design for the switch:

- ship `<stem>_cov.xz` in the repository (Git LFS) and gitignore the
  decompressed `<stem>_cov`;
- decompress in the project start script (start_lssty1xdesi.sh, sourced
  by start_cocoa.sh), only when the plain file is missing or older than
  the archive;
- never decompress inside the likelihood: every MPI rank of a chain
  would race to write the same file. When the plain file is missing,
  the likelihood fails with a message naming the fix.

A smaller alternative if size ever matters more: SSC is low rank (in
dr1_dz0p05, 83 of 2,982 eigenmodes exceed 1e-6 of the largest), so it
could ship as a few hundred eigenvectors, a few MB even for
dr1_dz0p02, but that needs the covariance reader to accept a low-rank
term.
