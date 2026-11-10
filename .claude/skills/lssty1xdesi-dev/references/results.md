# Results with provenance

Every number here is read from an executed notebook or a figure
folder's `data.json`; the notes quote the same values. Before quoting a
number elsewhere, read it again from its source. Layout order
throughout: Delta d_com = 62 Mpc / Delta z = 0.02 / 0.05 / 0.2. DESI is
C_ell^gg alone. k_max is quoted at z = 0: each bin is cut at
k_max/D(zbar_i) by the UV-fraction criterion (alpha = 0.05) within the
shipped mask. Shifts are Fisher linear responses, (w0, wa) in units of
the fit's marginalized ellipse and Omega_m, A_s in their own sigma;
the threshold is sigma/3. The covariance is the Gaussian one on the
one-loop fiducial (SKILL.md, "The covariance"). Figures and text quote
no FoM above k_max = 0.2.

| result | notebook (section) | data |
|---|---|---|
| linear bias against k_max | EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb (4, 5) | `notes/linear_bias_kmax/plots/{fob_kmax,fom_kmax,ladder_bias_params}/data.json` |
| model A at k_max = 0.075 (a record) | the removed EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb (6, 7), commit 567751f | the removed v2 folders baseline_ladder, ladder_bias_scatter, goodness_of_fit, drift_shifts, calibration_drift (same commit) |
| model B against k_max; where model B breaks | EXAMPLE_EVALUATE_FISHER_MODEL.ipynb (6, 7) | `notes/correlated_bias_prior_v2/plots/fom_kmax_models/data.json` |
| the paper model | EXAMPLE_EVALUATE_FISHER_MODEL.ipynb (9) | `notes/bias_model_summary/plots/{summary_fom_kmax,summary_fom_untied}/data.json` |
| linear-bias FoMs, engine Fisher biases, amplitude-cut contours | EXAMPLE_EVALUATE_FISHER.ipynb (4-8) | printed tables; `fisher_and_scale_cuts.md` |
| LSST Y1 3x2pt references | lsst_y1 EXAMPLE_EVALUATE2.ipynb | `lsst_y1_3x2pt_fom*` keys of the v2 data.json files |

## 1. Linear bias against k_max

The truths are model A (calibrated: b2, bs2 on the calibrations at the
fiducial b1; plus brackets b2 +- 1 and alpha_0 = +-12.5 h^-2 Mpc^2), the
fit is linear bias under each of the four priors.

- **Safe only at k_max = 0.05**, and for every prior and truth only on
  62 Mpc (largest shift 0.14 sigma). Delta z = 0.02 keeps no band at
  0.05 and fails at 0.075. Delta z = 0.05 is safe at 0.05 for every
  truth under growth-tied and baseline, for the calibrated truth only
  under pessimistic, never under free. Delta z = 0.2 is safe at 0.05
  for the calibrated truth only. The worst truths at 0.05 are the
  negative brackets.
- **At 0.075 every prior fails on every layout**, a free bias per bin
  included: largest shifts 0.39 sigma (Delta z = 0.2, growth-tied) to
  1.79 sigma; on 62 Mpc Omega_m moves by 1.50-1.79 sigma, A_s by
  0.95-1.23 and (w0, wa) by 0.27-0.54. More bias freedom moves the
  error between parameters (Delta z = 0.05: growth-tied Omega_m 1.28,
  (w0, wa) 0.29; baseline Omega_m 0.19, (w0, wa) 0.98): a constant b_i^2
  cannot absorb the k dependence of the loops.
- **The cost of the safe cut**: on 62 Mpc FoM(w0, wa) falls from 5.10 /
  3.37 / 2.78 at 0.075 to 0.264 / 0.213 / 0.036 at 0.05 (baseline,
  pessimistic, free), 1-6% of its value.
- **The accuracy test of the linear-bias priors scored for Omega_m and
  A_s** (linear-bias truths at 0.075) gives the verdict of (w0, wa):
  growth-tied excluded, baseline honest only under its own scatter,
  pessimistic honest under both. Matched-prior identity
  E[chi2_w] - drift^2 = 2 to 5.6e-11.

## 2. Model A at k_max = 0.075 h/Mpc

A record: the model A notebook, EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb,
and its five figure folders were removed on 2026-11-10 (their last
version is commit 567751f). The values below predate its final-tier
rerun of that commit and are not refreshed; SKILL.md, "Results", quotes
the final-tier values.

**FoM(w0, wa)** (`baseline_ladder/data.json`, key `fom`):

| variant | growth-tied | baseline | pessimistic | free |
|---|---|---|---|---|
| model A | 6.58 / 6.46 / 1.99 / 0.390 | 4.33 / 4.00 / 0.522 / 0.0884 | 2.78 / 3.01 / 0.346 / 0.0884 | 2.24 / 1.29 / 0.280 / 0.0884 |
| model A + tilts | 4.95 / 4.29 / 0.628 / 0.104 | 4.13 / 3.68 / 0.426 / 0.0536 | 2.59 / 2.70 / 0.268 / 0.0536 | 2.04 / 1.03 / 0.219 / 0.0536 |

**FoM(Omega_m, 10^9 A_s)** (key `fom_om_as`):

| variant | baseline | pessimistic | free |
|---|---|---|---|
| model A | 10.1 / 8.57 / 3.10 / 0.716 | 8.57 / 7.52 / 2.23 / 0.716 | 7.67 / 4.12 / 1.85 / 0.716 |
| model A + tilts | 9.56 / 7.61 / 2.57 / 0.457 | 8.09 / 6.58 / 1.77 / 0.457 | 7.19 / 3.39 / 1.48 / 0.457 |

LSST Y1 3x2pt alone: 17.26 and 214.9. The growth-tied column is printed
for completeness only: it fails the accuracy test, so no figure draws
its FoM. Model A carries one SN2 per group; adding it lowered
FoM(w0, wa) at 0.075 by 4-8% on the thin layouts and under 1% on the
wide ones.

Diagnostics, not results (`baseline_ladder/data.json`, keys
`v1_linear` and the model A calibrated rows): linear bias at the same
cut gives 5.10 / 4.65 / 0.712 / 0.177 on the baseline prior (model A
keeps about 85 / 86 / 73 / 50% of it; model A + tilts on the
pessimistic prior about 77 / 76 / 55 / 30% of linear bias on that
prior); exact calibrations give 6.93 / 5.09 / 1.42 / 0.295 on the
baseline prior, above linear bias.

**The accuracy test** (2000 random truths per scatter family, seed
20261107; each truth the known drift plus a residual drawn from the
baseline or the pessimistic prior, b2 and bs2 following b1 through the
calibrations; `goodness_of_fit/data.json`, `ladder_bias_scatter/data.json`).
The statistic eta = sqrt(<chi^2>/n_dof), chi^2 = Delta p^T C_p^-1
Delta p (recovered minus true, the fit's marginalized covariance;
n_dof = 2 for (w0, wa), 1 for Omega_m or A_s), is an exact average over
the data noise and the bias history: <chi^2> = drift^2 + the residual
scatter term + the noise term, not a Monte Carlo. The 2000 draws only
make the 16-84% bands of the scatter figure, and carry no noise.

- **Growth-tied fails** under every truth: eta(w0, wa) 3.4-6.6 under
  model A, 1.4-2.3 under model A + tilts; under the
  pessimistic scatter its Omega_m bar covers the truth in 5-40% of
  universes. The conserved-population drift alone moves its (w0, wa)
  by 4.0-6.1 sigma (model A), 1.2-1.8 sigma (model A + tilts).
- **Baseline is honest only if the residual is as smooth as it
  assumes**: eta 1.00-1.02 under its own scatter; under the pessimistic
  scatter 1.24-1.70 in (w0, wa), and the Omega_m bar covers the truth
  in 55 / 63 / 40% of universes (A_s: 56 / 64 / 54%) on the first
  three layouts; on Delta z = 0.2 the priors coincide. Model A + tilts
  does not change this.
- **Pessimistic is honest under both scatters**: eta <= 1.05 in every
  score (the 1.05 is the BGS flux-limit drift on Delta z = 0.05, which
  this prior leaves just above sigma/3: 0.38 sigma in (w0, wa), 0.35 in
  Omega_m under model A), and 0.89-1.00 in (w0, wa) under
  baseline scatter. Free is exactly 1.
- **Checks**: matched-prior identity to 5.7e-10; the free prior's
  largest mean shift 8.4e-12; one of the 48 systems drops one null
  direction (62 Mpc, model A + tilts, free prior; largest component
  on the scored parameters 7.2e-12).

**Coverage.** For one parameter with mean shift mu and eta = s,
coverage = Phi((1 - mu)/sqrt(v)) - Phi((-1 - mu)/sqrt(v)), v = s^2 -
mu^2; with the drift absorbed this is 2 Phi(1/s) - 1 (s = 1.3 -> 56%,
1.75 -> 43%, 2.5 -> 31%), exact to 0.1% for the baseline, pessimistic
and free priors and only an upper bound for growth-tied.

**The calibration drift** (Delta_2 linear across each group from about
-0.5 to +0.5, Delta_s half of it, zero mean, b1 at the fiducial;
`calibration_drift/data.json`): under model A it moves
(w0, wa) by 0.40-0.41 sigma on Delta z = 0.05 (A_s 0.35-0.37) and
0.40 on Delta z = 0.2 under the baseline, pessimistic and free priors,
at most 0.24 on 62 Mpc and 0.15 on Delta z = 0.02, and 0.57-1.09 under
growth-tied. Model A + tilts absorbs it exactly (1e-12), since its
tilts have the drift's shape; a nonlinear drift within a group is
untested.

## 3. Model B against k_max; where model B breaks

Source: EXAMPLE_EVALUATE_FISHER_MODEL.ipynb (sections 6, 7) and
`fom_kmax_models/data.json` (`fom`, `fom_om_as`, `shift`, `worst`,
`worst_in_prior`, `break`, for both variants and the three priors, the
free prior included). The cuts are the UV-fraction cuts of the paper
fiducial (`chains/EXAMPLE_EVALUATE_FISHER_MODEL_SCALE_CUTS.pkl`). The
scan: seven cuts 0.05, 0.075, ..., 0.2 (`KMAX_MAX`), two variants
(model B, model B + tilts), three priors (growth-tied omitted): 168
systems, 162 with a fit, 6 without any band (Delta z = 0.02 at 0.05);
11 drop a null direction (largest component on the scored parameters
3.3e-10). Counterterm widths follow 0.5/k_max^2 at every cut. The
baseline scenario is the baseline prior with model B; the pessimistic
scenario is the pessimistic prior with model B + tilts.

**FoM(w0, wa)** in the two scenarios:

| k_max | pessimistic scenario (model B + tilts) | baseline scenario (model B) |
|---|---|---|
| 0.075 | 0.407 / 0.237 / 0.155 / 0.0423 | 0.595 / 0.489 / 0.253 / 0.0676 |
| 0.1 | 1.29 / 1.48 / 0.503 / 0.101 | 1.78 / 2.01 / 0.708 / 0.162 |
| 0.15 | 4.80 / 5.14 / 1.68 / 0.229 | 5.66 / 6.43 / 1.99 / 0.347 |
| 0.2 | 7.53 / 8.42 / 2.45 / 0.300 | 8.46 / 9.91 / 2.77 / 0.478 |

- Model B + tilts lies below model B at every cut, and every FoM(w0, wa)
  stays below LSST Y1 3x2pt (17.26). FoM(Omega_m, 10^9 A_s) of model B
  stays below 214.9 at every cut (at most 67.5 at 0.2).

**Where model B breaks.** The fixed truths, each at one prior width and
the same in every bin or a drift across each group: (a) alpha_2 at
+-1 sigma of model B's prior, 0.5/k_max^2; (b) SN2 = +-5 units; (c) the
offsets at +-(2, 1); (d) the calibration drift, Delta_2 and Delta_s
linear in redshift across each group with zero mean, changing by 1 and
0.5 from its first bin to its last (the tilts of model B + tilts absorb
it exactly; its alpha_0 and alpha_2 tilts are not exercised). (a)-(c)
lie inside the prior and are reported; (d) is tested. The shift is
exact for (a) and (b) and first order for (c) and (d); +- truths score
identically. A model holds at a cut while every score of every tested
truth stays below sigma/3, there and at every smaller fitted cut:

| layout | baseline scenario: model B | pessimistic scenario: model B + tilts |
|---|---|---|
| Delta d_com = 62 Mpc | 0.075; at 0.1 the drift moves (w0, wa) by 0.82 sigma | every cut to 0.2 |
| Delta z = 0.02 | 0.075; at 0.1 the drift moves (w0, wa) by 0.86 | every cut to 0.2 |
| Delta z = 0.05 | 0.05; at 0.075 the drift moves (w0, wa) by 0.35 | every cut to 0.2 |
| Delta z = 0.2 | 0.05; at 0.075 the drift moves (w0, wa) by 0.36 | every cut to 0.2 |

- In the baseline scenario the largest tested shift from 0.1 on is
  0.53-2.79 sigma. The pessimistic scenario's pass carries no
  information on its own: its in-prior shifts reach 0.98 sigma at 0.075
  on 62 Mpc and 1.10 sigma at 0.1 on Delta z = 0.02 (both alpha_2),
  0.33-0.73 sigma from 0.125 on on the three thinner layouts, and at
  most 0.25 sigma on Delta z = 0.2.
- SN2 at +-5 sets no break point: it moves no score by more than
  0.26 sigma at any cut, in either scenario.
- The break points under the free prior are in data.json.

The model A scans of this section (model A and model A + tilts against
k_max, model A's alpha_2 truth, the prior-mismatch and random alpha_2
diagnostics) left the notebook with the rest of its model A content on
2026-11-10; their last values, on the linear-bias cuts, are in
EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb of commit 567751f and in the git
history of this file.

## 4. The paper model

Model B + tilts without alpha_2 (its group values and tilts fixed at 0)
and every sigma_v prior width x6, on the pessimistic prior: the example
yamls EXAMPLE_EVALUATE1-4 and the variant 'B tilts noA2 sv x6'. Source:
EXAMPLE_EVALUATE_FISHER_MODEL.ipynb, section 9;
`notes/bias_model_summary/plots/summary_fom_kmax/data.json` and
`summary_fom_untied/data.json`.

- FoM(w0, wa): 0.564 / 0.352 / 0.173 / 0.0445 at 0.075 and 6.70 / 7.95 /
  2.34 / 0.307 at 0.2; FoM(Omega_m, 10^9 A_s) 50.3 / 53.2 / 19.9 / 3.20
  at 0.2.
- Against the linear-bias cuts the paper-fiducial cuts change these
  FoMs by a factor 0.85-1.04 (0.97-1.00 on 62 Mpc and Delta z = 0.02;
  Delta z = 0.2 at 0.075: 0.85).
- eta <= 1.030 (Omega_m on 62 Mpc at 0.175) under the known drifts plus
  the pessimistic scatter; coverage of the 1 sigma region at least 66.8%.
- alpha_2 = +-12.5 h^-2 Mpc^2 moves no score by more than 0.327 sigma;
  alpha_2 at one model B width breaks the model at 0.05 on 62 Mpc and
  Delta z = 0.05 and below the smallest fitted cut on Delta z = 0.02
  (A_s 1.64 sigma at 0.075 on 62 Mpc, (w0, wa) 1.31 at 0.1 on
  Delta z = 0.02).
- The in-prior truths (SN2 and the offsets) move a score by at most
  0.69 sigma, and by at most 0.42 sigma from 0.125 on.
