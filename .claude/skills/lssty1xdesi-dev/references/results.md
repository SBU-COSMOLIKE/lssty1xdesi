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
model A fiducial (SKILL.md, "The covariance"). Figures and text quote
no FoM above k_max = 0.2.

| result | notebook (section) | data |
|---|---|---|
| linear bias against k_max | EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb (4, 5) | `notes/linear_bias_kmax/plots/{fob_kmax,fom_kmax,ladder_bias_params}/data.json` |
| model A at k_max = 0.075 | EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb (6, 7) | `notes/correlated_bias_prior_v2/plots/{baseline_ladder,ladder_bias_scatter,goodness_of_fit,drift_shifts,calibration_drift}/data.json` |
| models A and B against k_max; where each model breaks | EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb (6, 7) | `notes/correlated_bias_prior_v2/plots/fom_kmax_models/data.json` |
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

## 3. Models A and B against k_max; where each model breaks

The scan: seven cuts 0.05, 0.075, ..., 0.2 (`KMAX_MAX`), four variants
(model A, model A + tilts, model B, model B + tilts), three priors
(growth-tied omitted): 336 systems, 324 with a fit, 12 without any band
(Delta z = 0.02 at 0.05); 20 drop a null direction (largest component
on the scored parameters 7.0e-10); the model A FoMs at 0.075 equal the
model A notebook's to 7.9e-10. Counterterm widths follow 0.5/k_max^2 at
every cut. The baseline scenario is the baseline prior with model A and
model B; the pessimistic scenario is the pessimistic prior with model A
+ tilts and model B + tilts. Source: the model B notebook (sections 6,
7) and `fom_kmax_models/data.json` (`fom`, `fom_om_as`, `shift`,
`worst`, `break`, for every variant and prior, the free prior included).

**FoM(w0, wa)** of model B in the two scenarios:

| k_max | pessimistic scenario (model B + tilts) | baseline scenario (model B) |
|---|---|---|
| 0.075 | 1.29 / 1.34 / 0.227 / 0.0495 | 2.92 / 2.64 / 0.489 / 0.0841 |
| 0.1 | 2.71 / 4.62 / 0.623 / 0.101 | 6.17 / 9.00 / 1.17 / 0.162 |
| 0.15 | 7.17 / 9.57 / 1.81 / 0.229 | 12.9 / 18.7 / 2.51 / 0.347 |
| 0.2 | 10.9 / 13.9 / 2.64 / 0.289 | 17.5 / 24.6 / 3.33 / 0.458 |

Model A at 0.2, beyond its break on the thin layouts (what extending
model A would claim, not a result): model A + tilts on the pessimistic
prior 13.3 / 17.0 / 2.82 / 0.303; model A on the baseline prior 19.0 /
26.8 / 3.48 / 0.471.

- Every model B variant lies below the same model A variant at every
  cut (wider priors, more parameters), and the baseline scenario gives
  the larger FoMs at every cut.
- FoM(w0, wa) above LSST Y1 3x2pt (17.26) occurs only in the baseline
  scenario on the thin layouts at the largest cuts (model B: 17.5 and
  24.6 at 0.2, Delta z = 0.02 from 0.15), beyond that scenario's break
  point. FoM(Omega_m, 10^9 A_s) of model B stays below 214.9 at every
  cut (at most 116 at 0.2).

**Where each model breaks.** The fixed truths, each at one prior width
and the same in every bin or a drift across each group: (a) alpha_2 =
+-12.5 h^-2 Mpc^2 (model A only: model B fits alpha_2); (b) SN2 = +-5
units; (c) the offsets at +-1 sigma of the variant's own widths,
+-(1, 0.5) for model A and +-(2, 1) for model B; (d) the calibration
drift, Delta_2 and Delta_s linear in redshift across each group with
zero mean, changing by 1 and 0.5 from its first bin to its last (the
tilts of the + tilts variants absorb it exactly; model B + tilts'
alpha_0 and alpha_2 tilts are not exercised). The shift is exact for
(a) and (b) and first order for (c) and (d); +- truths score
identically. A model holds at a cut while every score of every truth
stays below sigma/3, there and at every smaller fitted cut:

| layout | baseline scenario: model A / model B | pessimistic scenario: model A + tilts / model B + tilts |
|---|---|---|
| Delta d_com = 62 Mpc | 0.075 / 0.075; at 0.1 the drift moves (w0, wa) by 0.92 / 0.68 sigma | 0.075 / 0.075; at 0.1 alpha_2 moves A_s by 0.40 / the offsets move Omega_m by 0.35 |
| Delta z = 0.02 | 0.075 / 0.075; at 0.1 alpha_2 moves (w0, wa) by 1.03 / the drift by 0.88 | 0.075 / 0.075; at 0.1 alpha_2 moves (w0, wa) by 1.10 / the offsets move Omega_m by 0.43 |
| Delta z = 0.05 | 0.05 / 0.05; at 0.075 the drift moves (w0, wa) by 0.40 / 0.38 | 0.15 / 0.075; alpha_2 moves A_s by 0.53 at 0.175 / the offsets move Omega_m by 0.43 at 0.1 |
| Delta z = 0.2 | 0.05 / 0.05; at 0.075 the drift moves (w0, wa) by 0.40 / 0.36 | every cut to 0.2 |

- In the baseline scenario the calibration drift sets the largest shift
  in every fit from 0.125 on, up to 2.75 sigma; in the pessimistic
  scenario the largest shift reaches 1.26 sigma (model A + tilts) and
  0.43 sigma (model B + tilts), and beyond 0.1 model B + tilts stays at
  0.25-0.34 sigma, near the threshold.
- SN2 at +-5 sets no break point: it moves no score by more than
  0.30 sigma at any cut, in either scenario.
- The break points under the free prior are in data.json.

Diagnostics outside the break point:

- **Prior mismatch.** If the calibrations scatter as widely as model B
  allows, +-(2, 1), model A's tighter offset priors shift its largest
  score by 0.23-0.44 sigma at 0.075 and 0.31-0.75 sigma from 0.1 up.
- **Random alpha_2 truths.** With alpha_2 drawn per group from model B's
  prior (seven times 12.5 h^-2 Mpc^2 at 0.075) and the rest from model
  A's own prior, model A's error bars are honest at 0.05 (eta <= 1.09)
  and on Delta z = 0.2 at every cut (<= 1.004); Delta z = 0.05 stays
  within 1.12 (coverage at least 61%). On the thin layouts they are too
  small at 0.075 and recover as the prior narrows: 62 Mpc 2.1-2.9
  (coverage 25-36%) at 0.075 and 1.02-1.22 at 0.2; Delta z = 0.02
  2.4-3.7 (coverage 15-33%) at 0.075, up to 3.5 for A_s at 0.1, and
  1.02-1.34 at 0.2. Check: with alpha_2 removed, eta = 1 to 4.1e-9.

**The earlier three-truth test** (committed notes, model A alone, truths
(a)-(c)): model A held to 0.075 on the thin layouts, to 0.1 on
Delta z = 0.05 and at every cut on Delta z = 0.2, and model A + tilts as
above; the calibration drift was scored separately in the accuracy test
(Sec. 2). Without SN2 in model A, SN2 = +-5 in the truth moved Omega_m
by 1.25-1.68 sigma at 0.075 on the thin layouts, and model A broke at
0.05 on 62 Mpc and at no fitted cut on Delta z = 0.02
(`fom_and_bias_review.md`, Sec. 6).
