# The one-loop bias model: implementation reference

Implementation detail behind SKILL.md, "The bias model". The physics,
derivations and literature are in
`notes/higher_order_bias/higher_order_bias.tex` (sections "The model as
implemented", "Projection to thin-bin angular spectra", "Validation")
and in `notes/correlated_bias_prior_v2/correlated_bias_prior_v2.tex`
(section "Model A", "Model B and the reach of model A", appendix
"Implementation and the Fisher construction"). When this file and a
note disagree on physics, the note is right and this file is fixed.

## 1. Option, parameters and files

`likelihood/desi_cl.py` has two bias options, `higher_order_bias`:

- **0, linear bias.** b_{1,i} = B_g/D(zbar_i) per bin; b2 = bs2 = 0
  (b2 does not follow the calibrations); alpha_2 is still set to zero
  at every point.
- **1, the one-loop model, every term per bin.** b_{1,i}; b_{2,i} =
  b2_Lazeyras(b_{1,i}) + DESI_DB2_i; b_{s^2,i} = -(4/7)(b_{1,i} - 1) +
  DESI_DBS_i; b3 = 0; alpha_0 (DESI_A0_i); alpha_2 (DESI_A2_i); SN0
  (DESI_SN0_i); SN2 (DESI_SN2_i). Every per-bin parameter must be
  declared; a missing one raises by name.

A third option (`higher_order_bias: 2`, a separate model B) existed
briefly and was merged into option 1: models A and B are two parameter
files of the one option. Checks of the merge: model A at SN2 = 0 and
option 1 with the model B file are bitwise identical to the previous
model A and model B, at the fiducial and with nonzero offsets,
alpha_2 and SN2.

Option 1 requires `bias_model: [0,0,0,0,0,0]`: the likelihood hands
cosmolike b2, bs2, b3 = 0 and bK per bin, and bias code 1 of b3
(b3 = b1 - 1) or of bs2 would silently replace them. It also reads the
bin densities from `<stem>_ngal.txt` beside `<stem>_bins.txt` and the
tracer sample of each bin from the bin table.

**Groups are yaml.** `scripts/make_desi_params.py --model-a` and
`--model-b` write `likelihood/params_desi[_<case>]_model_{a,b}.yaml`
(model A, model B) and `..._model_{a,b}_tilts.yaml` (model A + tilts,
model B + tilts; the files were named `_pessimistic` before the
variants were renamed): one sampled offset per group and family
(DESI_DB2_<g>, DESI_DBS_<g>, DESI_A0_<g>, DESI_A2_<g>, DESI_SN0_<g>,
DESI_SN2_<g>) and, for the tilted families, a tilt (DESI_TB2_<g>,
DESI_TBS_<g>, DESI_TA0_<g>, DESI_TA2_<g>) fixed at zero in model A and
model B and free with a flat prior in the + tilts variants, a straight
line in redshift per group like the linear bias's s_g; every bin n of
group g takes
DESI_X_<n> = DESI_X_<g> + DESI_TX_<g> (z_n - zbar_g), with zbar_g the
mean of the group's bin redshifts. Flat tilt ranges per unit redshift:
model A 50 (TB2, TBS) and 5000 (TA0); model B 100 (TB2, TBS) and 1000
(TA0, TA2). Model A fixes DESI_A2 to 0 in every bin. "Group" is one of
BGS, LRG1, LRG2, LRG3, ELG1, ELG2 (the code and file names keep
"chunk").

The thin classes `desi_cl[_dr1_dz0p0x]_model_a` and `_model_b` name the
files; the examples EXAMPLE_EVALUATE_MODEL_A1-A4 and B1-B4 run them on
the four layouts in layout order (62 Mpc, Delta z = 0.02, 0.05, 0.2).
Their chi2 is not near 0: the shipped data vectors are linear bias.

## 2. Terms, units and conventions

- **b2** is the Lazeyras et al. separate-universe fit
  (arXiv:1511.01096), 0.412 - 2.143 b1 + 0.929 b1^2 + 0.008 b1^3, in
  the CCL/FAST-PT normalization cosmolike uses (delta_g contains
  b2 delta^2/2). **bs2** is the co-evolution value of a population
  formed with local bias (arXiv:1405.1447). Both take the full
  b_{1,i}, deviations included: a deviation is a real change of the
  halos a selection reaches, and the calibrations map populations at
  given b1. Near b1 ~ 2 the Lazeyras slope is ~1.7, so a 3% residual
  moves b2 by ~0.10, a tenth of model A's offset width.
- **b3 = 0**, DESI's baseline. C-FASTPT's b3 template (`tab_d1d3` in
  `pt_cfastpt.h`) is a fixed, cosmology-independent table that
  disagrees with Python FAST-PT (Sec. 5); the discarded co-evolution
  value is worth at most 0.3% of P_gg at k <= 0.1 h/Mpc.
- **alpha_0** [h^-2 Mpc^2] rides cosmolike's derivative-bias slot
  2 b1 bK k^2 P_nl. cosmolike measures k in units of H0/c, so the
  likelihood sets bK = alpha_0/[2 b1 (c/H0)^2], c/H0 = 2997.92 Mpc/h
  (`C_OVER_H0`); the model A width 88.9 h^-2 Mpc^2 is 2 b1 bK = 9.9e-6
  in cosmolike units. alpha_0 multiplies the nonlinear P_nl, where
  DESI's multiplies (b1 + f mu^2) b1 P_lin: the coefficients differ by
  b1^2 and by P_nl/P_lin <= 1.16 at k <= 0.2 h/Mpc. At bin i's own cut
  the width 0.5/k_max^2 is a correction of 0.5/B_g^2 of the linear
  galaxy power, the same in every bin of a group: 28% BGS, 17% LRG,
  71% ELG at model A's cut.
- **alpha_2** [h^-2 Mpc^2], the DESI DR1 field convention (Maus et al.,
  arXiv:2404.07312): the galaxy density gains (1/2) f mu^2 alpha_2 k^2
  delta_L, so the power gains P_CT = (b1 + f mu^2) f alpha_2 mu^2 k^2
  P_lin. In this convention alpha_2 k^2 is a fractional correction of
  the share of the Kaiser power carried by the velocity term, whatever
  b1 and f, so a width 0.5/k_max^2 is a 50% correction at k_max in
  every bin (DESI's 12.5 at 0.2 h/Mpc); as a bare power
  alpha_2 mu^2 k^2 P_lin the same width would change meaning from bin
  to bin. The likelihood passes one value per bin through
  `ci.set_nuisance_bias_alpha2(A2=numpy array)`; the C code (`galpha2`
  in bias.c, stored in `nuisance.gb[6]` in (c/H0)^2 units) applies b1
  and f. alpha_2 = 0 in every bin skips the term in C.
- **SN0**, white stochastic power in units of the bin's Poisson power
  P_Pois,i = 1/nbar_i: the deviation from Poisson, since the covariance
  carries the Poisson part. nbar_i is fixed by the bin's surface
  density so that 1/nbar_i projected with the raw histogram is exactly
  the angular shot noise.
- **SN2**, the line-of-sight stochastic power SN2 k^2 mu^2 in units of
  U_i = f_sat sigma_sat^2 P_Pois,i, with f_sat the satellite fraction
  and sigma_sat the satellites' velocity dispersion in distance units
  (`SN2_UNIT`, from the stochastic-prior table of Maus et al.,
  arXiv:2404.07312): LRG 0.1 x 7.8^2 = 6.08 (Mpc/h)^2, ELG
  0.1 x 2.9^2 = 0.84 (Mpc/h)^2. That table has no BGS row; BGS takes the
  LRG values (the larger unit, the wider prior). The choice does not
  matter: SN2 in BGS alone moves no parameter (0.00 sigma at
  k_max = 0.075), with the borrowed unit or with one built from the
  BGS bins' own FoG dispersion (1.5-1.7 (Mpc/h)^2). The code and some
  comments write sigma_v for sigma_sat; it is not the FoG sigma_v.
- **Stochastic cross terms** exist only inside one tracer sample (the
  mean of the two bins' units); an LRG and an ELG bin at the same
  redshift share none, since different galaxies share no stochastic
  term.
- **Zero amplitudes cost nothing**: the SN0 or SN2 templates are not
  computed when every amplitude of the family is zero.
- **Not modeled**: alpha_4 and SN4 (DESI DR1 drops them with the
  hexadecapole), the redshift-space one-loop terms (CosmoLike has none;
  FAST-PT's are not ported to C), and b_nabla^2, exactly degenerate
  with alpha_0 at one loop. Redshift space is linear Kaiser plus the
  sampled FoG convolution of each bin's n(z), a positive damping of
  fixed form that is only partly degenerate with alpha_2.

## 3. The model A / model B prior widths

All Gaussian, centered on zero, per group:

| family | model A | model B |
|---|---|---|
| Delta_2 / Delta_s | 1.0 / 0.5 | 2.0 / 1.0 |
| alpha_0 [h^-2 Mpc^2] | 0.5/k_max^2 (88.9 at 0.075) | 0.5/k_max^2 (yaml: 16.1 at the shipped mask's 0.176 h/Mpc; per cut in the Fisher scan) |
| alpha_2 [h^-2 Mpc^2] | fixed to 0 | 0.5/k_max^2 |
| SN0 [P_Pois] | 2 | 2 |
| SN2 [f_sat sigma_sat^2 P_Pois] | 5 | 5 |
| tilted (+ tilts variant) | Delta_2, Delta_s, alpha_0 | Delta_2, Delta_s, alpha_0, alpha_2 |
| per group, without / with tilts | 5 / 8 | 6 / 10 |

Sources: the offsets cover the scatter of the Lazeyras fit and the 30%
spread between co-evolution and excursion-set bs2 at LRG biases; model
B doubles them, demoting the calibrations to prior centers for cuts
where the loops reach 15% of P_gg. The counterterms take the DESI rule
of a 50% correction at the cut; the stochastic widths are DESI DR1's.
SN0 and SN2 never tilt (their Poisson part is known bin by bin).

The bias-sector budget on Delta d_com = 62 Mpc (76 bins, six groups;
the 64 residuals are the 76 per-bin deviations minus the free amplitude
and tilt of each group):

| | model A | model A + tilts | model B | model B + tilts |
|---|---:|---:|---:|---:|
| linear tilts s_g | 6 | 6 | 6 | 6 |
| residuals held by the kernel prior | 64 | 64 | 64 | 64 |
| Delta_2, Delta_s | 12 | 24 | 12 | 24 |
| alpha_0 / alpha_2 | 6 / - | 12 / - | 6 / 6 | 12 / 12 |
| SN0 / SN2 | 6 / 6 | 6 / 6 | 6 / 6 | 6 / 6 |
| new parameters | 100 | 118 | 106 | 130 |
| without a tight prior, incl. the 6 B_g | 42 | 60 | 48 | 72 |

DESI DR1 full shape uses seven nuisances per tracer, 42 in total at
k_max = 0.2 h/Mpc (arXiv:2411.12021, 2411.12022); model B is that
budget plus one b1 tilt per group. (Tilts exist only in groups of more
than one bin, so the + tilts counts are upper limits on the wide
layouts.)

## 4. Projections and their checks

- **One-loop autos.** The low-rank non-Limber engine factorizes each
  bin's full one-loop P_gg, counterterm included, less the white-noise
  constant P_inf = -(sigma4/2) D^4 (b2 + 2 bs2/3)^2 (sigma4 = 4391
  (Mpc/h)^3 at the likelihood fiducial). P_g - P_inf, the spectrum with
  the unsubtracted templates, stays positive, so its square root
  exists; the Limber density-only term subtracted in the splice uses
  the same unsubtracted templates, and the full Limber term keeps the
  subtracted ones, so P_inf reaches C_ell through Limber, which is
  exact for a constant power. Checks: linear-bias vectors (low-rank,
  FKEM, Limber) and FKEM one-loop vectors bitwise unchanged; no clamps
  at the calibrations; 1 and 8 threads identical; the low-rank one-loop
  vector moves by at most 5e-3 of the shot noise against flooring.
- **One-loop cross pairs.** The Limber pair batch
  (C_gg_cross_tomo_limber_work) adds the pair's one-loop spectrum, the
  auto's loop sum with each product of two biases shared between the
  bins; its density-only output, which the low-rank splice subtracts,
  is the geometric mean W_i W_j sqrt(P_g,i P_g,j), the high-ell limit
  of the low-rank pair term. A pair carries the bilinear constant
  -(sigma4/2) D^4 (b2_i + 2 bs2_i/3)(b2_j + 2 bs2_j/3). Checks: a pair
  of two bins with the same n(z) equals the polarization of the autos
  to 1e-14 of the one-loop increment; linear bias bitwise.
  `set_nuisance_bias_fastpt_tidal` takes bs2 per bin (bs2 != 0 with
  b2 = 0 everywhere is fatal).
- **alpha_0 needs the low-rank engine.** FKEM projects the linear term
  exactly and moves every nonlinear term, the counterterm included,
  through its Limber difference. On Delta z = 0.05 the low-rank
  response dC/dalpha_0 is 7.3x the Limber one at ell = 20, 1.8x at 50,
  1.26x at 100 and ~1.05x at 200-400 (median over kept autos): a thin,
  sharp-edged bin passes radial modes of higher k, which the k^2 weight
  amplifies. It is converged (nonlimber_accuracyboost 32 -> 128, rank
  5 -> 10, accuracyboost 1 -> 2 move it far below one chi2); the two
  engines differ by tens of chi2 for a one-sigma alpha_0.
- **alpha_2** reuses the linear Kaiser transfer Delta_ell^i and its RSD
  row R_ell^i: an auto gains alpha_2 k^2 Delta R under the k integral,
  a pair (1/2) k^2 (alpha_j Delta_i R_j + alpha_i R_i Delta_j); no new
  Bessel integral. It is in every gg engine (C_cl_tomo_core, the
  low-rank core and pair pass, the FKEM pair core, and the Limber gg
  and gg-cross kernels with P_lin). Checks: alpha_2 = 0 bitwise
  (low-rank, FKEM, Limber; Delta z = 0.05 and 62 Mpc); linear to 1e-12;
  a brute-force j_l / j_l'' double integral agrees to <= 6e-4 for one
  thin auto and one adjacent pair at ell = 2-500; 1, 4 and 8 threads
  bitwise; -O0 UBSan clean; in the notebook the forward and backward
  differences of the columns agree to 3.1e-10. Size at 12.5 h^-2 Mpc^2:
  a median 0.7% (Delta z = 0.05) and 1.1% (62 Mpc) of the shipped-mask
  autos, at most 2.9%.
- **SN0** projects exactly: a constant 3D power N gives
  C_ell^ij = N int dchi W_i W_j / chi^2, independent of ell (Bessel
  orthogonality makes Limber exact for white noise), with the
  FoG-convolved kernels of the point (`desi_lens_nz_fog`).
- **SN2** projects exactly too: k^2 mu^2 = k_par^2 is the power of the
  radial derivative of a white noise, so integration by parts gives
  C_ell^ij = U_ij int dchi W_i' W_j' / chi^2 (W' = dW/dchi), the SN0
  formula with each kernel replaced by its radial derivative
  (`stochastic_templates(order=2)`; `order=0` reproduces the SN0
  templates bitwise; linear to 3e-14). Per unit, the SN2 auto template
  is a median 0.022, 0.017, 0.0065 and 0.0017 times the SN0 one
  (62 Mpc, Delta z = 0.02, 0.05, 0.2): sharper edges, larger template.
  A thin bin's W' peaks at its two edges with opposite signs, and
  adjacent bins share an edge, so the normalized adjacent term
  K_ij/sqrt(K_ii K_jj) has a median of -0.50 over the 73 adjacent pairs
  of one sample on 62 Mpc (range -0.54 to -0.46), against +0.05 for
  SN0. White noise adds only positive cross terms, so no SN0, per group
  or per bin, can mimic it. About half of each bin's SN2 auto kernel
  comes from k_par above its own cut at 0.075 (0.58 BGS bin 10, 0.46
  LRG bin 34, 0.51 ELG bin 63; 0.18 / 0.09 / 0.12 at 0.2).
- **Notebook columns.** The Fisher notebooks build per-bin operator
  columns dm/db1_i, dm/db2_i, dm/dbs2_i, dm/dalpha0_i (and dm/dalpha2_i
  for model B) by central differences through the wrapped setters, on
  the coloring of the cross-pair graph (steps: 1% of b1, 0.02 in b2 and
  bs2, 5 h^-2 Mpc^2 in alpha_0); the model A column of a bin's b1 is
  the chain-rule sum dm/db1 + b2_L'(b1) dm/db2 - (4/7) dm/dbs2, and
  summed over a group with weights 1/D(zbar_i) it reproduces the
  stencil derivative of B_g to 1e-8-1e-7 of the column. A step of
  alpha_0 = -5 turns b1^2 + alpha_0 k^2 negative above k ~ 0.6 h/Mpc,
  where the engine floors; a step of 5 against 1 changes the columns by
  at most 3.5e-4 of their largest entry.

## 5. Validation against Python FAST-PT

The reference comparison runs the cosmolike vector against Python
FAST-PT through TATT (equal to NLA with no source sample), IA_code 1
and the fastpt theory block, on the 62 Mpc thin bins, cross pairs
included. No extra reset knob was added for it.

- b3 = 0: delta chi2 <= 3.5e-5, so no retune is needed; doubling the
  C-FASTPT grid moves delta chi2 by ~1e-8; accuracyboost 3-4 moves the
  one-loop part by <= 4.5e-3; the low-rank rank is converged at 5
  (5 -> 12: 1e-10).
- b3 = b1 - 1: delta chi2 = 3.4, because the fixed `tab_d1d3` table is
  wrong even at the fiducial. Hence b3 = 0 in both models.
- `internal_accuracyboost: 4` breaks the C-FASTPT bias tables
  (delta chi2 ~ 1e12); 2 is converged. An open core bug; the likelihood
  ships 1.

## 6. The white-noise constant and positivity

CosmoLike subtracts the low-k constants of the quadratic templates (as
CCL with `sub_lowk`), so the subtracted loop sum tends to P_inf as
k -> infinity. Velocileptors keeps the b2^2 loop unsubtracted and its
SN0 absorbs the constant: the subtracted spectrum with SN0 centered at
zero equals the unsubtracted one with SN0 centered at P_inf, a center
that moves with the biases, so a fixed-center SN0 prior states
different physics in the two conventions. The subtracted convention is
why SN0 keeps DESI's width in Poisson units: unsubtracted, it would
have to absorb 5-42 Poisson units in this forecast's BGS.

**Where P_gg turns negative.** At the calibrations |P_inf| is 5-42x
the Poisson power in the forecast's BGS bins (1/nbar = 33-170
(Mpc/h)^3) and 3-15% of it in ELG; LRG stays positive to
k > 1.5 h/Mpc. BGS one-loop P_gg turns negative near k ~ 0.3 h/Mpc at
the calibrations and from k ~ 0.15 h/Mpc for Delta_2 = -1 (figure
`notes/higher_order_bias/plots/k_neg`; only negative offsets matter,
since b2 + 2 bs2/3 vanishes near Delta_2 ~ +0.9). One model A width
below the calibrations, the exact treatment shows negative C_ell in the
lowest BGS bins beyond k ~ 0.15 h/Mpc on the shipped mask (-29 times
the shot noise at ell = 50; -25 with FKEM), which a floor would hide.
At the model A cut every kept C_ell stays positive (moved by at most
0.27 of the shot noise), and the region beyond k_neg carries at most
0.34% of a BGS bin's white-noise projection.

**Why DESI never sees it.** Its full-shape fits use only
0.02 < k < 0.20 h/Mpc in 3D, where a thin bin's radial modes do not
enter, and DR1 is sparse (BGS 1/nbar ~ 3e3 (Mpc/h)^3), so the constant
is about Poisson-sized and fits the SN0 prior in either convention. The
forecast's BGS is 20-100 times denser.

**Negative alpha_0.** At alpha_0 = -88.9 h^-2 Mpc^2 (one model A width
below zero) P_gg - P_inf turns negative above k ~ b1/sqrt(|alpha_0|)
~ 0.14 h/Mpc on 47% of the engine's grid, the engine floors it, and
the vector curves in alpha_0 (delta chi2 18.9 of nonlinearity over
+-1 sigma) while FKEM's stays linear. Harmless for the Fisher forecast
at the fiducial; a chain that explores such points leaves the validated
model.

**Remedies (not implemented).** A taper of the loop terms above the
scale where one-loop perturbation theory is trusted (near 0.15 h/Mpc
for BGS), or a positivity prior on P_gg for chains, like the amplitude
cut. A smaller BGS k_max avoids the region; a tighter b2 prior only
shifts k_neg, and SN0 adds the same constant at every k, so neither
fixes the sign.

## 7. Prior choices that were not adopted

- **DESI DR1's agnostic priors** b2 sigma8^2, bs sigma8^2 ~ N(0, 5^2)
  (Lagrangian; widths of order 10-25 in this b2) are not adopted for
  model A: at k_max = 0.075 the data barely measure b2, and for the
  dense BGS a low b2 drives P_gg negative. A DESI-like variant is only
  an optional bound.
- **No DESI chain values as fiducials.** Fiducial higher-order values
  are the calibrations (every fiducial offset, counterterm and
  stochastic amplitude is zero). Converted DESI posteriors are prior-
  and degeneracy-dominated (b_s ~ +4 to +6) and set prior widths only.
  The draft paper's tidal-bias column holds DESI sampling combinations,
  not an Eulerian b_s (`fom_and_bias_review.md`, Sec. 4).
- **Free b2, bs2 per bin** (152 parameters on 62 Mpc) is a bound, not a
  model: a bin's loop terms are nearly degenerate with its b1 and
  counterterm, so the prior would set the per-bin posteriors.
- **Exact calibrations** (no offsets, counterterms or stochastic terms)
  give FoMs above linear bias, because the loops grow as D^4 against
  b1^2 D^2 and carry amplitude information that is only as good as the
  calibrations. No FoM rests on exact calibrations; they appeared only
  as diagnostics of the model A notebook and its
  `baseline_ladder/data.json`, both removed on 2026-11-10 (commit
  567751f holds them).
