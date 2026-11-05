# The lssty1xdesi project

A joint forecast of DESI DR3 spectroscopic galaxy clustering and LSST
Y1 cosmic shear in Fourier band powers. The galaxy sample is split
into 76 thin comoving lens bins (BGS, LRG1-3, ELG1-2; mean redshifts
0.11 to 1.55; `data/lssty1xdesi_desi_dr3_bins.txt` names each bin's
chunk and `data/lssty1xdesi_desi_dr3_ngal.txt` its density). The data
vector holds 15 shear pairs, 76 clustering autos and 97 adjacent-bin
clustering crosses, each in 53 bands of width $\Delta\ell = 10$
covering $15 \le \ell \le 544$ (9,964 points).

The thin bins are the project's defining numerical challenge: their
radial kernels keep real non-Limber power far past the multipole
where photometric surveys hand off to Limber, and the finger-of-god
smearing couples neighboring bins. The likelihood therefore computes
the full density-density term with the low-rank non-Limber engine
(`adopt_limber_gg: 0`, `nonlimber_lowrank_gg: 1`) and rebuilds the 76
lens distributions through the FoG convolution at every evaluation
(`external_nz_modeling: 1`). The bins share their nuisance
parameters by sample: one linear bias per sample chunk (`DESI_B1_*`:
BGS, LRG1-3, ELG1-2) and seven velocity dispersions (`DESI_SIGV_*`,
with BGS split in two).

- `likelihood/desi_cl.py` scores the band-power vector;
  the `combo_*` variants expose the probe subsets.
- `EXAMPLE_EVALUATE1.yaml` evaluates the fiducial point (a commented
  `bfmt` theory block shows how to turn each baryonic-feedback model
  on); `EXAMPLE_EMUL2_EVALUATE1.yaml` is the hybrid-emulator variant,
  with MCMC, Nautilus and profile examples alongside.
- [tests/](tests/README.md) documents the test suites: frozen-reference
  and race checks, accuracy-knob scans (including the project's own
  non-Limber controls), Halofit vs EuclidEmulator2, baryonic feedback,
  photo-z conventions, the non-Limber and low-rank engine checks, the
  cache ladder, and the optional n(z) reconstruction study.
- [covariance/](covariance/README.md) documents covariance generation:
  the Gaussian part with RSD and non-Limber clustering spectra, Limber
  SSC and cNG, the comparison against the source forecast, and the
  `EXAMPLE_EVALUATE_COVARIANCE.ipynb` notebook.
- `EXAMPLE_EVALUATE2.yaml` to `EXAMPLE_EVALUATE4.yaml` evaluate the
  DESI DR1 binnings of fixed width in redshift, and
  `EXAMPLE_EVALUATE5.yaml` the LSST Y1 3x2pt case (the two sections
  below).
- `EXAMPLE_FISHER1.ipynb` forecasts the DESI clustering, LSST 3x2pt
  and combined constraints, and the parameter biases of the
  projection approximations (the last section below).

The survey inputs (densities, fiducial biases, $f_{\rm sky} = 0.3394$,
the supplied comparison covariance) are ported from the source DESI
DR3 x LSST Y1 forecast; `covariance/README.md` records the
model-choice differences the comparison revealed. Scale-cut choices
and release notes belong here as the analysis matures.

## The fixed-width Δz cases <a name="dz_cases"></a>

The comoving DR3 layout above splits the samples into bins of fixed
comoving width. The DESI DR1 forecast distributions instead split
BGS, LRG and ELG into bins of fixed width $\Delta z$ in redshift. Three
widths are set up as separate cases of the same likelihood, each with
its own data files (`data/lssty1xdesi_desi_dr1_dz0pXX*`), likelihood
class (`likelihood/desi_cl_dr1_dz0pXX.py`), parameter file and
EXAMPLE yaml:

| Case | $\Delta z$ | Lens bins | Cross pairs | Entries (unmasked) | `nonlimber_accuracyboost` | EXAMPLE |
|---|---:|---:|---:|---:|---:|---|
| `dr1_dz0p20` | 0.2 | 10 | 13 | 2,014 (744) | 8 | `EXAMPLE_EVALUATE4.yaml` |
| `dr1_dz0p05` | 0.05 | 36 | 52 | 5,459 (2,982) | 32 | `EXAMPLE_EVALUATE3.yaml` |
| `dr1_dz0p02` | 0.02 | 90 | 133 | 12,614 (7,621) | 64 | `EXAMPLE_EVALUATE2.yaml` |

The distributions come without bandpowers, pairs or scale cuts, so
`scripts/make_desi_dz_case.py` builds them with the conventions of the
DR3 sacc file. Its docstring states each rule; in short:

- **bands**: the 53 DR3 bands;
- **cross pairs**: every two bins whose distributions overlap or
  touch;
- **scale cuts**: an auto spectrum keeps the bands with
  $\ell + 1/2 \le k_\mathrm{max}\,\chi(\bar z)$, with
  $k_\mathrm{max} = 0.1185\ \mathrm{Mpc}^{-1}$, $\chi$ the comoving
  distance and $\bar z$ the bin's mean redshift; a cross pair keeps
  the bands both of its autos keep;
- **densities**: only the per-bin `ngal` of the forecast's
  `probes_properties` files is used. Their other entries are not
  current.

Narrower bins need a finer FFTLog grid along the line of sight. The
non-Limber integrand oscillates on the scale of the bin's width, and
a coarse grid aliases it. Each case sets the smallest
`nonlimber_accuracyboost` whose $\Delta\chi^2$ against boost 128 stays
below 0.01 at that boost and at every larger one. The measured ladders
are in each likelihood yaml.

The covariance of each case is the Gaussian term of the project's
covariance code, at the DESI footprint of 14,000 deg². It is stored
sparse in the cosmolike text format (only nonzero entries, upper
triangle). Like every covariance file under `data/`, it is built
locally and not tracked by git (the second and third steps below);
the test snapshot in `tests/frozen/data/` keeps a copy.

A case is rebuilt in four steps, shown here for $\Delta z = 0.02$. The
DR1 distribution files are not distributed with the project:
`--input-dir` names their folder. The second step needs the project
compiled with its covariance interface ([covariance/](covariance/README.md)).

```bash
python ./projects/lssty1xdesi/scripts/make_desi_dz_case.py --width 0.02 --stage files --input-dir /path/to/dr1_inputs
```

```bash
python ./projects/lssty1xdesi/covariance/compute_covariance.py ./projects/lssty1xdesi/EXAMPLE_EVALUATE_COVARIANCE2.yaml
```

```bash
python ./projects/lssty1xdesi/scripts/make_desi_dz_case.py --width 0.02 --stage covariance
```

```bash
python ./projects/lssty1xdesi/scripts/make_desi_dz_case.py --width 0.02 --stage fiducial
```

The last step writes the model vector at the EXAMPLE's fiducial point
as the case's data vector, so the EXAMPLE evaluates to
$\chi^2 \approx 0$.

## The LSST Y1 3x2pt case <a name="lsst_case"></a>

`EXAMPLE_EVALUATE5.yaml` evaluates LSST Y1 alone. It has cosmic shear,
galaxy-galaxy lensing and clustering, all in band powers on the same
53 bands, which is what lets its Fisher matrix add to a DESI case's.
The lens and source distributions are the lsst_y1 project's
(`data/lssty1xdesi_lens.nz`, `data/lssty1xdesi_source.nz`, five bins
each, read with that project's convention: the z column is the lower
edge of each histogram bin). The likelihood
`likelihood/lsst_y1_3x2pt.py` is `combo_3x2pt` in Fourier space: FKEM
non-Limber clustering and galaxy-galaxy lensing below $\ell = 150$.

| Block | Spectra | Entries | Kept by the mask |
|---|---:|---:|---|
| cosmic shear | 15 | 795 | every band |
| galaxy-galaxy lensing | 25 | 1,325 | the lens bin's clustering cut |
| clustering | 5 | 265 | $\ell + 1/2 \le k_\mathrm{max}\,\chi(\bar z)$ |

That leaves 1,629 unmasked entries out of 2,385.

The covariance is the Gaussian term at 12,300 deg², with the
shape noise and densities of the LSST Y1 science requirements
(sources 2 arcmin⁻² per bin and $\sigma_e = 0.26$ per component,
lenses 3.6 arcmin⁻² per bin). The shear and lensing signals in it
include nonlinear-alignment intrinsic alignments, with the fiducial
power law evaluated at each source bin's mean redshift.
`scripts/make_lsst_3x2pt_case.py` builds the case with the three
stages of the Δz builder (`--stage files`, then the covariance CLI on
`EXAMPLE_EVALUATE_COVARIANCE5.yaml`, then `--stage covariance` and
`--stage fiducial`).

## Runtimes <a name="runtimes"></a>

The EXAMPLE yamls evaluated one at a time with `cobaya-run`, on an
Apple M2 Pro laptop on AC power with 8 OpenMP threads (measured
2026-11-05). The likelihood column is cobaya's timing of the
likelihood alone. CAMB adds 0.7 s to every evaluation. The whole run
includes the one-time initialization, which dominates it.

| EXAMPLE | case | likelihood evaluation | whole run |
|---|---|---:|---:|
| `EXAMPLE_EVALUATE1.yaml` | DR3, comoving bins | 7.4 s | 132 s |
| `EXAMPLE_EVALUATE2.yaml` | DR1, $\Delta z = 0.02$ | 8.9 s | 264 s |
| `EXAMPLE_EVALUATE3.yaml` | DR1, $\Delta z = 0.05$ | 2.3 s | 27 s |
| `EXAMPLE_EVALUATE4.yaml` | DR1, $\Delta z = 0.2$ | 0.44 s | 4.9 s |
| `EXAMPLE_EVALUATE5.yaml` | LSST Y1 3x2pt | 0.054 s | 4.7 s |

These are laptop numbers, not a benchmark of the production
supercomputers. The test guide
([tests/data_vector/](tests/data_vector/README.md#lowrank_check))
compares the low-rank engine's and FKEM's steady-state times on each
DESI layout.

## Fisher forecasts <a name="fisher"></a>

`EXAMPLE_FISHER1.ipynb` forecasts the $w_0w_a$CDM constraints of
three probes on the Δz = 0.2 binning:

- DESI clustering alone;
- LSST Y1 3x2pt;
- their combination as independent data sets (a block-diagonal
  "6x2pt": no LSST×DESI cross spectra and no cross-covariance).

Every probe uses one area, 12,300 deg², and is marginalized over all
of its nuisance parameters. The derivatives are recomputed at four
stencil steps and at two accuracy boosts, and a forecast is trusted
only where neither moves it. At accuracy boost 3 the dark-energy
figure of merit FoM($w_0$, $w_a$) agrees to 0.6% across the steps:
about 5.7 for DESI clustering, 17.4 for LSST 3x2pt and 32.2 for the
combination.

> [!WARNING]
> The forecast needs CAMB installed with
> `PATCH_CAMB_HALOFIT_TOLERANCE` in `set_installation_options.sh`.
> Stock CAMB stops halofit's search for the nonlinear scale at a
> tolerance of $10^{-3}$, and the data vectors jump with it between
> nearby cosmologies. The derivatives then depend on the step: the
> LSST figure of merit ranged from 14 to 52 across the four steps.
> The notebook's first cell reports whether the patch is installed.

One Fisher matrix at one step costs four evaluations per varied
parameter: 20 for DESI, 29 for LSST. On the laptop of the
[runtime table](#runtimes), one evaluation takes:

| probe | accuracy boost 1 | accuracy boost 3 |
|---|---:|---:|
| DESI, $\chi$-grid boost 64 | 3.8 s (5 min per step) | 14.3 s (19 min per step) |
| LSST Y1 3x2pt | 0.8 s (1.5 min per step) | 2.4 s (4.6 min per step) |

The notebook's four steps at both boosts take about two hours. It
saves the derivatives to one cache file per boost, so a second run
only redraws the results.

The same derivatives give **Fisher biases**: the shift of the best-fit
cosmology when the model fitted to the data projects differently from
the one that generated them. The notebook compares the FKEM split with
the low-rank engine, and Limber with non-Limber, for each probe and
for the combination. At an equal multipole cap, FKEM shifts no DESI
parameter by more than 0.01σ on the Δz = 0.2 bins, and the low-rank
engine shifts no LSST parameter by more than 0.003σ. Limber clustering
on the DESI bins shifts parameters by up to 5.6σ alone, and by up to
2.1σ in the combination.

The machinery is `interface/cosmolike_lssty1xdesi_fisher.py`. Its
`ModelFisher` builds the cobaya model of an EXAMPLE yaml and
differentiates exactly the data vector the likelihood scores. The
compiled interface holds one model at a time, so each probe's Fisher
matrix is computed while its model is the latest one built.
