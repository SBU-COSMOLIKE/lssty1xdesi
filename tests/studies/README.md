# DESI DR3 x LSST Y1: optional n(z) reconstruction study

## :interrobang: FAQ: What are optional studies? <a name="optional_studies"></a>

Optional studies explore how numerical settings and modeling choices
affect predictions; they can be expensive, so they run separately from
the routine tests. The shared guide to this n(z) reconstruction study -
the reconstruction definitions, the distance metric, the area
accounting, the figure construction and the run options - is
[The n(z) reconstruction study](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/cosmolike_notebook_utils/README.md#nz_study)
in the CosmoLike core. This file records this survey's measurements.

This survey is the study's stress case: the 76 DESI lens
distributions are 0.05-wide spectroscopic bins, so each histogram is
a near-top-hat a smooth interpolant must trace without ringing, while
the 5 LSST source distributions behave like the photometric surveys
the study was built on.

## The survey figure

![Compiled reconstructions and their bin-area changes](results/reconstructions.png)

## Measured model-vector distances

All arms use the stored NLA example point, this survey's histogram
files, mask and supplied covariance, at the shared study settings;
the distance is the guide's $`\Delta\chi^2`$, judged against practical
stability at 0.2.

| Original → 3 | 3 → 5 | 3 → 7 | 5 → 7 |
|---:|---:|---:|---:|
| 1.5452461 | 0.063152472 | 0.13030892 | 0.014820323 |

Original at its YAML defaults versus the refined study settings:
**Δχ² = 2.2978655** - numerical accuracy alone, without changing
the histogram interpretation.

Both numbers are two to three orders of magnitude above their LSST Y1
analogs (0.021 and 0.0022), and the figure shows why: on a 0.05-wide
spectroscopic bin the original cubic spline rings around the top-hat
edge - overshooting the plateau and dipping below zero outside it -
while the Akima reconstructions trace the edge monotonically. The
interpolant choice and the default numerical settings both sit near
this survey's statistical precision, so neither is a free convention
here: the frozen references pin the original cubic at the YAML
defaults, and moving to an Akima reconstruction is a deliberate,
re-freezing change.

## Measured area accounting

| Reconstruction | Total-area range | Largest redistributed fraction |
|---|---:|---:|
| Original cubic | 0.999948163-1 | 0.441034% |
| 3-point Akima | 1-1 | 0.147087% |
| 5-point Akima | 1-1 | 0.088252% |
| 7-point Akima | 1-1 | 0.063037% |

## Run the study <a name="run_study"></a>

The environment setup, stage selection and accuracy options are in the
shared guide ([Run the study](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/cosmolike_notebook_utils/README.md#nz_run)). For this survey, from
`cocoa/Cocoa/` with the cocoa environment active:

```bash
python -m pytest projects/lssty1xdesi/tests/studies/study_nz_reconstruction.py -v -s
```

or, keeping the results in a chosen folder:

```bash
python projects/lssty1xdesi/tests/studies/run_nz_reconstruction.py --output <folder>
```

To redraw the committed figure without a cosmology calculation:

```bash
python projects/lssty1xdesi/tests/studies/generate_nz_reconstruction_figures.py
```
