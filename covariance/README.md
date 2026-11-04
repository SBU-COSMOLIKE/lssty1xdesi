# Table of contents
1. [Overview](#overview)
2. [Running from the command line](#command_line)
3. [Running the covariance notebook](#running)
4. [Checking against the source forecast](#source_forecast)
5. [Files](#files)
6. [Appendix](#appendix)
    1. [FAQ: Which survey does the calculation use?](#survey)
    2. [FAQ: What does the Gaussian term include?](#gaussian)
    3. [FAQ: Which accuracy settings are available?](#accuracy-settings)

# Overview <a name="overview"></a>

The covariance matrix $`C`$ of the data vector gives the expected
scatter of each measured bandpower and the correlations between
entries; the likelihood's $\chi^2$ uses its inverse. The calculation
sums three terms: Gaussian (G: cosmic variance, shape noise and shot
noise), super-sample covariance (SSC) and the connected non-Gaussian
term (cNG). The shared kernels live in
`external_modules/code/cosmolike_core/cosmolike/covariances/`;
`lssty1xdesi_covariance.py` states this survey and forwards the
calculation.

The measured layout has 9964 Fourier entries, 53 bandpowers per
spectrum:

| block | spectra | entries |
|---|---|---|
| EE | 15 LSST Y1 source pairs | 0 - 794 |
| gg | 76 DESI thin-bin autos | 795 - 4822 |
| gg cross | 97 DESI measured bin pairs | 4823 - 9963 |

No galaxy-galaxy lensing rows are measured. The cross pairs follow
`data/lssty1xdesi_desi_dr3_pairs.txt`, the order of the data vector,
the mask and the shipped covariance.

The shipped covariance (`data/lssty1xdesi_desi_dr3_cov`) is imported
from the source forecast's sacc file. This module computes an
independent matrix with cosmolike's own kernels, so the two can be
compared entry by entry (see
[Checking against the source forecast](#source_forecast)).

Two model choices differ from the wide-bin projects:

- the Gaussian spectra carry the lens redshift-space distortion
  window (`gaussian: rsd: true`): it moves the lowest bands of the
  thin-bin rows onto the source forecast (the measured table below);
- the density-only non-Limber recompute stays off
  (`gaussian: nonlimber: false`): the covariance non-Limber transfers
  carry density only and refuse the distortion window.

The covariance cosmology sets $`m_\nu = 0`$: the halo forecast behind
SSC and cNG requires massless neutrinos, and the sub-percent
$`C_\ell`$ change sits far below the parameter-error accuracy a
covariance needs.

# Running from the command line <a name="command_line"></a>

The CLI evaluates the full matrix at one cosmology and saves G, SSC,
cNG, their sum, the row coordinates and the resolved settings in one
`.npz` archive.

**Step :one:**: activate Cocoa.

    conda activate cocoa
    source start_cocoa.sh

**Step :two:**: enable covariance generation.

    unset IGNORE_COSMOLIKE_LSSTY1XDESI_COVARIANCE

**Step :three:**: compile the project interface.

    source projects/lssty1xdesi/scripts/compile_lssty1xdesi.sh

**Step :four:**: select the number of OpenMP cores (below, 8).

    export OMP_NUM_THREADS=8

**Step :five:**: run the CLI.

    python projects/lssty1xdesi/covariance/compute_covariance.py \
        projects/lssty1xdesi/EXAMPLE_EVALUATE_COVARIANCE.yaml

> [!NOTE]
> A library compiled without covariance support stops the script with
> a message naming the key of Step :two:.

# Running the covariance notebook <a name="running"></a>

**Step :one:**: activate Cocoa and enable covariance generation as in
Steps :one: and :two: above, then compile the interface (Step
:three:).

**Step :two:**: start Jupyter.

    jupyter lab

**Step :three:**: open `projects/lssty1xdesi/EXAMPLE_EVALUATE_COVARIANCE.ipynb`
and select **Kernel → Restart Kernel and Run All Cells**.

# Checking against the source forecast <a name="source_forecast"></a>

The Knox formula with the project's own exact band values closes the
bookkeeping: building the Gaussian diagonal from the fiducial data
vector's spectra (plus shot noise and the per-band mode count
$`f_{\rm sky}\sum_\ell (2\ell+1)`$) reproduces the shipped diagonal at
the lowest-redshift rows,

| row | band | Knox from the data vector / shipped |
|---|---|---|
| gg (0,0) | first | 1.007 |
| gg cross (0,1) | first | 1.018 |

so the footprint, the noise units, the band supports and the mode
counting all match the source forecast exactly.

The computed Gaussian diagonal then brackets the source forecast's
within the band a covariance needs. Per model choice,
$`G_{\rm computed}/G_{\rm shipped}`$ at the unmasked bands of seven
representative clustering rows:

| Gaussian spectra | lowest band | middle bands |
|---|---|---|
| Limber + distortion window (the default) | 0.72 - 1.18 | 1.1 - 1.9 |
| Limber, density only | 0.60 - 1.10 | 1.1 - 1.9 |
| non-Limber, density only | 0.40 - 0.63 | 0.87 - 1.4 |

The distortion window closes the lowest bands; the density-only
non-Limber recompute instead repairs the middle bands, where the
Limber projection genuinely overestimates a thin bin's
density $`C_\ell`$, but loses the lowest ones because the distortion
dominates there and its transfers cannot carry it. A distortion
transfer in `cosmolike/covariances/nonlimber_cov.c` would reconcile
the two regimes; the residual middle-band excess of the default also
contains the fiducial-model differences the data-vector comparison
against the sacc file shows (one-loop bias terms, the finger-of-god
damping).

One qualitative difference matters at high multipole: adjacent-bin
cross bandpowers of the source forecast turn negative (finger-of-god
smearing anticorrelates neighboring thin bins), while the project's
geometric-mean cross spectra are nonnegative by construction. The
covariance model shares whichever choice the data vector makes.

# Files <a name="files"></a>

| file | role |
|---|---|
| `lssty1xdesi_covariance.py` | survey adapter: layout rows, bands, densities, biases, cosmology |
| `default.yaml` | numerical accuracy of this survey's calculation |
| `compute_covariance.py` | the production CLI |
| `../EXAMPLE_EVALUATE_COVARIANCE.yaml` | cosmology and run choices of the CLI |
| `../EXAMPLE_EVALUATE_COVARIANCE.ipynb` | the notebook workflow |
| `../data/lssty1xdesi_desi_dr3_ngal.txt` | per-bin DESI densities [arcmin^-2] |

# Appendix <a name="appendix"></a>

## FAQ: Which survey does the calculation use? <a name="survey"></a>

The DESI DR3 forecast footprint of 14000 deg$`{}^2`$
($`f_{\rm sky} = 0.3394`$), the 76 thin comoving lens bins of
`data/lssty1xdesi_desi_dr3.nz` with the per-bin densities of
`data/lssty1xdesi_desi_dr3_ngal.txt`, and the five LSST Y1 source
bins of `data/lssty1xdesi_source.nz` with density 2.0 per
arcmin$`{}^2`$ and shape noise 0.26 per ellipticity component per
bin. The fiducial linear biases are the `DESI_B1` leader values of
the EXAMPLE yaml files, one value per sample chunk:

    BGS: 1.34
    LRG1, LRG2, LRG3: 1.70
    ELG1, ELG2: 0.84

The bandpowers average $`C_\ell`$ with $`(2\ell+1)`$ weights over ten
consecutive multipoles per band, band $`b`$ covering
$`[15+10b,\ 24+10b]`$ for $`b = 0..52`$ - the supports read from
`data/lssty1xdesi_desi_dr3_windows.npz`.

## FAQ: What does the Gaussian term include? <a name="gaussian"></a>

Linear per-bin galaxy bias over the nonlinear matter power, the lens
redshift-space distortion window in its Limber form, shot and shape
noise, and the exact per-band mode counting. The data vector's
finger-of-god dispersion and one-loop bias terms are not part of the
covariance model: covariance accuracy targets are parameter-error
level, far looser than the data-vector $\chi^2$ rule.

Two engine limits shape the choices here:

- the covariance non-Limber transfers carry density only, so
  `gaussian: rsd: true` requires `gaussian: nonlimber: false`; adding
  a distortion transfer to
  `cosmolike/covariances/nonlimber_cov.c` would reconcile the lowest
  and middle bands (the comparison table above);
- the distortion window's Limber form decays by $`\ell \approx 25`$,
  so at higher multipoles the default's spectra are plain Limber.
  Supplying the Gaussian assembly with the data-vector engine's exact
  spectra is the other path to one model across all bands.

## FAQ: Which accuracy settings are available? <a name="accuracy-settings"></a>

`default.yaml` holds the numerical controls; `configuration()`
accepts single-entry overrides (for example
`integration_accuracy=1`). The non-Limber controls
(`nonlimber_lmax`, `nonlimber_accuracyboost`) are read only when
`gaussian: nonlimber` is true, which this project's default is not.
