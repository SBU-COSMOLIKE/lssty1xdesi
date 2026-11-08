# Table of contents <a name="table_of_contents"></a>

1. [The DESI × LSST Y1 forecast](#forecast)
2. [Reading path through the examples](#reading_path)
3. [Running Cosmolike projects (Basic instructions)](#running_cosmolike_projects)
4. [The DESI DR1 fixed-width Δz cases](#dz_cases)
5. [The LSST Y1 3×2pt case](#lsst_case)
6. [The one-loop bias model](#higher_order_bias)
7. [Baryonic feedback on EXAMPLE_EVALUATE1](#baryonic_feedback)
8. [Nested sampling with Nautilus](#nautilus)
9. [Running Fisher](#fisher)
10. [Unit tests](#unit_tests)
11. [Computing covariances](#computing_covariances)
12. [Exploring notebooks](#notebooks)
13. [Appendix: Which accuracy settings are available?](#accuracy)

# The DESI × LSST Y1 forecast <a name="forecast"></a>

This project forecasts DESI spectroscopic galaxy clustering joint
with LSST Y1 cosmic shear in Fourier band powers: angular
clustering in thin tomographic bins, not a 3D power-spectrum
analysis. The galaxy sample is split into 76 lens bins of equal
comoving width, about 62.3 Mpc - the layout the project calls
$`\Delta d_\mathrm{com} = 62`$ Mpc after that width, with
$`d_\mathrm{com}`$ the comoving distance (BGS, LRG1-3,
ELG1-2; mean redshifts 0.11 to 1.55;
`data/lssty1xdesi_desi_dr3_bins.txt` names each bin's sample chunk and
`data/lssty1xdesi_desi_dr3_ngal.txt` its density). The data vector
holds 15 shear pairs, 76 clustering autos and 97 measured cross-bin
clustering pairs, each in 53 bands of width $`\Delta\ell = 10`$
covering $`15 \le \ell \le 544`$: 9,964 entries, 5,309 of them
unmasked. No galaxy-galaxy lensing rows are measured
(`ggl_exclude: all`).

Each DESI layout's data vector is a model at the fiducial point, not a
measurement. `data/lssty1xdesi_desi_<case>.dataset`, which
`EXAMPLE_EVALUATE1.yaml` to `EXAMPLE_EVALUATE4.yaml` read, holds the
paper's bias model (model B + tilts on the pessimistic linear-bias
prior, [one-loop bias](#higher_order_bias)) at its fiducial, so those
examples evaluate to $`\chi^2 \approx 0`$.
`data/lssty1xdesi_desi_<case>_linear.dataset` differs only in its data
vector, the growth-tied linear bias at the same point, and every
example in [exploration/](exploration/README.md) reads it.
`scripts/make_desi_fiducial.py` writes either vector from its example
(`<case>` = `dr3` for $`\Delta d_\mathrm{com} = 62`$ Mpc, `dr1_dz0p02`,
`dr1_dz0p05`, `dr1_dz0p20`).

The thin bins are the project's defining numerical challenge. Their
radial kernels keep real non-Limber power far past the multipole where
photometric surveys hand off to Limber, and the Fingers-of-God
smearing couples neighboring bins. The likelihood therefore computes
the full density-density term with the low-rank non-Limber engine
(`adopt_limber_gg: 0`, `nonlimber_lowrank_gg: 1`) and rebuilds the 76
lens distributions through the Fingers-of-God convolution at every
evaluation (`external_nz_modeling: 1`): a velocity dispersion
$`\sigma_v`$ smears observed redshifts by
$`\sigma_z = (1 + \bar z)\,\sigma_v/c`$, and the bins are far narrower
than that smearing.

The bins share their nuisance parameters by sample: one linear bias
per sample chunk (`DESI_B1_*`: BGS, LRG1-3, ELG1-2) and seven sampled
velocity dispersions (`DESI_SIGV_*`, with BGS split in two).
`likelihood/desi_cl.py` scores the band-power vector; the `combo_*`
variants expose the probe subsets. The survey inputs (densities,
fiducial biases, $`f_{\rm sky} = 0.3394`$, the supplied comparison
covariance) are ported from the sacc file of the source forecast,
a DESI DR3-like × LSST Y1 forecast;
the [covariance guide](covariance/README.md#source_forecast) records
the model-choice differences the comparison revealed.

# Reading path through the examples <a name="reading_path"></a>

Start with one evaluation of `EXAMPLE_EVALUATE1.yaml`
([basic instructions](#running_cosmolike_projects)): it checks the
installation that every later example uses.

```mermaid
flowchart TB
  A["EXAMPLE_EVALUATE1.yaml: first evaluation (Δd_com = 62 Mpc comoving bins)"] --> B["EXAMPLE_EVALUATE2-4.yaml: DESI DR1 fixed-width bins"]
  A --> C["EXAMPLE_EVALUATE5.yaml: LSST Y1 3x2pt in band powers"]
  A --> D["EXAMPLE_MCMC1.yaml: chains"]
  A --> E["EXAMPLE_NAUTILUS1.py: nested sampling"]
  A --> F["EXAMPLE_EVALUATE_COVARIANCE.ipynb: covariance terms"]
  F --> G["covariance/compute_covariance.py: production covariance"]
  B --> H["EXAMPLE_EVALUATE_FISHER.ipynb: forecasts and projection biases on every DESI layout"]
  C --> H
```

- `EXAMPLE_EVALUATE1.yaml` evaluates the fiducial point of the
  $`\Delta d_\mathrm{com} = 62`$ Mpc comoving binning with the paper's
  bias model, the one-loop model B + tilts on the pessimistic
  linear-bias prior ([one-loop bias](#higher_order_bias)); a commented
  `bfmt` theory block shows how to turn each baryonic-feedback model on
  ([baryonic feedback](#baryonic_feedback)).
- `EXAMPLE_EVALUATE2.yaml` to `EXAMPLE_EVALUATE4.yaml` evaluate the
  DESI DR1 binnings of fixed width in redshift with the same model, and
  `EXAMPLE_EVALUATE5.yaml` the LSST Y1 3×2pt case
  ([Δz cases](#dz_cases), [LSST case](#lsst_case)).
- [exploration/](exploration/README.md) holds the examples of the other
  bias models, on the linear data vectors: the growth-tied linear bias
  (`EXAMPLE_EVALUATE_LINEAR1.yaml` to `EXAMPLE_EVALUATE_LINEAR4.yaml`),
  model A, and model B on the growth-tied linear bias.
- `EXAMPLE_MCMC1.yaml` samples the paper's model on the
  $`\Delta d_\mathrm{com} = 62`$ Mpc layout, from the Fisher proposal
  covariance `EXAMPLE_MCMC1.covmat`.
- `EXAMPLE_NAUTILUS1.py` runs the Nautilus nested sampler on the model
  of `EXAMPLE_EVALUATE1.yaml` ([Nautilus](#nautilus)). The linear-bias
  MCMC is in [exploration/](exploration/README.md).
- `EXAMPLE_EVALUATE_FISHER.ipynb` forecasts, on every DESI layout, the
  constraints of DESI clustering, LSST Y1 3×2pt and their combination,
  and the parameter biases of the projection approximations
  ([running Fisher](#fisher)).
- The covariance notebook and the production CLI need the optional
  covariance build ([computing covariances](#computing_covariances)).
- [tests/](tests/README.md) checks the predictions against stored
  references ([unit tests](#unit_tests)).

# Running Cosmolike projects (Basic instructions) <a name="running_cosmolike_projects"></a>

From `Cocoa/Readme` instructions:

> [!Note]
> `setup_cocoa.sh` and `compile_cocoa.sh` install and compile the cosmolike projects selected in `set_installation_options.sh`. A commented `IGNORE_COSMOLIKE_*_CODE` key enables a project; an active key skips it. This project ships skipped: comment out its key before running the two scripts.
>
>     [Adapted from Cocoa/set_installation_options.sh shell script]
>     (...)
>     # WARNING: lssty1xdesi is under active development.
>     export IGNORE_COSMOLIKE_LSSTY1XDESI_CODE=1
>     (...)
>     export LSSTY1XDESI_URL="https://github.com/SBU-COSMOLIKE/lssty1xdesi.git"
>     export LSSTY1XDESI_NAME="lssty1xdesi"
>     export LSSTY1XDESI_GIT_BRANCH="main"
>
> To select another revision, set only one of the project's `GIT_COMMIT`, `GIT_BRANCH` or `GIT_TAG` keys (here `LSSTY1XDESI_GIT_COMMIT`, `LSSTY1XDESI_GIT_BRANCH` or `LSSTY1XDESI_GIT_TAG`): a commit takes precedence over a branch, and a branch over a tag.

> [!NOTE]
> If users want to recompile cosmolike, there is no need to rerun the Cocoa general scripts. Instead, run the following three commands:
>
>      source start_cocoa.sh
>
> and
>
>      source ./installation_scripts/setup_cosmolike_projects.sh
>
> and
>
>       source ./installation_scripts/compile_all_projects.sh
>
> or (in case users just want to compile this project)
>
>       source ./projects/lssty1xdesi/scripts/compile_lssty1xdesi.sh

> [!TIP]
> Assuming Cocoa is installed on a local (not remote!) machine, type the command below after step 2️⃣ to run Jupyter Notebooks.
>
>     jupyter notebook --no-browser --port=8888
>
> The terminal will then show a message similar to the following template:
>
>     (...)
>     [... NotebookApp] Jupyter Notebook 6.1.1 is running at:
>     [... NotebookApp] http://f0a13949f6b5:8888/?token=XXX
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>     [... NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
>
> Now go to the local internet browser and type `http://127.0.0.1:8888/?token=XXX`, where XXX is the previously saved token displayed on the line
>
>     [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
>
> The project contains jupyter notebook examples located at `projects/lssty1xdesi`.

> [!NOTE]
> The example notebooks load their shared support functions from
> `Cocoa/external_modules/code/cosmolike_core/cosmolike_notebook_utils/`,
> called through the `cnu` namespace: the CAMB run packaged for
> cosmolike, the data-vector helpers, and the Fisher-forecast helpers.
> What is specific to this project lives in
> `interface/cosmolike_lssty1xdesi_fisher.py` (the `ModelFisher` class
> of the [Fisher examples](#fisher)).

To run the example

 **Step :one:**: activate the Cocoa Conda environment,  and the private Python environment

      conda activate cocoa

and

      source start_cocoa.sh

 **Step :two:**: Select the number of OpenMP cores (the commands below set it to 8).

  - Linux

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

 **Step :three:**: The folder `projects/lssty1xdesi` contains examples. So, run the `cobaya-run` on the first example following the commands below.

> [!Warning]
> (Linux only) In some HPC nodes, `numa` can cause you problems. If that is the case,
> replace `numa` with `slot`

- **One model evaluation**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 1 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/lssty1xdesi/EXAMPLE_EVALUATE1.yaml -f

  - macOS (arm)

        mpirun -n 1 --oversubscribe \
          cobaya-run ./projects/lssty1xdesi/EXAMPLE_EVALUATE1.yaml -f

- **MCMC (Metropolis-Hastings Algorithm)**:

  - Linux

        "${CONDA_PREFIX}"/bin/mpirun -n 4 --oversubscribe --mca pml ob1 --mca btl vader,tcp,self \
          --bind-to core:overload-allowed --report-bindings \
          --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
          cobaya-run ./projects/lssty1xdesi/EXAMPLE_MCMC1.yaml -f

  - macOS (arm)

        mpirun -n 4 --oversubscribe \
          cobaya-run ./projects/lssty1xdesi/EXAMPLE_MCMC1.yaml -f

# The DESI DR1 fixed-width Δz cases <a name="dz_cases"></a>

The $`\Delta d_\mathrm{com} = 62`$ Mpc layout above splits the
samples into bins of fixed
comoving width. The DESI DR1 forecast distributions instead split BGS,
LRG and ELG into bins of fixed width $`\Delta z`$ in redshift. Three
widths are set up as separate cases of the same likelihood, each with
its own data files (`data/lssty1xdesi_desi_dr1_dz0pXX*`), likelihood
class (`likelihood/desi_cl_dr1_dz0pXX.py`, and
`likelihood/desi_cl_dr1_dz0pXX_model_b.py` for the paper's model),
parameter files and EXAMPLE yaml (the paper's model; the linear one is
`exploration/EXAMPLE_EVALUATE_LINEAR<N>.yaml`, with the same number):

| Case | $`\Delta z`$ | Lens bins | Cross pairs | Entries | Unmasked | `accuracyboost` | `nonlimber_accuracyboost` | EXAMPLE |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dr1_dz0p20` | 0.2 | 10 | 13 | 2,014 | 744 | 2 | 4 | `EXAMPLE_EVALUATE4.yaml` |
| `dr1_dz0p05` | 0.05 | 36 | 52 | 5,459 | 2,982 | 2 | 16 | `EXAMPLE_EVALUATE3.yaml` |
| `dr1_dz0p02` | 0.02 | 90 | 133 | 12,614 | 7,621 | 2 | 32 | `EXAMPLE_EVALUATE2.yaml` |

The distributions come without bandpowers, pairs or scale cuts, so
`scripts/make_desi_dz_case.py` builds them with the conventions of the
$`\Delta d_\mathrm{com} = 62`$ Mpc sacc file. Its docstring states
each rule; in short:

- **bands**: the 53 bands of the $`\Delta d_\mathrm{com} = 62`$ Mpc
  sacc;
- **cross pairs**: every two bins whose distributions overlap or
  touch;
- **scale cuts**: an auto spectrum keeps the bands with
  $`\ell + 1/2 \le k_\mathrm{max}\,\chi(\bar z)`$, with
  $`k_\mathrm{max} = 0.1185\ \mathrm{Mpc}^{-1}`$, $`\chi`$ the comoving
  distance and $`\bar z`$ the bin's mean redshift; a cross pair keeps
  the bands both of its autos keep;
- **densities**: only the per-bin `ngal` of the forecast's
  `probes_properties` files is used. Their other entries are not
  current.

Narrower bins need a finer FFTLog grid along the line of sight: the
non-Limber integrand oscillates on the scale of the bin's width, and a
coarse grid aliases it. Each case's likelihood yaml sets the reference
`accuracyboost` and `nonlimber_accuracyboost` of the table above and
records the measured $`\Delta\chi^2`$ ladder behind them; the
[appendix](#accuracy) explains how the two settings combine into the
line-of-sight grid.

The covariance of each case is the Gaussian term of the project's
covariance code, at the DESI footprint of 14,000 deg². It is stored
sparse in the cosmolike text format (only nonzero entries, upper
triangle). Like every covariance file under `data/`, it is built
locally and not tracked by git (steps two and three below); the test
snapshot in `tests/frozen/data/` keeps a copy.

A case is rebuilt in four steps, shown here for $`\Delta z = 0.02`$.
The DR1 distribution files are not distributed with the project:
`--input-dir` names their folder. The second step needs the project
compiled with its covariance interface
([covariance guide](covariance/README.md)).

**Step :one:**: build the case's data files.

```bash
python ./projects/lssty1xdesi/scripts/make_desi_dz_case.py --width 0.02 --stage files --input-dir /path/to/dr1_inputs
```

**Step :two:**: compute the case's Gaussian covariance.

```bash
python ./projects/lssty1xdesi/covariance/compute_covariance.py ./projects/lssty1xdesi/EXAMPLE_EVALUATE_COVARIANCE2.yaml
```

**Step :three:**: install the covariance into the case's data files.

```bash
python ./projects/lssty1xdesi/scripts/make_desi_dz_case.py --width 0.02 --stage covariance
```

**Step :four:**: write the fiducial data vectors.

```bash
python ./projects/lssty1xdesi/scripts/make_desi_dz_case.py --width 0.02 --stage fiducial
```

The last step writes, with `scripts/make_desi_fiducial.py`, the model
vector of the linear example (`exploration/EXAMPLE_EVALUATE_LINEAR<N>.yaml`)
at its fiducial point as the data vector of the case's linear dataset,
and, when the paper's example `EXAMPLE_EVALUATE<N>.yaml` exists (it is
written by hand for a new case), that example's vector as the data
vector of the case's main dataset. Each example then evaluates to
$`\chi^2 \approx 0`$.

# The LSST Y1 3×2pt case <a name="lsst_case"></a>

`EXAMPLE_EVALUATE5.yaml` evaluates LSST Y1 alone. It has cosmic shear,
galaxy-galaxy lensing and clustering, all in band powers on the same
53 bands, which is what lets its Fisher matrix add to a DESI case's.
The lens and source distributions are the lsst_y1 project's
(`data/lssty1xdesi_lens.nz`, `data/lssty1xdesi_source.nz`, five bins
each, read with that project's convention: the z column is the lower
edge of each histogram bin). The likelihood
`likelihood/lsst_y1_3x2pt.py` is `combo_3x2pt` in Fourier space: FKEM
non-Limber clustering and galaxy-galaxy lensing below $`\ell = 150`$.

| Block | Spectra | Entries | Kept by the mask |
|---|---:|---:|---|
| cosmic shear | 15 | 795 | every band |
| galaxy-galaxy lensing | 25 | 1,325 | the lens bin's clustering cut |
| clustering | 5 | 265 | $`\ell + 1/2 \le k_\mathrm{max}\,\chi(\bar z)`$ |

That leaves 1,629 unmasked entries out of 2,385.

The covariance is the Gaussian term at 12,300 deg², with the shape
noise and densities of the LSST Y1 science requirements (sources
2 arcmin⁻² per bin and $`\sigma_e = 0.26`$ per component, lenses
3.6 arcmin⁻² per bin). The shear and lensing signals in it include
nonlinear-alignment intrinsic alignments, with the fiducial power law
evaluated at each source bin's mean redshift.
`scripts/make_lsst_3x2pt_case.py` builds the case with the four steps
of the Δz builder (`--stage files`, then the covariance CLI on
`EXAMPLE_EVALUATE_COVARIANCE5.yaml`, then `--stage covariance` and
`--stage fiducial`).

# The one-loop bias model <a name="higher_order_bias"></a>

The likelihood option `higher_order_bias: 1` switches on the one-loop
galaxy-bias model of
[the higher-order bias note](notes/higher_order_bias/higher_order_bias.pdf)
on every DESI layout; `higher_order_bias: 0` is linear bias. The model
keeps the per-bin linear bias $`b_{1,i}`$ of the linear examples, with
$`i`$ the lens bin, and adds per bin:

- the quadratic bias $`b_{2,i}`$ of the Lazeyras et al. relation on
  $`b_{1,i}`$ (arXiv:1511.01096) and the co-evolution tidal bias
  $`b_{s^2,i} = -\tfrac{4}{7}(b_{1,i} - 1)`$, each plus a free offset;
- the counterterm $`\alpha_0 k^2 P_\mathrm{NL}`$;
- the radial counterterm $`\alpha_2`$ in the DESI DR1 convention
  (arXiv:2404.07312): the redshift-space galaxy power gains
  $`(b_{1,i} + f\mu^2)\,f\alpha_2\,\mu^2 k^2 P_\mathrm{lin}`$;
- a white-noise power $`\mathrm{SN}_0`$;
- the line-of-sight stochastic power $`\mathrm{SN}_2\,k^2\mu^2`$.

Here $`k`$ is the wavenumber, $`\mu`$ the cosine between the
wavevector and the line of sight, $`f`$ the linear growth rate, and
$`P_\mathrm{lin}`$ and $`P_\mathrm{NL}`$ the linear and nonlinear
matter power spectra; the third-order bias $`b_{3\mathrm{nl}}`$ is 0.
Zero amplitudes switch off $`\alpha_2`$, $`\mathrm{SN}_0`$ and
$`\mathrm{SN}_2`$ at no cost per evaluation. Adjacent thin bins share
an edge, where the radial derivatives of their kernels have opposite
signs, so the projected
$`\mathrm{SN}_2`$ of two neighbors is anticorrelated (correlation
coefficient about -0.5), which no $`\mathrm{SN}_0`$ can mimic: without
it the fit breaks already at
$`k_\mathrm{max} = 0.075\,h\,\mathrm{Mpc}^{-1}`$ on the thin layouts.
The redshift-space one-loop terms of the note are not computed
(CosmoLike has no redshift-space loop templates): the sampled velocity
dispersions and $`\alpha_2`$ stand in for them.

Models A and B of the note are two parameter files of this one
option. **Model A**, for
$`k_\mathrm{max} = 0.075\,h\,\mathrm{Mpc}^{-1}`$, fixes $`\alpha_2 = 0`$.
**Model B**, for the shipped masks, which keep every bin up to
$`k = 0.1185\ \mathrm{Mpc}^{-1}`$ ($`\approx 0.176\,h\,\mathrm{Mpc}^{-1}`$)
at every redshift, frees $`\alpha_2`$ and widens the $`b_2`$ and
$`b_{s^2}`$ offset priors to the DESI DR1 widths.

The likelihood reads one value per bin (`DESI_A2_<n>` and so on, with
$`n`$ counting the bins from 1); the parameter files tie the bins of a
sample chunk $`g`$ (BGS, LRG1-3, ELG1-2) to one sampled parameter per
chunk. Every prior is a Gaussian centered on zero:

| Parameter | Meaning | Unit | Width, model A | Width, model B |
|---|---|---|---:|---:|
| `DESI_DB2_<g>` | offset of $`b_2`$ from the Lazeyras relation | none | 1.0 | 2.0 |
| `DESI_DBS_<g>` | offset of $`b_{s^2}`$ from co-evolution | none | 0.5 | 1.0 |
| `DESI_A0_<g>` | counterterm $`\alpha_0`$ | $`h^{-2}\,\mathrm{Mpc}^2`$ | 88.9 | 16.1 |
| `DESI_A2_<g>` | radial counterterm $`\alpha_2`$ | $`h^{-2}\,\mathrm{Mpc}^2`$ | fixed to 0 | 16.1 |
| `DESI_SN0_<g>` | white noise $`\mathrm{SN}_0`$ | $`1/\bar n_i`$ | 2 | 2 |
| `DESI_SN2_<g>` | line-of-sight noise $`\mathrm{SN}_2`$ | $`f_\mathrm{sat}\sigma_v^2/\bar n_i`$ | 5 | 5 |

The counterterm widths follow the DESI rule, a 50% correction to the
linear spectrum at the scale cut: $`0.5/k_\mathrm{max}^2`$, with
$`k_\mathrm{max} = 0.075\,h\,\mathrm{Mpc}^{-1}`$ for model A and
$`0.176\,h\,\mathrm{Mpc}^{-1}`$ for model B. In the units,
$`\bar n_i`$ is the bin's number density (so $`1/\bar n_i`$ is its
Poisson power), $`f_\mathrm{sat}`$ the satellite fraction and
$`\sigma_v`$ the satellites' velocity dispersion in
$`h^{-1}\,\mathrm{Mpc}`$: 0.1 and 7.8 for BGS and LRG, 0.1 and 2.9 for
ELG.

The paper uses one of these variants, model B + tilts on the pessimistic
linear-bias prior. Each layout has one likelihood class and one example
of it:

| Layout | Example | Likelihood class |
|---|---|---|
| $`\Delta d_\mathrm{com} = 62`$ Mpc | `EXAMPLE_EVALUATE1.yaml` | `desi_cl_model_b` |
| $`\Delta z = 0.02`$ | `EXAMPLE_EVALUATE2.yaml` | `desi_cl_dr1_dz0p02_model_b` |
| $`\Delta z = 0.05`$ | `EXAMPLE_EVALUATE3.yaml` | `desi_cl_dr1_dz0p05_model_b` |
| $`\Delta z = 0.2`$ | `EXAMPLE_EVALUATE4.yaml` | `desi_cl_dr1_dz0p20_model_b` |

Each class sets `higher_order_bias: 1` and reads two parameter files
(cobaya takes a likelihood's parameters from its class):

- `likelihood/params_desi[_<case>]_model_b_tilts.yaml`, model B + tilts:
  the offsets of $`b_2`$, $`b_{s^2}`$ and the counterterms each get a
  free tilt in redshift per chunk (`DESI_TB2_<g>`, `DESI_TBS_<g>`,
  `DESI_TA0_<g>`, `DESI_TA2_<g>`); the stochastic amplitudes never tilt.
- `likelihood/params_desi[_<case>]_bias_pessimistic.yaml`, the
  pessimistic linear-bias prior of
  [the correlated-prior note](notes/correlated_bias_prior_v2/correlated_bias_prior_v2.pdf):
  bin $`n`$ of chunk $`g`$ has
  $`b_{1,n} = B_g\,(1 + \delta_n)/D(\bar z_n)`$ with
  $`\delta_n = s_g\,(\bar z_n - \bar z_g) + \varepsilon_n`$. The amplitude
  $`B_g`$ (`DESI_B1_<g>`) and the tilt $`s_g`$ (`DESI_TB1_<g>`) have flat
  priors, $`\bar z_g`$ is the mean of the chunk's bin redshifts, and the
  residual $`\varepsilon_n`$ has a Gaussian prior of width 3% whose
  correlation between two bins of the chunk is
  $`\exp(-|\bar z_m - \bar z_n|/0.05)`$, sampled as one unit Gaussian
  `DESI_UB1_<n>` per bin. A chunk of one or two bins has no residuals,
  so its bins are free.

The examples run as in the
[basic instructions](#running_cosmolike_projects), with every offset,
tilt and residual at zero, where every bin's bias is its chunk's
$`B_g/D(\bar z_n)`$. The data vectors are this model at this point, so
the examples evaluate to $`\chi^2 \approx 0`$ (1.0e-11, 1.1e-11, 4.1e-12
and 7.0e-13 in layout order).

The linear examples (growth-tied linear bias) and model A, model A +
tilts and model B without tilts, with their classes and examples, are
in `likelihood/exploration/` and [exploration/](exploration/README.md).
`scripts/make_desi_params.py` writes the parameter files, with
`--bias-pessimistic`, `--model-a` or `--model-b` (and `--case <name>`
for a $`\Delta z`$ layout, e.g. `--case dr1_dz0p05`).

# Baryonic feedback on EXAMPLE_EVALUATE1 <a name="baryonic_feedback"></a>

`EXAMPLE_EVALUATE1.yaml` can apply an external baryonic feedback
suppression to the matter power spectrum via the `bfmt` theory block
(SP(k), BCEmu, Flamingo, BACCOemu, or BCemu2025). By default, the
example runs without feedback.

**Step :one:**: ensure the lines below are commented out in
`set_installation_options.sh` before running `setup_cocoa.sh` and
`compile_cocoa.sh`. *By default, these lines should be commented out,
but it is worth checking*.

      [Adapted from Cocoa/set_installation_options.sh shell script]
      #export IGNORE_PYSPK_CODE=1
      #export IGNORE_BCEMU_CODE=1
      #export IGNORE_FBRE_CODE=1
      #export IGNORE_BACCOEMU_CODE=1
      (...)
      #export IGNORE_BFMT_CODE=1 # Baryon Feedback Theory Block from Nihar&JVicente

**Step :two:**: in `EXAMPLE_EVALUATE1.yaml`, uncomment the `bfmt`
theory block and select the model:

      theory:
        bfmt:
          baryon_model: 2 # 1 = SP(k), 2 = BCEmu, 3 = FlamingoEmulator, 4 = BACCOemu, 5 = BCemu2025

**Step :three:**: set `external_baryon_suppression: True` on the
`lssty1xdesi.desi_cl_model_b` likelihood block.

**Step :four:**: add the selected model's sampled parameters to the
`params` block and to the `sampler: evaluate: override` block.

> [!NOTE]
> BACCOemu (`baryon_model: 4`) rejects this example's fiducial
> `omegab: 0.04`, the edge of its baryon-density training box. Move
> `omegab` inside that box before selecting it.

> [!TIP]
> For the sampled parameters of each model, their validity ranges, and
> the `bfmt` options, see
> `Cocoa/external_modules/code/baryon_suppression/README.md`. The
> baryonic-feedback drift tests of the
> [test guide](tests/README.md) exercise every model against stored
> per-model vectors.

# Nested sampling with Nautilus <a name="nautilus"></a>

`EXAMPLE_NAUTILUS1.py` samples the model of `EXAMPLE_EVALUATE1.yaml` -
the paper's model on the $`\Delta d_\mathrm{com} = 62`$ Mpc layout, with
CAMB - with the Nautilus nested sampler, which estimates the Bayesian
evidence and weighted posterior samples. Its options and steps follow
the Nautilus driver of the other Cocoa projects
(`cosmolike_core/cocoa_hybrid_sampling.py`, which accepts only emulator
likelihoods); every MPI rank builds its own model. The model samples 163
parameters. The default budget (`--nlive 1000`, `--maxfeval 100000`) is
the shared default, and a run that reaches `--maxfeval` stops without
convergence.

We assume Cocoa and this project are installed, the Cocoa Conda
environment is active, the shell is Bash, and the current folder is
`cocoa/Cocoa/`.

**Step :one:**: activate Cocoa.

```bash
source start_cocoa.sh
```

**Step :two:**: Select the number of OpenMP cores (below, we set it to 8).

  - Linux

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

**Step :three:**: check the setup before a long run. The check builds the
model, evaluates the fiducial point ($`\chi^2 \approx 10^{-11}`$) and builds
the Nautilus sampler, without sampling.

```bash
python ./projects/lssty1xdesi/EXAMPLE_NAUTILUS1.py --check
```

**Step :four:**: run Nautilus.

- Linux

      "${CONDA_PREFIX}"/bin/mpirun -n 90 --oversubscribe \
        --mca pml ob1 --mca btl vader,tcp,self \
        -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x CONDA_PREFIX -x ROOTDIR \
        -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES -x OMP_DYNAMIC \
        -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x CLIK_PATH -x CLIK_DATA \
        -x CLIK_PLUGIN --mca mpi_yield_when_idle 1 \
        --mca btl_tcp_if_exclude lo,docker0,virbr0,ib0 \
        --bind-to core:overload-allowed --report-bindings \
        --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} \
        python ./projects/lssty1xdesi/EXAMPLE_NAUTILUS1.py --nlive 1000 --neff 10000 --maxfeval 100000 --outroot nautilus1

- macOS (arm)

      mpirun -n 12 --oversubscribe \
        python ./projects/lssty1xdesi/EXAMPLE_NAUTILUS1.py --nlive 1000 --neff 10000 --maxfeval 100000 --outroot nautilus1

The results go to `projects/lssty1xdesi/chains/<outroot>.*`; an
existing result is never overwritten, so use a new `--outroot` for
each run. Nautilus writes weighted GetDist-compatible rows and a JSON
record of the run. Reaching `--maxfeval` is not convergence.

# Running Fisher <a name="fisher"></a>

`EXAMPLE_EVALUATE_FISHER.ipynb` forecasts the $`w_0w_a`$CDM
constraints of three probes on every DESI layout:
$`\Delta d_\mathrm{com} = 62`$ Mpc (76 comoving
bins) and the DR1 fixed-width binnings $`\Delta z = 0.02`$ (90 bins),
$`\Delta z = 0.05`$ (36) and $`\Delta z = 0.2`$ (10). The probes are:

- DESI clustering alone;
- LSST Y1 3×2pt;
- their combination as independent data sets (block-diagonal: no
  LSST×DESI cross spectra and no cross-covariance).

For each layout the notebook gives the marginalized errors and the
figure of merit of DESI alone and of the combination, the Fisher
biases of the DESI projection approximations (FKEM with the cap at
550, Limber), the amplitude cut and the prior walls against the
forecast, the figure of merit against the non-Limber UV-fraction
scale cuts, the cost of bias freedom on DESI clustering alone at
$`k_\mathrm{max} = 0.075\,h\,\mathrm{Mpc}^{-1}`$ (biases tied per
group against one independent bias per bin and a correlated prior
within each group), and the $`w_0`$ - $`w_a`$ ellipses.

Every probe uses one area, 12,300 deg², and is marginalized over all
of its nuisance parameters. On the $`\Delta z = 0.2`$ binning the
derivatives are recomputed at four stencil steps and at two CAMB
accuracy boosts, and a forecast is trusted only where neither moves
it; the other layouts run at accuracy boost 1 and the single step
$`h = 0.02`$. At `AccuracyBoost: 3` the dark-energy figure of merit
FoM($`w_0`$, $`w_a`$) agrees to 0.6% across the steps: on
$`\Delta z = 0.2`$, about 5.7 for DESI clustering, 17.4 for LSST
3×2pt and 32.2 for the combination.

> [!WARNING]
> The forecast needs CAMB installed with
> `PATCH_CAMB_HALOFIT_TOLERANCE` in `set_installation_options.sh`.
> Stock CAMB stops halofit's search for the nonlinear scale at a
> tolerance of $`10^{-3}`$, and the data vectors jump with it between
> nearby cosmologies. The derivatives then depend on the step: the
> LSST figure of merit ranged from 14 to 52 across the four steps.
> The notebook's first cell reports whether the patch is installed.

One Fisher matrix at one step costs four evaluations per varied
parameter (20 varied parameters for DESI, 29 for LSST). On an Apple M2
Pro laptop on AC power with 8 OpenMP threads, one evaluation takes:

| Probe | `AccuracyBoost: 1` | `AccuracyBoost: 3` |
|---|---:|---:|
| DESI, chi-grid boost 64 | 3.8 s (5 min per step) | 14.3 s (19 min per step) |
| LSST Y1 3×2pt | 0.8 s (1.5 min per step) | 2.4 s (4.6 min per step) |

The notebook's four steps at both boosts take about two hours. It
saves the derivatives to one cache file per boost, so a second run
only redraws the results.

The same derivatives give **Fisher biases**: the shift of the best-fit
cosmology when the model fitted to the data projects differently from
the one that generated them. The notebook compares the FKEM split with
the low-rank engine, and Limber with non-Limber, for each probe and
for the combination. At an equal multipole cap, FKEM shifts no DESI
parameter by more than 0.01σ on the $`\Delta z = 0.2`$ bins, and the
low-rank engine shifts no LSST parameter by more than 0.003σ. Limber
clustering on the DESI bins shifts parameters by up to 5.6σ alone, and
by up to 2.1σ in the combination. The two DESI comparisons repeat on
every layout: FKEM with the cap at 550 stays below 0.02σ on
$`\Delta z = 0.2`$ but shifts the thin-bin layouts by up to 0.85σ in
the combination ($`\Omega_b`$ on $`\Delta z = 0.02`$), and Limber
clustering reaches tens of σ there, so Limber is excluded for DESI
clustering on every layout. The notebook tabulates the per-layout
shifts and their $`\Delta\chi^2`$.

Section 8 of the notebook tests a cut on the clustering amplitude
against the Fisher forecast of every DESI layout. Here $`A_s`$ is
the primordial amplitude (sampled as `As_1e9` $`= 10^9 A_s`$) and
$`b_i`$ is a bin's linear galaxy bias. The flat priors (`As_1e9` in
$`[0.5, 5]`$, every $`b_i`$ in $`[0.8, 4]`$) would let the amplitude
ratio $`A_s b_i^2/(A_s b_i^2)_\mathrm{fid}`$ reach about 50 for the
ELG samples (fiducial bias 0.84). The galaxy power of a bin
scales as $`A_s b_i^2`$, so a fixed relative numerical error of the
data vector costs a $`\Delta\chi^2`$ that grows as the square of that
ratio against the fiducial covariance: at a ratio of 50 it is many
units of $`\chi^2`$, far past the 0.2 budget of the N-random-models
check of [the data-vector tests](tests/data_vector/README.md#nmodels_check).

For every DESI layout the section draws two figures: a triangle of
$`A_s`$ and the six bias groups that shades the region the cut
(`AMPLITUDE_CUT`, 8 by default) excludes, under the $`1\sigma`$ and
$`2\sigma`$ contours of that layout's DESI clustering forecast, and a
triangle of ($`\Omega_m`$, $`w_0`$, $`w_a`$) that draws the
DESI-alone and combination contours against the cosmological prior
walls
($`\Omega_m \in [0.2, 0.5]`$, $`w_0 \le -1/3`$,
$`-3 \le w_0 + w_a \le -0.01`$). Measured at the step $`h = 0.02`$:

- The DESI $`2\sigma`$ contours reach an amplitude ratio of 1.33 on
  $`\Delta d_\mathrm{com} = 62`$ Mpc, 1.36 on $`\Delta z = 0.02`$,
  1.54 on $`\Delta z = 0.05`$ and
  2.11 on $`\Delta z = 0.2`$, so a cut at 8 removes only prior
  volume the data already exclude while bounding the amplitude,
  and with it the numerical error, of every bin. The value 8 is
  the common cut of the cosmolike projects (lsst_y1 samples its
  lens biases under the same cut), not a tuning of this survey.
- On $`\Delta d_\mathrm{com} = 62`$ Mpc, $`\Delta z = 0.02`$ and
  $`\Delta z = 0.05`$ no DESI or
  combination $`2\sigma`$ contour touches a prior wall. On the wide
  $`\Delta z = 0.2`$ bins DESI alone is partly prior-bounded: its
  $`2\sigma`$ contours cross the $`w_0 = -1/3`$ wall and both
  $`w_0 + w_a`$ walls, while the combination contour crosses only
  $`w_0 + w_a = -0.01`$.
- The marginalized errors and FoM($`w_0`$, $`w_a`$); each cell is
  DESI alone / the combination:

| layout | $`\sigma(\Omega_m)`$ | $`\sigma(w_0)`$ | $`\sigma(w_a)`$ | FoM($`w_0`$, $`w_a`$) |
|---|---|---|---|---|
| $`\Delta d_\mathrm{com} = 62`$ Mpc (76 comoving bins) | 0.0100 / 0.0066 | 0.065 / 0.062 | 0.210 / 0.193 | 135 / 203 |
| $`\Delta z = 0.02`$ (90 bins) | 0.0104 / 0.0066 | 0.063 / 0.060 | 0.206 / 0.187 | 126 / 198 |
| $`\Delta z = 0.05`$ (36 bins) | 0.0186 / 0.0117 | 0.138 / 0.115 | 0.437 / 0.357 | 32.0 / 72.5 |
| $`\Delta z = 0.2`$ (10 bins) | 0.0361 / 0.0170 | 0.316 / 0.190 | 1.109 / 0.622 | 5.67 / 32.2 |

The errors shrink as the binning gets finer, and the two finest
layouts, $`\Delta d_\mathrm{com} = 62`$ Mpc and $`\Delta z = 0.02`$,
agree to about 7%. The
$`\Delta z = 0.2`$ matrices are the main forecast's, at the overall
accuracy boost 3 (at boost 1 that layout moves by about 1%: DESI FoM
5.61); the other three layouts run at accuracy boost 1 with a
combined non-Limber $`\chi`$-grid factor of 64. The notebook caches
each matrix in `chains/` on its first run; without the cache the
$`\Delta d_\mathrm{com} = 62`$ Mpc
and $`\Delta z = 0.02`$ matrices take about 45 and 65 minutes on a
laptop with 4 threads.

The section ends with the non-Limber UV-fraction scale cuts,
following arXiv:2603.13031. Each band power $`C_\ell`$ is split in
two by a step on the 3D power spectrum: an infrared part
$`C_\ell^\mathrm{IR}`$ sourced by the modes
$`k \le k_\mathrm{max}/D(z)`$, with $`D`$ the linear growth factor
($`D(0) = 1`$) and $`k_\mathrm{max}`$ quoted in
$`h\,\mathrm{Mpc}^{-1}`$ at $`z = 0`$, and the ultraviolet rest
$`C_\ell^\mathrm{UV}`$; a band is kept while its UV fraction
$`|C_\ell^\mathrm{UV}| / (|C_\ell^\mathrm{UV}| + |C_\ell^\mathrm{IR}|)`$
stays below $`\alpha = 0.05`$. The step multiplies
$`P(k)`$ inside every $`C_\ell^{gg}`$ engine, Limber and non-Limber
alike, so the criterion needs no Limber relation: it counts the
radial modes $`k_\parallel \sim 2\pi/\Delta\chi`$ a thin bin
collects, which a Limber-relation cut
$`\ell \le k_\mathrm{max}\,\chi(\bar z) - 1/2`$ misses, and it
removes most bands of the cross spectra of adjacent thin bins, whose
signal lives on small radial separations across the shared edge. The
fit applies the cut on top of the shipped masks, which stop every
bin at the fixed $`k = 0.1185\ \mathrm{Mpc}^{-1}`$
($`\approx 0.176\,h\,\mathrm{Mpc}^{-1}`$, no growth scaling).
Inside the reach of linear bias, at
$`k_\mathrm{max} = 0.1\,h\,\mathrm{Mpc}^{-1}`$, the DESI-alone
FoM($`w_0`$, $`w_a`$) is 54.9 on $`\Delta d_\mathrm{com} = 62`$ Mpc,
57.0 on $`\Delta z = 0.02`$,
14.2 on $`\Delta z = 0.05`$ and 3.06 on $`\Delta z = 0.2`$, against
135 / 126 / 32.0 / 5.67 at the shipped masks; at
$`k_\mathrm{max} = 0.05\,h\,\mathrm{Mpc}^{-1}`$ the
$`\Delta z = 0.02`$ layout keeps no band at all.

Freeing one bias per bin instead of one amplitude per group divides
the DESI-alone figure of merit at
$`k_\mathrm{max} = 0.075\,h\,\mathrm{Mpc}^{-1}`$ by 3 to 7,
depending on the layout; the correlated prior
$`b_i = B_g\,(1 + \delta_i)/D(\bar z_i)`$, with
$`\langle\delta_i\delta_j\rangle = \sigma_b^2\,e^{-|\bar z_i - \bar z_j|/L}`$
within a group ($`\sigma_b`$ the fluctuation amplitude, $`L`$ the
correlation length in redshift) and the group amplitude $`B_g`$
free, perturbs the constant-clustering-amplitude shape
$`B_g/D(\bar z_i)`$ bin by bin. These results are linear bias only.
[The correlated-prior
note](notes/correlated_bias_prior_v2/correlated_bias_prior_v2.pdf)
holds the derivation, the physics behind the plausible $`L`$ and
$`\sigma_b`$, and the prior ladder under the one-loop bias model A;
[the $`k_\mathrm{max}`$ note](notes/linear_bias_kmax/linear_bias_kmax.pdf)
shows that a linear-bias fit is already biased at
$`k_\mathrm{max} = 0.075\,h\,\mathrm{Mpc}^{-1}`$.

**Known limitation - the covariance bias mismatch.** The DESI
covariance files are computed with each bin's linear bias set to its
group's raw value (1.34 / 1.70 / 0.84), while the likelihood's data
vector uses $`b_i = B_1/D(\bar z_i)`$: the group amplitude $`B_1`$
(`DESI_B1`) divided by the growth factor $`D`$ at the bin's mean
redshift $`\bar z_i`$. At high redshift the covariance's signal term
is therefore too small by $`D(\bar z_i)^2`$, which understates the
variances and inflates every layout's FoM by a factor of about
2.4-3.2: a self-consistent Gaussian covariance gives DESI clustering
alone FoM about 2.3 ($`\Delta z = 0.2`$), 13 ($`\Delta z = 0.05`$),
52 ($`\Delta z = 0.02`$) and 42 ($`\Delta d_\mathrm{com} = 62`$ Mpc)
instead of 5.7 / 32 / 126 /
135; the ratios between the layouts are unchanged. The fix will be
made together with the higher-order galaxy bias model, the next
modeling step. Until then the absolute FoM values of this section
are upper bounds of a rigid linear-bias model, not forecasts; the
UV-fraction scale cuts locate the model's defensible range,
$`k_\mathrm{max} \approx 0.08`$-$`0.12\,h\,\mathrm{Mpc}^{-1}`$,
but do not repair the covariance.

The pass/fail line of the N-random-models check is its 0.2 budget,
and at the extreme corners a cut at 8 still admits - a high amplitude
or $`\Omega_m`$ at the box edge, mostly with large neutrino masses -
the pushed-settings $`\Delta\chi^2`$ exceeds it (measured
2026-11-06, 50 models per layout): the worst point, an amplitude
ratio of 6.2 at $`\Omega_m = 0.50`$, reaches 47.4 on
$`\Delta d_\mathrm{com} = 62`$ Mpc and 37.6 on
$`\Delta z = 0.02`$, and the worst on $`\Delta z = 0.05`$ is 12.1.
On the thin layouts the residual is mainly cosmolike's table
resolution (`accuracyboost`); on the wide $`\Delta z = 0.2`$ bins it
is mainly CAMB's precision. The project accepts these values at the
data-excluded corners - the DESI Fisher $`2\sigma`$ contours reach
an amplitude ratio of only 1.33 to 2.11, with $`\Omega_m`$ inside
0.21-0.39 - as a documented judgment: the held models stay at or
below 0.304, every drawn model with an amplitude ratio of 3.5 or less
stays at or below 1.45, and raising `accuracyboost` on the thin
layouts is the lever that would reduce the corners, at a cost per
evaluation.
Every example that samples linear biases carries this cut as a cobaya
prior, `amplitude_cut` (0 inside, $`-\infty`$ outside, so cobaya skips
the likelihood of an excluded point); the LSST Y1 3×2pt example applies
it to its lens biases.

The machinery is `interface/cosmolike_lssty1xdesi_fisher.py`. Its
`ModelFisher` builds the cobaya model of an EXAMPLE yaml and
differentiates exactly the data vector the likelihood scores. The
compiled interface holds one model at a time, so each probe's Fisher
matrix is computed while its model is the latest one built.

# Unit tests <a name="unit_tests"></a>

The `tests/` folder compares likelihood predictions against stored
reference values, checks for race conditions from OpenMP threading,
and measures the numerical error of the default accuracy settings,
including the project's own non-Limber controls: the low-rank engine
switches, the FKEM comparison at an equal multipole cap on the four
DESI layouts, the chi-grid ladder, the photo-z conventions and the
cache ladder. [tests/README.md](tests/README.md) describes the checks,
how to run them, the input data snapshots, and how to refresh them.

# Computing covariances <a name="computing_covariances"></a>

The covariance tools compute Gaussian (G), super-sample (SSC),
connected non-Gaussian (cNG) and total matrices before scale cuts; the
Gaussian term carries the lens redshift-space distortions and the
non-Limber recompute of the low multipoles.
[covariance/README.md](covariance/README.md) explains compilation, the
production CLI, the `EXAMPLE_EVALUATE_COVARIANCE.ipynb` notebook,
accuracy settings, output files, and the entry-by-entry comparison
against the source forecast's covariance.

# Exploring notebooks <a name="notebooks"></a>

**Armadillo** was chosen to make a convenient Python API for notebook
exploration. This C++ library provides vectors, matrices and
three-dimensional arrays called cubes. A small interface layer
connects them to NumPy through **pybind11**, with **CARMA** handling
array conversion. The notebooks expose intermediate quantities;
production calculations use the CLI interfaces.

We assume Cocoa and this project are installed, the Cocoa Conda
environment is active, the shell is Bash, and the current folder is
`cocoa/Cocoa/`.

Compile the project first; the covariance notebook also needs the
optional covariance build described
[above](#computing_covariances).

**Step :one:**: activate Cocoa.

```bash
source start_cocoa.sh
```

**Step :two:**: Select the number of OpenMP cores (below, we set it to 8).

  - Linux

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

  - macOS (arm)

        export OMP_NUM_THREADS=8; export OMP_PROC_BIND=disabled; \
        export OMP_PLACES=cores; export OMP_DYNAMIC=FALSE; \
        export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1

**Step :three:**: start Jupyter.

```bash
jupyter notebook --no-browser --port=8888
```

**Step :four:**: open the printed URL and choose a notebook below.

**Step :five:**: select **Kernel → Restart Kernel and Run All Cells**.

| Notebook | Contents |
|---|---|
| [EXAMPLE_EVALUATE_COVARIANCE.ipynb](EXAMPLE_EVALUATE_COVARIANCE.ipynb) | The covariance terms of the $`\Delta d_\mathrm{com} = 62`$ Mpc layout and matrix diagnostics; needs the covariance build ([covariance guide](covariance/README.md#running)). |
| [EXAMPLE_EVALUATE_FISHER.ipynb](EXAMPLE_EVALUATE_FISHER.ipynb) | Fisher forecasts of DESI clustering, LSST 3×2pt and their combination on every DESI layout, and the Fisher biases of the projection approximations ([running Fisher](#fisher)). |

Choose the Python kernel from the activated Cocoa environment and
restart it after recompiling.

# Appendix <a name="appendix"></a>

## FAQ: Which accuracy settings are available? <a name="accuracy"></a>

Data-vector options belong to the selected `likelihood` block.
Covariance options belong to the covariance yaml and are documented in
the [covariance guide](covariance/README.md#accuracy-settings); the
two sets are separate, and changing one does not refine the other.

| Data-vector setting | What it changes |
|---|---|
| `accuracyboost` | Overall interpolation-table resolution; it also multiplies the non-Limber chi grid below. |
| `internal_accuracyboost` | C-FAST-PT convolution grid. |
| `nonlimber_accuracyboost` | Non-Limber line-of-sight (chi) sampling. |
| `integration_accuracy` | Quadrature resolution of the Limber integrals. |
| `pk_z_refinement` | Nested redshift refinement of matter-power inputs. |
| `lmax` | Angular-transform table bound; in Fourier mode it only bounds `nonlimber_lowrank_lmax` from above. |
| `kmax_boltzmann` | Requested Boltzmann power range; coordinate it with the theory settings. |

The non-Limber chi grid is log-spaced with
$`N_{\chi} - 1 = 511\,m`$ nodes, where $`m`$ is the product of
`accuracyboost` and `nonlimber_accuracyboost` rounded up to a power of
two: the grid refines by nested interval doubling, so consecutive
settings reuse the same nodes. The log spacing makes the
highest-redshift thin bins the narrowest in $`\ln\chi`$, which is why
the thin layouts need large combined factors. The reference settings
of the shipped EXAMPLEs are:

| EXAMPLE | Layout | `accuracyboost` | `nonlimber_accuracyboost` | `kmax_boltzmann` |
|---|---|---:|---:|---:|
| `EXAMPLE_EVALUATE1.yaml` | $`\Delta d_\mathrm{com} = 62`$ Mpc bins | 2 | 32 | 20 |
| `EXAMPLE_EVALUATE2.yaml` | DR1, $`\Delta z = 0.02`$ | 2 | 32 | 20 |
| `EXAMPLE_EVALUATE3.yaml` | DR1, $`\Delta z = 0.05`$ | 2 | 16 | 20 |
| `EXAMPLE_EVALUATE4.yaml` | DR1, $`\Delta z = 0.2`$ | 2 | 4 | 20 |

With `kmax_boltzmann: 20` and `accuracyboost: 2`, CAMB tabulates the
matter power up to `kmax_boltzmann` $`\times`$ `accuracyboost`
$`= 40\ \mathrm{Mpc}^{-1}`$; past the table the power is
extrapolated. The range protects the prior corners rather than the
fiducial: at a low-$`\Omega_m`$ point ($`\Omega_m = 0.237`$,
$`w_0 = -2.39`$, $`w_0 + w_a = -2.87`$,
$`H_0 = 78.6\ \mathrm{km\,s^{-1}\,Mpc^{-1}}`$), a table ending at
$`10\ \mathrm{Mpc}^{-1}`$ moves the $`\chi^2`$ through that
extrapolated tail by $`\Delta\chi^2 = 2.09`$
($`\Delta d_\mathrm{com} = 62`$ Mpc) and 2.06
($`\Delta z = 0.02`$), and the $`40\ \mathrm{Mpc}^{-1}`$ table
brings them to $`\Delta\chi^2 = 0.12`$ and 0.10.

`EXAMPLE_MCMC1.yaml` samples with the settings of
`EXAMPLE_EVALUATE1.yaml` (`accuracyboost: 2`, `kmax_boltzmann: 20`),
so a chain evaluates the model with the same settings that generated
its data vector.

Each likelihood yaml documents the measured $`\Delta\chi^2`$ ladder
behind its values, and the accuracy checks of the
[test guide](tests/README.md) re-measure them.

`adopt_limber_gg` chooses between non-Limber (0) and Limber (1)
clustering. With `nonlimber_lowrank_gg: 1` the low-rank engine
computes the full density-density term up to
`nonlimber_lowrank_lmax` with a rank-`nonlimber_lowrank_rank`
factorization and no internal Limber hand-off; `lmax_nonlimber` (the
FKEM split point) is read only with the engine off. These are modeling
choices, not interchangeable accuracy boosts, and the
[test guide](tests/README.md) pins their differences.

`photoz_zmid_convention` describes the n(z) file's z column: the DESI
files hold bin midpoints (`1`), while the LSST Y1 case reads its donor
files as lower bin edges (`0`). `external_nz_modeling: 1` routes each
thin bin's histogram through the Fingers-of-God convolution at every
sampled point and is mandatory for the DESI likelihoods.
`photoz_histogram_akima: 0` retains the original midpoint n(z)
reconstruction; `1` selects histogram preparation followed by Akima
interpolation, with `photoz_histogram_akima_factor` (3, 5 or 7)
preparation cells per original bin. The selected factor never changes
with an accuracy boost.
