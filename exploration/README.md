# Exploration examples

The paper uses one bias model, the fiducial model of
[the one-loop bias section](../README.md#higher_order_bias), run by
`EXAMPLE_EVALUATE1.yaml` to `EXAMPLE_EVALUATE4.yaml` in the project
folder. This folder holds the growth-tied linear-bias examples the
forecast explored next to it. Each example runs from `Cocoa/` like the
project's examples, for example

      cobaya-run ./projects/lssty1xdesi/exploration/EXAMPLE_EVALUATE_LINEAR1.yaml -f

The layouts are, in example order, $`\Delta d_\mathrm{com} = 62`$ Mpc
and $`\Delta z = 0.02`$, $`0.05`$, $`0.2`$ (`<case>` = `dr1_dz0p02`,
`dr1_dz0p05`, `dr1_dz0p20`).

Every example here reads the linear datasets
(`data/lssty1xdesi_desi_[dr3|<case>]_linear.dataset`), whose data
vectors are the growth-tied linear bias at the fiducial point;
`scripts/make_desi_fiducial.py` rewrites them. The examples evaluate to
$`\chi^2 \approx 0`$. On $`\Delta d_\mathrm{com} = 62`$ Mpc the linear
dataset keeps the DESI forecast's 97 cross pairs with their mask and
covariance; the main dataset adds 10 LRG3 x ELG1 pairs
(`scripts/make_desi_pairs.py`).

| File | Model | Served |
|---|---|---|
| `EXAMPLE_EVALUATE_LINEAR1.yaml` to `4.yaml` | the growth-tied linear bias, the model of the linear data vectors ($`\chi^2 \approx 0`$ at the fiducial); likelihood `desi_cl[_<case>]` | the linear-bias forecasts of `EXAMPLE_EVALUATE_FISHER.ipynb` and `EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb`; note `linear_bias_kmax`; the model that `scripts/make_desi_dz_case.py` turns into the linear Δz data vectors; snapshot as `tests/frozen/EXAMPLE_EVALUATE1.yaml` to `4.yaml` |
| `EXAMPLE_MCMC_LINEAR1.yaml`, `EXAMPLE_MCMC_LINEAR1.covmat` | the MCMC of `EXAMPLE_EVALUATE_LINEAR1.yaml`; its yaml sets no `covmat`, and the `.covmat` file, which lists another project's parameters, is not read | the linear-bias chains |
| `likelihood/exploration/params_desi_dr3_forecast.yaml` | the source forecast's own bias fiducials with the seven-group velocity dispersions | no example reads it |

The subpackage `likelihood/exploration/` holds only that last fragment;
the linear likelihood classes `desi_cl[_<case>]` live in `likelihood/`.

The linear-bias Fisher notebooks, `EXAMPLE_EVALUATE_FISHER.ipynb` and
`EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb`, build their cobaya models
from the linear examples of this folder and their priors in code with
the widths of the project's parameter files; the k_max notebook builds
its one-loop truth vectors from the paper yamls `EXAMPLE_EVALUATE1.yaml`
to `EXAMPLE_EVALUATE4.yaml`, with the linear vector's own settings,
which point each layout at its paper data set. The paper notebook,
`EXAMPLE_EVALUATE_FISHER_MODEL.ipynb`, builds from the paper yamls.
`scripts/make_desi_params.py` with no flag writes
`likelihood/exploration/params_desi_dr3_forecast.yaml` next to the
project default `likelihood/params_desi.yaml`;
`scripts/make_desi_dz_case.py` writes `EXAMPLE_EVALUATE_LINEAR2.yaml` to
`4.yaml`.
