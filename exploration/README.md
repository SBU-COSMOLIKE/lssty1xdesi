# Exploration examples and likelihoods

The paper uses one bias model, the paper model of
[the one-loop bias section](../README.md#higher_order_bias), run by
`EXAMPLE_EVALUATE1.yaml` to `EXAMPLE_EVALUATE4.yaml` in the project
folder. This folder and the subpackage `likelihood/exploration/` hold
the other models the forecast explored. Each example runs from
`Cocoa/` like the project's examples, for example

      cobaya-run ./projects/lssty1xdesi/exploration/EXAMPLE_EVALUATE_MODEL_A1.yaml -f

A class in `likelihood/exploration/` is named
`lssty1xdesi.exploration.<class>` in a yaml. The layouts are, in example
order, $`\Delta d_\mathrm{com} = 62`$ Mpc and $`\Delta z = 0.02`$,
$`0.05`$, $`0.2`$ (`<case>` = `dr1_dz0p02`, `dr1_dz0p05`, `dr1_dz0p20`).

Every example here reads the linear datasets
(`data/lssty1xdesi_desi_[dr3|<case>]_linear.dataset`), whose data
vectors are the growth-tied linear bias at the fiducial point;
`scripts/make_desi_fiducial.py` rewrites them. The linear examples
evaluate to $`\chi^2 \approx 0`$; the model A and model B examples
score their one-loop vector against the linear data vector, so their
$`\chi^2`$ is far from zero by construction. On
$`\Delta d_\mathrm{com} = 62`$ Mpc the linear dataset keeps the DESI
forecast's 97 cross pairs with their mask and covariance; the main
dataset adds 10 LRG3 x ELG1 pairs (`scripts/make_desi_pairs.py`).

| File | Model | Served |
|---|---|---|
| `EXAMPLE_EVALUATE_LINEAR1.yaml` to `4.yaml` | the growth-tied linear bias, the model of the linear data vectors ($`\chi^2 \approx 0`$ at the fiducial); likelihood `desi_cl[_<case>]` | the linear-bias forecasts of `EXAMPLE_EVALUATE_FISHER.ipynb` and `EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb`; note `linear_bias_kmax`; the model that `scripts/make_desi_dz_case.py` turns into the linear Δz data vectors; snapshot as `tests/frozen/EXAMPLE_EVALUATE1.yaml` to `4.yaml` |
| `EXAMPLE_EVALUATE_MODEL_A1.yaml` to `A4.yaml` | model A on the growth-tied linear bias; likelihood `exploration.desi_cl[_<case>]_model_a` | notes `correlated_bias_prior_v2` (Model A), `linear_bias_kmax`, `higher_order_bias` |
| `EXAMPLE_EVALUATE_MODEL_B_GROWTH_TIED1.yaml` to `4.yaml` | model B on the growth-tied linear bias; likelihood `exploration.desi_cl[_<case>]_model_b_growth_tied` | model B of `notes/higher_order_bias` (The model as implemented) |
| `EXAMPLE_MCMC_LINEAR1.yaml`, `EXAMPLE_MCMC_LINEAR1.covmat` | the MCMC of `EXAMPLE_EVALUATE_LINEAR1.yaml`; its yaml sets no `covmat`, and the `.covmat` file, which lists another project's parameters, is not read | the linear-bias chains |
| `likelihood/exploration/desi_cl[_<case>]_model_a.py`, `.yaml` | the model A classes | the model A examples |
| `likelihood/exploration/desi_cl[_<case>]_model_b_growth_tied.py`, `.yaml` | the model B classes on the growth-tied linear bias | the model B growth-tied examples |
| `likelihood/exploration/params_desi[_<case>]_model_a.yaml` | model A: one offset per group | the model A examples |
| `likelihood/exploration/params_desi[_<case>]_model_a_tilts.yaml` | model A + tilts | no example reads it by default; a model A class comment names it as the swap-in |
| `likelihood/exploration/params_desi[_<case>]_model_b.yaml` | model B without tilts | the model B growth-tied examples |
| `likelihood/exploration/params_desi_dr3_forecast.yaml` | the source forecast's own bias fiducials with the seven-group velocity dispersions | no example reads it |

The linear-bias Fisher notebooks, `EXAMPLE_EVALUATE_FISHER.ipynb` and
`EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb`, build their cobaya models
from the linear examples of this folder and their priors in code with
the widths of these parameter files; the paper notebook,
`EXAMPLE_EVALUATE_FISHER_MODEL.ipynb`, builds from the project's
`EXAMPLE_EVALUATE1.yaml` to `EXAMPLE_EVALUATE4.yaml`.
`scripts/make_desi_params.py` writes every parameter file listed here
(`--model-a`, `--model-b`, and no flag for the DR3 forecast fiducials);
`scripts/make_desi_dz_case.py` writes `EXAMPLE_EVALUATE_LINEAR2.yaml` to
`4.yaml`.
