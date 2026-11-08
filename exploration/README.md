# Exploration examples and likelihoods

The paper uses one bias model: model B + tilts on the pessimistic
linear-bias prior, run by `EXAMPLE_EVALUATE1.yaml` to
`EXAMPLE_EVALUATE4.yaml` in the project folder (likelihoods
`lssty1xdesi.desi_cl*_model_b`). This folder and the subpackage
`likelihood/exploration/` hold the other models the forecast explored.
They moved here from the project folder and from `likelihood/` without
changes to their parameters, and each example still runs from `Cocoa/`
like the project's examples, for example

      cobaya-run ./projects/lssty1xdesi/exploration/EXAMPLE_EVALUATE_MODEL_A1.yaml -f

A class in `likelihood/exploration/` is named
`lssty1xdesi.exploration.<class>` in a yaml. The layouts are, in example
order, $`\Delta d_\mathrm{com} = 62`$ Mpc and $`\Delta z = 0.02`$,
$`0.05`$, $`0.2`$ (`<case>` = `dr1_dz0p02`, `dr1_dz0p05`, `dr1_dz0p20`).

Every example here reads the linear datasets
`data/lssty1xdesi_desi_dr3_linear.dataset` and
`data/lssty1xdesi_desi_<case>_linear.dataset`, whose data vectors
(`..._linear_fid.datavector`) are the growth-tied linear bias at the
fiducial point. They are the data vectors the project shipped before
the paper's model became the main dataset, so every chi2 here is what it
was: about 0 for the linear examples, and 4817.064, 5408.959, 2166.0383
and 704.4138 (in layout order) for model A and model B on the
growth-tied linear bias at their fiducial points.
`scripts/make_desi_fiducial.py` rewrites a linear vector from its linear
example.

| File | Model | Served |
|---|---|---|
| `EXAMPLE_EVALUATE_LINEAR1.yaml` to `4.yaml` | the growth-tied linear bias, the model of the linear data vectors ($`\chi^2 \approx 0`$ at the fiducial), formerly `EXAMPLE_EVALUATE1.yaml` to `4.yaml`; likelihood `desi_cl[_<case>]` | the linear-bias forecasts of `EXAMPLE_EVALUATE_FISHER.ipynb` and `EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb`; note `linear_bias_kmax`; the model that `scripts/make_desi_dz_case.py` turns into the linear Δz data vectors; snapshot as `tests/frozen/EXAMPLE_EVALUATE1.yaml` to `4.yaml` |
| `EXAMPLE_EVALUATE_MODEL_A1.yaml` to `A4.yaml` | model A on the growth-tied linear bias; likelihood `exploration.desi_cl[_<case>]_model_a` | the cobaya models of `EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb`, `EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb` and `EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb`; notes `correlated_bias_prior_v2` (Model A), `linear_bias_kmax`, `higher_order_bias` |
| `EXAMPLE_EVALUATE_MODEL_B_GROWTH_TIED1.yaml` to `4.yaml` | model B on the growth-tied linear bias, formerly `EXAMPLE_EVALUATE_MODEL_B1.yaml` to `B4.yaml`; likelihood `exploration.desi_cl[_<case>]_model_b_growth_tied` | model B of `notes/higher_order_bias` (The model as implemented) |
| `likelihood/exploration/desi_cl[_<case>]_model_a.py`, `.yaml` | the model A classes | the model A examples |
| `likelihood/exploration/desi_cl[_<case>]_model_b_growth_tied.py`, `.yaml` | the model B classes on the growth-tied linear bias | the model B growth-tied examples |
| `likelihood/exploration/params_desi[_<case>]_model_a.yaml` | model A: one offset per group | the baseline scenario of `EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb` and `EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb` |
| `likelihood/exploration/params_desi[_<case>]_model_a_tilts.yaml` | model A + tilts | the pessimistic scenario of the same notebooks |
| `likelihood/exploration/params_desi[_<case>]_model_b.yaml` | model B without tilts | the baseline scenario of `EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb` |
| `likelihood/exploration/params_desi_dr3_forecast.yaml` | the source forecast's own bias fiducials with the seven-group velocity dispersions | no example reads it |

The Fisher notebooks build their priors in code with the widths of these
parameter files, and their cobaya models from the linear and model A
examples of this folder.
`scripts/make_desi_params.py` writes every parameter file listed here
(`--model-a`, `--model-b`, and no flag for the DR3 forecast fiducials);
`scripts/make_desi_dz_case.py` writes `EXAMPLE_EVALUATE_LINEAR2.yaml` to
`4.yaml`.
