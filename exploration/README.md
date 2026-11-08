# Exploration examples and likelihoods

The paper uses one bias model: model B + tilts on the pessimistic
linear-bias prior, run by `EXAMPLE_EVALUATE_MODEL_B1.yaml` to
`EXAMPLE_EVALUATE_MODEL_B4.yaml` in the project folder (likelihoods
`lssty1xdesi.desi_cl*_model_b`). This folder and the subpackage
`likelihood/exploration/` hold the other models the forecast explored.
They moved here from the project folder and from `likelihood/` without
changes to their parameters, and each example still runs from `Cocoa/`
like `EXAMPLE_EVALUATE1.yaml`, for example

      cobaya-run ./projects/lssty1xdesi/exploration/EXAMPLE_EVALUATE_MODEL_A1.yaml -f

A class in `likelihood/exploration/` is named
`lssty1xdesi.exploration.<class>` in a yaml. The layouts are, in example
order, $`\Delta d_\mathrm{com} = 62`$ Mpc and $`\Delta z = 0.02`$,
$`0.05`$, $`0.2`$ (`<case>` = `dr1_dz0p02`, `dr1_dz0p05`, `dr1_dz0p20`).

| File | Model | Served |
|---|---|---|
| `EXAMPLE_EVALUATE_MODEL_A1.yaml` to `A4.yaml` | model A on the growth-tied linear bias; likelihood `exploration.desi_cl[_<case>]_model_a` | the cobaya models of `EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb`, `EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb` and `EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb`; notes `correlated_bias_prior_v2` (Model A), `linear_bias_kmax`, `higher_order_bias` |
| `EXAMPLE_EVALUATE_MODEL_B_GROWTH_TIED1.yaml` to `4.yaml` | model B on the growth-tied linear bias, formerly `EXAMPLE_EVALUATE_MODEL_B1.yaml` to `B4.yaml`; likelihood `exploration.desi_cl[_<case>]_model_b_growth_tied` | model B of `notes/higher_order_bias` (The model as implemented) |
| `likelihood/exploration/desi_cl[_<case>]_model_a.py`, `.yaml` | the model A classes | the model A examples |
| `likelihood/exploration/desi_cl[_<case>]_model_b_growth_tied.py`, `.yaml` | the model B classes on the growth-tied linear bias | the model B growth-tied examples |
| `likelihood/exploration/params_desi[_<case>]_model_a.yaml` | model A: one offset per group | the baseline scenario of `EXAMPLE_EVALUATE_FISHER_MODEL_A.ipynb` and `EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb` |
| `likelihood/exploration/params_desi[_<case>]_model_a_tilts.yaml` | model A + tilts | the pessimistic scenario of the same notebooks |
| `likelihood/exploration/params_desi[_<case>]_model_b.yaml` | model B without tilts | the baseline scenario of `EXAMPLE_EVALUATE_FISHER_MODEL_B.ipynb` |
| `likelihood/exploration/params_desi_dr3_forecast.yaml` | the source forecast's own bias fiducials with the seven-group velocity dispersions | no example reads it |

The Fisher notebooks build their priors in code with the widths of these
parameter files. They name the model A examples by their former path in
the project folder and open them only when they rebuild their derivative
caches; a rebuild needs the path `exploration/EXAMPLE_EVALUATE_MODEL_A<n>.yaml`.
`scripts/make_desi_params.py` writes every parameter file listed here
(`--model-a`, `--model-b`, and no flag for the DR3 forecast fiducials).
