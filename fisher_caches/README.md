# Fisher cache archive

This folder archives the final-tier caches of the three Fisher
notebooks and of the pair-axes study: the derivatives and vectors that
cost hours to compute. A fresh clone copies them to working copies in
`chains/` instead of recomputing them. The `.npz` files are stored through Git LFS; the
`<name>.json` provenance records and `checksums.json` are plain git.

> [!NOTE]
> An `.npz` here that opens as a short text file starting with
> `version https://git-lfs.github.com/...` is a Git LFS pointer, not
> the cache: run `git lfs pull` in the project folder, from the cocoa
> environment (which provides `git-lfs`).

Only final-tier caches enter the archive: results computed at the
converged settings of the derivative convergence study in
[tests/studies/](../tests/studies/). Exploration-tier results stay in
`chains/`.

## The archived caches <a name="caches"></a>

In the names, `<layout>` is `dcom62` ($`\Delta d_\mathrm{com} = 62`$
Mpc), `dz0p02`, `dz0p05` or `dz0p20` ($`\Delta z`$ = 0.02, 0.05, 0.2).
The last column abbreviates the notebooks:
[MODEL](../EXAMPLE_EVALUATE_FISHER_MODEL.ipynb) is
`EXAMPLE_EVALUATE_FISHER_MODEL.ipynb`,
[FISHER](../EXAMPLE_EVALUATE_FISHER.ipynb) is
`EXAMPLE_EVALUATE_FISHER.ipynb`,
[KMAX](../EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb) is
`EXAMPLE_EVALUATE_FISHER_LINEAR_KMAX.ipynb` and PAIRS is the script
`scripts/compute_pair_axes_fits.py`.

| Cache | Holds | Size | Read by |
|---|---|---:|---|
| `model_derivatives_dcom62_final.npz` | the derivative base of $`\Delta d_\mathrm{com} = 62`$ Mpc: stencil columns of the 20 sampled parameters, per-bin $`b_1`$, $`b_2`$, $`b_{s^2}`$ and $`\alpha_0`$ columns, the $`\mathrm{SN}_0`$ templates, the fiducial vector and biases | 0.97 MB | MODEL |
| `model_derivatives_dz0p02_final.npz` | the same, $`\Delta z = 0.02`$ | 1.28 MB | MODEL |
| `model_derivatives_dz0p05_final.npz` | the same, $`\Delta z = 0.05`$ | 0.49 MB | MODEL |
| `model_derivatives_dz0p20_final.npz` | the same, $`\Delta z = 0.2`$ | 0.13 MB | MODEL |
| `model_templates_dcom62_final.npz` | the exact $`\alpha_2`$ columns and the $`\mathrm{SN}_0`$ and $`\mathrm{SN}_2`$ templates, $`\Delta d_\mathrm{com} = 62`$ Mpc | 0.22 MB | MODEL |
| `model_templates_dz0p02_final.npz` | the same, $`\Delta z = 0.02`$ | 0.29 MB | MODEL |
| `model_templates_dz0p05_final.npz` | the same, $`\Delta z = 0.05`$ | 0.11 MB | MODEL |
| `model_templates_dz0p20_final.npz` | the same, $`\Delta z = 0.2`$ | 0.03 MB | MODEL |
| `model_scale_cuts_final.npz` | the UV-fraction scale cuts on the paper fiducial, final tier | 3.57 MB | MODEL |
| `model_derivatives_dz0p02_sigv600_final.npz` | the derivative base of $`\Delta z = 0.02`$ on the $`\sigma_v = 600`$ km/s pair list | 1.91 MB | none (the record of that study) |
| `model_templates_dz0p02_sigv600_final.npz` | the same templates, $`\Delta z = 0.02`$ on the $`\sigma_v = 600`$ km/s pair list | 0.39 MB | none (the record of that study) |
| `model_scale_cuts_dz0p02_sigv600_final.npz` | the UV-fraction scale cuts on the paper fiducial, final tier, $`\Delta z = 0.02`$ on the $`\sigma_v = 600`$ km/s pair list | 2.17 MB | none (the record of that study) |
| `model_derivatives_dcom62_pairmax_final.npz` | the derivative base of $`\Delta d_\mathrm{com} = 62`$ Mpc on its maximal pair list | 2.53 MB | PAIRS |
| `model_derivatives_dz0p02_pairmax_final.npz` | the same, $`\Delta z = 0.02`$ | 2.77 MB | PAIRS |
| `model_derivatives_dz0p05_pairmax_final.npz` | the same, $`\Delta z = 0.05`$ | 1.24 MB | PAIRS |
| `model_derivatives_dz0p20_pairmax_final.npz` | the same, $`\Delta z = 0.2`$ | 0.23 MB | PAIRS |
| `model_templates_dcom62_pairmax_final.npz` | the $`\alpha_2`$ columns and the $`\mathrm{SN}_0`$ and $`\mathrm{SN}_2`$ templates on the maximal pair list, $`\Delta d_\mathrm{com} = 62`$ Mpc | 0.47 MB | PAIRS |
| `model_templates_dz0p02_pairmax_final.npz` | the same, $`\Delta z = 0.02`$ | 0.54 MB | PAIRS |
| `model_templates_dz0p05_pairmax_final.npz` | the same, $`\Delta z = 0.05`$ | 0.22 MB | PAIRS |
| `model_templates_dz0p20_pairmax_final.npz` | the same, $`\Delta z = 0.2`$ | 0.04 MB | PAIRS |
| `model_scale_cuts_dcom62_pairmax_final.npz` | the UV-fraction scale cuts on the paper fiducial, final tier, on the maximal pair list of $`\Delta d_\mathrm{com} = 62`$ Mpc | 3.02 MB | PAIRS, and through it the figure data of [the cross-correlation note](../notes/cross_correlations) |
| `model_scale_cuts_dz0p02_pairmax_final.npz` | the same, $`\Delta z = 0.02`$ | 3.16 MB | PAIRS, as above |
| `model_scale_cuts_dz0p05_pairmax_final.npz` | the same, $`\Delta z = 0.05`$ | 1.48 MB | PAIRS, as above |
| `model_scale_cuts_dz0p20_pairmax_final.npz` | the same, $`\Delta z = 0.2`$ | 0.31 MB | PAIRS, as above |
| `linear_fisher_dcom62_final.npz` | the linear-bias derivatives of DESI $`C_\ell^{gg}`$, $`\Delta d_\mathrm{com} = 62`$ Mpc | 0.54 MB | FISHER, KMAX |
| `linear_fisher_dz0p02_final.npz` | the same, $`\Delta z = 0.02`$ | 0.71 MB | FISHER, KMAX |
| `linear_fisher_dz0p05_final.npz` | the same, $`\Delta z = 0.05`$ | 0.29 MB | FISHER, KMAX |
| `linear_probes_dz0p20_lsst_ab3_final.npz` | the $`\Delta z = 0.2`$ DESI and the LSST Y1 3×2pt derivatives (LSST at accuracy boost 3, outside the tier system) | 1.15 MB | FISHER, KMAX |
| `linear_bin_bias_final.npz` | the per-bin linear-bias derivative columns of every layout | 0.31 MB | FISHER, KMAX |
| `linear_scale_cuts_final.npz` | the UV-fraction scale cuts on the linear-bias fiducial | 1.41 MB | FISHER, KMAX, MODEL |
| `linear_fom_scale_cuts_final.npz` | the linear-bias FoM scan over the cuts, with each layout's covariance sha256 | 0.03 MB | FISHER |
| `linear_alternatives_final.npz` | the FKEM and Limber alternative vectors of every layout | 0.31 MB | FISHER |
| `linear_kmax_truths_final.npz` | the fiducial model's one-loop truth vectors against $`k_\mathrm{max}`$, from `EXAMPLE_EVALUATE<n>.yaml` at the linear vector's own settings (each layout's paper data set) | 0.89 MB | KMAX |

The three `*_dz0p02_sigv600_final` caches belong to the
$`\sigma_v = 600`$ km/s study. They use the $`\Delta z = 0.02`$ layout
with a longer cross-pair list: 219 pairs, one for every bin pair whose
blurred kernels overlap when every velocity dispersion is 600 km/s.
They are kept next to today's caches, which they do not replace.

The twelve `*_pairmax_final` caches belong to the pair-axes study of
[the cross-correlation note](../notes/cross_correlations/cross_correlations.pdf):
the derivative base, the templates and the final-tier scale cuts of
each layout on its maximal pair list
(`data/<stem>_pairmax_pairs.txt`: 378, 370, 178 and 32 cross pairs in
layout order). `scripts/compute_pair_axes_fits.py` reads them (its docstring
holds the usage and the prerequisites) and writes
`chains/pair_axes_fits_<layout>.json`, from which the three
`make_pair_axes_*_data.py` generators of
`notes/cross_correlations/plots/` write the figures' data.

## The file format <a name="format"></a>

- `<name>.npz`: numpy arrays plus one JSON header (the settings, tier,
  fingerprints, parameter names, array shapes and the creation date).
  Plain arrays, no pickles, so reading a cache runs no code.
- `<name>.json`: the header's provenance record, readable on GitHub.
  For every model it holds the fully resolved cobaya input with its
  fingerprint, the sha256 of every dataset file, the stored settings,
  and the git state of the repositories, the CAMB version and the
  halofit patch at write time.
- `checksums.json`: the sha256 of every archived `.npz`, verified
  before any copy out of the archive.

## Using the caches <a name="usage"></a>

Every copy, read, write and archive is an explicit notebook cell:
nothing loads, computes or writes a cache on its own (section
"2b. The caches" in each notebook).

**Step :one:**: copy an archived cache to its working copy:
`fm.load_fisher_from_cache('<name>')` copies `fisher_caches/<name>.npz`
to `chains/<name>.npz` after checking its checksum. A different
existing working copy is replaced only with `overwrite=True`.

**Step :two:**: read the working copy: the notebook's read cell calls
`fm.read_fisher_cache`, which compares the stored settings, tier and
fingerprints with the notebook's and raises on any mismatch. It never
recomputes.

**Step :three:**: to recompute, run the notebook's commented compute
cell in place of the read cell, then its write cell:
`fm.write_fisher_cache` writes the working copy, refuses an existing
file without `overwrite=True`, and refuses to write when a stored
fingerprint no longer matches the current files.

**Step :four:**: archive a final-tier result:
`fm.save_fisher_to_cache('<name>')` copies the working copy here,
writes `<name>.json` and updates `checksums.json`. It is the only call
that changes the archive; it refuses any tier other than final and
replaces a different archived file only with `overwrite=True`. Stage
the three files with git.

The design (the fingerprint, the tiers, and what makes a cache out of
date) is section "Using the caches" of
[the Fisher-machinery note](../notes/fisher_machinery/fisher_machinery.pdf).
