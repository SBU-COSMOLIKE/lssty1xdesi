"""Build one fixed-width Delta z case of the DESI thin-bin layout.

The DESI DR1 forecast distributions split BGS, LRG and ELG into bins
of fixed width in redshift - 0.02, 0.05 or 0.2 - in files
desi_<sample>_<width>_binned_distribution_DR1.txt (the format of the
comoving DR3 files: a shared redshift grid, then one row per bin).

Unlike the comoving DR3 layout, whose bandpowers, pairs, scale cuts
and covariance come from the source forecast's sacc file
(scripts/make_desi_dataset.py), these cases come with distributions
only. Every other file is built here, with the conventions of that
sacc file:

  - cross pairs: every two bins whose raw n(z) supports overlap or
    touch. In the DR3 sacc this rule reproduces 96 of the 97 measured
    pairs; the missing one, the BGS -> LRG boundary pair, touches here
    because fixed-width bins tile the redshift range exactly.
  - scale cuts: an auto spectrum keeps the bands with
    l + 1/2 <= KMAX chi(zbar); a cross pair keeps the bands both of its
    autos keep. KMAX reproduces 59 of the 76 DR3 cuts exactly and the
    other 17 within one band.
  - bands: the 53 DR3 bands (data/lssty1xdesi_desi_dr3_ells.txt).
  - cosmic shear: present in the layout and masked, as in DR3.

Of the forecast's probes_properties yaml only ngal is read: its author
states the other entries are not current. Bias and velocity-dispersion
groups and fiducials are the project's (scripts/make_desi_params.py).

Three stages, run in order (the project README's section on the
Delta z cases lists the full command sequence):

  files      - the n(z) (refined five-fold, make_desi_nz.py's rule),
               bin table, densities, pairs, mask, the two datasets
               (<stem>.dataset and <stem>_linear.dataset, which differ
               only in their data vector) with placeholder data vectors
               and covariance, the parameter fragment, the likelihood
               class and yaml, and the EXAMPLE yamls (the linear example
               exploration/EXAMPLE_EVALUATE_LINEAR<N>.yaml and
               EXAMPLE_EVALUATE_COVARIANCE<N>.yaml);
  covariance - writes the cosmolike covariance file from the Gaussian
               term of the case's archive (compute_covariance.py on
               EXAMPLE_EVALUATE_COVARIANCE<N>.yaml);
  fiducial   - writes the case's two data vectors with
               make_desi_fiducial.py: the linear example's model at its
               fiducial point as <stem>_linear_fid.datavector, and, when
               the paper's example EXAMPLE_EVALUATE<N>.yaml exists (it is
               written by hand), the paper's model at its fiducial point
               as <stem>_fid.datavector.

Usage (from the Cocoa/ folder, cocoapy311 active, start_cocoa.sh
sourced, the project compiled with its covariance interface):

    python ./projects/lssty1xdesi/scripts/make_desi_dz_case.py \\
        --width 0.02 --stage files --input-dir /path/to/dr1_inputs
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
from scipy.integrate import quad

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SCRIPTS)
DATA = os.path.join(PROJECT, "data")
LIKE = os.path.join(PROJECT, "likelihood")
sys.path.insert(0, SCRIPTS)
import make_desi_nz as nzmod        # noqa: E402  loader, refinement, chunks
import make_desi_params as parmod   # noqa: E402  parameter groups, fiducials

# width (as spelled in the input file names) -> (case name, the
# probes_properties file holding the per-bin ngal, the number of the
# case's EXAMPLE yamls: EXAMPLE_EVALUATE<N>.yaml (the paper's model),
# exploration/EXAMPLE_EVALUATE_LINEAR<N>.yaml (linear bias) and
# EXAMPLE_EVALUATE_COVARIANCE<N>.yaml; 1 is the comoving DR3 layout)
WIDTHS = {
    "0.02": ("dr1_dz0p02", "probes_properties_002.yaml", 2),
    "0.05": ("dr1_dz0p05", "probes_properties_005.yaml", 3),
    "0.2": ("dr1_dz0p20", "probes_properties_0.2.yaml", 4),
}

# the scale-cut wavenumber [1/Mpc] (see the module docstring)
KMAX = 0.1185

# the fiducial background behind chi(zbar) for the scale cuts: the
# EXAMPLE yaml cosmology (flat LCDM; the massive-neutrino correction to
# chi moves no cut)
H0 = 67.32
OMEGAM = 0.3
C_KMS = 299792.458

# the LSST Y1 sources shared by every layout
N_SOURCE = 5


def chi_mpc(z):
    """Comoving distance [Mpc] at redshift z in the fiducial background."""
    integrand = lambda x: 1.0/np.sqrt(OMEGAM*(1 + x)**3 + 1 - OMEGAM)
    return C_KMS/H0*quad(integrand, 0.0, z)[0]


def case_names(width):
    """Return (case, stem, class name) of one width."""
    case = WIDTHS[width][0]
    return case, f"lssty1xdesi_desi_{case}", f"desi_cl_{case}"


# ---- stage 1: the files -------------------------------------------------


def stack(input_dir, width):
    """Read the three samples' DR1 files: refined n(z), table, supports.

    Returns:
      (z, rows, table_lines, supports, samples): the refined grid, the
      refined n(z) rows in canonical bin order (BGS, LRG, ELG), the bin
      table lines, the raw (unrefined) [first, last] nonzero redshift of
      each bin, and each bin's sample.
    """
    z0, rows_all, lines, supports, samples = None, [], [], [], []
    n = 0
    for sample in nzmod.SAMPLES:
        fname = f"desi_{sample}_{width}_binned_distribution_DR1.txt"
        z, rows = nzmod.load_sample(input_dir, sample, filename=fname)
        if z0 is not None and not np.array_equal(z0, z):
            raise AssertionError("the samples do not share one grid")
        z0 = z
        raw = np.loadtxt(os.path.join(input_dir, fname))
        for r in raw[1:]:
            live = np.nonzero(r > 0)[0]
            supports.append((raw[0][live[0]], raw[0][live[-1]]))
        for row in rows:
            # the bin-table rule of make_desi_nz.py: mean redshift
            # weighted by the distribution, chunk by that mean
            zbar = float(np.sum(z*row)/np.sum(row))
            index, name = 0, None
            for chunk, upper in nzmod.CHUNKS[sample]:
                if zbar < upper:
                    name = chunk
                    break
                index += 1
            lines.append(f"{n:4d}  {sample:4s}  {index:2d}  "
                         f"{name:5s}  {zbar:.8f}")
            rows_all.append(row)
            samples.append(sample)
            n += 1
    return z0, rows_all, lines, np.array(supports), samples


def read_ngal(input_dir, width, samples):
    """Return the per-bin densities [arcmin^-2] of the probes file.

    A sample may list more densities than its distribution file has
    bins: the Delta z = 0.05 file carries one LRG density for a bin
    above z = 1.10 that the LRG distribution does not contain. The
    densities follow the bins in redshift order, so the trailing extras
    are dropped, with a warning; fewer densities than bins is an error.

    Returns:
      (ngal, dropped): the densities in bin order, and
      {sample: [dropped values]}.
    """
    import yaml
    path = os.path.join(input_dir, WIDTHS[width][1])
    tracers = yaml.safe_load(open(path))["probes"]["desi"]["tracers"]
    ngal, dropped = [], {}
    for sample in nzmod.SAMPLES:
        values = [float(v) for v in tracers[f"spec_{sample}"]["ngal"]]
        expected = samples.count(sample)
        if len(values) < expected:
            raise AssertionError(
                f"{path}: {sample} lists {len(values)} densities for "
                f"{expected} bins")
        if len(values) > expected:
            dropped[sample] = values[expected:]
            print(f"WARNING {path}: {sample} lists {len(values)} densities "
                  f"for {expected} bins; dropping the trailing "
                  f"{values[expected:]}", flush=True)
        ngal.extend(values[:expected])
    return np.array(ngal), dropped


def cross_pairs(supports, dz):
    """Bins whose raw supports overlap or touch (one grid step slack)."""
    pairs = []
    for i in range(len(supports)):
        for j in range(i + 1, len(supports)):
            if (supports[j][0] <= supports[i][1] + 1.5*dz and
                    supports[i][0] <= supports[j][1] + 1.5*dz):
                pairs.append((i, j))
    return pairs


def layout(nz_file, nbin, ells, pairs):
    """Ask the compiled interface for the block sizes of the layout."""
    sys.path.insert(0, os.path.join(PROJECT, "interface"))
    import cosmolike_lssty1xdesi_interface as ci
    ci.initial_setup()
    ci.init_accuracy_boost(accuracy_boost=1.0, integration_accuracy=0)
    ci.init_probes(possible_probes="3x2pt")
    ci.init_binning_fourier_ells(ells=ells,
                                 lmax_shear=int(np.ceil(ells[-1])))
    # clustering only: every (lens, source) ggl pair excluded, as in
    # desi_cl.yaml (ggl_exclude: all)
    ci.init_ggl_exclude(np.array(
        [[zl, zs] for zl in range(nbin) for zs in range(N_SOURCE)],
        dtype=int).flatten())
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=nz_file, lens_ntomo=nbin,
        source_multihisto_file=os.path.join(DATA, "lssty1xdesi_source.nz"),
        source_ntomo=N_SOURCE)
    ci.init_gg_cross_pairs(pairs=np.asarray(pairs, dtype=int).flatten())
    return [int(s) for s in
            ci.compute_data_vector_fourier_with_gg_cross_sizes()]


def leaders(case):
    """The sampled leaders of the case's fragment: {name: fiducial}."""
    table = os.path.join(DATA, f"lssty1xdesi_desi_{case}_bins.txt")
    rows = np.loadtxt(table, usecols=(1, 3), dtype=str, ndmin=2)
    zbars = np.loadtxt(table, usecols=4, ndmin=1)
    model = parmod.MODELS["params_desi.yaml"]
    out = {}
    seen = set()
    for n, (sample, chunk) in enumerate(rows, start=1):
        g = parmod.group_key(sample, chunk, model["bias_groups"])
        if ("b", g) not in seen:
            seen.add(("b", g))
            out[f"DESI_B1_{n}"] = model["bias_values"][g]
    for n, ((sample, chunk), zb) in enumerate(zip(rows, zbars), start=1):
        g = parmod.sigv_group_key(sample, chunk, zb)
        if ("s", g) not in seen:
            seen.add(("s", g))
            out[f"DESI_SIGV_{n}"] = model["sigv_values"][g]
    return out


def write_examples(width, case, stem, cls, nbin):
    """Write the case's evaluate and covariance EXAMPLE yamls."""
    number = WIDTHS[width][2]
    # the evaluate example: the linear example of the Delta d_com = 62 Mpc
    # layout on the case's likelihood (growth-tied linear bias, the model
    # the fiducial stage turns into the case's data vector). The paper's
    # model on the case, EXAMPLE_EVALUATE<N>.yaml, is not written here.
    template = os.path.join(PROJECT, "exploration", "EXAMPLE_EVALUATE_LINEAR1.yaml")
    text = open(template).read()
    for old, new in (
            ("  lssty1xdesi.desi_cl:\n", f"  lssty1xdesi.{cls}:\n"),
            ("data_file: lssty1xdesi_desi_dr3_linear.dataset",
             f"data_file: {stem}_linear.dataset"),
            ("chains/desi_theory.modelvector",
             f"chains/desi_{case}_theory.modelvector"),
            ("output: ./projects/lssty1xdesi/chains/EXAMPLE_EVALUATE_LINEAR1",
             f"output: ./projects/lssty1xdesi/chains/EXAMPLE_EVALUATE_LINEAR{number}")):
        if text.count(old) != 1:
            raise AssertionError(f"{template}: anchor {old!r}")
        text = text.replace(old, new)
    text = re.sub(r"      DESI_(B1|SIGV)_\d+: [^\n]*\n", "", text)
    lead = "".join(f"      {k}: {v}\n" for k, v in leaders(case).items())
    anchor = "      w: -1.0\n"
    if text.count(anchor) != 1:
        raise AssertionError(f"{template}: override anchor")
    text = text.replace(anchor, anchor + lead)
    header = (f"# The DESI DR1 fixed-width Delta z = {width} case ({nbin} lens "
              "bins):\n# scripts/make_desi_dz_case.py builds its files.\n")
    out = os.path.join(PROJECT, "exploration",
                       f"EXAMPLE_EVALUATE_LINEAR{number}.yaml")
    open(out, "w").write(header + text)

    # the covariance example: the production yaml on the case
    text = open(os.path.join(PROJECT,
                             "EXAMPLE_EVALUATE_COVARIANCE.yaml")).read()
    for old, new in (
            ("  space: fourier\n",
             f"  space: fourier\n  # the layout and data files of the case\n"
             f"  case: {case}\n"),
            ("projects/lssty1xdesi/EXAMPLE_EVALUATE_COVARIANCE.yaml",
             f"projects/lssty1xdesi/EXAMPLE_EVALUATE_COVARIANCE{number}.yaml"),
            ("output: ./projects/lssty1xdesi/covariance/production_covariance.npz",
             f"output: ./projects/lssty1xdesi/covariance/"
             f"production_covariance_{case}.npz")):
        if text.count(old) != 1:
            raise AssertionError(
                f"EXAMPLE_EVALUATE_COVARIANCE.yaml: anchor {old!r}")
        text = text.replace(old, new)
    out = os.path.join(PROJECT, f"EXAMPLE_EVALUATE_COVARIANCE{number}.yaml")
    open(out, "w").write(header + text)


def write_likelihood(width, case, stem, cls, nbin, counts):
    """Write the case's likelihood class and its yaml of defaults."""
    open(os.path.join(LIKE, f"{cls}.py"), "w").write(f'''"""desi_cl on the DESI DR1 fixed-width Delta z = {width} bins.

{nbin} lens bins ({counts}), built by scripts/make_desi_dz_case.py.
Everything is desi_cl's; the class exists so its yaml can name the
case's dataset and its own per-bin parameter fragment (cobaya merges
desi_cl.yaml underneath, and a class's params block replaces the
parent's, so no parameter of another binning survives).
"""
from cobaya.likelihoods.lssty1xdesi.desi_cl import desi_cl


class {cls}(desi_cl):
  """desi_cl on the Delta z = {width} layout (see the module docstring)."""
''')
    open(os.path.join(LIKE, f"{cls}.yaml"), "w").write(f'''# desi_cl on the DESI DR1 fixed-width Delta z = {width} bins ({nbin} lens
# bins), built by scripts/make_desi_dz_case.py; every option not set
# here is desi_cl.yaml's.
data_file: {stem}.dataset
# the non-Limber FFTLog chi grid of this binning (see desi_cl.yaml):
# provisional until the case's convergence ladder sets it
nonlimber_accuracyboost: 64.0
params: !defaults [params_desi_{case}]
''')


def stage_files(input_dir, width):
    case, stem, cls = case_names(width)
    z, rows, lines, supports, samples = stack(input_dir, width)
    nbin = len(rows)
    counts = ", ".join(f"{s.upper()} {samples.count(s)}"
                       for s in nzmod.SAMPLES)

    nz_file = os.path.join(DATA, f"{stem}.nz")
    np.savetxt(nz_file, np.column_stack([z] + rows), fmt="%.17e")
    with open(os.path.join(DATA, f"{stem}_bins.txt"), "w") as out:
        out.write(f"# bin  sample  chunk_index  chunk_name  mean_redshift\n"
                  f"# DESI DR1 fixed-width Delta z = {width} bins; "
                  "chunk_index counts chunks within the sample\n")
        out.write("\n".join(lines) + "\n")

    ngal, dropped = read_ngal(input_dir, width, samples)
    with open(os.path.join(DATA, f"{stem}_ngal.txt"), "w") as out:
        out.write(f"# bin  ngal [arcmin^-2]: the DESI DR1 forecast density "
                  f"of each Delta z = {width} bin\n# (the forecast's "
                  f"{WIDTHS[width][1]}, whose ngal is its only "
                  "current entry)\n")
        for sample, extra in dropped.items():
            out.write(f"# dropped: {sample} densities {extra} listed beyond "
                      "the distribution file's bins\n")
        for n, v in enumerate(ngal):
            out.write(f"{n:4d}  {v:.10e}\n")

    raw_dz = 3.5e-4
    pairs = cross_pairs(supports, raw_dz)
    with open(os.path.join(DATA, f"{stem}_pairs.txt"), "w") as out:
        out.write("# bin_i  bin_j (i < j): bins whose n(z) supports overlap "
                  "or touch;\n# row n is spectrum n of the gg-cross block\n")
        for i, j in pairs:
            out.write(f"{i} {j}\n")

    ells = np.loadtxt(os.path.join(DATA, "lssty1xdesi_desi_dr3_ells.txt"))
    ncl = len(ells)
    s_ee, s_gs, s_gg, s_ggx = layout(nz_file, nbin, ells, pairs)
    if s_gg != ncl*nbin or s_ggx != ncl*len(pairs):
        raise AssertionError(f"unexpected layout {[s_ee, s_gs, s_gg, s_ggx]}")
    ndata = s_ee + s_gs + s_gg + s_ggx

    zbar = np.array([float(l.split()[4]) for l in lines])
    keep = np.array([[ells[b] + 0.5 <= KMAX*chi_mpc(zb) for b in range(ncl)]
                     for zb in zbar], dtype=int)
    mask = np.zeros(ndata, dtype=int)
    g0 = s_ee + s_gs
    for i in range(nbin):
        mask[g0 + ncl*i: g0 + ncl*(i + 1)] = keep[i]
    x0 = g0 + s_gg
    for n, (i, j) in enumerate(pairs):
        mask[x0 + ncl*n: x0 + ncl*(n + 1)] = keep[i]*keep[j]
    np.savetxt(os.path.join(DATA, f"{stem}.mask"),
               np.column_stack([np.arange(ndata), mask]), fmt="%d")

    # placeholders until the covariance and fiducial stages: the data
    # vector of the paper's model (<stem>_fid) and of the growth-tied
    # linear bias (<stem>_linear_fid)
    for suffix in ("", "_linear"):
        np.savetxt(os.path.join(DATA, f"{stem}{suffix}_fid.datavector"),
                   np.column_stack([np.arange(ndata), np.zeros(ndata)]),
                   fmt=["%d", "%.8e"])
    with open(os.path.join(DATA, f"{stem}_cov"), "w") as out:
        for i in range(ndata):
            out.write(f"{i} {i} 1.0\n")

    # two datasets that differ only in data_file: <stem>.dataset for the
    # paper's example, <stem>_linear.dataset for the exploration examples
    for suffix in ("", "_linear"):
        with open(os.path.join(DATA, f"{stem}{suffix}.dataset"), "w") as out:
            out.write(f"data_file = {stem}{suffix}_fid.datavector\n")
            out.write(f"cov_file = {stem}_cov\n")
            out.write(f"mask_file = {stem}.mask\n")
            out.write(f"nz_lens_file = {stem}.nz\n")
            out.write("nz_source_file = lssty1xdesi_source.nz\n")
            out.write(f"lens_ntomo = {nbin}\n")
            out.write(f"source_ntomo = {N_SOURCE}\n")
            out.write("ell_file = lssty1xdesi_desi_dr3_ells.txt\n")
            out.write(f"lens_bin_table = {stem}_bins.txt\n")
            out.write(f"gg_cross_pairs_file = {stem}_pairs.txt\n")

    subprocess.run([sys.executable, os.path.join(SCRIPTS, "make_desi_params.py"),
                    "--case", case], check=True)
    write_likelihood(width, case, stem, cls, nbin, counts)
    write_examples(width, case, stem, cls, nbin)
    print(f"{case}: {nbin} bins ({counts}), {len(pairs)} cross pairs, "
          f"{ndata} entries, {int(mask.sum())} unmasked; ngal total "
          f"{ngal.sum():.4f} arcmin^-2")


# ---- stage 2: the covariance file ---------------------------------------


def stage_covariance(width):
    case, stem, _ = case_names(width)
    # the adapter imports the shared cosmolike_notebook_utils package,
    # which lives in the cosmolike_core checkout (compute_covariance.py's
    # path rule)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(PROJECT)),
                                    "external_modules", "code",
                                    "cosmolike_core"))
    sys.path.insert(0, os.path.join(PROJECT, "covariance"))
    import lssty1xdesi_covariance as adapter
    archive = np.load(os.path.join(PROJECT, "covariance",
                                   f"production_covariance_{case}.npz"),
                      allow_pickle=False)
    if not np.array_equal(archive["rows"], adapter.observable_rows(case)):
        raise AssertionError("the archive's rows are not the case's layout")
    gauss = archive["gaussian"]
    mask = np.loadtxt(os.path.join(DATA, f"{stem}.mask"), dtype=int)[:, 1]
    if gauss.shape != (mask.size, mask.size):
        raise AssertionError("archive and mask sizes differ")
    live = np.nonzero(mask)[0]
    sub = gauss[np.ix_(live, live)]
    # the masked Gaussian term must be a valid covariance
    np.linalg.cholesky(sub)
    # Gaussian band powers couple no two bands, so most entries are exact
    # zeros: only the upper triangle's nonzeros are written, the reader's
    # format (i j value, i <= j), plus a 1.0 placeholder diagonal on every
    # masked index (never reaches the chi2)
    a, b = np.nonzero(np.triu(sub))
    with open(os.path.join(DATA, f"{stem}_cov"), "w") as out:
        for i, j, v in zip(live[a], live[b], sub[a, b]):
            out.write(f"{i} {j} {v:.12e}\n")
        for i in np.nonzero(mask == 0)[0]:
            out.write(f"{i} {i} 1.0\n")
    print(f"{case}: covariance file with {a.size} entries over {live.size} "
          f"unmasked rows (Gaussian term, positive definite)")


# ---- stage 3: the fiducial data vector ----------------------------------


def stage_fiducial(width):
    """Write the case's two fiducial data vectors (make_desi_fiducial.py).

    The linear example exploration/EXAMPLE_EVALUATE_LINEAR<N>.yaml
    (growth-tied linear bias) writes <stem>_linear_fid.datavector, the
    vector of <stem>_linear.dataset; the paper's example
    EXAMPLE_EVALUATE<N>.yaml (model B + tilts on the pessimistic
    linear-bias prior) writes <stem>_fid.datavector, the vector of
    <stem>.dataset. Each example then evaluates to chi2 ~ 0 on its own
    dataset. The paper's example is written by hand for a new case; until
    it exists only the linear vector is written.

    Arguments:
      width = the case's Delta z, as spelled in WIDTHS.

    Side effects: overwrites the data vectors named above.
    """
    import make_desi_fiducial as fiducial
    number = WIDTHS[width][2]
    case = case_names(width)[0]
    examples = [
        os.path.join(PROJECT, "exploration", f"EXAMPLE_EVALUATE_LINEAR{number}.yaml"),
        os.path.join(PROJECT, f"EXAMPLE_EVALUATE{number}.yaml"),
    ]
    for example in examples:
        if not os.path.exists(example):
            print(f"{case}: {example} does not exist; its data vector is "
                  "not written")
            continue
        target, rows, live, chi2_old = fiducial.write_fiducial(example)
        print(f"{case}: wrote {target} from {os.path.basename(example)} "
              f"({rows} rows, {live} nonzero)")


def main():
    parser = argparse.ArgumentParser(
        description="Build one DESI DR1 fixed-width Delta z case.")
    parser.add_argument("--width", required=True, choices=sorted(WIDTHS))
    parser.add_argument("--stage", required=True,
                        choices=("files", "covariance", "fiducial"))
    parser.add_argument("--input-dir",
                        help="folder of the DR1 distribution and "
                             "probes_properties files (stage files)")
    args = parser.parse_args()
    if args.stage == "files":
        if not args.input_dir:
            parser.error("--input-dir is required for --stage files")
        stage_files(args.input_dir, args.width)
    elif args.stage == "covariance":
        stage_covariance(args.width)
    else:
        stage_fiducial(args.width)


if __name__ == "__main__":
    main()
