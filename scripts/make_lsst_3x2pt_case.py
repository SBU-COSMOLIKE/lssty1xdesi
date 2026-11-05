"""Build the LSST Y1 3x2pt case in Fourier band powers.

The LSST Y1 lens and source samples (data/lssty1xdesi_lens.nz and
data/lssty1xdesi_source.nz, five bins each, the lsst_y1 project's files,
read with the Z_LOW convention) on the bands of the DESI layouts (the 53
DR3 bands, data/lssty1xdesi_desi_dr3_ells.txt), so a Fisher matrix of
this case and one of a DESI case share one multipole grid and can be
summed (the block-diagonal 6x2pt).

The layout is cosmolike's Fourier 3x2pt: cosmic shear (15 source
pairs), galaxy-galaxy lensing (every lens-source pair: 25, the
enumeration with no ggl_exclude list) and the 5 clustering autos. The
scale cuts follow scripts/make_desi_dz_case.py's rule for clustering and
apply it to galaxy-galaxy lensing through the lens bin: a band is kept
when l + 1/2 <= KMAX chi(zbar of the lens bin). Cosmic shear keeps every
band (the bands stop at l = 544).

Three stages, as make_desi_dz_case.py's:

  files      - mask, dataset with placeholder data vector and
               covariance, the likelihood class and yaml, and the
               EXAMPLE yamls (EXAMPLE_EVALUATE5.yaml,
               EXAMPLE_EVALUATE_COVARIANCE5.yaml);
  covariance - the cosmolike covariance file from the Gaussian term of
               compute_covariance.py's archive (case lsst_y1_3x2pt);
  fiducial   - the model vector at the EXAMPLE's fiducial point as the
               case's data vector.

Usage (from the Cocoa/ folder, cocoapy311 active, start_cocoa.sh
sourced, the project compiled with its covariance interface):

    python ./projects/lssty1xdesi/scripts/make_lsst_3x2pt_case.py --stage files
"""
import argparse
import os
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
import make_desi_dz_case as dz      # noqa: E402  KMAX, chi_mpc, paths

PROJECT, DATA, LIKE = dz.PROJECT, dz.DATA, dz.LIKE
CASE = "lsst_y1_3x2pt"
STEM = f"lssty1xdesi_{CASE}"
CLS = "lsst_y1_3x2pt"
NUMBER = 5
LENS_NZ = "lssty1xdesi_lens.nz"
SOURCE_NZ = "lssty1xdesi_source.nz"

# The fiducial point of the EXAMPLE: the DESI cases' cosmology (the
# paper's fiducial, so the case's Fisher sums with theirs), the lsst_y1
# 3x2pt example's lens biases and NLA amplitudes, and zero photo-z shifts
# and shear calibrations (the forecast's centered priors)
COSMOLOGY = {"As_1e9": 2.1, "ns": 0.96605, "H0": 67.32, "omegab": 0.04,
             "omegam": 0.3, "mnu": 0.06, "w0pwa": -1.0, "w": -1.0}
LSST_BIAS = [1.72716, 1.65168, 1.61423, 1.92886, 2.11633]
NUISANCE = {
    **{f"LSST_DZ_S{i}": 0.0 for i in range(1, 6)},
    **{f"LSST_M{i}": 0.0 for i in range(1, 6)},
    **{f"LSST_DZ_L{i}": 0.0 for i in range(1, 6)},
    **{f"LSST_B1_{i}": b for i, b in enumerate(LSST_BIAS, start=1)},
    # NLA (A1 at the pivot and its redshift power eta1); the TATT
    # amplitudes are fixed parameters of params_source.yaml
    "LSST_A1_1": 0.606102,
    "LSST_A1_2": -1.51541,
}


def mean_z(filename):
    """Mean redshift of each bin of an n(z) file, with the Z_LOW reading."""
    t = np.loadtxt(os.path.join(DATA, filename))
    z = t[:, 0] + 0.5*(t[1, 0] - t[0, 0])
    return np.array([np.sum(z*t[:, i])/np.sum(t[:, i])
                     for i in range(1, t.shape[1])])


def lens_zbar():
    """Mean redshift of each lens bin."""
    return mean_z(LENS_NZ)


# 1 + z0 of cosmolike's IA power law (nuisance.oneplusz0_ia)
ONEPLUSZ0_IA = 1.62


def source_nla_amplitudes():
    """The fiducial NLA power law at each source bin's mean redshift.

    The likelihood's NLA amplitude evolves as A1 ((1+z)/(1+z0))^eta1
    inside every bin; the Gaussian covariance takes one constant per
    source bin, so each bin gets the power law at its mean redshift.
    Only the covariance's signal term carries this approximation.
    """
    zbar = mean_z(SOURCE_NZ)
    return NUISANCE["LSST_A1_1"]*((1.0 + zbar)/ONEPLUSZ0_IA)**NUISANCE["LSST_A1_2"]


def layout(ells):
    """Block sizes of the Fourier 3x2pt layout, from the interface."""
    sys.path.insert(0, os.path.join(PROJECT, "interface"))
    import cosmolike_lssty1xdesi_interface as ci
    ci.initial_setup()
    ci.init_accuracy_boost(accuracy_boost=1.0, integration_accuracy=0)
    ci.init_probes(possible_probes="3x2pt")
    ci.init_binning_fourier_ells(ells=ells,
                                 lmax_shear=int(np.ceil(ells[-1])))
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=os.path.join(DATA, LENS_NZ), lens_ntomo=5,
        source_multihisto_file=os.path.join(DATA, SOURCE_NZ),
        source_ntomo=dz.N_SOURCE)
    return [int(s) for s in ci.compute_data_vector_3x2pt_fourier_sizes()]


def write_examples():
    """Write EXAMPLE_EVALUATE5.yaml and EXAMPLE_EVALUATE_COVARIANCE5.yaml."""
    text = open(os.path.join(PROJECT, "EXAMPLE_EVALUATE1.yaml")).read()
    start = text.index("likelihood:\n")
    end = text.index("\nparams:\n")
    block = f"""likelihood:
  lssty1xdesi.{CLS}:
    path: ./external_modules/data/lssty1xdesi
    data_file: {STEM}.dataset
    print_datavector: False
    print_datavector_file: "./projects/lssty1xdesi/chains/{CASE}_theory.modelvector"
"""
    text = text[:start] + block + text[end:]
    o0 = text.index("    override:\n") + len("    override:\n")
    o1 = text.index("\noutput:")
    point = {**COSMOLOGY, **NUISANCE}
    over = "".join(f"      {k}: {v}\n" for k, v in point.items())
    # text[o1:] starts with "\noutput:", which keeps the blank line the
    # template has between the override block and the output key
    text = text[:o0] + over + text[o1:]
    text = text.replace("output: ./projects/lssty1xdesi/chains/EXAMPLE_EVALUATE1",
                        f"output: ./projects/lssty1xdesi/chains/EXAMPLE_EVALUATE{NUMBER}")
    header = ("# The LSST Y1 3x2pt case in Fourier band powers (the DESI bands):\n"
              "# scripts/make_lsst_3x2pt_case.py builds its files.\n")
    open(os.path.join(PROJECT, f"EXAMPLE_EVALUATE{NUMBER}.yaml"), "w").write(header + text)

    text = open(os.path.join(PROJECT, "EXAMPLE_EVALUATE_COVARIANCE.yaml")).read()
    a1 = ", ".join(f"{a:.6f}" for a in source_nla_amplitudes())
    gaussian = f"""  # Gaussian spectra only; SSC/cNG keep their existing model (zero
  # IA, Limber). The shear and galaxy-galaxy lensing signals carry the
  # fiducial NLA: A1 ((1+z)/1.62)^eta1 at each source bin's mean
  # redshift (A1 = {NUISANCE["LSST_A1_1"]}, eta1 = {NUISANCE["LSST_A1_2"]}).
  gaussian:
    nonlimber: true
    rsd: true
    ia: NLA
    A1: [{a1}]
"""
    g0 = text.index("  # Gaussian spectra only;")
    g1 = text.index("    ia: none\n") + len("    ia: none\n")
    text = text[:g0] + gaussian + text[g1:]
    for old, new in (
            ("  space: fourier\n",
             f"  space: fourier\n  # the layout and data files of the case\n  case: {CASE}\n"),
            ("projects/lssty1xdesi/EXAMPLE_EVALUATE_COVARIANCE.yaml",
             f"projects/lssty1xdesi/EXAMPLE_EVALUATE_COVARIANCE{NUMBER}.yaml"),
            ("output: ./projects/lssty1xdesi/covariance/production_covariance.npz",
             f"output: ./projects/lssty1xdesi/covariance/production_covariance_{CASE}.npz")):
        if text.count(old) != 1:
            raise AssertionError(f"covariance yaml anchor {old!r}")
        text = text.replace(old, new)
    open(os.path.join(PROJECT, f"EXAMPLE_EVALUATE_COVARIANCE{NUMBER}.yaml"),
         "w").write(header + text)


def write_likelihood():
    """The Fourier subclass of combo_3x2pt and its yaml of defaults."""
    open(os.path.join(LIKE, f"{CLS}.py"), "w").write(f'''"""combo_3x2pt in Fourier band powers on the LSST Y1 samples.

The LSST Y1 lens and source samples on the DESI layouts' bands, built by
scripts/make_lsst_3x2pt_case.py. Everything is combo_3x2pt's; the class
exists so its yaml can declare the Fourier binning and name the case's
dataset (cobaya merges combo_3x2pt.yaml underneath).
"""
from cobaya.likelihoods.lssty1xdesi.combo_3x2pt import combo_3x2pt


class {CLS}(combo_3x2pt):
  """combo_3x2pt on the Fourier LSST Y1 layout (see the module docstring)."""
''')
    open(os.path.join(LIKE, f"{CLS}.yaml"), "w").write(f'''# combo_3x2pt in Fourier band powers on the LSST Y1 lens and source
# samples, built by scripts/make_lsst_3x2pt_case.py; every option not
# set here is combo_3x2pt.yaml's (its photoz_zmid_convention: 0 is the
# Z_LOW reading of the LSST Y1 n(z) files).
data_file: {STEM}.dataset
# band powers at the dataset's explicit centers (the DESI bands)
binning_space: fourier
''')


def stage_files():
    ells = np.loadtxt(os.path.join(DATA, "lssty1xdesi_desi_dr3_ells.txt"))
    ncl = len(ells)
    s_ee, s_gs, s_gg = layout(ells)
    nlens = s_gg // ncl
    if s_gs != 25*ncl or nlens != 5:
        raise AssertionError(f"unexpected layout {[s_ee, s_gs, s_gg]}")
    zbar = lens_zbar()
    keep = np.array([[ells[b] + 0.5 <= dz.KMAX*dz.chi_mpc(z) for b in range(ncl)]
                     for z in zbar], dtype=int)
    ndata = s_ee + s_gs + s_gg
    mask = np.zeros(ndata, dtype=int)
    mask[:s_ee] = 1
    for lens in range(5):
        for source in range(dz.N_SOURCE):
            n = lens*dz.N_SOURCE + source
            mask[s_ee + ncl*n: s_ee + ncl*(n + 1)] = keep[lens]
        mask[s_ee + s_gs + ncl*lens: s_ee + s_gs + ncl*(lens + 1)] = keep[lens]
    np.savetxt(os.path.join(DATA, f"{STEM}.mask"),
               np.column_stack([np.arange(ndata), mask]), fmt="%d")
    np.savetxt(os.path.join(DATA, f"{STEM}_fid.datavector"),
               np.column_stack([np.arange(ndata), np.zeros(ndata)]),
               fmt=["%d", "%.8e"])
    with open(os.path.join(DATA, f"{STEM}_cov"), "w") as out:
        for i in range(ndata):
            out.write(f"{i} {i} 1.0\n")
    with open(os.path.join(DATA, f"{STEM}.dataset"), "w") as out:
        out.write(f"data_file = {STEM}_fid.datavector\n")
        out.write(f"cov_file = {STEM}_cov\n")
        out.write(f"mask_file = {STEM}.mask\n")
        out.write(f"nz_lens_file = {LENS_NZ}\n")
        out.write(f"nz_source_file = {SOURCE_NZ}\n")
        out.write("lens_ntomo = 5\n")
        out.write(f"source_ntomo = {dz.N_SOURCE}\n")
        out.write("ell_file = lssty1xdesi_desi_dr3_ells.txt\n")
    write_likelihood()
    write_examples()
    print(f"{CASE}: EE {s_ee} + gs {s_gs} + gg {s_gg} = {ndata} entries, "
          f"{int(mask.sum())} unmasked; lens zbar {np.round(zbar, 3)}")


def stage_covariance():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(PROJECT)),
                                    "external_modules", "code", "cosmolike_core"))
    sys.path.insert(0, os.path.join(PROJECT, "covariance"))
    import lssty1xdesi_covariance as adapter
    archive = np.load(os.path.join(PROJECT, "covariance",
                                   f"production_covariance_{CASE}.npz"),
                      allow_pickle=False)
    if not np.array_equal(archive["rows"], adapter.observable_rows(CASE)):
        raise AssertionError("the archive's rows are not the case's layout")
    gauss = archive["gaussian"]
    mask = np.loadtxt(os.path.join(DATA, f"{STEM}.mask"), dtype=int)[:, 1]
    live = np.nonzero(mask)[0]
    sub = gauss[np.ix_(live, live)]
    np.linalg.cholesky(sub)
    a, b = np.nonzero(np.triu(sub))
    with open(os.path.join(DATA, f"{STEM}_cov"), "w") as out:
        for i, j, v in zip(live[a], live[b], sub[a, b]):
            out.write(f"{i} {j} {v:.12e}\n")
        for i in np.nonzero(mask == 0)[0]:
            out.write(f"{i} {i} 1.0\n")
    print(f"{CASE}: covariance file with {a.size} entries over {live.size} "
          f"unmasked rows (Gaussian term, positive definite)")


def stage_fiducial():
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file
    info = yaml_load_file(os.path.join(PROJECT, f"EXAMPLE_EVALUATE{NUMBER}.yaml"))
    point = dict(info["sampler"]["evaluate"]["override"])
    for key in ("sampler", "output", "force", "resume", "debug"):
        info.pop(key, None)
    target = os.path.join(DATA, f"{STEM}_fid.datavector")
    like = info["likelihood"][f"lssty1xdesi.{CLS}"]
    like["print_datavector"] = True
    like["print_datavector_file"] = target
    model = get_model(info)
    point = {k: v for k, v in point.items()
             if k in model.parameterization.sampled_params()}
    model.logposterior(point, cached=False)
    v = np.loadtxt(target)
    print(f"{CASE}: fiducial data vector written ({v.shape[0]} rows, "
          f"{int(np.sum(v[:, 1] != 0))} nonzero)")


def main():
    parser = argparse.ArgumentParser(
        description="Build the LSST Y1 3x2pt Fourier case.")
    parser.add_argument("--stage", required=True,
                        choices=("files", "covariance", "fiducial"))
    args = parser.parse_args()
    {"files": stage_files, "covariance": stage_covariance,
     "fiducial": stage_fiducial}[args.stage]()


if __name__ == "__main__":
    main()
