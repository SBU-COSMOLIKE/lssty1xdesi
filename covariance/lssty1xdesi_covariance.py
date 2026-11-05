"""Supplies the DESI DR3 x LSST Y1 survey choices to the shared covariance code.

The covariance matrix C of the data vector gives the expected scatter of
each measured bandpower and the correlations between entries; the
likelihood's chi2 uses its inverse. The shared package
cosmolike_notebook_utils.covariance computes C as the sum of a Gaussian
term (G: cosmic variance, shape noise and shot noise), the super-sample
covariance (SSC) and the connected non-Gaussian term (cNG). This module is
the project's adapter to that package (the "survey" argument of its
functions): it states the survey and forwards the calculation.
compute_covariance.py and the notebook EXAMPLE_EVALUATE_COVARIANCE.ipynb
use it.

The measured layout has 9964 Fourier entries over 53 bandpowers each:

  EE        15 LSST Y1 source-pair spectra      entries     0 -  794
  gg        76 DESI thin-bin auto spectra       entries   795 - 4822
  gg cross  97 DESI measured bin pairs          entries  4823 - 9963

No galaxy-galaxy lensing rows are measured. The cross pairs are the rows
of data/lssty1xdesi_desi_dr3_pairs.txt, in file order - the same order as
the data vector, the mask and the shipped covariance.

The shipped covariance (data/lssty1xdesi_desi_dr3_cov) is imported from
the source forecast's sacc file; this module computes an independent one
with cosmolike's own kernels, so the two can be compared entry by entry.

The Gaussian term's spectra model linear per-bin galaxy bias over the
nonlinear matter power, with the lens redshift-space distortions in
both the Limber window and the non-Limber recompute's j_l'' transfer
(gaussian rsd = nonlimber = true): the exact distortion term lands
the lowest-redshift clustering rows on the source forecast's
covariance (the measured four-model comparison lives in
covariance/README.md). The lens kernels carry the fiducial
finger-of-god smearing: each thin bin's n(z) is convolved once, at
the fixed fiducial dispersions, matching what the data vector's
sampled convolution computes at its fiducial point
(_fog_convolved_lens_file). The one-loop bias terms remain outside
the covariance model - covariance accuracy targets are
parameter-error level, far looser than the data-vector chi2 rule.

Cases. The layout above is the default case, "dr3" (the comoving bins
of the source forecast's sacc file). The fixed-width Delta z binnings
of the DR1 distributions are further cases ("dr1_dz0p02", "dr1_dz0p05",
"dr1_dz0p20"; data built by scripts/make_desi_dz_case.py). A DESI case
selects only the data files - the n(z), bin table, densities and cross
pairs - and with them the number of lens bins and of rows.

One case has a different lens sample: "lsst_y1_3x2pt", the LSST Y1
photometric lenses of a Fourier 3x2pt layout (scripts/
make_lsst_3x2pt_case.py). Its lens kernels carry no FoG smearing, its
biases, densities and footprint are the LSST Y1 values of the lsst_y1
project's adapter, its n(z) files are read with the Z_LOW convention,
and its rows add galaxy-galaxy lensing for every lens-source pair.

The model, the bands and the accuracy controls are the same for every
case. A covariance yaml picks one with the key covariance: case: <name>.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    gaussian_model,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov

# the project folder (one level above covariance/), where data/ lives
_PROJECT = Path(__file__).resolve().parents[1]

# The cases this adapter covers: case name -> (the stem of its data
# files, its lens sample). A "desi" case reads data/<stem>.nz,
# <stem>_bins.txt, <stem>_ngal.txt and <stem>_pairs.txt; the "lsst" case
# reads the LSST Y1 lens n(z) and the constants below.
CASES = {
    "dr3": ("lssty1xdesi_desi_dr3", "desi"),
    "dr1_dz0p02": ("lssty1xdesi_desi_dr1_dz0p02", "desi"),
    "dr1_dz0p05": ("lssty1xdesi_desi_dr1_dz0p05", "desi"),
    "dr1_dz0p20": ("lssty1xdesi_desi_dr1_dz0p20", "desi"),
    "lsst_y1_3x2pt": ("lssty1xdesi_lsst_y1_3x2pt", "lsst"),
}

# The LSST Y1 lens sample of the "lsst" case, the values of the lsst_y1
# project's covariance adapter: five photometric bins at 3.6 galaxies per
# arcmin^2 each, its fiducial linear biases (the LSST_B1 values of its
# 3x2pt example), and the LSST Y1 footprint
LSST_LENS_NZ = "data/lssty1xdesi_lens.nz"
LSST_LENS_DENSITY = 3.6
LSST_LENS_BIAS = [1.72716, 1.65168, 1.61423, 1.92886, 2.11633]
LSST_AREA_DEG2 = 12300.0

# number of lens bins of the default case (DESI thin comoving), and of
# source (LSST Y1) bins in every case
NLENS = 76
NSOURCE = 5

# fiducial linear bias of each sample chunk, the DESI_B1 leader values of
# the EXAMPLE yaml files (one value per chunk of
# data/lssty1xdesi_desi_dr3_bins.txt; every bin of a chunk shares it)
CHUNK_BIAS = {
    "BGS": 1.34,
    "LRG1": 1.70,
    "LRG2": 1.70,
    "LRG3": 1.70,
    "ELG1": 0.84,
    "ELG2": 0.84,
}

# Fiducial Fingers-of-God velocity dispersions [km/s]: the forecast's
# simulation-fit values, the same fiducials the likelihood's sampled
# dispersions start from (the DESI_SIGV leaders of
# likelihood/params_desi.yaml; keep groups and values in sync with
# scripts/make_desi_params.py). The measured sigma_v(zbar) steps
# inside BGS at SIGMA_V_BGS_SPLIT_Z, so the dispersions carry one
# more group than the bias. The thin bins make the smearing a
# kernel-level effect the covariance must share with the data vector.
CHUNK_SIGMA_V = {
    "BGS0": 371.0,
    "BGS1": 358.0,
    "LRG1": 357.0,
    "LRG2": 348.0,
    "LRG3": 343.0,
    "ELG1": 304.0,
    "ELG2": 289.0,
}
SIGMA_V_BGS_SPLIT_Z = 0.255

# speed of light [km/s]: a line-of-sight velocity dispersion sigma_v
# smears observed redshifts by sigma_z = (1 + zbar) sigma_v / c
C_KMS = 299792.458


def _stem(case):
    """Return the data-file stem of one case (see CASES).

    Raises:
      ValueError for a case this adapter does not know.
    """
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}: one of {sorted(CASES)}")
    return CASES[case][0]


def _lsst(case):
    """Whether the case's lens sample is the LSST Y1 photometric one."""
    _stem(case)
    return CASES[case][1] == "lsst"


def _nlens(case):
    """Return the number of lens bins of one case.

    A DESI case counts its bin-table rows; the LSST case counts the n(z)
    file's distribution columns.
    """
    if _lsst(case):
        return np.loadtxt(_PROJECT/LSST_LENS_NZ).shape[1] - 1
    return len(np.loadtxt(_PROJECT/f"data/{_stem(case)}_bins.txt",
                          dtype=str, usecols=3, ndmin=1))


def observable_rows(case="dr3"):
    """Return the (probe,A,B) rows of one case's measured layout.

    Field IDs put the lenses before the 5 sources, the shared
    package's convention. Probe IDs are 0 = EE bandpowers (cosmic
    shear) and 3 = galaxy clustering; no gamma_t (probe 2) rows are
    measured in a DESI case. Clustering carries one auto row per lens
    bin first, then the measured cross pairs with A < B - exactly the
    data-vector block order, so row r of the covariance is entry 53*r of
    the vector. The default case has 76 lens bins and 97 pairs: 188
    rows. The LSST case inserts galaxy-galaxy lensing (probe 2, rows
    (lens, source)) for every pair, lens bin outer, between EE and the
    clustering autos, and has no cross pairs.

    Arguments:
      case = one of CASES (default "dr3").

    Returns:
      int32 [n_rows, 3] array, columns (probe, A, B).
    """
    nlens = _nlens(case)
    rows = []
    # EE: each unordered source pair once, first <= second; source
    # field IDs follow the lens IDs
    for first in range(NSOURCE):
        for second in range(first, NSOURCE):
            rows.append((0, nlens + first, nlens + second))
    # galaxy-galaxy lensing, every pair (the data vector's enumeration
    # with no ggl_exclude list), lens bin outer
    if _lsst(case):
        for lens in range(nlens):
            for source in range(NSOURCE):
                rows.append((2, lens, nlens + source))
    # clustering autos, bin order
    for lens in range(nlens):
        rows.append((3, lens, lens))
    if _lsst(case):
        return np.array(rows, dtype=np.int32)
    # clustering cross pairs, file order (sorted, A < B)
    pairs = np.loadtxt(_PROJECT/f"data/{_stem(case)}_pairs.txt",
                       dtype=np.int32, ndmin=2)
    for a, b in pairs:
        rows.append((3, int(a), int(b)))
    return np.array(rows, dtype=np.int32)


def _band_edges():
    """Return (band_first, band_last) of the 53 measured bandpowers.

    The bandpower windows (data/lssty1xdesi_desi_dr3_windows.npz) store,
    per spectrum and band, weights over integer multipoles; each band's
    nonzero weights cover ten consecutive multipoles with (2l+1)
    weighting, and every spectrum shares the same band definitions - a
    spectrum with a scale cut just keeps fewer of them (the data-vector
    mask handles that). The inclusive support endpoints therefore
    describe each band completely; the shared Fourier covariance applies
    its own (2l+1)-weighting inside [first, last], the same weighting
    that built the band centers.

    Returns:
      two int32 [53] arrays, inclusive first and last multipole per band.

    Raises:
      ValueError when a band's support is not one consecutive-integer
      range, its weights are not (2l+1)-proportional, or two spectra
      disagree on a band's endpoints.
    """
    archive = np.load(_PROJECT/"data/lssty1xdesi_desi_dr3_windows.npz")

    def support(spectrum, band):
        """Return (first, last) of the band's nonzero-weight multipoles."""
        ells = np.rint(
            archive[f"{spectrum}_band{band}_ells"]).astype(np.int64)
        weights = archive[f"{spectrum}_band{band}_weights"]
        live = np.nonzero(weights)[0]
        if live.size == 0:
            raise ValueError(f"{spectrum} band {band}: empty window")
        inside = ells[live]
        if inside.max() - inside.min() + 1 != inside.size:
            raise ValueError(
                f"{spectrum} band {band}: the window support is not one "
                "consecutive range")
        # the weighting rule the band centers were built with
        ratio = weights[live]/(2.0*inside + 1.0)
        if not np.allclose(ratio, ratio[0], rtol=1.0e-6):
            raise ValueError(
                f"{spectrum} band {band}: weights are not "
                "(2l+1)-proportional")
        return int(inside.min()), int(inside.max())

    # the highest-redshift auto spectrum keeps every union band
    reference = f"bin{NLENS - 1}"
    nband = len([key for key in archive.keys()
                 if key.startswith(reference + "_band")
                 and key.endswith("_ells")])
    first = np.zeros(nband, dtype=np.int32)
    last = np.zeros(nband, dtype=np.int32)
    for band in range(nband):
        first[band], last[band] = support(reference, band)
    # one shared definition: spot-check band 0 of two other spectra
    for spectrum in ("bin0", "bin0x1"):
        if support(spectrum, 0) != (int(first[0]), int(last[0])):
            raise ValueError(
                f"{spectrum} band 0 endpoints differ from {reference}: "
                "the spectra do not share one band definition")
    return first, last


def _lens_bias(case="dr3"):
    """Return the fiducial linear bias of each lens bin of one case.

    The case's bin table names each bin's chunk (BGS, LRG1..LRG3,
    ELG1, ELG2); CHUNK_BIAS maps the chunk to its leader value. Every
    downstream bin order follows that file.

    Arguments:
      case = one of CASES (default "dr3").

    Returns:
      float list, one entry per lens bin (the LSST case: LSST_LENS_BIAS).
    """
    if _lsst(case):
        return list(LSST_LENS_BIAS)
    chunk_names = np.loadtxt(_PROJECT/f"data/{_stem(case)}_bins.txt",
                             dtype=str, usecols=3, ndmin=1)
    return [CHUNK_BIAS[str(name)] for name in chunk_names]


def _fog_convolved_lens_file(case="dr3"):
    """Write and return the fiducial-FoG-convolved lens n(z) of a case.

    The likelihood convolves each thin bin's raw histogram with a
    Gaussian in redshift of width sigma_z = (1 + zbar) sigma_v / c at
    every sampled point (sigma_v sampled per chunk). A covariance is
    built once, at the fiducial, so this adapter bakes the same
    convolution at the fixed CHUNK_SIGMA_V values into a derived file
    and hands that to the shared machinery. The smearing is this
    project's radial systematic, so it lives here, in the adapter:
    the shared covariance code never models FoG.

    Arguments:
      case = one of CASES (default "dr3").

    Returns:
      the derived file's path, relative to the project folder (the
      shared machinery joins it onto the project path). The LSST case's
      photometric lenses carry no FoG: its lens n(z) is returned as is.
    """
    if _lsst(case):
        return LSST_LENS_NZ
    stem = _stem(case)
    raw = np.loadtxt(_PROJECT/f"data/{stem}.nz")
    z = raw[:, 0]
    dz = np.diff(z)
    # the kernel width is expressed in grid steps, so the grid must be
    # uniform; 1e-6 relative is far looser than the text round trip
    if not np.all(dz > 0) or np.max(np.abs(dz - dz[0])) > 1.0e-6*dz[0]:
        raise ValueError(
            "the lens n(z) grid must be uniform and increasing")
    bins_file = _PROJECT/f"data/{stem}_bins.txt"
    chunk_names = np.loadtxt(bins_file, dtype=str, usecols=3, ndmin=1)
    zbar = np.loadtxt(bins_file, usecols=4, ndmin=1)
    nlens = len(chunk_names)
    if raw.shape[1] - 1 != nlens:
        raise ValueError(
            f"n(z) holds {raw.shape[1] - 1} bins and the bins file "
            f"{nlens}")
    out = raw.copy()
    for i in range(nlens):
        group = str(chunk_names[i])
        if group == "BGS":
            group = ("BGS0" if float(zbar[i]) < SIGMA_V_BGS_SPLIT_Z
                     else "BGS1")
        sigma_v = CHUNK_SIGMA_V[group]
        sigma_z = (1.0 + float(zbar[i]))*sigma_v/C_KMS
        # mode="constant" treats the histogram as zero beyond the
        # grid, which is what a distribution that ends there means
        out[:, i + 1] = gaussian_filter1d(
            raw[:, i + 1], sigma=sigma_z/float(dz[0]), mode="constant")
    path = f"covariance/{stem}_fog.nz"
    # full double precision: the uniform grid must survive the text
    # round trip
    np.savetxt(_PROJECT/path, out, fmt="%.17e")
    return path


def configuration(accuracy_boost=None, gaussian=None, case="dr3",
                  **accuracy_overrides):
    """Return resolved survey, cosmology and YAML accuracy choices.

    "Resolved" means that every value the calculation uses is written
    out, defaults included, so an archive saved with these settings
    records exactly what ran.

    Arguments:
        accuracy_boost = None uses the value in default.yaml; 1, 2, 4 or
            8 refines every boosted control of that file.
        gaussian = optional dictionary of Gaussian-term choices:
            rsd and nonlimber (both true here: thin lens bins need the
            redshift distortions, and the non-Limber recompute carries
            them through its j_l'' transfer), ia ("none", "NLA" or
            "TATT", default "none"), A1, A2 and B_TA (IA amplitudes,
            default 0). gaussian_model checks it. The project fiducial
            has zero IA amplitudes, so the default matches the
            likelihood examples.
        case = one of CASES (default "dr3"): the layout and data files
            the covariance describes.
        accuracy_overrides = keyword arguments that replace single
            entries of default.yaml (for example nonlimber_accuracyboost
            = 16); an unknown name raises TypeError.
    Returns:
        dict of fully resolved settings: the survey and cosmology
        entries below, the accuracy entries of default.yaml (boosted,
        and also their unboosted values), "gaussian" and "case".
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )
    band_first, band_last = _band_edges()
    # per-bin DESI densities [arcmin^-2], shipped in the bin order of
    # every project data file (see the file's header)
    if _lsst(case):
        lens_density = np.full(_nlens(case), LSST_LENS_DENSITY)
    else:
        lens_density = np.loadtxt(
            _PROJECT/f"data/{_stem(case)}_ngal.txt", usecols=1, ndmin=1)
    settings = {
        # Fiducial cosmology and CAMB settings, the EXAMPLE yaml
        # override point: As_1e9 = 10^9 A_s, w0pwa = w0 + wa (-1 with
        # w = -1 means wa = 0), kmax [1/Mpc] and k_per_logint = CAMB's
        # k range and sampling, non_linear_emul = 2 = CAMB's Halofit
        # (halofit_version) for the nonlinear P(k).
        #
        # mnu = 0, not the likelihood fiducial's 0.06 eV: the halo
        # forecast behind SSC and cNG requires massless neutrinos. The
        # sub-percent C_l change this causes sits far below the
        # parameter-error accuracy a covariance needs.
        "cosmology": {
            "omegam": 0.3,
            "omegab": 0.04,
            "H0": 67.32,
            "ns": 0.96605,
            "As_1e9": 2.1,
            "w": -1.0,
            "w0pwa": -1.0,
            "mnu": 0.0,
            "AccuracyBoost": 1.0,
            "CLAccuracyBoost": 1.0,
            "CAMBAccuracyBoost": 1.0,
            "kmax": 20.0,
            "k_per_logint": 20,
            "non_linear_emul": 2,
            "lens_potential_accuracy": 1.0,
            "halofit_version": "takahashi",
        },
        # n(z) files, relative to the project folder. The lens file
        # is the derived fiducial-FoG convolution of the raw
        # histograms, rewritten on every call
        # (_fog_convolved_lens_file). The likelihood reads the raw
        # files with photoz_interpolation_type = 0 (cubic spline) and
        # photoz_zmid_convention = 1 (the z column holds bin
        # midpoints); the covariance matches both conventions.
        "lens_file": _fog_convolved_lens_file(case),
        "source_file": "data/lssty1xdesi_source.nz",
        "photoz_interpolation": 0,
        # the LSST Y1 n(z) files of the lsst case use Z_LOW (lsst_y1's
        # convention, and its likelihood's photoz_zmid_convention: 0)
        "photoz_zmid": 0 if _lsst(case) else 1,
        # no gamma_t rows are measured; internal field spectra keep
        # every pair regardless
        "excluded_gammat": [],
        "band_first": band_first,
        "band_last": band_last,
        # the project measures Fourier bandpowers only; the shared
        # resolver still derives both geometries, so angular bin edges
        # must exist and be valid - nothing in the measured layout
        # reads them
        "theta_edges_arcmin": np.geomspace(start=2.5, stop=900.0, num=27),
        # ln(M/[Msun/h]) edges of the halo-mass integration panels
        "lnm_edges": cov.halo_mass_edges(),
        # DESI DR3 footprint [deg^2] (fsky = 0.3394); DESI densities
        # per bin; the LSST Y1 sources split the SRD total equally
        # (10/5 = 2.0 per arcmin^2) with shape noise 0.26 per
        # ellipticity component
        "area_deg2": LSST_AREA_DEG2 if _lsst(case) else 14000.0,
        "lens_density_arcmin2": lens_density.tolist(),
        "source_density_arcmin2": [2.0]*NSOURCE,
        "sigma_e_component": [0.26]*NSOURCE,
        "bias": _lens_bias(case),
        # scale-factor edges of the line-of-sight integration panels,
        # at z = 3.5, 2, 1.5, 1, 0.7, 0.4, 0.2 and 1e-5 (the last edge
        # stops just short of the observer); the DESI bins live inside
        # z < 1.7 and the LSST sources inside z < 3.5
        "a_edges": 1.0/(1.0+np.array([3.5, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    settings.update(numerical)
    if gaussian is None:
        # the project default: redshift distortions on, in both the
        # Limber window and the non-Limber recompute's j_l'' transfer
        gaussian = {"rsd": True, "nonlimber": True}
    settings["gaussian"] = gaussian_model(
        gaussian=gaussian, nsource=NSOURCE,
    )
    settings["case"] = case
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the complete forecast state without a covariance.

    Arguments:
        interface = imported cosmolike_lssty1xdesi_interface module.
        settings = resolved mapping from configuration().
    Returns:
        CAMB input tables as a dict, suitable for saving beside results.
    Side effects:
        Replaces the interface's global cosmology and nuisance state. The
        likelihood covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings, project=_PROJECT,
    )


def compute(interface, settings, space="fourier", rows=None, progress=None,
            backend=None):
    """Return the forecast with G, SSC, connected and total matrices.

    Arguments:
        interface = the compiled project module, after initialize().
        settings = the dictionary returned by configuration().
        space = "fourier" (the measured E-mode and clustering
            bandpowers); the project measures no real-space vector.
        rows = optional subset of measured rows, an int32 [n_rows, 3]
            table of (type, A, B); None selects the full 188-row layout
            of observable_rows() for settings["case"], the
            data-vector order.
        progress = optional function called as progress(stage,
            elapsed_seconds) while the calculation runs.
        backend = None for the notebook wrappers, interface.covariance
            for the direct production bindings (compute_covariance.py).
    Returns:
        the shared forecast dict (compute_forecast): the G, SSC, cNG and
        total matrices, mean signals, the coordinate of each row (band
        centers in multipole) and the resolved settings. The full layout
        has 9964 entries in the default case.
    """
    if rows is None:
        rows = observable_rows(settings.get("case", "dr3"))
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )
