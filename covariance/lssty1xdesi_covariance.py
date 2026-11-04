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
covariance/README.md). The data vector's finger-of-god dispersion and
one-loop bias terms are not part of the covariance model - covariance
accuracy targets are parameter-error level, far looser than the
data-vector chi2 rule.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    gaussian_model,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov

# the project folder (one level above covariance/), where data/ lives
_PROJECT = Path(__file__).resolve().parents[1]

# number of lens (DESI thin comoving) and source (LSST Y1) bins
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


def observable_rows():
    """Return the (probe,A,B) rows of the measured 9964-entry layout.

    Field IDs put the 76 lenses before the 5 sources, the shared
    package's convention. Probe IDs are 0 = EE bandpowers (cosmic
    shear) and 3 = galaxy clustering; no gamma_t (probe 2) rows are
    measured. Clustering carries the 76 auto rows first, then the 97
    measured cross pairs with A < B - exactly the data-vector block
    order, so row r of the covariance is entry 53*r of the vector.

    Returns:
      int32 [188, 3] array, columns (probe, A, B).
    """
    rows = []
    # EE: each unordered source pair once, first <= second; source
    # field IDs follow the NLENS lens IDs
    for first in range(NSOURCE):
        for second in range(first, NSOURCE):
            rows.append((0, NLENS + first, NLENS + second))
    # clustering autos, bin order
    for lens in range(NLENS):
        rows.append((3, lens, lens))
    # clustering cross pairs, file order (sorted, A < B)
    pairs = np.loadtxt(_PROJECT/"data/lssty1xdesi_desi_dr3_pairs.txt",
                       dtype=np.int32)
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


def _lens_bias():
    """Return the fiducial linear bias of each of the 76 lens bins.

    data/lssty1xdesi_desi_dr3_bins.txt names each bin's chunk (BGS,
    LRG1..LRG3, ELG1, ELG2); CHUNK_BIAS maps the chunk to its leader
    value. Every downstream bin order follows that file.

    Returns:
      float list [76].
    """
    chunk_names = np.loadtxt(_PROJECT/"data/lssty1xdesi_desi_dr3_bins.txt",
                             dtype=str, usecols=3)
    if len(chunk_names) != NLENS:
        raise ValueError(
            f"bins file lists {len(chunk_names)} bins, expected {NLENS}")
    return [CHUNK_BIAS[str(name)] for name in chunk_names]


def configuration(accuracy_boost=None, gaussian=None, **accuracy_overrides):
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
        accuracy_overrides = keyword arguments that replace single
            entries of default.yaml (for example nonlimber_accuracyboost
            = 16); an unknown name raises TypeError.
    Returns:
        dict of fully resolved settings: the survey and cosmology
        entries below, the accuracy entries of default.yaml (boosted,
        and also their unboosted values) and "gaussian".
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )
    band_first, band_last = _band_edges()
    # per-bin DESI densities [arcmin^-2], shipped in the bin order of
    # every project data file (see the file's header)
    lens_density = np.loadtxt(
        _PROJECT/"data/lssty1xdesi_desi_dr3_ngal.txt", usecols=1)
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
        # n(z) files, relative to the project folder. The likelihood
        # reads the same files with photoz_interpolation_type = 0
        # (cubic spline) and photoz_zmid_convention = 1 (the z column
        # holds bin midpoints); the covariance matches both.
        "lens_file": "data/lssty1xdesi_desi_dr3.nz",
        "source_file": "data/lssty1xdesi_source.nz",
        "photoz_interpolation": 0,
        "photoz_zmid": 1,
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
        "area_deg2": 14000.0,
        "lens_density_arcmin2": lens_density.tolist(),
        "source_density_arcmin2": [2.0]*NSOURCE,
        "sigma_e_component": [0.26]*NSOURCE,
        "bias": _lens_bias(),
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
            of observable_rows(), the data-vector order.
        progress = optional function called as progress(stage,
            elapsed_seconds) while the calculation runs.
        backend = None for the notebook wrappers, interface.covariance
            for the direct production bindings (compute_covariance.py).
    Returns:
        the shared forecast dict (compute_forecast): the G, SSC, cNG and
        total matrices, mean signals, the coordinate of each row (band
        centers in multipole) and the resolved settings. The full layout
        has 9964 entries.
    """
    if rows is None:
        rows = observable_rows()
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )
