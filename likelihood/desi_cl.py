"""Defines the Cobaya likelihood lssty1xdesi.desi_cl (DESI spectroscopic
angular clustering band powers).

The entry `lssty1xdesi.desi_cl` in the likelihood block of a yaml file
makes Cobaya import this module and build the class of the same name,
with the options and defaults of desi_cl.yaml (next to this file). The
data vector is the DESI-DR3 thin-bin forecast: 76 tomographic bins of
equal comoving width (18 BGS + 31 LRG + 27 ELG), each an angular
clustering auto spectrum in band powers of width Delta ell = 10, read
together with its Gaussian covariance and per-bin scale-cut mask from
the files make_desi_dataset.py imported out of the group's sacc file.
Cross-bin spectra have no slot yet (the cross wiring is a held design
discussion); their information is simply absent from the data vector.

The base class _cosmolike_prototype_base does the heavy lifting in its
fourier mode (binning_space: fourier): band centers from the dataset's
ell_file, init_data_fourier, and the Fourier 3x2pt data vector, of
which this probe ("wtheta") fills the clustering block alone. This
subclass adds the two pieces of DESI-specific physics, both applied at
every sampled point:

  galaxy bias - one sampled amplitude per bias chunk (BGS, LRG1-3,
    ELG1-2; the paper's Table 2 ranges), with the per-bin linear bias
    b1_i = amplitude / D(zbar_i): the amplitude is the bias at z = 0 in
    a constant-clustering-amplitude (passive) evolution, and D(zbar_i)
    is the linear growth at the bin's mean redshift, measured from the
    linear P(k) ratio at one large scale where growth is scale
    independent;

  Fingers of God - each thin bin's distribution is the raw histogram
    convolved with a Gaussian in redshift of width
    sigma_z,i = (1 + zbar_i) sigma_v / c, where sigma_v (km/s) is
    sampled once per sample (BGS, LRG, ELG): a line-of-sight peculiar
    velocity v shifts an observed redshift by (1 + z) v / c, and the
    bins are far narrower than the dispersion's effect, which is what
    makes the convolution the dominant small-scale radial systematic
    (the paper's eq. 20). The convolved distributions reach cosmolike
    through the external-n(z) path (set_lens_sample), so the raw
    histograms on disk are never modified.
"""
import os

import numpy as np
from cobaya.likelihoods.lssty1xdesi._cosmolike_prototype_base import \
    _cosmolike_prototype_base
from cobaya.log import LoggedError
from getdist import IniFile
from scipy.ndimage import gaussian_filter1d

import cosmolike_lssty1xdesi_interface as ci

# speed of light in km/s: a line-of-sight peculiar velocity v shifts the
# observed redshift by (1 + z) v / c, so a velocity dispersion sigma_v
# becomes a redshift dispersion sigma_z = (1 + zbar) sigma_v / c
C_KMS = 299792.458

# (sample, chunk index within the sample) -> sampled-parameter suffix:
# the six bias chunks of the paper's Table 2, assigned to bins by the
# bin table make_desi_nz.py wrote
CHUNK_SUFFIX = {
    ("bgs", 0): "BGS",
    ("lrg", 0): "LRG1",
    ("lrg", 1): "LRG2",
    ("lrg", 2): "LRG3",
    ("elg", 0): "ELG1",
    ("elg", 1): "ELG2",
}

# sample -> suffix of its sampled velocity-dispersion parameter
SIGMA_SUFFIX = {"bgs": "BGS", "lrg": "LRG", "elg": "ELG"}

# Fiducial values used when a run does not sample a parameter, from the
# paper's Table 1: bias amplitudes b(z) = amplitude / D(z) per sample
# (every chunk of a sample starts at the sample's value) and velocity
# dispersions in km/s (BGS: Hawkins et al. 2003; LRG: Reid et al. 2014;
# ELG: Avila et al. 2020)
B1_FIDUCIAL = {"BGS": 1.34, "LRG1": 1.70, "LRG2": 1.70, "LRG3": 1.70,
               "ELG1": 0.84, "ELG2": 0.84}
SIGMA_FIDUCIAL = {"BGS": 357.0, "LRG": 326.0, "ELG": 200.0}

# the large scale [1/Mpc] where D(z) is read off the linear P(k) ratio;
# growth is scale independent there to far better than the sub-percent
# level even with massive neutrinos
KREF = 5.0e-3


class desi_cl(_cosmolike_prototype_base):
  """Evaluates the DESI thin-bin clustering likelihood (probe "wtheta")."""

  def initialize(self):
    """Configure the base for the Fourier clustering-only probe.

    Cobaya calls initialize once. super(...) runs the base initialize,
    which reads the .dataset file, sets up cosmolike in fourier mode and
    loads the data vector, covariance and mask; this override then reads
    the bin table (sample, bias chunk and mean redshift of each thin
    bin) and prepares the chunk-to-bin parameter maps.

    Raises:
      LoggedError when the dataset is not in fourier mode, when the
      external-n(z) path is off (the FoG convolution needs it), or when
      the bin table does not cover every lens bin.
    """
    super(desi_cl, self).initialize(probe="wtheta")
    if self.binning_space != "fourier":
      raise LoggedError(
          self.log, "desi_cl needs binning_space: fourier (the data "
          "vector is band powers; got %s)", self.binning_space)
    if not self.external_nz_modeling:
      raise LoggedError(
          self.log, "desi_cl needs external_nz_modeling: 1 - the "
          "Fingers-of-God convolution rebuilds the n(z) at every point")
    if self.use_baryon_pca or self.create_baryon_pca:
      raise LoggedError(
          self.log, "the baryon principal components live on the "
          "real-space data vector; no fourier counterpart exists")

    # the bin table written by make_desi_nz.py, named by the dataset:
    # one row per thin bin with its sample, chunk and mean redshift
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))
    table_file = ini.relativeFileName("lens_bin_table")
    samples = np.loadtxt(table_file, usecols=1, dtype=str)
    chunk_index = np.loadtxt(table_file, usecols=2, dtype=int)
    self.desi_zbar = np.atleast_1d(np.loadtxt(table_file, usecols=4))
    if len(self.desi_zbar) != int(self.lens_ntomo):
      raise LoggedError(
          self.log, "the bin table lists %d bins; the dataset declares "
          "lens_ntomo = %d", len(self.desi_zbar), int(self.lens_ntomo))
    self.desi_b1_suffix = [
        CHUNK_SUFFIX[(s, c)] for s, c in zip(samples, chunk_index)]
    self.desi_sigma_suffix = [SIGMA_SUFFIX[s] for s in samples]

    # the n(z) table the base read for the external-n(z) path: column 0
    # is the shared redshift grid, column i + 1 the raw histogram of bin
    # i; the FoG kernel width is expressed in steps of this grid
    zgrid = np.asarray(self.lens_nz)[:, 0]
    dz = np.diff(zgrid)
    # 1e-6 relative: far looser than any text-format rounding of the
    # grid (12 printed digits leave ~1e-9 relative wobble) and far
    # tighter than anything that could move a kernel of ~4 grid steps
    if not np.all(dz > 0) or np.max(np.abs(dz - dz[0])) > 1.0e-6 * dz[0]:
      raise LoggedError(
          self.log, "the lens n(z) grid must be uniform and increasing "
          "(the Gaussian FoG kernel works in grid steps)")
    self.desi_dz = float(dz[0])

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_lens_related(self, **params):
    """Send the DESI nuisance parameters of this point to cosmolike.

    Replaces the base implementation (LSST lens parameters) with the
    DESI model: per-bin linear bias from the sampled chunk amplitudes
    divided by the growth at each bin's mean redshift, the
    Fingers-of-God convolution of every raw thin-bin n(z) with the
    sample's sampled velocity dispersion, and zeros for every lens
    nuisance this forecast does not model (higher-order bias,
    magnification, point masses, photo-z shifts - spectroscopic
    redshifts carry no shift parameter).

    Every array crosses the compiled-interface boundary as a numpy
    array, never a plain Python list: the carma list-to-arma conversion
    of this environment silently zero-fills sequences longer than
    Armadillo's 16-element preallocation, and this project's lens
    arrays carry 76 entries (the 5-bin donor never saw the bug).

    Arguments:
      **params = the sampled parameters, as keyword arguments:
        DESI_B1_{BGS,LRG1,LRG2,LRG3,ELG1,ELG2} = bias amplitudes
                      (b1 at z = 0 under passive evolution);
        DESI_SIGV_{BGS,LRG,ELG} = velocity dispersions in km/s.
      A parameter the run does not sample takes its Table-1 fiducial.

    Side effects: changes cosmolike's nuisance parameters and replaces
    the lens n(z) of this point.
    """
    ntomo = int(self.lens_ntomo)
    zeros = np.zeros(ntomo)

    # growth at the bin means, normalized to z = 0, from the linear
    # P(k) the pipeline already requests (nonlinear=(True, False))
    pk = self.provider.get_Pk_interpolator(
        ("delta_tot", "delta_tot"), nonlinear=False)
    growth = np.sqrt(pk.P(self.desi_zbar, KREF) / pk.P(0.0, KREF))

    b1 = np.empty(ntomo)
    for i in range(ntomo):
      suffix = self.desi_b1_suffix[i]
      amplitude = float(params.get("DESI_B1_" + suffix,
                                   B1_FIDUCIAL[suffix]))
      b1[i] = amplitude / float(growth[i])
    if not np.all(np.isfinite(b1)) or not np.all(b1 > 0):
      raise LoggedError(
          self.log, "per-bin b1 must be finite and positive; got "
          "b1[:3] = %s from growth[:3] = %s at zbar[:3] = %s",
          b1[:3], np.asarray(growth)[:3], self.desi_zbar[:3])

    ci.set_point_mass(PMV=zeros)
    ci.set_nuisance_bias(B1=b1, B2=zeros, B_MAG=zeros,
                         B3nl=zeros, BK=zeros)

    # Fingers of God: convolve a COPY of each raw histogram (self.lens_nz
    # must keep the file's distributions for the next point).
    # gaussian_filter1d smooths with a Gaussian whose width is given in
    # grid steps; mode="constant" treats the distribution as zero beyond
    # the grid, which is what a histogram that ends there means.
    lens_nz_local = np.asarray(self.lens_nz).copy()
    for i in range(ntomo):
      suffix = self.desi_sigma_suffix[i]
      sigma_v = float(params.get("DESI_SIGV_" + suffix,
                                 SIGMA_FIDUCIAL[suffix]))
      sigma_z = (1.0 + float(self.desi_zbar[i])) * sigma_v / C_KMS
      lens_nz_local[:, i + 1] = gaussian_filter1d(
          lens_nz_local[:, i + 1], sigma=sigma_z / self.desi_dz,
          mode="constant")
    ci.set_lens_sample(lens_nz_local)

    # spectroscopic redshifts: no photo-z shift is modeled, but the
    # setter must run so a value left by another likelihood in the same
    # process cannot survive
    ci.set_nuisance_clustering_photoz(bias=zeros)
