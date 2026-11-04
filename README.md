# The lssty1xdesi project

A joint forecast of DESI DR3 spectroscopic galaxy clustering and LSST
Y1 cosmic shear in Fourier band powers. The galaxy sample is split
into 76 thin comoving lens bins (BGS, LRG1-3, ELG1-2; mean redshifts
0.11 to 1.55; `data/lssty1xdesi_desi_dr3_bins.txt` names each bin's
chunk and `data/lssty1xdesi_desi_dr3_ngal.txt` its density). The data
vector holds 15 shear pairs, 76 clustering autos and 97 adjacent-bin
clustering crosses, each in 53 bands of width $\Delta\ell = 10$
covering $15 \le \ell \le 544$ (9,964 points).

The thin bins are the project's defining numerical challenge: their
radial kernels keep real non-Limber power far past the multipole
where photometric surveys hand off to Limber, and the finger-of-god
smearing couples neighboring bins. The likelihood therefore computes
the full density-density term with the low-rank non-Limber engine
(`adopt_limber_gg: 0`, `nonlimber_lowrank_gg: 1`) and rebuilds the 76
lens distributions through the FoG convolution at every evaluation
(`external_nz_modeling: 1`), with one linear bias (`DESI_B1_*`) and
one velocity dispersion (`DESI_SIGV_*`) sampled per bin.

- `likelihood/desi_cl.py` scores the band-power vector;
  the `combo_*` variants expose the probe subsets.
- `EXAMPLE_EVALUATE1.yaml` evaluates the fiducial point (a commented
  `bfmt` theory block shows how to turn each baryonic-feedback model
  on); `EXAMPLE_EMUL2_EVALUATE1.yaml` is the hybrid-emulator variant,
  with MCMC, Nautilus and profile examples alongside.
- [tests/](tests/README.md) documents the test suites: frozen-reference
  and race checks, accuracy-knob scans (including the project's own
  non-Limber controls), Halofit vs EuclidEmulator2, baryonic feedback,
  photo-z conventions, the non-Limber and low-rank engine checks, the
  cache ladder, and the optional n(z) reconstruction study.
- [covariance/](covariance/README.md) documents covariance generation:
  the Gaussian part with RSD and non-Limber clustering spectra, Limber
  SSC and cNG, the comparison against the source forecast, and the
  `EXAMPLE_EVALUATE_COVARIANCE.ipynb` notebook.

The survey inputs (densities, fiducial biases, $f_{\rm sky} = 0.3394$,
the supplied comparison covariance) are ported from the source DESI
DR3 x LSST Y1 forecast; `covariance/README.md` records the
model-choice differences the comparison revealed. Scale-cut choices
and release notes belong here as the analysis matures.
