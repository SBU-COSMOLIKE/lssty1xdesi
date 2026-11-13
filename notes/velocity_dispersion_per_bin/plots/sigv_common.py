"""Shared pieces of the velocity-dispersion data generators: the fiducial
cosmology, one CAMB linear P(k), the linear velocity sigma_lin(z), the
Evrard et al. (2008) virial velocity, and the project's dispersion groups.

The fiducial is the paper yamls' override block (notes fiducial section):
10^9 A_s = 2.1, n_s = 0.96605, H_0 = 67.32 km/s/Mpc, Omega_b = 0.04,
Omega_m = 0.3, sum m_nu = 0.06 eV, w_0 = -1, w_a = 0, tau = 0.0697186.
Only the generators import this module; the plot scripts read data.json.
"""
import os

import numpy as np

H0, OMEGA_M, OMEGA_B = 67.32, 0.3, 0.04
NS, AS_1E9, MNU, TAU = 0.96605, 2.1, 0.06, 0.0697186
h = H0/100.0
C_KMS = 299792.458
KREF = 5.0e-3  # 1/Mpc, where the likelihood reads the growth factor
PRJ = ("/Users/vivianmiranda/data/COCOA/november2026/sbu_cosmolike/cocoa/"
       "Cocoa/projects/lssty1xdesi")

GROUPS = ["BGS0", "BGS1", "LRG1", "LRG2", "LRG3", "ELG1", "ELG2"]
SIGV = dict(zip(GROUPS, [371.0, 358.0, 357.0, 348.0, 343.0, 304.0, 289.0]))
# dispersion-group bins (1-based ranges) on Delta z = 0.02; the other
# layouts' ranges are in the note's fiducial section
GROUP_BINS_DZ0P02 = {"BGS0": (1, 8), "BGS1": (9, 15), "LRG1": (16, 25),
                     "LRG2": (26, 35), "LRG3": (36, 50), "ELG1": (51, 65),
                     "ELG2": (66, 90)}
BIN_FILE_DZ0P02 = os.path.join(PRJ, "data",
                               "lssty1xdesi_desi_dr1_dz0p02_bins.txt")


class LinearVelocity:
    """sigma_lin(z) = a H(z) f(z) sigma_d(z) from one CAMB linear P(k)."""

    def __init__(self):
        import camb
        omch2 = (OMEGA_M - OMEGA_B)*h**2 - (MNU*(3.046/3.0)**0.75)/94.0708
        pars = camb.set_params(H0=H0, ombh2=OMEGA_B*h**2, omch2=omch2,
                               ns=NS, As=AS_1E9*1e-9, mnu=MNU, tau=TAU,
                               num_massive_neutrinos=1, nnu=3.046,
                               kmax=100.0, k_per_logint=30, AccuracyBoost=2.0)
        zgrid = np.linspace(0.0, 2.0, 81)
        pars.set_matter_power(redshifts=zgrid[::-1], kmax=100.0,
                              k_per_logint=30, nonlinear=False)
        self.res = camb.get_results(pars)
        self.pk = self.res.get_matter_power_interpolator(
            nonlinear=False, hubble_units=False, k_hunit=False,
            var1="delta_tot", var2="delta_tot")
        self.lnk = np.linspace(np.log(1e-5), np.log(80.0), 4000)
        self.k = np.exp(self.lnk)

    def efunc(self, z):
        return float(self.res.hubble_parameter(z))/H0

    def growth(self, z):
        """D(z) as the likelihood reads it: sqrt(P(z, KREF)/P(0, KREF))."""
        return float(np.sqrt(self.pk.P(z, KREF)/self.pk.P(0.0, KREF)))

    def growth_rate(self, z):
        """f(z) = -(1 + z) dlnD/dz, five-point stencil."""
        dz = 0.01
        zs = np.clip(z + dz*np.array([2.0, 1.0, -1.0, -2.0]), 0.0, None)
        lnD = np.log([self.growth(zz) for zz in zs])
        return -(1.0 + z)*(-lnD[0] + 8.0*lnD[1] - 8.0*lnD[2] + lnD[3])/(12.0*dz)

    def sigma_d(self, z):
        """One-dimensional linear displacement [Mpc]:
        sigma_d^2 = (1/6 pi^2) int P_lin(k, z) dk."""
        integrand = self.pk.P(z, self.k)*self.k
        return float(np.sqrt(np.trapz(integrand, self.lnk)/(6.0*np.pi**2)))

    def sigma_lin(self, z):
        """One-dimensional rms linear velocity [km/s]: a H f sigma_d."""
        return (H0*self.efunc(z)/(1.0 + z)*self.growth_rate(z)
                *self.sigma_d(z))


def efunc_lcdm(z):
    """E(z) of flat LambdaCDM with Omega_m = 0.3 (enough for sigma_halo)."""
    return np.sqrt(OMEGA_M*(1.0 + z)**3 + 1.0 - OMEGA_M)


def sigma_halo(log_m_h, z):
    """Evrard et al. (2008): the 1D dark-matter dispersion [km/s] of a halo
    of mass M_200c = 10^log_m_h Msun/h; h(z) M[Msun] = E(z) M[Msun/h]."""
    return 1082.9*(efunc_lcdm(z)*10.0**np.asarray(log_m_h)/1e15)**0.3361


def read_zbar_dz0p02():
    """Mean redshift of every Delta z = 0.02 bin, from the bin table."""
    return np.loadtxt(BIN_FILE_DZ0P02, usecols=(0, 4))[:, 1]
