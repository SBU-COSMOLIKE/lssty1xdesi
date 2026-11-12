"""Write plots/limber_toy/data.json: a toy that shows why two redshift shells
with no overlap still have a cross spectrum, and why the Limber
approximation sets it to zero.

The toy is flat-sky. Two shells, each a top hat of comoving width WIDTH,
sit at mean comoving distance CHI_BAR with a gap of g widths between their
edges (g = 0: touching neighbours; g = 1: two bins apart; g = 2: three
apart), so their centers are d = (1 + g) WIDTH apart. With the shells'
radial profiles n_i(chi), normalized to unit integral, and their Fourier
transforms along the line of sight,

    W_i(k_par) = integral dchi n_i(chi) exp(i k_par chi),

the cross spectrum at multipole ell (transverse wavenumber k_perp =
ell/CHI_BAR) is, up to a factor common to every spectrum,

    C_ij(ell) ~ integral dk_par/(2 pi) P_s(k_perp, k_par) Re[W_i W_j*],
    Re[W_i W_j*] = cos(k_par d) sinc^2(k_par WIDTH/2),

the sum over line-of-sight modes k_par of the power of each mode times its
weight: cos(k_par d) is +1 for modes that take the same value in both
shells and -1 for modes with a crest in one and a trough in the other.
Without overlap the positive and negative weights have equal total area.
The Limber approximation replaces P_s by its value at k_par = 0, so those
areas cancel and C_ij = 0. The exact integral weights each mode by its own
power and does not cancel.

The power is P_s = (b + f mu^2)^2 P(k) with k = sqrt(k_perp^2 + k_par^2)
and mu = k_par/k: real space for f = 0, the linear Kaiser redshift-space
distortion for f > 0 (b = 1.5, f = 0.8, roughly an ELG bin at z = 0.9).
P(k) is the BBKS fit (Bardeen et al. 1986, ApJ 304, 15) with shape
Gamma = Omega_m h and n_s = 1: a smooth spectrum without baryon wiggles,
enough for the toy. Distances are in Mpc; the shells mimic one Delta z =
0.02 ELG bin at z = 0.93 (CHI_BAR, WIDTH). The correlation coefficient
r_ell = C_ij/sqrt(C_ii C_jj) removes the common factors.

    python make_limber_toy_data.py      (from any folder; numpy only)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHI_BAR = 3200.0       # Mpc, about z = 0.93
WIDTH = 55.0           # Mpc, one Delta z = 0.02 bin there
GAPS = [0, 1, 2]       # gap between the shells' edges, in widths
H = 0.674
OMEGA_M = 0.315
BIAS = 1.5
GROWTH_RATE = 0.8
ELLS = np.unique(np.round(np.geomspace(2, 400, 120)).astype(int))
ELL_INTEGRAND = [20, 200]   # the multipoles of the integrand panel
GAP_SHOWN = 1               # the pair of the shell and integrand panels
K_PAR = np.linspace(-0.8, 0.8, 40001)   # 1/Mpc, fine enough for cos(k d) sinc^2
K_PAR_SHOWN = 0.12          # 1/Mpc, the range drawn in the integrand panel


def p_bbks(k):
    '''BBKS power spectrum shape (arbitrary normalization), k in 1/Mpc.'''
    q = np.maximum(k/(OMEGA_M*H*H), 1e-8)   # k in h/Mpc over Gamma = Omega_m h
    t = (np.log(1 + 2.34*q)/(2.34*q)
         * (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**-0.25)
    return k*t**2


def sinc2(x):
    return np.sinc(x/np.pi)**2


def power(k_perp, k_par, f):
    k = np.sqrt(k_perp**2 + k_par**2)
    mu2 = np.where(k > 0, k_par**2/np.maximum(k, 1e-30)**2, 0.0)
    return (BIAS + f*mu2)**2*p_bbks(k)


def r_ell(ell, gap, f):
    '''Exact flat-sky correlation coefficient of two top hats.'''
    p = power(ell/CHI_BAR, K_PAR, f)
    w = sinc2(K_PAR*WIDTH/2)
    d = (1 + gap)*WIDTH
    return float(np.trapz(p*w*np.cos(K_PAR*d), K_PAR)/np.trapz(p*w, K_PAR))


def negative_intervals(k, w):
    '''The [start, end] intervals of k where w < 0.'''
    out, start = [], None
    for kk, ww in zip(k, w):
        if ww < 0 and start is None:
            start = float(kk)
        elif ww >= 0 and start is not None:
            out.append([start, float(kk)])
            start = None
    if start is not None:
        out.append([start, float(k[-1])])
    return out


def main():
    d = (1 + GAP_SHOWN)*WIDTH
    # panel 1: the two shells and three line-of-sight modes
    x = np.linspace(-1.2*WIDTH, d + 1.2*WIDTH, 600)
    modes = [{'k_par_over_pi_d': m, 'value': np.cos(m*np.pi*x/d).tolist()} for m in (0, 1, 2)]
    shells = {'near': [-WIDTH/2, WIDTH/2], 'far': [d - WIDTH/2, d + WIDTH/2]}
    # panel 2: the integrand over k_par >= 0 (it is even), over P at k_par = 0
    k = np.linspace(0.0, K_PAR_SHOWN, 2401)
    weight = np.cos(k*d)*sinc2(k*WIDTH/2)
    integrand = []
    for ell in ELL_INTEGRAND:
        k_perp = ell/CHI_BAR
        ratio = power(k_perp, k, 0.0)/power(k_perp, 0.0, 0.0)
        full = power(k_perp, K_PAR, 0.0)
        w_full = sinc2(K_PAR*WIDTH/2)
        integrand.append({
            'ell': ell, 'k_perp': k_perp, 'exact': (weight*ratio).tolist(),
            'power_ratio': ratio.tolist(),
            'r_exact': float(np.trapz(full*w_full*np.cos(K_PAR*d), K_PAR)
                             / np.trapz(full*w_full, K_PAR))})
    # panel 3: r_ell against ell, real space and Kaiser
    curves = []
    for gap in GAPS:
        for f in (0.0, GROWTH_RATE):
            curves.append({'gap': gap, 'f': f, 'r': [r_ell(ell, gap, f) for ell in ELLS],
                           'ell_scale': CHI_BAR/((1 + gap)*WIDTH)})
    data = {
        'about': (
            'Flat-sky toy of two top-hat shells of width WIDTH at mean comoving distance '
            'CHI_BAR with an edge gap of g widths: the line-of-sight modes, the integrand of '
            'the exact cross spectrum against the Limber one (whose positive and negative '
            'parts cancel), and the correlation coefficient r_ell in real space and with the '
            'linear Kaiser term. P(k): BBKS shape, Gamma = Omega_m h, n_s = 1. Written by '
            'notes/cross_correlations/plots/make_limber_toy_data.py.'),
        'chi_bar_Mpc': CHI_BAR, 'width_Mpc': WIDTH, 'bias': BIAS, 'growth_rate': GROWTH_RATE,
        'gap_shown': GAP_SHOWN, 'distance_Mpc': d,
        'shell_x_Mpc': x.tolist(), 'shells': shells, 'modes': modes,
        'k_par': k.tolist(), 'limber': weight.tolist(), 'integrand': integrand,
        # the k_par intervals where the weight cos(k_par d) sinc^2 is negative:
        # the modes with a crest in one shell and a trough in the other
        'negative_weight': negative_intervals(k, weight),
        'ell': ELLS.tolist(), 'curves': curves,
    }
    folder = os.path.join(HERE, 'limber_toy')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.join(folder, 'data.json'))
    for it in integrand:
        print(f"ell {it['ell']}: r_exact {it['r_exact']:+.4f} (Limber 0)")
    for c in curves:
        print(f"gap {c['gap']} f {c['f']}: r(20) {c['r'][int(np.argmin(np.abs(ELLS - 20)))]:+.3f} "
              f"r(100) {c['r'][int(np.argmin(np.abs(ELLS - 100)))]:+.3f} chi/d {c['ell_scale']:.0f}")


if __name__ == '__main__':
    main()
