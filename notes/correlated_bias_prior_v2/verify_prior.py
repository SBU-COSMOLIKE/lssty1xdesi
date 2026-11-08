"""Numerical checks of the correlated galaxy-bias prior.

This script verifies, one by one, the identities derived in the companion
note ``correlated_bias_prior_v2.tex``: the two-bin inverse covariance and
projected precision, the three-bin inverse with its exactly zero
(1, 3) entry, the properties of the projected precision P for a
general exponential kernel on unevenly spaced
redshifts, the tridiagonal form of the inverse covariance, the
Pythagorean split of the prior penalty, the equivalence of P with the
Schur complement obtained by marginalizing a flat-prior group amplitude,
the variance of the jump between two bins (by Monte Carlo), the
decorrelation and variance of the neighbor-prediction leftover used by
the appendix's chain factorization (from the kernel matrix, no
sampling), and the numbers of the worked example in the note.  The
note's section on physical values of L and sigma_b adds three more
checks: the conserved-population bias law b(z) - 1 = (b_c - 1) D_c/D(z)
against a direct integration of the continuity equation, the
conserved-population deviation table, and the neighbor rms table.  The
memory-model section adds nine more: the projection P = Q - Q U
(U^T Q U)^-1 U^T Q with U = [1, z - zbar] annihilates U and has rank
n - 2; the Woodbury inverse approaches it as 1/W^2; the null-space
identity P = N (N^T C N)^-1 N^T; the three-bin curvature penalty with
closed-form variance 2 sigma_b^2 (1 - r)(3 - r); P = 0 for one and two
bins; the universal-kriging predictor against a direct Gaussian
conditional; the end-to-end correlation tending to +-1 with a shared
slope; and the straight-line fits of the note's conserved-population
and BGS flux-limit drifts, plus the LRG3 magnitude-limit rate.

Run with the cocoa environment:

    source /Users/vivianmiranda/miniforge/etc/profile.d/conda.sh && \
    conda activate cocoapy311 && python verify_prior.py

Each check prints exactly one PASS or FAIL line with the measured
deviation, so the output documents how tightly every identity holds.
"""

import numpy as np

# Tolerance for identities that hold to floating-point accuracy. The
# matrices here are small (n <= 10) and well conditioned, so machine
# precision times a modest safety factor is the right scale.
ALGEBRA_TOLERANCE = 1.0e-10

# Tolerance for the Monte Carlo variance check. With 10^5 draws the
# relative scatter of a sample variance is about sqrt(2/10^5) = 0.45%,
# so a 3% band is a comfortable five-sigma margin.
MONTE_CARLO_RELATIVE_TOLERANCE = 0.03

NUMBER_OF_MONTE_CARLO_DRAWS = 100000

RANDOM_SEED = 20261106


def report(label, passed, deviation_text):
    """Print one PASS or FAIL line for a named check."""
    if passed:
        status = "PASS"
    else:
        status = "FAIL"
    print(f"{status}: {label} ({deviation_text})")
    return passed


def exponential_covariance(redshifts, spread, correlation_length):
    """Prior covariance C_ij = spread^2 exp(-|z_i - z_j|/L).

    ``redshifts`` is the 1D array of bin redshifts, ``spread`` is the
    per-bin prior standard deviation s = sigma_b * B_g, and
    ``correlation_length`` is L. Returns an (n, n) array.
    """
    separation = np.abs(redshifts[:, None] - redshifts[None, :])
    return spread**2 * np.exp(-separation / correlation_length)


def projected_precision(covariance):
    """Projection P = C^-1 - C^-1 1 1^T C^-1 / (1^T C^-1 1).

    This is the prior precision after removing (marginalizing with a
    flat prior) the uniform-shift direction along the ones vector.
    """
    inverse_covariance = np.linalg.inv(covariance)
    number_of_bins = covariance.shape[0]
    ones_vector = np.ones(number_of_bins)

    weighted_ones = inverse_covariance @ ones_vector
    uniform_norm = ones_vector @ weighted_ones

    return inverse_covariance - np.outer(weighted_ones, weighted_ones) / uniform_norm


def tridiagonal_precision_uneven(redshifts, spread, correlation_length):
    """Analytic inverse of the exponential kernel on uneven redshifts.

    The chain factorization of the exponential kernel gives, with
    r_k = exp(-(z_{k+1} - z_k)/L) for each adjacent pair, the
    tridiagonal precision (in units of 1/spread^2):

        Q_11 = 1/(1 - r_1^2),     Q_nn = 1/(1 - r_{n-1}^2),
        Q_kk = 1/(1 - r_{k-1}^2) + r_k^2/(1 - r_k^2)   (interior k),
        Q_{k,k+1} = Q_{k+1,k} = -r_k/(1 - r_k^2).
    """
    number_of_bins = len(redshifts)
    neighbor_correlation = np.exp(-np.diff(redshifts) / correlation_length)

    precision = np.zeros((number_of_bins, number_of_bins))
    precision[0, 0] = 1.0 / (1.0 - neighbor_correlation[0]**2)
    precision[-1, -1] = 1.0 / (1.0 - neighbor_correlation[-1]**2)

    for k in range(1, number_of_bins - 1):
        left_term = 1.0 / (1.0 - neighbor_correlation[k - 1]**2)
        right_term = neighbor_correlation[k]**2 / (1.0 - neighbor_correlation[k]**2)
        precision[k, k] = left_term + right_term

    for k in range(number_of_bins - 1):
        off_diagonal = -neighbor_correlation[k] / (1.0 - neighbor_correlation[k]**2)
        precision[k, k + 1] = off_diagonal
        precision[k + 1, k] = off_diagonal

    return precision / spread**2


def check_two_bin_formulas():
    """Two-bin closed forms for C^-1 and the projected precision P."""
    spread = 0.3
    correlation = 0.6

    covariance = spread**2 * np.array([[1.0, correlation],
                                       [correlation, 1.0]])

    # Closed form of the inverse: 1/(s^2 (1 - r^2)) [[1, -r], [-r, 1]].
    inverse_closed_form = np.array([[1.0, -correlation],
                                    [-correlation, 1.0]])
    inverse_closed_form /= spread**2 * (1.0 - correlation**2)
    inverse_numeric = np.linalg.inv(covariance)
    inverse_deviation = np.max(np.abs(inverse_closed_form - inverse_numeric))

    # Closed form of the projection: 1/(2 s^2 (1 - r)) [[1, -1], [-1, 1]].
    projection_closed_form = np.array([[1.0, -1.0],
                                       [-1.0, 1.0]])
    projection_closed_form /= 2.0 * spread**2 * (1.0 - correlation)
    projection_numeric = projected_precision(covariance)
    projection_deviation = np.max(np.abs(projection_closed_form - projection_numeric))

    deviation = max(inverse_deviation, projection_deviation)
    return report("two-bin C^-1 and P closed forms",
                  deviation < ALGEBRA_TOLERANCE,
                  f"max deviation {deviation:.3e}")


def check_three_bin_inverse():
    """Three-bin closed-form inverse and its zero (1, 3) entry.

    For three evenly spaced bins with neighbor correlation r the
    covariance is s^2 [[1, r, r^2], [r, 1, r], [r^2, r, 1]] and the
    note's closed-form inverse is

        1/(s^2 (1 - r^2)) [[1, -r, 0], [-r, 1 + r^2, -r], [0, -r, 1]].

    Bins 1 and 3 are correlated in the covariance (the r^2 entry), yet
    their entry in the inverse is exactly zero: once the middle bin is
    known, the outer bins carry no direct link.
    """
    spread = 0.3
    correlation = 0.67

    covariance = spread**2 * np.array([
        [1.0, correlation, correlation**2],
        [correlation, 1.0, correlation],
        [correlation**2, correlation, 1.0],
    ])

    inverse_closed_form = np.array([
        [1.0, -correlation, 0.0],
        [-correlation, 1.0 + correlation**2, -correlation],
        [0.0, -correlation, 1.0],
    ])
    inverse_closed_form /= spread**2 * (1.0 - correlation**2)

    inverse_numeric = np.linalg.inv(covariance)
    formula_deviation = np.max(np.abs(inverse_closed_form - inverse_numeric))

    # The corner entry of the numeric inverse, relative to the largest
    # entry, measures how exactly the (1, 3) direct link vanishes.
    corner_entry = abs(inverse_numeric[0, 2]) / np.max(np.abs(inverse_numeric))

    passed = (formula_deviation < ALGEBRA_TOLERANCE
              and corner_entry < ALGEBRA_TOLERANCE)
    detail = (f"max deviation {formula_deviation:.3e}, "
              f"relative (1,3) entry {corner_entry:.3e}")
    return report("three-bin C^-1 closed form and zero (1, 3) entry",
                  passed, detail)


def check_projection_properties(redshifts, spread, correlation_length):
    """P annihilates the ones vector, is PSD, and has rank n - 1."""
    covariance = exponential_covariance(redshifts, spread, correlation_length)
    projection = projected_precision(covariance)
    number_of_bins = len(redshifts)
    ones_vector = np.ones(number_of_bins)

    null_residual = np.max(np.abs(projection @ ones_vector))

    eigenvalues = np.linalg.eigvalsh(projection)
    scale = np.max(eigenvalues)
    smallest_eigenvalue = eigenvalues[0]

    # Rank n - 1 means exactly one eigenvalue at zero (the ones
    # direction) and all others strictly positive.
    zero_count = int(np.sum(np.abs(eigenvalues) < ALGEBRA_TOLERANCE * scale))

    passed = (null_residual < ALGEBRA_TOLERANCE * scale
              and smallest_eigenvalue > -ALGEBRA_TOLERANCE * scale
              and zero_count == 1)
    detail = (f"|P 1|_max/|P| {null_residual/scale:.3e}, "
              f"min eig/max eig {smallest_eigenvalue/scale:.3e}, "
              f"zero eigenvalues {zero_count}")
    return report("P 1 = 0, positive semidefinite, rank n - 1", passed, detail)


def check_tridiagonal_inverse_even():
    """AR(1) tridiagonal precision inverts the evenly spaced kernel."""
    number_of_bins = 8
    spread = 0.25
    correlation_length = 0.05
    bin_width = 0.02

    redshifts = 0.2 + bin_width * np.arange(number_of_bins)
    covariance = exponential_covariance(redshifts, spread, correlation_length)

    neighbor_correlation = np.exp(-bin_width / correlation_length)

    precision = np.zeros((number_of_bins, number_of_bins))
    precision[0, 0] = 1.0
    precision[-1, -1] = 1.0
    for k in range(1, number_of_bins - 1):
        precision[k, k] = 1.0 + neighbor_correlation**2
    for k in range(number_of_bins - 1):
        precision[k, k + 1] = -neighbor_correlation
        precision[k + 1, k] = -neighbor_correlation
    precision /= spread**2 * (1.0 - neighbor_correlation**2)

    identity_residual = np.max(np.abs(precision @ covariance - np.eye(number_of_bins)))
    return report("tridiagonal inverse, evenly spaced (AR(1))",
                  identity_residual < ALGEBRA_TOLERANCE,
                  f"max |Q C - I| {identity_residual:.3e}")


def check_tridiagonal_inverse_uneven(redshifts, spread, correlation_length):
    """Uneven-spacing tridiagonal precision inverts the kernel."""
    covariance = exponential_covariance(redshifts, spread, correlation_length)
    precision = tridiagonal_precision_uneven(redshifts, spread, correlation_length)

    number_of_bins = len(redshifts)
    identity_residual = np.max(np.abs(precision @ covariance - np.eye(number_of_bins)))
    return report("tridiagonal inverse, uneven redshifts",
                  identity_residual < ALGEBRA_TOLERANCE,
                  f"max |Q C - I| {identity_residual:.3e}")


def check_pythagorean_split(redshifts, spread, correlation_length, rng):
    """Penalty of the residual: dbeta^T P dbeta = w^T C^-1 w."""
    covariance = exponential_covariance(redshifts, spread, correlation_length)
    inverse_covariance = np.linalg.inv(covariance)
    projection = projected_precision(covariance)

    number_of_bins = len(redshifts)
    ones_vector = np.ones(number_of_bins)
    uniform_norm = ones_vector @ inverse_covariance @ ones_vector

    worst_deviation = 0.0
    for trial in range(20):
        delta_beta = rng.normal(0.0, 1.0, number_of_bins)

        # Generalized-least-squares amplitude of the uniform component,
        # then the residual w, orthogonal to 1 in the C^-1 metric.
        shift_amplitude = (ones_vector @ inverse_covariance @ delta_beta) / uniform_norm
        residual = delta_beta - shift_amplitude * ones_vector

        penalty_from_projection = delta_beta @ projection @ delta_beta
        penalty_from_residual = residual @ inverse_covariance @ residual

        deviation = abs(penalty_from_projection - penalty_from_residual)
        deviation /= max(abs(penalty_from_residual), 1.0)
        worst_deviation = max(worst_deviation, deviation)

    return report("Pythagorean split: dbeta^T P dbeta = w^T C^-1 w",
                  worst_deviation < ALGEBRA_TOLERANCE,
                  f"worst relative deviation {worst_deviation:.3e} over 20 draws")


def check_schur_complement(redshifts, spread, correlation_length):
    """P equals the Schur complement of the (beta, B) joint precision.

    With beta = B 1 + epsilon, epsilon ~ N(0, C) and a flat prior on B,
    the joint quadratic form in (beta, B) has the precision blocks

        [[ C^-1,        -C^-1 1      ],
         [ -1^T C^-1,   1^T C^-1 1   ]].

    Marginalizing B takes the Schur complement over the B block, which
    must reproduce the projection P.
    """
    covariance = exponential_covariance(redshifts, spread, correlation_length)
    inverse_covariance = np.linalg.inv(covariance)
    projection = projected_precision(covariance)

    number_of_bins = len(redshifts)
    ones_vector = np.ones(number_of_bins)

    joint_precision = np.zeros((number_of_bins + 1, number_of_bins + 1))
    joint_precision[:number_of_bins, :number_of_bins] = inverse_covariance
    joint_precision[:number_of_bins, number_of_bins] = -inverse_covariance @ ones_vector
    joint_precision[number_of_bins, :number_of_bins] = -ones_vector @ inverse_covariance
    joint_precision[number_of_bins, number_of_bins] = (
        ones_vector @ inverse_covariance @ ones_vector)

    beta_block = joint_precision[:number_of_bins, :number_of_bins]
    cross_block = joint_precision[:number_of_bins, number_of_bins]
    amplitude_block = joint_precision[number_of_bins, number_of_bins]

    schur_complement = beta_block - np.outer(cross_block, cross_block) / amplitude_block

    deviation = np.max(np.abs(schur_complement - projection))
    return report("P equals the Schur complement over the flat-prior B",
                  deviation < ALGEBRA_TOLERANCE,
                  f"max deviation {deviation:.3e}")


def check_conditional_leftover(redshifts, sigma_b, correlation_length):
    """The leftover e = delta_{k+1} - r_k delta_k decorrelates the past.

    The appendix writes each bin as its neighbor's prediction plus a
    leftover.  Two facts make the chain factorization work, and both
    are plain covariance algebra on the kernel matrix (no sampling):

        Cov(e, delta_j) = 0 for every j <= k   (the past adds nothing),
        Var(e) = sigma_b^2 (1 - r_k^2)         (the new scatter).
    """
    kernel = exponential_covariance(redshifts, sigma_b, correlation_length)
    neighbor_correlation = np.exp(-np.diff(redshifts) / correlation_length)
    number_of_bins = len(redshifts)

    worst_covariance = 0.0
    worst_variance_deviation = 0.0
    for k in range(number_of_bins - 1):
        # Covariance of the leftover with every bin up to and including
        # bin k, read straight from the kernel rows of bins k+1 and k.
        for j in range(k + 1):
            leftover_covariance = (kernel[k + 1, j]
                                   - neighbor_correlation[k] * kernel[k, j])
            worst_covariance = max(worst_covariance,
                                   abs(leftover_covariance) / sigma_b**2)

        leftover_variance = (kernel[k + 1, k + 1]
                             - 2.0 * neighbor_correlation[k] * kernel[k + 1, k]
                             + neighbor_correlation[k]**2 * kernel[k, k])
        expected_variance = sigma_b**2 * (1.0 - neighbor_correlation[k]**2)
        variance_deviation = abs(leftover_variance - expected_variance)
        worst_variance_deviation = max(worst_variance_deviation,
                                       variance_deviation / sigma_b**2)

    passed = (worst_covariance < ALGEBRA_TOLERANCE
              and worst_variance_deviation < ALGEBRA_TOLERANCE)
    detail = (f"worst |Cov(e, delta_j)|/sigma_b^2 {worst_covariance:.3e}, "
              f"worst variance deviation {worst_variance_deviation:.3e}")
    return report("leftover e = delta_(k+1) - r_k delta_k: Cov(e, past) = 0 "
                  "and Var(e) = sigma_b^2 (1 - r_k^2)", passed, detail)


def check_jump_variance_monte_carlo(rng):
    """Var(delta_i - delta_j) = 2 sigma_b^2 (1 - exp(-dz/L)) by sampling."""
    sigma_b = 0.10
    correlation_length = 0.05
    redshifts = np.array([0.30, 0.32, 0.37, 0.45])

    covariance = exponential_covariance(redshifts, sigma_b, correlation_length)
    draws = rng.multivariate_normal(np.zeros(len(redshifts)), covariance,
                                    size=NUMBER_OF_MONTE_CARLO_DRAWS)

    worst_relative_deviation = 0.0
    for i in range(len(redshifts)):
        for j in range(i + 1, len(redshifts)):
            separation = redshifts[j] - redshifts[i]
            expected = 2.0 * sigma_b**2 * (1.0 - np.exp(-separation / correlation_length))
            measured = np.var(draws[:, i] - draws[:, j])
            relative_deviation = abs(measured - expected) / expected
            worst_relative_deviation = max(worst_relative_deviation,
                                           relative_deviation)

    return report("jump variance 2 sigma_b^2 (1 - exp(-dz/L)), Monte Carlo",
                  worst_relative_deviation < MONTE_CARLO_RELATIVE_TOLERANCE,
                  f"worst relative deviation {worst_relative_deviation:.3e} "
                  f"over {NUMBER_OF_MONTE_CARLO_DRAWS} draws")


def check_worked_example():
    """The note's numbers: r = 0.670, exact jump 8.1%, Taylor 8.9%."""
    bin_separation = 0.02
    correlation_length = 0.05
    sigma_b = 0.10

    neighbor_correlation = np.exp(-bin_separation / correlation_length)
    exact_jump = sigma_b * np.sqrt(2.0 * (1.0 - neighbor_correlation))
    taylor_jump = sigma_b * np.sqrt(2.0 * bin_separation / correlation_length)

    correlation_ok = round(neighbor_correlation, 3) == 0.670
    exact_ok = round(100.0 * exact_jump, 1) == 8.1
    taylor_ok = round(100.0 * taylor_jump, 1) == 8.9
    small_sigma_ok = round(100.0 * exact_jump / 10.0, 2) == 0.81

    passed = correlation_ok and exact_ok and taylor_ok and small_sigma_ok
    detail = (f"r {neighbor_correlation:.4f}, exact jump "
              f"{100.0*exact_jump:.2f}%, Taylor {100.0*taylor_jump:.2f}%")
    return report("worked example: r = 0.670, jump 8.1% (Taylor 8.9%)",
                  passed, detail)


def growth_solution(omega_matter):
    """Linear growth factor D of matter fluctuations in flat LCDM.

    Solves the growth equation in x = ln a with a fourth-order
    Runge-Kutta integrator,

        D'' + (2 + dlnH/dx) D' - (3/2) Omega_m(a) D = 0,

    starting at a = 10^-3 with the matter-era solution D = a (so
    D' = D), and normalizing D(a = 1) = 1.  Returns the ln a grid, D on
    that grid, and dD/d ln a in the same normalization.
    """
    number_of_steps = 20000
    ln_a_grid = np.linspace(np.log(1.0e-3), 0.0, number_of_steps + 1)
    step = ln_a_grid[1] - ln_a_grid[0]

    def growth_derivatives(ln_a, state):
        scale_factor = np.exp(ln_a)
        hubble_sq = omega_matter*scale_factor**-3 + (1.0 - omega_matter)
        omega_m_of_a = omega_matter*scale_factor**-3/hubble_sq
        dlnH_dlna = -1.5*omega_matter*scale_factor**-3/hubble_sq
        growth, growth_rate = state
        acceleration = (-(2.0 + dlnH_dlna)*growth_rate
                        + 1.5*omega_m_of_a*growth)
        return np.array([growth_rate, acceleration])

    state = np.array([np.exp(ln_a_grid[0]), np.exp(ln_a_grid[0])])
    states = [state.copy()]
    for index in range(number_of_steps):
        ln_a = ln_a_grid[index]
        slope_1 = growth_derivatives(ln_a, state)
        slope_2 = growth_derivatives(ln_a + step/2.0,
                                     state + step/2.0*slope_1)
        slope_3 = growth_derivatives(ln_a + step/2.0,
                                     state + step/2.0*slope_2)
        slope_4 = growth_derivatives(ln_a + step, state + step*slope_3)
        state = state + step/6.0*(slope_1 + 2.0*slope_2
                                  + 2.0*slope_3 + slope_4)
        states.append(state.copy())

    states = np.array(states)
    normalization = states[-1, 0]
    return ln_a_grid, states[:, 0]/normalization, states[:, 1]/normalization


def growth_at_redshift(redshifts, ln_a_grid, growth_grid):
    """Interpolate the normalized growth factor at the given redshifts."""
    ln_a_requested = -np.log1p(np.atleast_1d(redshifts))
    return np.interp(ln_a_requested, ln_a_grid, growth_grid)


def check_bias_continuity():
    """b(z) - 1 = (b_c - 1) D_c/D(z) from integrating the continuity law.

    For a conserved population, d delta_g/d ln a = d delta_m/d ln a.
    This check integrates that statement numerically: delta_m = D/D_c
    (normalized to one at the reference a_c), delta_g starts at b_c at
    a_c, and the increment of delta_g is the cumulative trapezoid of
    d delta_m/d ln a.  The resulting b = delta_g/delta_m must match the
    closed form at every redshift.
    """
    omega_matter = 0.3
    reference_redshift = 0.5
    reference_bias = 2.2

    ln_a_grid, growth_grid, growth_rate_grid = growth_solution(omega_matter)
    growth_reference = growth_at_redshift(reference_redshift,
                                          ln_a_grid, growth_grid)[0]

    # Cumulative trapezoid of d delta_m/d ln a along the grid (a
    # vectorized prefix sum; each entry adds one trapezoid panel).
    step = ln_a_grid[1] - ln_a_grid[0]
    panel_areas = 0.5*step*(growth_rate_grid[1:] + growth_rate_grid[:-1])
    integral_grid = np.concatenate([[0.0], np.cumsum(panel_areas)])

    ln_a_reference = -np.log1p(reference_redshift)
    integral_reference = np.interp(ln_a_reference, ln_a_grid, integral_grid)

    delta_matter = growth_grid/growth_reference
    delta_galaxy = (reference_bias
                    + (integral_grid - integral_reference)/growth_reference)

    worst_deviation = 0.0
    for redshift in [0.0, 0.1, 0.3, 0.7, 1.0]:
        ln_a_requested = -np.log1p(redshift)
        bias_integrated = (np.interp(ln_a_requested, ln_a_grid, delta_galaxy)
                           / np.interp(ln_a_requested, ln_a_grid,
                                       delta_matter))
        growth_here = growth_at_redshift(redshift, ln_a_grid, growth_grid)[0]
        bias_closed_form = (1.0 + (reference_bias - 1.0)
                            * growth_reference/growth_here)
        deviation = abs(bias_integrated - bias_closed_form)/bias_closed_form
        worst_deviation = max(worst_deviation, deviation)

    return report("continuity equation: b(z) - 1 = (b_c - 1) D_c/D(z)",
                  worst_deviation < 1.0e-6,
                  f"worst relative deviation {worst_deviation:.3e}")


def check_conserved_deviation_table():
    """The note's conserved-population table, recomputed from scratch.

    delta(z) = (D(z)/D(z_c) - 1)/b_c with z_c at the group midpoint and
    b_c = B/D(z_c), for the three groups quoted in the note.  The table
    rounds to one decimal in percent, so the comparison allows 0.05
    percentage points, half a rounding unit (the same margin on b_c).
    """
    omega_matter = 0.3
    ln_a_grid, growth_grid, _ = growth_solution(omega_matter)

    # Group name, amplitude B, z range, midpoint, then the quoted
    # numbers: b_c, delta at the low and high ends (percent), and the
    # change across one Delta z = 0.05 step at the midpoint (percent).
    groups = [
        ("BGS",  1.34, 0.10, 0.40, 0.25, 1.5, +5.3, -4.9, 1.7),
        ("LRG1", 1.70, 0.40, 0.60, 0.50, 2.2, +2.4, -2.2, 1.1),
        ("LRG3", 1.70, 0.80, 1.10, 0.95, 2.7, +2.6, -2.3, 0.8),
    ]
    tolerance = 0.05

    worst_deviation = 0.0
    for (name, amplitude, z_low, z_high, z_mid,
         quoted_bias, quoted_low, quoted_high, quoted_step) in groups:
        growth_mid = growth_at_redshift(z_mid, ln_a_grid, growth_grid)[0]
        bias_mid = amplitude/growth_mid

        growth_low = growth_at_redshift(z_low, ln_a_grid, growth_grid)[0]
        growth_high = growth_at_redshift(z_high, ln_a_grid, growth_grid)[0]
        delta_low = 100.0*(growth_low/growth_mid - 1.0)/bias_mid
        delta_high = 100.0*(growth_high/growth_mid - 1.0)/bias_mid

        growth_minus = growth_at_redshift(z_mid - 0.025,
                                          ln_a_grid, growth_grid)[0]
        growth_plus = growth_at_redshift(z_mid + 0.025,
                                         ln_a_grid, growth_grid)[0]
        step_percent = 100.0*abs(growth_minus - growth_plus)/growth_mid/bias_mid

        for computed, quoted in [(bias_mid, quoted_bias),
                                 (delta_low, quoted_low),
                                 (delta_high, quoted_high),
                                 (step_percent, quoted_step)]:
            worst_deviation = max(worst_deviation, abs(computed - quoted))

    return report("conserved-population deviation table",
                  worst_deviation < tolerance,
                  f"worst |computed - quoted| {worst_deviation:.3f}")


def check_neighbor_rms_table():
    """The note's neighbor rms table, sigma_b sqrt(2(1 - exp(-dz/L))).

    Rows: the baseline (3%, L = 0.05), the plausible-region corners
    (1%, 0.1) and (5%, 0.02), and the two larger spreads (10%, 0.05)
    and (20%, 0.02).  Quoted values round to one decimal in percent,
    so the comparison allows 0.06 percentage points.
    """
    # (sigma_b, L, quoted percent over dz = 0.02, over dz = 0.05)
    rows = [
        (0.03, 0.05, 2.4, 3.4),
        (0.01, 0.10, 0.6, 0.9),
        (0.05, 0.02, 5.6, 6.8),
        (0.10, 0.05, 8.1, 11.2),
        (0.20, 0.02, 22.5, 27.1),
    ]
    tolerance = 0.06

    worst_deviation = 0.0
    for sigma_b, correlation_length, quoted_002, quoted_005 in rows:
        for separation, quoted in [(0.02, quoted_002), (0.05, quoted_005)]:
            rms_change = 100.0*sigma_b*np.sqrt(
                2.0*(1.0 - np.exp(-separation/correlation_length)))
            worst_deviation = max(worst_deviation, abs(rms_change - quoted))

    return report("neighbor rms table",
                  worst_deviation < tolerance,
                  f"worst |computed - quoted| {worst_deviation:.3f}")


def memory_design(redshifts):
    """The memory model's design matrix U = [1, v], v = z - mean(z).

    Column one is the ones vector (the free amplitude direction) and
    column two the centered redshifts (the free tilt direction).
    Returns an (n, 2) array.
    """
    number_of_bins = len(redshifts)
    centered = redshifts - np.mean(redshifts)
    design = np.column_stack([np.ones(number_of_bins), centered])
    return design


def memory_projection(covariance, design):
    """The memory projection P = Q - Q U (U^T Q U)^-1 U^T Q."""
    precision = np.linalg.inv(covariance)
    weighted_design = precision @ design
    normal_matrix = design.T @ weighted_design
    return precision - weighted_design @ np.linalg.inv(normal_matrix) @ weighted_design.T


def design_nullspace(design):
    """Columns spanning the vectors Euclidean-orthogonal to U's columns.

    Computed from the full singular value decomposition of U: the left
    singular vectors beyond U's rank span the orthogonal complement.
    Returns an (n, n - rank) array (empty for a single bin).
    """
    left_vectors, singular_values, _ = np.linalg.svd(design)
    tolerance = np.max(design.shape) * np.finfo(float).eps * np.max(singular_values)
    rank = int(np.sum(singular_values > tolerance))
    return left_vectors[:, rank:]


def luminosity_distance_mpc_over_h(redshifts, omega_matter):
    """Luminosity distance in Mpc/h for flat LCDM (trapezoid integral)."""
    redshift_grid = np.linspace(0.0, 2.0, 200001)
    inverse_e = 1.0/np.sqrt(omega_matter*(1.0 + redshift_grid)**3
                            + 1.0 - omega_matter)
    hubble_distance_mpc_over_h = 2997.92458
    panel_areas = 0.5*(inverse_e[1:] + inverse_e[:-1])*np.diff(redshift_grid)
    comoving_grid = hubble_distance_mpc_over_h*np.concatenate(
        [[0.0], np.cumsum(panel_areas)])
    comoving = np.interp(redshifts, redshift_grid, comoving_grid)
    return (1.0 + np.asarray(redshifts))*comoving


def zehavi_threshold_bias(absolute_magnitude):
    """The threshold-bias fit b = 1.06 + 0.21 (L/L*)^1.12, M* = -20.44."""
    luminosity_ratio = 10.0**(-0.4*(absolute_magnitude + 20.44))
    return 1.06 + 0.21*luminosity_ratio**1.12


def check_memory_projection_properties(redshifts, spread, correlation_length):
    """P U = 0, positive semidefinite, rank n - 2 on uneven redshifts."""
    covariance = exponential_covariance(redshifts, spread, correlation_length)
    design = memory_design(redshifts)
    projection = memory_projection(covariance, design)

    null_residual = np.max(np.abs(projection @ design))
    eigenvalues = np.linalg.eigvalsh(projection)
    scale = np.max(eigenvalues)
    smallest = eigenvalues[0]
    zero_count = int(np.sum(np.abs(eigenvalues) < ALGEBRA_TOLERANCE*scale))

    passed = (null_residual < ALGEBRA_TOLERANCE*scale
              and smallest > -ALGEBRA_TOLERANCE*scale
              and zero_count == 2)
    detail = (f"|P U|_max/|P| {null_residual/scale:.3e}, "
              f"min eig/max eig {smallest/scale:.3e}, "
              f"zero eigenvalues {zero_count}")
    return report("memory projection: P U = 0, PSD, rank n - 2", passed, detail)


def check_memory_woodbury_limit(redshifts, spread, correlation_length):
    """The Woodbury inverse approaches the projection as 1/W^2."""
    covariance = exponential_covariance(redshifts, spread, correlation_length)
    design = memory_design(redshifts)
    projection = memory_projection(covariance, design)

    deviations = []
    for prior_width in [1.0, 10.0, 100.0, 1000.0]:
        widened = covariance + prior_width**2 * (design @ design.T)
        deviations.append(np.max(np.abs(np.linalg.inv(widened) - projection)))

    decreasing = (deviations[0] > deviations[1] > deviations[2] > deviations[3])
    final_ratio = deviations[2]/deviations[3]
    passed = decreasing and 80.0 < final_ratio < 125.0
    detail = ("max deviations " +
              ", ".join(f"{d:.3e}" for d in deviations) +
              f" at W = 1, 10, 100, 1000; last ratio {final_ratio:.1f}")
    return report("memory projection: Woodbury inverse approaches P as 1/W^2",
                  passed, detail)


def check_memory_nullspace_identity(redshifts, spread, correlation_length):
    """P equals N (N^T C N)^-1 N^T with N spanning the non-line space."""
    covariance = exponential_covariance(redshifts, spread, correlation_length)
    design = memory_design(redshifts)
    projection = memory_projection(covariance, design)

    nullspace = design_nullspace(design)
    middle = np.linalg.inv(nullspace.T @ covariance @ nullspace)
    alternative = nullspace @ middle @ nullspace.T

    deviation = np.max(np.abs(projection - alternative))/np.max(np.abs(projection))
    return report("memory projection: null-space identity",
                  deviation < ALGEBRA_TOLERANCE,
                  f"max relative deviation {deviation:.3e}")


def check_memory_three_bin_curvature():
    """Three even bins: the penalty is the curvature over its variance."""
    spacing = 0.02
    correlation_length = 0.05
    sigma_b = 0.10
    redshifts = np.array([0.30, 0.32, 0.34])
    neighbor_correlation = np.exp(-spacing/correlation_length)

    covariance = exponential_covariance(redshifts, sigma_b, correlation_length)
    design = memory_design(redshifts)
    projection = memory_projection(covariance, design)

    curvature_vector = np.array([1.0, -2.0, 1.0])
    curvature_variance = curvature_vector @ covariance @ curvature_vector
    closed_form = 2.0*sigma_b**2*(1.0 - neighbor_correlation)*(3.0 - neighbor_correlation)
    variance_deviation = abs(curvature_variance - closed_form)/closed_form

    rng = np.random.default_rng(RANDOM_SEED + 1)
    worst_penalty_deviation = 0.0
    for trial in range(10):
        delta = rng.normal(0.0, 1.0, 3)
        penalty = delta @ projection @ delta
        curvature = curvature_vector @ delta
        penalty_closed = curvature**2/closed_form
        deviation = abs(penalty - penalty_closed)/max(penalty_closed, 1.0e-6)
        worst_penalty_deviation = max(worst_penalty_deviation, deviation)

    passed = (variance_deviation < ALGEBRA_TOLERANCE
              and worst_penalty_deviation < ALGEBRA_TOLERANCE)
    detail = (f"variance vs 2 sigma_b^2 (1-r)(3-r): {variance_deviation:.3e}, "
              f"penalty vs curvature form: {worst_penalty_deviation:.3e}")
    return report("memory projection: three-bin curvature closed form",
                  passed, detail)


def check_memory_small_groups(spread, correlation_length):
    """One and two bins are fully free: the memory projection vanishes."""
    # One bin: nothing is orthogonal to both columns, so N is empty and
    # the null-space form gives the zero matrix.
    single = np.array([0.5])
    covariance_one = exponential_covariance(single, spread, correlation_length)
    nullspace_one = design_nullspace(memory_design(single))
    if nullspace_one.shape[1] == 0:
        projection_one = np.zeros((1, 1))
    else:
        middle = np.linalg.inv(nullspace_one.T @ covariance_one @ nullspace_one)
        projection_one = nullspace_one @ middle @ nullspace_one.T

    # Two bins: U is square and invertible, and the formula collapses.
    pair = np.array([0.40, 0.46])
    covariance_two = exponential_covariance(pair, spread, correlation_length)
    projection_two = memory_projection(covariance_two, memory_design(pair))
    precision_scale = np.max(np.abs(np.linalg.inv(covariance_two)))

    one_bin_size = np.max(np.abs(projection_one))
    two_bin_size = np.max(np.abs(projection_two))/precision_scale
    passed = one_bin_size == 0.0 and two_bin_size < ALGEBRA_TOLERANCE
    detail = (f"one bin |P| {one_bin_size:.1e}, "
              f"two bins |P|/|Q| {two_bin_size:.3e}")
    return report("memory projection: P = 0 for one and two bins",
                  passed, detail)


def check_memory_kriging(redshifts, sigma_b, correlation_length, rng):
    """The universal-kriging predictor against a wide-prior conditional."""
    design = memory_design(redshifts)
    centered = design[:, 1]
    prior_width = 1.0e4

    worst_deviation = 0.0
    for known_count in [2, 3, 4]:
        known_block = redshifts[:known_count + 1]
        covariance = exponential_covariance(known_block, sigma_b,
                                            correlation_length)
        past_covariance = covariance[:known_count, :known_count]
        past_design = design[:known_count, :]
        past_precision = np.linalg.inv(past_covariance)

        for trial in range(5):
            past_values = rng.normal(0.0, sigma_b, known_count)

            # The formula: the generalized-least-squares line through
            # the past, then one Markov step from the last residual.
            normal_matrix = past_design.T @ past_precision @ past_design
            line_fit = np.linalg.solve(normal_matrix,
                                       past_design.T @ past_precision @ past_values)
            step_correlation = np.exp(-(redshifts[known_count]
                                        - redshifts[known_count - 1])
                                      / correlation_length)
            last_residual = (past_values[-1] - line_fit[0]
                             - line_fit[1]*centered[known_count - 1])
            predicted = (line_fit[0] + line_fit[1]*centered[known_count]
                         + step_correlation*last_residual)

            # The direct conditional mean of the wide-prior Gaussian.
            widened = (covariance
                       + prior_width**2 * (design[:known_count + 1, :]
                                           @ design[:known_count + 1, :].T))
            conditional = (widened[known_count, :known_count]
                           @ np.linalg.solve(widened[:known_count, :known_count],
                                             past_values))
            worst_deviation = max(worst_deviation,
                                  abs(predicted - conditional))

    return report("memory model: universal-kriging predictor",
                  worst_deviation < 1.0e-6,
                  f"worst |formula - conditional| {worst_deviation:.3e}")


def check_memory_end_correlation(redshifts, sigma_b, correlation_length):
    """A shared slope correlates the group ends toward +-1."""
    centered = redshifts - np.mean(redshifts)
    first, last = 0, len(redshifts) - 1
    separation = redshifts[last] - redshifts[first]
    kernel_cross = sigma_b**2*np.exp(-separation/correlation_length)

    def full_correlation(slope_width):
        covariance_ij = (slope_width**2*centered[first]*centered[last]
                         + kernel_cross)
        variance_i = slope_width**2*centered[first]**2 + sigma_b**2
        variance_j = slope_width**2*centered[last]**2 + sigma_b**2
        return covariance_ij/np.sqrt(variance_i*variance_j)

    def asymptotic_correlation(slope_width):
        covariance_ij = slope_width**2*centered[first]*centered[last]
        variance_i = slope_width**2*centered[first]**2 + sigma_b**2
        variance_j = slope_width**2*centered[last]**2 + sigma_b**2
        return covariance_ij/np.sqrt(variance_i*variance_j)

    moderate = 5.0*sigma_b
    formula_deviation = abs(full_correlation(moderate)
                            - asymptotic_correlation(moderate))
    formula_tolerance = 2.0*np.exp(-separation/correlation_length)

    strong = 1000.0*sigma_b
    limiting = full_correlation(strong)
    expected_sign = np.sign(centered[first]*centered[last])

    passed = (formula_deviation < formula_tolerance
              and np.sign(limiting) == expected_sign
              and abs(limiting) > 0.999)
    detail = (f"formula deviation {formula_deviation:.3e} "
              f"(tolerance {formula_tolerance:.3e}), "
              f"rho at large tau {limiting:+.4f}")
    return report("memory model: end-to-end correlation tends to +-1",
                  passed, detail)


def check_memory_line_removes_drift():
    """The note's table of straight-line fits to the conserved drift."""
    omega_matter = 0.3
    ln_a_grid, growth_grid, _ = growth_solution(omega_matter)

    # Group name, amplitude B, z range, then the quoted numbers: span
    # of delta, fitted slope per unit z, residual rms, residual max,
    # all in percent, at the Delta z = 0.05 bin centers.
    groups = [
        ("BGS",  1.34, 0.10, 0.40,  8.49, -34.0, 0.062, 0.084),
        ("LRG1", 1.70, 0.40, 0.60,  3.41, -22.7, 0.020, 0.020),
        ("LRG2", 1.70, 0.60, 0.80,  2.94, -19.6, 0.018, 0.019),
        ("LRG3", 1.70, 0.80, 1.10,  4.07, -16.3, 0.038, 0.052),
        ("ELG1", 0.84, 0.80, 1.10,  8.23, -32.9, 0.077, 0.104),
        ("ELG2", 0.84, 1.10, 1.60, 11.09, -24.6, 0.158, 0.270),
    ]

    worst_deviation = 0.0
    for (name, amplitude, z_low, z_high,
         quoted_span, quoted_slope, quoted_rms, quoted_max) in groups:
        centers = np.arange(z_low + 0.025, z_high, 0.05)
        z_mid = 0.5*(z_low + z_high)
        growth_mid = growth_at_redshift(z_mid, ln_a_grid, growth_grid)[0]
        bias_mid = amplitude/growth_mid

        growth_centers = growth_at_redshift(centers, ln_a_grid, growth_grid)
        deviation = (growth_centers/growth_mid - 1.0)/bias_mid

        line = np.polyfit(centers, deviation, 1)
        residual = deviation - np.polyval(line, centers)

        span_percent = 100.0*(deviation[0] - deviation[-1])
        slope_percent = 100.0*line[0]
        rms_percent = 100.0*np.sqrt(np.mean(residual**2))
        max_percent = 100.0*np.max(np.abs(residual))

        for computed, quoted, tol in [(span_percent, quoted_span, 0.01),
                                      (slope_percent, quoted_slope, 0.1),
                                      (rms_percent, quoted_rms, 0.001),
                                      (max_percent, quoted_max, 0.001)]:
            worst_deviation = max(worst_deviation, abs(computed - quoted)/tol)

    return report("memory model: a line removes the conserved drift (table)",
                  worst_deviation < 1.0,
                  f"worst deviation {worst_deviation:.2f} tolerance units")


def check_memory_bgs_flux_line():
    """The BGS flux-limit drift and its straight-line residual."""
    omega_matter = 0.3
    ln_a_grid, growth_grid, _ = growth_solution(omega_matter)

    centers = np.arange(0.125, 0.40, 0.05)
    distances = luminosity_distance_mpc_over_h(centers, omega_matter)
    magnitudes = 19.5 - 25.0 - 5.0*np.log10(distances)
    biases = zehavi_threshold_bias(magnitudes)

    # The deviation from the tied trend with the group amplitude set so
    # the mean deviation vanishes (the free B_g absorbs the mean).
    amplitude_track = biases*growth_at_redshift(centers, ln_a_grid, growth_grid)
    deviation = amplitude_track/np.mean(amplitude_track) - 1.0

    line = np.polyfit(centers, deviation, 1)
    residual = deviation - np.polyval(line, centers)

    quoted = [(magnitudes[0], -18.56, 0.008), (magnitudes[-1], -21.24, 0.008),
              (biases[0], 1.09, 0.008), (biases[-1], 1.54, 0.008),
              (100.0*(deviation[-1] - deviation[0]), 22.3, 0.08),
              (100.0*line[0], 89.0, 0.8),
              (100.0*np.sqrt(np.mean(residual**2)), 2.4, 0.08),
              (100.0*np.max(np.abs(residual)), 3.3, 0.08)]
    worst_deviation = 0.0
    for computed, expected, tol in quoted:
        worst_deviation = max(worst_deviation, abs(computed - expected)/tol)

    return report("memory model: BGS flux-limit drift and line residual",
                  worst_deviation < 1.0,
                  f"worst deviation {worst_deviation:.2f} tolerance units; "
                  f"M {magnitudes[0]:.2f} to {magnitudes[-1]:.2f}, "
                  f"b {biases[0]:.2f} to {biases[-1]:.2f}")


def check_lrg3_flux_drift():
    """The LRG3 magnitude-limit drift: 0.27 mag per dz = 0.1, 6-11% bias.

    Past z = 0.8 the LRG selection is a fixed flux limit, so the
    luminosity threshold rises with the distance modulus.  The note
    quotes the rate 0.27 magnitudes per Delta z = 0.1 at z = 1, and
    converts it with the Zehavi slope of 0.5-1.0 in bias per magnitude
    into a 6-11% drift of b = 2.4.  Both numbers are recomputed here.
    """
    omega_matter = 0.3
    lower, upper = 0.95, 1.05
    distances = luminosity_distance_mpc_over_h(np.array([lower, upper]),
                                               omega_matter)
    magnitude_step = 5.0*np.log10(distances[1]/distances[0])

    reference_bias = 2.4
    drift_low = 100.0*0.5*magnitude_step/reference_bias
    drift_high = 100.0*1.0*magnitude_step/reference_bias

    passed = (abs(magnitude_step - 0.27) < 0.01
              and abs(drift_low - 6.0) < 0.6
              and abs(drift_high - 11.0) < 0.6)
    detail = (f"magnitude step {magnitude_step:.3f} per dz = 0.1, "
              f"drift {drift_low:.1f} to {drift_high:.1f} percent")
    return report("memory model: LRG3 magnitude-limit drift rate",
                  passed, detail)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # A deliberately uneven redshift layout for the n-bin checks: thin
    # neighboring bins plus a few larger gaps, as in a real tracer group.
    uneven_redshifts = np.array([0.10, 0.13, 0.17, 0.25, 0.26, 0.40, 0.55])
    spread = 0.23
    correlation_length = 0.07

    results = []
    results.append(check_two_bin_formulas())
    results.append(check_three_bin_inverse())
    results.append(check_projection_properties(uneven_redshifts, spread,
                                               correlation_length))
    results.append(check_tridiagonal_inverse_even())
    results.append(check_tridiagonal_inverse_uneven(uneven_redshifts, spread,
                                                    correlation_length))
    results.append(check_pythagorean_split(uneven_redshifts, spread,
                                           correlation_length, rng))
    results.append(check_schur_complement(uneven_redshifts, spread,
                                          correlation_length))
    results.append(check_conditional_leftover(uneven_redshifts, spread,
                                              correlation_length))
    results.append(check_jump_variance_monte_carlo(rng))
    results.append(check_worked_example())
    results.append(check_bias_continuity())
    results.append(check_conserved_deviation_table())
    results.append(check_neighbor_rms_table())
    results.append(check_memory_projection_properties(uneven_redshifts, spread,
                                                      correlation_length))
    results.append(check_memory_woodbury_limit(uneven_redshifts, spread,
                                               correlation_length))
    results.append(check_memory_nullspace_identity(uneven_redshifts, spread,
                                                   correlation_length))
    results.append(check_memory_three_bin_curvature())
    results.append(check_memory_small_groups(spread, correlation_length))
    results.append(check_memory_kriging(uneven_redshifts, spread,
                                        correlation_length, rng))
    results.append(check_memory_end_correlation(uneven_redshifts, spread,
                                                correlation_length))
    results.append(check_memory_line_removes_drift())
    results.append(check_memory_bgs_flux_line())
    results.append(check_lrg3_flux_drift())

    number_failed = results.count(False)
    if number_failed == 0:
        print(f"All {len(results)} checks passed.")
    else:
        print(f"{number_failed} of {len(results)} checks FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
