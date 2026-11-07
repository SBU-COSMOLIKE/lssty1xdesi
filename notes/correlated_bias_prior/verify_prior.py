"""Numerical checks of the correlated galaxy-bias prior.

This script verifies, one by one, the identities derived in the companion
note ``correlated_bias_prior.tex``: the two-bin inverse covariance and
projected precision, the three-bin inverse with its exactly zero
(1, 3) entry, the properties of the projected precision P for a
general exponential (Ornstein--Uhlenbeck) kernel on unevenly spaced
redshifts, the tridiagonal form of the inverse covariance, the
Pythagorean split of the prior penalty, the equivalence of P with the
Schur complement obtained by marginalizing a flat-prior group amplitude,
the variance of the jump between two bins (by Monte Carlo), and the
numbers of the worked example in the note.

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

    The Markov factorization of the Ornstein--Uhlenbeck process gives,
    with r_k = exp(-(z_{k+1} - z_k)/L) for each adjacent pair, the
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
    results.append(check_jump_variance_monte_carlo(rng))
    results.append(check_worked_example())

    number_failed = results.count(False)
    if number_failed == 0:
        print(f"All {len(results)} checks passed.")
    else:
        print(f"{number_failed} of {len(results)} checks FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
