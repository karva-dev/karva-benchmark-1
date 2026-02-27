"""Extended statistics tests with heavy parametrization."""

import math
import random

import pytest

from finlib.statistics import (
    mean,
    variance,
    std,
    covariance,
    correlation,
    covariance_matrix,
    correlation_matrix,
    rolling_mean,
    rolling_std,
    ewma,
    skewness,
    kurtosis,
    percentile,
    quartiles,
    iqr,
    zscore,
    zscore_outliers,
    iqr_outliers,
    weighted_mean,
    geometric_mean,
    harmonic_mean,
)


def generate_data(n: int, mean_val: float, std_val: float, seed: int) -> list:
    """Generate random data for testing."""
    random.seed(seed)
    return [random.gauss(mean_val, std_val) for _ in range(n)]


def generate_positive_data(n: int, mean_val: float, std_val: float, seed: int) -> list:
    """Generate positive random data for geometric/harmonic mean."""
    random.seed(seed)
    return [abs(random.gauss(mean_val, std_val)) + 0.1 for _ in range(n)]


# Mean tests
@pytest.mark.parametrize("n", [10, 50, 100, 500])
@pytest.mark.parametrize("mean_val", [-10, 0, 10, 100])
@pytest.mark.parametrize("std_val", [1, 5, 10])
def test_mean_converges(n: int, mean_val: float, std_val: float):
    """Mean should converge to true mean for large n."""
    data = generate_data(n, mean_val, std_val, 42)
    result = mean(data)
    # Allow wider tolerance for smaller samples
    tolerance = 3 * std_val / math.sqrt(n)
    assert abs(result - mean_val) < tolerance


# Variance tests
@pytest.mark.parametrize("n", [20, 50, 100, 500])
@pytest.mark.parametrize("std_val", [1, 5, 10, 20])
@pytest.mark.parametrize("ddof", [0, 1])
def test_variance_positive(n: int, std_val: float, ddof: int):
    """Variance should be positive."""
    data = generate_data(n, 0, std_val, 42)
    var = variance(data, ddof=ddof)
    assert var >= 0


@pytest.mark.parametrize("n", [50, 100, 500])
@pytest.mark.parametrize("std_val", [1, 5, 10])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_variance_converges(n: int, std_val: float, seed: int):
    """Variance should converge to true variance for large n."""
    data = generate_data(n, 0, std_val, seed)
    var = variance(data)
    true_var = std_val**2
    # Wide tolerance for variance estimation
    assert abs(var - true_var) < true_var


# Standard deviation tests
@pytest.mark.parametrize("n", [20, 50, 100, 500])
@pytest.mark.parametrize("std_val", [1, 5, 10, 20])
@pytest.mark.parametrize("ddof", [0, 1])
def test_std_positive(n: int, std_val: float, ddof: int):
    """Standard deviation should be positive."""
    data = generate_data(n, 0, std_val, 42)
    s = std(data, ddof=ddof)
    assert s >= 0


@pytest.mark.parametrize("n", [50, 100, 500])
@pytest.mark.parametrize("std_val", [1, 5, 10])
@pytest.mark.parametrize("ddof", [0, 1])
def test_std_is_sqrt_variance(n: int, std_val: float, ddof: int):
    """Std should be sqrt of variance."""
    data = generate_data(n, 0, std_val, 42)
    v = variance(data, ddof=ddof)
    s = std(data, ddof=ddof)
    assert abs(s - math.sqrt(v)) < 0.0001


# Covariance tests
@pytest.mark.parametrize("n", [20, 50, 100])
@pytest.mark.parametrize("std_val", [1, 5, 10])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_covariance_self_equals_variance(n: int, std_val: float, seed: int):
    """Covariance of series with itself equals variance."""
    data = generate_data(n, 0, std_val, seed)
    cov = covariance(data, data)
    var = variance(data)
    assert abs(cov - var) < 0.0001


@pytest.mark.parametrize("n", [50, 100])
@pytest.mark.parametrize("multiplier", [0.5, 1.0, 2.0, -1.0])
@pytest.mark.parametrize("seed", [42, 123])
def test_covariance_scaled_series(n: int, multiplier: float, seed: int):
    """Covariance with scaled series."""
    x = generate_data(n, 0, 5, seed)
    y = [xi * multiplier for xi in x]
    cov = covariance(x, y)
    var_x = variance(x)
    expected = multiplier * var_x
    assert abs(cov - expected) < 0.01


# Correlation tests
@pytest.mark.parametrize("n", [50, 100, 200])
@pytest.mark.parametrize("seed", [42, 123, 456, 789])
def test_correlation_self_is_one(n: int, seed: int):
    """Correlation of series with itself should be 1."""
    data = generate_data(n, 0, 5, seed)
    corr = correlation(data, data)
    assert abs(corr - 1.0) < 0.0001


@pytest.mark.parametrize("n", [50, 100, 200])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_correlation_inverse_is_negative_one(n: int, seed: int):
    """Correlation of series with inverse should be -1."""
    data = generate_data(n, 0, 5, seed)
    inverse = [-x for x in data]
    corr = correlation(data, inverse)
    assert abs(corr - (-1.0)) < 0.0001


@pytest.mark.parametrize("n", [50, 100])
@pytest.mark.parametrize("seed1", [42, 123])
@pytest.mark.parametrize("seed2", [456, 789])
def test_correlation_bounds(n: int, seed1: int, seed2: int):
    """Correlation should be between -1 and 1."""
    x = generate_data(n, 0, 5, seed1)
    y = generate_data(n, 0, 5, seed2)
    corr = correlation(x, y)
    assert -1 <= corr <= 1


# Covariance matrix tests
@pytest.mark.parametrize("n", [30, 50, 100])
@pytest.mark.parametrize("num_series", [2, 3, 4])
@pytest.mark.parametrize("seed", [42, 123])
def test_covariance_matrix_symmetric(n: int, num_series: int, seed: int):
    """Covariance matrix should be symmetric."""
    data = [generate_data(n, 0, 5, seed + i) for i in range(num_series)]
    matrix = covariance_matrix(data)
    for i in range(num_series):
        for j in range(num_series):
            assert abs(matrix[i][j] - matrix[j][i]) < 0.0001


@pytest.mark.parametrize("n", [30, 50, 100])
@pytest.mark.parametrize("num_series", [2, 3, 4])
@pytest.mark.parametrize("seed", [42, 123])
def test_covariance_matrix_diagonal_positive(n: int, num_series: int, seed: int):
    """Diagonal elements (variances) should be positive."""
    data = [generate_data(n, 0, 5, seed + i) for i in range(num_series)]
    matrix = covariance_matrix(data)
    for i in range(num_series):
        assert matrix[i][i] >= 0


# Correlation matrix tests
@pytest.mark.parametrize("n", [30, 50, 100])
@pytest.mark.parametrize("num_series", [2, 3, 4])
@pytest.mark.parametrize("seed", [42, 123])
def test_correlation_matrix_diagonal_ones(n: int, num_series: int, seed: int):
    """Diagonal of correlation matrix should be 1."""
    data = [generate_data(n, 0, 5, seed + i) for i in range(num_series)]
    matrix = correlation_matrix(data)
    for i in range(num_series):
        assert abs(matrix[i][i] - 1.0) < 0.0001


@pytest.mark.parametrize("n", [30, 50, 100])
@pytest.mark.parametrize("num_series", [2, 3, 4])
@pytest.mark.parametrize("seed", [42, 123])
def test_correlation_matrix_bounds(n: int, num_series: int, seed: int):
    """All correlation matrix elements should be between -1 and 1."""
    data = [generate_data(n, 0, 5, seed + i) for i in range(num_series)]
    matrix = correlation_matrix(data)
    for row in matrix:
        for val in row:
            assert -1 <= val <= 1


# Rolling mean tests
@pytest.mark.parametrize("n", [50, 100, 200])
@pytest.mark.parametrize("window", [5, 10, 20])
@pytest.mark.parametrize("seed", [42, 123])
def test_rolling_mean_length(n: int, window: int, seed: int):
    """Rolling mean should have correct length."""
    data = generate_data(n, 0, 5, seed)
    result = rolling_mean(data, window)
    assert len(result) == n - window + 1


@pytest.mark.parametrize("n", [100])
@pytest.mark.parametrize("window", [5, 10, 20, 30])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_rolling_mean_values(n: int, window: int, seed: int):
    """Rolling mean values should be valid."""
    data = generate_data(n, 0, 5, seed)
    result = rolling_mean(data, window)
    for val in result:
        assert isinstance(val, float)


# Rolling std tests
@pytest.mark.parametrize("n", [50, 100, 200])
@pytest.mark.parametrize("window", [5, 10, 20])
@pytest.mark.parametrize("seed", [42, 123])
def test_rolling_std_length(n: int, window: int, seed: int):
    """Rolling std should have correct length."""
    data = generate_data(n, 0, 5, seed)
    result = rolling_std(data, window)
    assert len(result) == n - window + 1


@pytest.mark.parametrize("n", [100])
@pytest.mark.parametrize("window", [5, 10, 20, 30])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_rolling_std_positive(n: int, window: int, seed: int):
    """Rolling std values should be positive."""
    data = generate_data(n, 0, 5, seed)
    result = rolling_std(data, window)
    for val in result:
        assert val >= 0


# EWMA tests
@pytest.mark.parametrize("n", [50, 100, 200])
@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.9])
@pytest.mark.parametrize("seed", [42, 123])
def test_ewma_length(n: int, alpha: float, seed: int):
    """EWMA should have same length as input."""
    data = generate_data(n, 0, 5, seed)
    result = ewma(data, alpha)
    assert len(result) == n


@pytest.mark.parametrize("n", [50, 100])
@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.9])
@pytest.mark.parametrize("seed", [42, 123])
def test_ewma_first_equals_first(n: int, alpha: float, seed: int):
    """EWMA first value should equal first data value."""
    data = generate_data(n, 0, 5, seed)
    result = ewma(data, alpha)
    assert result[0] == data[0]


# Skewness tests
@pytest.mark.parametrize("n", [50, 100, 500])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_skewness_symmetric_near_zero(n: int, seed: int):
    """Skewness of symmetric distribution should be near zero."""
    data = generate_data(n, 0, 5, seed)
    sk = skewness(data)
    assert abs(sk) < 1.0  # Wide tolerance for random data


@pytest.mark.parametrize("n", [100, 500])
@pytest.mark.parametrize("seed", [42, 123])
def test_skewness_finite(n: int, seed: int):
    """Skewness should be finite."""
    data = generate_data(n, 0, 5, seed)
    sk = skewness(data)
    assert math.isfinite(sk)


# Kurtosis tests
@pytest.mark.parametrize("n", [50, 100, 500])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_kurtosis_finite(n: int, seed: int):
    """Kurtosis should be finite."""
    data = generate_data(n, 0, 5, seed)
    k = kurtosis(data)
    assert math.isfinite(k)


@pytest.mark.parametrize("n", [100, 500])
@pytest.mark.parametrize("excess", [True, False])
@pytest.mark.parametrize("seed", [42, 123])
def test_kurtosis_excess_vs_raw(n: int, excess: bool, seed: int):
    """Test both excess and raw kurtosis."""
    data = generate_data(n, 0, 5, seed)
    k = kurtosis(data, excess=excess)
    assert isinstance(k, float)


# Percentile tests
@pytest.mark.parametrize("n", [50, 100, 500])
@pytest.mark.parametrize("p", [0, 25, 50, 75, 100])
@pytest.mark.parametrize("seed", [42, 123])
def test_percentile_in_range(n: int, p: float, seed: int):
    """Percentile should be within data range."""
    data = generate_data(n, 0, 5, seed)
    pct = percentile(data, p)
    assert min(data) <= pct <= max(data)


@pytest.mark.parametrize("n", [100])
@pytest.mark.parametrize("seed", [42, 123, 456, 789])
def test_percentile_ordering(n: int, seed: int):
    """Percentiles should be ordered."""
    data = generate_data(n, 0, 5, seed)
    p25 = percentile(data, 25)
    p50 = percentile(data, 50)
    p75 = percentile(data, 75)
    assert p25 <= p50 <= p75


# Quartiles and IQR tests
@pytest.mark.parametrize("n", [50, 100, 500])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_quartiles_ordered(n: int, seed: int):
    """Quartiles should be ordered."""
    data = generate_data(n, 0, 5, seed)
    q1, q2, q3 = quartiles(data)
    assert q1 <= q2 <= q3


@pytest.mark.parametrize("n", [50, 100, 500])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_iqr_positive(n: int, seed: int):
    """IQR should be positive."""
    data = generate_data(n, 0, 5, seed)
    i = iqr(data)
    assert i >= 0


# Z-score tests
@pytest.mark.parametrize("n", [50, 100, 500])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_zscore_mean_zero(n: int, seed: int):
    """Z-scores should have mean near zero."""
    data = generate_data(n, 0, 5, seed)
    z = zscore(data)
    assert abs(mean(z)) < 0.0001


@pytest.mark.parametrize("n", [50, 100, 500])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_zscore_std_one(n: int, seed: int):
    """Z-scores should have std near 1."""
    data = generate_data(n, 0, 5, seed)
    z = zscore(data)
    assert abs(std(z) - 1) < 0.01


# Outlier detection tests
@pytest.mark.parametrize("n", [100, 500])
@pytest.mark.parametrize("threshold", [2.0, 2.5, 3.0])
@pytest.mark.parametrize("seed", [42, 123])
def test_zscore_outliers_indices(n: int, threshold: float, seed: int):
    """Z-score outlier indices should be valid."""
    data = generate_data(n, 0, 5, seed)
    outliers = zscore_outliers(data, threshold=threshold)
    for idx in outliers:
        assert 0 <= idx < n


@pytest.mark.parametrize("n", [100, 500])
@pytest.mark.parametrize("k", [1.5, 2.0, 3.0])
@pytest.mark.parametrize("seed", [42, 123])
def test_iqr_outliers_indices(n: int, k: float, seed: int):
    """IQR outlier indices should be valid."""
    data = generate_data(n, 0, 5, seed)
    outliers = iqr_outliers(data, k=k)
    for idx in outliers:
        assert 0 <= idx < n


# Weighted mean tests
@pytest.mark.parametrize("n", [10, 50, 100])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_weighted_mean_equals_mean_for_equal_weights(n: int, seed: int):
    """Weighted mean with equal weights should equal mean."""
    data = generate_data(n, 0, 5, seed)
    weights = [1.0] * n
    wm = weighted_mean(data, weights)
    m = mean(data)
    assert abs(wm - m) < 0.0001


@pytest.mark.parametrize("n", [10, 50, 100])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_weighted_mean_in_range(n: int, seed: int):
    """Weighted mean should be within data range."""
    data = generate_positive_data(n, 10, 3, seed)
    weights = generate_positive_data(n, 1, 0.5, seed + 100)
    wm = weighted_mean(data, weights)
    assert min(data) <= wm <= max(data)


# Geometric mean tests
@pytest.mark.parametrize("n", [10, 50, 100])
@pytest.mark.parametrize("mean_val", [5, 10, 20])
@pytest.mark.parametrize("seed", [42, 123])
def test_geometric_mean_positive(n: int, mean_val: float, seed: int):
    """Geometric mean should be positive for positive data."""
    data = generate_positive_data(n, mean_val, 2, seed)
    gm = geometric_mean(data)
    assert gm > 0


@pytest.mark.parametrize("n", [10, 50, 100])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_geometric_mean_leq_arithmetic(n: int, seed: int):
    """Geometric mean should be <= arithmetic mean."""
    data = generate_positive_data(n, 10, 3, seed)
    gm = geometric_mean(data)
    am = mean(data)
    assert gm <= am + 0.0001


# Harmonic mean tests
@pytest.mark.parametrize("n", [10, 50, 100])
@pytest.mark.parametrize("mean_val", [5, 10, 20])
@pytest.mark.parametrize("seed", [42, 123])
def test_harmonic_mean_positive(n: int, mean_val: float, seed: int):
    """Harmonic mean should be positive for positive data."""
    data = generate_positive_data(n, mean_val, 2, seed)
    hm = harmonic_mean(data)
    assert hm > 0


@pytest.mark.parametrize("n", [10, 50, 100])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_harmonic_mean_leq_geometric(n: int, seed: int):
    """Harmonic mean should be <= geometric mean."""
    data = generate_positive_data(n, 10, 3, seed)
    hm = harmonic_mean(data)
    gm = geometric_mean(data)
    assert hm <= gm + 0.0001


# Mean inequality: AM >= GM >= HM
@pytest.mark.parametrize("n", [20, 50, 100])
@pytest.mark.parametrize("mean_val", [5, 10, 20])
@pytest.mark.parametrize("seed", [42, 123, 456])
def test_mean_inequality(n: int, mean_val: float, seed: int):
    """AM >= GM >= HM should hold."""
    data = generate_positive_data(n, mean_val, 2, seed)
    am = mean(data)
    gm = geometric_mean(data)
    hm = harmonic_mean(data)
    assert am >= gm >= hm
