"""Mega statistics tests with heavy Cartesian product parametrization."""

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
    ewma_from_span,
    skewness,
    kurtosis,
    percentile,
    median,
    quartiles,
    iqr,
    zscore,
    zscore_outliers,
    iqr_outliers,
    weighted_mean,
    geometric_mean,
    harmonic_mean,
)


def _gen(n, mu, sigma, seed):
    random.seed(seed)
    return [random.gauss(mu, sigma) for _ in range(n)]


def _gen_pos(n, mu, sigma, seed):
    random.seed(seed)
    return [abs(random.gauss(mu, sigma)) + 0.1 for _ in range(n)]


SIZES = [5, 10, 20, 50, 100, 200, 500]
MEANS = [-100, -10, -1, 0, 1, 10, 100]
STDS = [0.1, 0.5, 1, 2, 5, 10, 50]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DDOFS = [0, 1]
WINDOWS = [3, 5, 10, 20, 50]
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
THRESHOLDS = [1.5, 2.0, 2.5, 3.0]


# 7*7*7*10 = 3430
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("mu", MEANS)
@pytest.mark.parametrize("sigma", STDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_mean_finite(n, mu, sigma, seed):
    """Mean should be finite."""
    data = _gen(n, mu, sigma, seed)
    m = mean(data)
    assert math.isfinite(m)


# 7*7*10*2 = 980
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", STDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("ddof", DDOFS)
def test_variance_nonneg(n, sigma, seed, ddof):
    """Variance should be non-negative."""
    if n <= ddof:
        return
    data = _gen(n, 0, sigma, seed)
    v = variance(data, ddof=ddof)
    assert v >= 0


# 7*7*10*2 = 980
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", STDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("ddof", DDOFS)
def test_std_nonneg(n, sigma, seed, ddof):
    """Std should be non-negative."""
    if n <= ddof:
        return
    data = _gen(n, 0, sigma, seed)
    s = std(data, ddof=ddof)
    assert s >= 0


# 7*7*10*2 = 980
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", STDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("ddof", DDOFS)
def test_std_is_sqrt_var(n, sigma, seed, ddof):
    """Std should be sqrt of variance."""
    if n <= ddof:
        return
    data = _gen(n, 0, sigma, seed)
    v = variance(data, ddof=ddof)
    s = std(data, ddof=ddof)
    assert abs(s - math.sqrt(v)) < 0.0001


# 7*7*10 = 490
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", STDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_covariance_self_equals_variance(n, sigma, seed):
    """Cov(X, X) == Var(X)."""
    data = _gen(n, 0, sigma, seed)
    cov = covariance(data, data)
    var = variance(data)
    assert abs(cov - var) < 0.0001


# 7*7*10 = 490
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", STDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_correlation_self_is_one(n, sigma, seed):
    """Corr(X, X) == 1."""
    if n < 2:
        return
    data = _gen(n, 0, sigma, seed)
    corr = correlation(data, data)
    assert abs(corr - 1.0) < 0.0001


# 7*7*10 = 490
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", STDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_correlation_bounds(n, sigma, seed):
    """Correlation should be between -1 and 1."""
    if n < 2:
        return
    x = _gen(n, 0, sigma, seed)
    y = _gen(n, 0, sigma, seed + 100)
    corr = correlation(x, y)
    assert -1.001 <= corr <= 1.001


# 5*4*10 = 200
@pytest.mark.parametrize("n", [20, 50, 100, 200, 500])
@pytest.mark.parametrize("num_series", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", SEEDS)
def test_cov_matrix_symmetric(n, num_series, seed):
    """Covariance matrix should be symmetric."""
    data = [_gen(n, 0, 5, seed + i) for i in range(num_series)]
    mat = covariance_matrix(data)
    for i in range(num_series):
        for j in range(num_series):
            assert abs(mat[i][j] - mat[j][i]) < 0.0001


# 5*4*10 = 200
@pytest.mark.parametrize("n", [20, 50, 100, 200, 500])
@pytest.mark.parametrize("num_series", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", SEEDS)
def test_corr_matrix_diagonal_ones(n, num_series, seed):
    """Correlation matrix diagonal should be 1."""
    data = [_gen(n, 0, 5, seed + i) for i in range(num_series)]
    mat = correlation_matrix(data)
    for i in range(num_series):
        assert abs(mat[i][i] - 1.0) < 0.0001


# 7*5*10 = 350
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rolling_mean_length(n, window, seed):
    """Rolling mean should have correct length."""
    if window > n:
        return
    data = _gen(n, 0, 5, seed)
    result = rolling_mean(data, window)
    assert len(result) == n - window + 1


# 7*5*10 = 350
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rolling_std_length(n, window, seed):
    """Rolling std should have correct length."""
    if window <= 1 or window > n:
        return
    data = _gen(n, 0, 5, seed)
    result = rolling_std(data, window)
    assert len(result) == n - window + 1


# 7*5*10 = 350
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rolling_std_nonneg(n, window, seed):
    """Rolling std values should be non-negative."""
    if window <= 1 or window > n:
        return
    data = _gen(n, 0, 5, seed)
    result = rolling_std(data, window)
    for val in result:
        assert val >= 0


# 7*7*10 = 490
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_ewma_length(n, alpha, seed):
    """EWMA should have same length as input."""
    data = _gen(n, 0, 5, seed)
    result = ewma(data, alpha)
    assert len(result) == n


# 7*7*10 = 490
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_ewma_first_equals_data(n, alpha, seed):
    """EWMA first value should equal first data value."""
    data = _gen(n, 0, 5, seed)
    result = ewma(data, alpha)
    assert result[0] == data[0]


# 7*5*10 = 350
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("span", [5, 10, 20, 50, 100])
@pytest.mark.parametrize("seed", SEEDS)
def test_ewma_from_span_length(n, span, seed):
    """EWMA from span should have same length as input."""
    data = _gen(n, 0, 5, seed)
    result = ewma_from_span(data, span)
    assert len(result) == n


# 5*10 = 50
@pytest.mark.parametrize("n", [50, 100, 200, 500, 1000])
@pytest.mark.parametrize("seed", SEEDS)
def test_skewness_finite(n, seed):
    """Skewness should be finite."""
    data = _gen(n, 0, 5, seed)
    sk = skewness(data)
    assert math.isfinite(sk)


# 5*10*2 = 100
@pytest.mark.parametrize("n", [50, 100, 200, 500, 1000])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("excess", [True, False])
def test_kurtosis_finite(n, seed, excess):
    """Kurtosis should be finite."""
    data = _gen(n, 0, 5, seed)
    k = kurtosis(data, excess=excess)
    assert math.isfinite(k)


# 7*9*10 = 630
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("p", PCTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_percentile_in_range(n, p, seed):
    """Percentile should be within data range."""
    data = _gen(n, 0, 5, seed)
    pct = percentile(data, p)
    assert min(data) <= pct <= max(data)


# 7*10 = 70
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_median_is_p50(n, seed):
    """Median should equal 50th percentile."""
    data = _gen(n, 0, 5, seed)
    med = median(data)
    p50 = percentile(data, 50)
    assert abs(med - p50) < 0.0001


# 7*10 = 70
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_quartiles_ordered(n, seed):
    """Quartiles should be ordered."""
    data = _gen(n, 0, 5, seed)
    q1, q2, q3 = quartiles(data)
    assert q1 <= q2 <= q3


# 7*10 = 70
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_iqr_nonneg(n, seed):
    """IQR should be non-negative."""
    data = _gen(n, 0, 5, seed)
    i = iqr(data)
    assert i >= 0


# 7*10 = 70
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_zscore_mean_zero(n, seed):
    """Z-scores should have mean near zero."""
    data = _gen(n, 10, 5, seed)
    z = zscore(data)
    assert abs(mean(z)) < 0.001


# 7*10 = 70
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_zscore_std_one(n, seed):
    """Z-scores should have std near 1."""
    data = _gen(n, 10, 5, seed)
    z = zscore(data)
    assert abs(std(z) - 1) < 0.01


# 5*4*10 = 200
@pytest.mark.parametrize("n", [50, 100, 200, 500, 1000])
@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_zscore_outlier_indices_valid(n, threshold, seed):
    """Z-score outlier indices should be valid."""
    data = _gen(n, 0, 5, seed)
    outliers = zscore_outliers(data, threshold)
    for idx in outliers:
        assert 0 <= idx < n


# 5*4*10 = 200
@pytest.mark.parametrize("n", [50, 100, 200, 500, 1000])
@pytest.mark.parametrize("k", [1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_iqr_outlier_indices_valid(n, k, seed):
    """IQR outlier indices should be valid."""
    data = _gen(n, 0, 5, seed)
    outliers = iqr_outliers(data, k)
    for idx in outliers:
        assert 0 <= idx < n


# 7*10 = 70
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_weighted_mean_equal_weights(n, seed):
    """Weighted mean with equal weights should equal mean."""
    data = _gen(n, 0, 5, seed)
    weights = [1.0] * n
    wm = weighted_mean(data, weights)
    m = mean(data)
    assert abs(wm - m) < 0.0001


# 5*7*10 = 350
@pytest.mark.parametrize("n", [5, 10, 20, 50, 100])
@pytest.mark.parametrize("mu", [1, 2, 5, 10, 20, 50, 100])
@pytest.mark.parametrize("seed", SEEDS)
def test_geometric_mean_positive(n, mu, seed):
    """Geometric mean should be positive for positive data."""
    data = _gen_pos(n, mu, mu / 3, seed)
    gm = geometric_mean(data)
    assert gm > 0


# 5*7*10 = 350
@pytest.mark.parametrize("n", [5, 10, 20, 50, 100])
@pytest.mark.parametrize("mu", [1, 2, 5, 10, 20, 50, 100])
@pytest.mark.parametrize("seed", SEEDS)
def test_harmonic_mean_positive(n, mu, seed):
    """Harmonic mean should be positive for positive data."""
    data = _gen_pos(n, mu, mu / 3, seed)
    hm = harmonic_mean(data)
    assert hm > 0


# 5*5*10 = 250
@pytest.mark.parametrize("n", [10, 20, 50, 100, 200])
@pytest.mark.parametrize("mu", [2, 5, 10, 20, 50])
@pytest.mark.parametrize("seed", SEEDS)
def test_mean_inequality_am_gm_hm(n, mu, seed):
    """AM >= GM >= HM should hold for positive data."""
    data = _gen_pos(n, mu, mu / 4, seed)
    am = mean(data)
    gm = geometric_mean(data)
    hm = harmonic_mean(data)
    assert am >= gm - 0.001
    assert gm >= hm - 0.001


# 7*10 = 70
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_percentile_monotonic(n, seed):
    """Higher percentiles should give higher values."""
    data = _gen(n, 0, 5, seed)
    pcts = [percentile(data, p) for p in [10, 25, 50, 75, 90]]
    for i in range(len(pcts) - 1):
        assert pcts[i] <= pcts[i + 1] + 0.001
