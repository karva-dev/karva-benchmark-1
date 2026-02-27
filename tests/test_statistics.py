"""Tests for financial statistics utilities."""

import math

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


# Mean Tests
@pytest.mark.parametrize(
    "data,expected",
    [
        ([1, 2, 3, 4, 5], 3),
        ([10, 20, 30], 20),
        ([0, 0, 0], 0),
        ([-2, -1, 0, 1, 2], 0),
        ([1.5, 2.5, 3.5], 2.5),
    ],
)
def test_mean_basic(data: list, expected: float):
    result = mean(data)
    assert abs(result - expected) < 0.0001


def test_mean_empty():
    try:
        mean([])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_mean_single():
    result = mean([42])
    assert result == 42


# Variance Tests
@pytest.mark.parametrize(
    "data,ddof,expected",
    [
        ([1, 2, 3, 4, 5], 0, 2),
        ([1, 2, 3, 4, 5], 1, 2.5),
        ([0, 0, 0], 0, 0),
        ([2, 4, 4, 4, 5, 5, 7, 9], 0, 4),
    ],
)
def test_variance(data: list, ddof: int, expected: float):
    result = variance(data, ddof)
    assert abs(result - expected) < 0.0001


def test_variance_insufficient_data():
    try:
        variance([1], ddof=1)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Standard Deviation Tests
@pytest.mark.parametrize(
    "data,ddof,expected",
    [
        ([1, 2, 3, 4, 5], 0, 1.4142),
        ([2, 4, 4, 4, 5, 5, 7, 9], 0, 2),
    ],
)
def test_std(data: list, ddof: int, expected: float):
    result = std(data, ddof)
    assert abs(result - expected) < 0.001


def test_std_is_sqrt_variance():
    data = [1, 3, 5, 7, 9]
    v = variance(data)
    s = std(data)
    assert abs(s - math.sqrt(v)) < 0.0001


# Covariance Tests
def test_covariance_same_series():
    data = [1, 2, 3, 4, 5]
    result = covariance(data, data)
    assert abs(result - variance(data)) < 0.0001


def test_covariance_opposite():
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    result = covariance(x, y)
    assert result < 0


@pytest.mark.parametrize(
    "x,y,expected",
    [
        ([1, 2, 3], [1, 2, 3], 0.6667),
        ([1, 2, 3], [3, 2, 1], -0.6667),
    ],
)
def test_covariance_values(x: list, y: list, expected: float):
    result = covariance(x, y)
    assert abs(result - expected) < 0.01


def test_covariance_mismatched_lengths():
    try:
        covariance([1, 2, 3], [1, 2])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Correlation Tests
def test_correlation_same_series():
    data = [1, 2, 3, 4, 5]
    result = correlation(data, data)
    assert abs(result - 1.0) < 0.0001


def test_correlation_opposite():
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    result = correlation(x, y)
    assert abs(result - (-1.0)) < 0.0001


def test_correlation_bounds():
    x = [1, 3, 5, 2, 4]
    y = [2, 1, 4, 3, 5]
    result = correlation(x, y)
    assert -1 <= result <= 1


def test_correlation_zero():
    x = [1, 2, 3, 4, 5]
    y = [1, 1, 1, 1, 1]
    result = correlation(x, y)
    assert result == 0.0


# Covariance Matrix Tests
def test_covariance_matrix_diagonal():
    data = [[1, 2, 3], [4, 5, 6]]
    matrix = covariance_matrix(data)
    # Diagonal should be variances
    assert abs(matrix[0][0] - variance(data[0])) < 0.0001
    assert abs(matrix[1][1] - variance(data[1])) < 0.0001


def test_covariance_matrix_symmetric():
    data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]]
    matrix = covariance_matrix(data)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            assert abs(matrix[i][j] - matrix[j][i]) < 0.0001


def test_covariance_matrix_size():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matrix = covariance_matrix(data)
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)


# Correlation Matrix Tests
def test_correlation_matrix_diagonal():
    data = [[1, 2, 3], [4, 5, 6]]
    matrix = correlation_matrix(data)
    # Diagonal should be 1
    assert matrix[0][0] == 1.0
    assert matrix[1][1] == 1.0


def test_correlation_matrix_bounds():
    data = [[1, 3, 5, 2], [2, 4, 1, 5], [3, 2, 4, 1]]
    matrix = correlation_matrix(data)
    for row in matrix:
        for val in row:
            assert -1 <= val <= 1


# Rolling Mean Tests
@pytest.mark.parametrize("window", [2, 3, 5])
def test_rolling_mean_length(window: int):
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = rolling_mean(data, window)
    assert len(result) == len(data) - window + 1


def test_rolling_mean_values():
    data = [1, 2, 3, 4, 5]
    result = rolling_mean(data, 3)
    assert abs(result[0] - 2) < 0.0001  # (1+2+3)/3
    assert abs(result[1] - 3) < 0.0001  # (2+3+4)/3
    assert abs(result[2] - 4) < 0.0001  # (3+4+5)/3


def test_rolling_mean_window_too_large():
    try:
        rolling_mean([1, 2, 3], 5)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Rolling Std Tests
@pytest.mark.parametrize("window", [3, 5, 10])
def test_rolling_std_length(window: int):
    data = list(range(20))
    result = rolling_std(data, window)
    assert len(result) == len(data) - window + 1


def test_rolling_std_positive():
    data = [1, 3, 2, 5, 4, 6, 3, 7]
    result = rolling_std(data, 3)
    assert all(r >= 0 for r in result)


# EWMA Tests
def test_ewma_length():
    data = [1, 2, 3, 4, 5]
    result = ewma(data, 0.3)
    assert len(result) == len(data)


def test_ewma_first_value():
    data = [10, 20, 30]
    result = ewma(data, 0.5)
    assert result[0] == data[0]


def test_ewma_alpha_one():
    # Alpha = 1 means EWMA = original data
    data = [1, 2, 3, 4, 5]
    result = ewma(data, 1.0)
    assert result == data


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.9])
def test_ewma_various_alpha(alpha: float):
    data = [1, 2, 3, 4, 5]
    result = ewma(data, alpha)
    assert len(result) == len(data)


def test_ewma_invalid_alpha():
    try:
        ewma([1, 2, 3], 0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# EWMA from Span Tests
def test_ewma_from_span():
    data = [1, 2, 3, 4, 5]
    span = 3
    result = ewma_from_span(data, span)
    expected_alpha = 2 / (span + 1)
    expected = ewma(data, expected_alpha)
    assert result == expected


# Skewness Tests
def test_skewness_symmetric():
    data = [1, 2, 3, 4, 5]
    result = skewness(data)
    assert abs(result) < 0.1


def test_skewness_right_tail():
    data = [1, 1, 1, 2, 2, 2, 3, 3, 10]  # Right skewed
    result = skewness(data)
    assert result > 0


def test_skewness_left_tail():
    data = [0, 7, 7, 8, 8, 8, 9, 9, 9]  # Left skewed
    result = skewness(data)
    assert result < 0


def test_skewness_insufficient_data():
    try:
        skewness([1, 2])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Kurtosis Tests
def test_kurtosis_normal_like():
    # Uniform distribution has excess kurtosis around -1.2
    data = list(range(1, 101))
    result = kurtosis(data)
    assert result < 0


def test_kurtosis_excess():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    excess = kurtosis(data, excess=True)
    non_excess = kurtosis(data, excess=False)
    assert abs(non_excess - excess - 3) < 0.01


def test_kurtosis_insufficient_data():
    try:
        kurtosis([1, 2, 3])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Percentile Tests
@pytest.mark.parametrize(
    "data,p,expected",
    [
        ([1, 2, 3, 4, 5], 0, 1),
        ([1, 2, 3, 4, 5], 100, 5),
        ([1, 2, 3, 4, 5], 50, 3),
    ],
)
def test_percentile(data: list, p: float, expected: float):
    result = percentile(data, p)
    assert abs(result - expected) < 0.0001


def test_percentile_interpolation():
    data = [1, 2, 3, 4]
    result = percentile(data, 50)
    assert abs(result - 2.5) < 0.0001


def test_percentile_invalid():
    try:
        percentile([1, 2, 3], 150)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Median Tests
def test_median_odd():
    result = median([1, 2, 3, 4, 5])
    assert result == 3


def test_median_even():
    result = median([1, 2, 3, 4])
    assert result == 2.5


# Quartiles Tests
def test_quartiles():
    data = list(range(1, 101))
    q1, q2, q3 = quartiles(data)
    assert abs(q1 - 25.75) < 0.5
    assert abs(q2 - 50.5) < 0.5
    assert abs(q3 - 75.25) < 0.5


def test_quartiles_order():
    data = [1, 3, 5, 7, 9, 11, 13, 15]
    q1, q2, q3 = quartiles(data)
    assert q1 <= q2 <= q3


# IQR Tests
def test_iqr_basic():
    data = list(range(1, 101))
    result = iqr(data)
    q1, _, q3 = quartiles(data)
    assert abs(result - (q3 - q1)) < 0.0001


def test_iqr_no_spread():
    data = [5, 5, 5, 5, 5]
    result = iqr(data)
    assert result == 0


# Z-Score Tests
def test_zscore_mean_zero():
    data = [1, 2, 3, 4, 5]
    z = zscore(data)
    assert abs(mean(z)) < 0.0001


def test_zscore_std_one():
    data = [1, 2, 3, 4, 5]
    z = zscore(data)
    assert abs(std(z) - 1) < 0.0001


def test_zscore_constant():
    data = [5, 5, 5, 5, 5]
    z = zscore(data)
    assert all(zi == 0 for zi in z)


# Z-Score Outliers Tests
def test_zscore_outliers_none():
    data = [1, 2, 3, 4, 5]
    result = zscore_outliers(data)
    assert len(result) == 0


def test_zscore_outliers_found():
    data = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
    result = zscore_outliers(data, threshold=2)
    assert 5 in result  # Index of 100


@pytest.mark.parametrize("threshold", [2.0, 2.5, 3.0])
def test_zscore_outliers_thresholds(threshold: float):
    data = [1, 2, 3, 4, 5, 50]
    result = zscore_outliers(data, threshold=threshold)
    assert isinstance(result, list)


# IQR Outliers Tests
def test_iqr_outliers_none():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = iqr_outliers(data)
    assert len(result) == 0


def test_iqr_outliers_found():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    result = iqr_outliers(data)
    assert 9 in result  # Index of 100


@pytest.mark.parametrize("k", [1.5, 2.0, 3.0])
def test_iqr_outliers_multiplier(k: float):
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 50]
    result = iqr_outliers(data, k=k)
    assert isinstance(result, list)


# Weighted Mean Tests
@pytest.mark.parametrize(
    "data,weights,expected",
    [
        ([1, 2, 3], [1, 1, 1], 2),
        ([1, 2, 3], [1, 2, 1], 2),
        (
            [10, 20, 30],
            [0.5, 0.3, 0.2],
            17,
        ),  # (10*0.5 + 20*0.3 + 30*0.2) / (0.5+0.3+0.2) = 17
    ],
)
def test_weighted_mean(data: list, weights: list, expected: float):
    result = weighted_mean(data, weights)
    assert abs(result - expected) < 0.01


def test_weighted_mean_mismatched():
    try:
        weighted_mean([1, 2, 3], [1, 2])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_weighted_mean_zero_weights():
    try:
        weighted_mean([1, 2, 3], [0, 0, 0])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Geometric Mean Tests
@pytest.mark.parametrize(
    "data,expected",
    [
        ([1, 1, 1], 1),
        ([2, 8], 4),
        ([1, 2, 4, 8], 2.828),
    ],
)
def test_geometric_mean(data: list, expected: float):
    result = geometric_mean(data)
    assert abs(result - expected) < 0.01


def test_geometric_mean_negative():
    try:
        geometric_mean([1, -2, 3])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_geometric_mean_empty():
    try:
        geometric_mean([])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Harmonic Mean Tests
@pytest.mark.parametrize(
    "data,expected",
    [
        ([1, 1, 1], 1),
        ([1, 2, 4], 1.714),
        ([2, 3, 6], 3),
    ],
)
def test_harmonic_mean(data: list, expected: float):
    result = harmonic_mean(data)
    assert abs(result - expected) < 0.01


def test_harmonic_mean_negative():
    try:
        harmonic_mean([1, -2, 3])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Integration Tests
def test_mean_relationship():
    data = [1, 2, 4, 8, 16]
    am = mean(data)
    gm = geometric_mean(data)
    hm = harmonic_mean(data)
    # AM >= GM >= HM
    assert am >= gm >= hm


def test_covariance_correlation_relationship():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    cov = covariance(x, y)
    corr = correlation(x, y)
    sx = std(x)
    sy = std(y)
    # Correlation = Covariance / (std_x * std_y)
    assert abs(corr - cov / (sx * sy)) < 0.0001


def test_percentile_quartiles_consistency():
    data = list(range(1, 101))
    q1, q2, q3 = quartiles(data)
    p25 = percentile(data, 25)
    p50 = percentile(data, 50)
    p75 = percentile(data, 75)
    assert abs(q1 - p25) < 0.0001
    assert abs(q2 - p50) < 0.0001
    assert abs(q3 - p75) < 0.0001
