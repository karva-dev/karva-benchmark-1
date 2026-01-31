"""Financial statistics utilities."""

import math


def mean(data: list[float]) -> float:
    """Calculate arithmetic mean."""
    if not data:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(data) / len(data)


def variance(data: list[float], ddof: int = 0) -> float:
    """Calculate variance.

    Args:
        data: List of values
        ddof: Delta degrees of freedom (0 for population, 1 for sample)

    Returns:
        Variance
    """
    if len(data) <= ddof:
        raise ValueError("Not enough data points")
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - ddof)


def std(data: list[float], ddof: int = 0) -> float:
    """Calculate standard deviation."""
    return math.sqrt(variance(data, ddof))


def covariance(x: list[float], y: list[float], ddof: int = 0) -> float:
    """Calculate covariance between two series.

    Args:
        x: First data series
        y: Second data series
        ddof: Delta degrees of freedom

    Returns:
        Covariance
    """
    if len(x) != len(y):
        raise ValueError("Series must have same length")
    if len(x) <= ddof:
        raise ValueError("Not enough data points")

    mean_x = mean(x)
    mean_y = mean(y)
    return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (len(x) - ddof)


def correlation(x: list[float], y: list[float]) -> float:
    """Calculate Pearson correlation coefficient.

    Args:
        x: First data series
        y: Second data series

    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(x) != len(y):
        raise ValueError("Series must have same length")
    if len(x) < 2:
        raise ValueError("Need at least 2 data points")

    std_x = std(x)
    std_y = std(y)

    if std_x == 0 or std_y == 0:
        return 0.0

    return covariance(x, y) / (std_x * std_y)


def covariance_matrix(data: list[list[float]], ddof: int = 0) -> list[list[float]]:
    """Calculate covariance matrix for multiple series.

    Args:
        data: List of data series (each inner list is a series)
        ddof: Delta degrees of freedom

    Returns:
        Covariance matrix
    """
    n = len(data)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            cov = covariance(data[i], data[j], ddof)
            matrix[i][j] = cov
            matrix[j][i] = cov

    return matrix


def correlation_matrix(data: list[list[float]]) -> list[list[float]]:
    """Calculate correlation matrix for multiple series.

    Args:
        data: List of data series

    Returns:
        Correlation matrix
    """
    n = len(data)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                corr = correlation(data[i], data[j])
                matrix[i][j] = corr
                matrix[j][i] = corr

    return matrix


def rolling_mean(data: list[float], window: int) -> list[float]:
    """Calculate rolling mean.

    Args:
        data: Input data
        window: Window size

    Returns:
        Rolling mean values (length = len(data) - window + 1)
    """
    if window > len(data):
        raise ValueError("Window size larger than data")
    if window <= 0:
        raise ValueError("Window size must be positive")

    result = []
    for i in range(len(data) - window + 1):
        result.append(mean(data[i : i + window]))
    return result


def rolling_std(data: list[float], window: int, ddof: int = 1) -> list[float]:
    """Calculate rolling standard deviation.

    Args:
        data: Input data
        window: Window size
        ddof: Delta degrees of freedom

    Returns:
        Rolling std values
    """
    if window > len(data):
        raise ValueError("Window size larger than data")
    if window <= ddof:
        raise ValueError("Window size must be larger than ddof")

    result = []
    for i in range(len(data) - window + 1):
        result.append(std(data[i : i + window], ddof))
    return result


def ewma(data: list[float], alpha: float) -> list[float]:
    """Calculate Exponentially Weighted Moving Average.

    Args:
        data: Input data
        alpha: Smoothing factor (0 < alpha <= 1)

    Returns:
        EWMA values
    """
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
    if not data:
        return []

    result = [data[0]]
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[-1])
    return result


def ewma_from_span(data: list[float], span: int) -> list[float]:
    """Calculate EWMA using span parameter.

    Args:
        data: Input data
        span: Span (decay in terms of center of mass)

    Returns:
        EWMA values
    """
    alpha = 2 / (span + 1)
    return ewma(data, alpha)


def skewness(data: list[float]) -> float:
    """Calculate skewness of data.

    Args:
        data: Input data

    Returns:
        Skewness (0 = symmetric, positive = right tail, negative = left tail)
    """
    if len(data) < 3:
        raise ValueError("Need at least 3 data points")

    m = mean(data)
    s = std(data, ddof=1)

    if s == 0:
        return 0.0

    n = len(data)
    return (n / ((n - 1) * (n - 2))) * sum(((x - m) / s) ** 3 for x in data)


def kurtosis(data: list[float], excess: bool = True) -> float:
    """Calculate kurtosis of data.

    Args:
        data: Input data
        excess: If True, return excess kurtosis (subtract 3)

    Returns:
        Kurtosis (excess kurtosis: 0 = normal, >0 = fat tails, <0 = thin tails)
    """
    if len(data) < 4:
        raise ValueError("Need at least 4 data points")

    m = mean(data)
    s = std(data, ddof=1)

    if s == 0:
        return 0.0

    n = len(data)
    k = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(
        ((x - m) / s) ** 4 for x in data
    )
    k -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))

    if not excess:
        k += 3

    return k


def percentile(data: list[float], p: float) -> float:
    """Calculate percentile of data.

    Args:
        data: Input data
        p: Percentile (0-100)

    Returns:
        Value at percentile
    """
    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    if not data:
        raise ValueError("Cannot calculate percentile of empty list")

    sorted_data = sorted(data)
    n = len(sorted_data)

    if p == 0:
        return sorted_data[0]
    if p == 100:
        return sorted_data[-1]

    k = (n - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return sorted_data[int(k)]

    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def median(data: list[float]) -> float:
    """Calculate median."""
    return percentile(data, 50)


def quartiles(data: list[float]) -> tuple[float, float, float]:
    """Calculate Q1, Q2 (median), Q3."""
    return percentile(data, 25), percentile(data, 50), percentile(data, 75)


def iqr(data: list[float]) -> float:
    """Calculate interquartile range."""
    q1, _, q3 = quartiles(data)
    return q3 - q1


def zscore(data: list[float]) -> list[float]:
    """Calculate z-scores for data."""
    m = mean(data)
    s = std(data)
    if s == 0:
        return [0.0] * len(data)
    return [(x - m) / s for x in data]


def zscore_outliers(data: list[float], threshold: float = 3.0) -> list[int]:
    """Detect outliers using z-score method.

    Args:
        data: Input data
        threshold: Z-score threshold for outlier detection

    Returns:
        Indices of outliers
    """
    z = zscore(data)
    return [i for i, zi in enumerate(z) if abs(zi) > threshold]


def iqr_outliers(data: list[float], k: float = 1.5) -> list[int]:
    """Detect outliers using IQR method.

    Args:
        data: Input data
        k: Multiplier for IQR (typically 1.5 for outliers, 3.0 for extreme)

    Returns:
        Indices of outliers
    """
    q1, _, q3 = quartiles(data)
    iqr_val = q3 - q1
    lower = q1 - k * iqr_val
    upper = q3 + k * iqr_val
    return [i for i, x in enumerate(data) if x < lower or x > upper]


def weighted_mean(data: list[float], weights: list[float]) -> float:
    """Calculate weighted mean."""
    if len(data) != len(weights):
        raise ValueError("Data and weights must have same length")
    if sum(weights) == 0:
        raise ValueError("Weights sum to zero")
    return sum(d * w for d, w in zip(data, weights)) / sum(weights)


def geometric_mean(data: list[float]) -> float:
    """Calculate geometric mean."""
    if not data:
        raise ValueError("Cannot calculate geometric mean of empty list")
    if any(x <= 0 for x in data):
        raise ValueError("All values must be positive")
    return math.exp(sum(math.log(x) for x in data) / len(data))


def harmonic_mean(data: list[float]) -> float:
    """Calculate harmonic mean."""
    if not data:
        raise ValueError("Cannot calculate harmonic mean of empty list")
    if any(x <= 0 for x in data):
        raise ValueError("All values must be positive")
    return len(data) / sum(1 / x for x in data)
