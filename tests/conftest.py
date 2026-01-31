"""Shared fixtures for finlib tests."""

from karva import fixture


@fixture
def sample_returns():
    """Sample daily returns for testing."""
    return [
        0.01, -0.005, 0.008, -0.002, 0.015,
        -0.01, 0.003, 0.007, -0.004, 0.002,
        0.005, -0.008, 0.012, -0.001, 0.006,
        -0.003, 0.009, -0.006, 0.004, 0.001
    ]


@fixture
def sample_prices():
    """Sample price series for testing."""
    return [
        100.0, 101.0, 100.5, 101.3, 101.1,
        102.5, 101.8, 102.3, 101.9, 102.1,
        103.0, 102.5, 103.2, 103.8, 104.1,
        103.5, 104.2, 104.8, 104.5, 105.0
    ]


@fixture
def sample_portfolio_values():
    """Portfolio values over time."""
    return [10000, 10150, 10080, 10250, 10180, 10350, 10420, 10380, 10500, 10550]


@fixture
def sample_weights():
    """Portfolio weights."""
    return [0.4, 0.3, 0.2, 0.1]


@fixture
def sample_covariance_matrix():
    """Sample covariance matrix for 4 assets."""
    return [
        [0.04, 0.01, 0.005, 0.002],
        [0.01, 0.09, 0.015, 0.008],
        [0.005, 0.015, 0.0625, 0.01],
        [0.002, 0.008, 0.01, 0.0225]
    ]


@fixture
def market_returns():
    """Market/benchmark returns."""
    return [
        0.008, -0.004, 0.006, -0.001, 0.012,
        -0.008, 0.002, 0.005, -0.003, 0.001,
        0.004, -0.006, 0.010, -0.002, 0.005,
        -0.002, 0.007, -0.004, 0.003, 0.002
    ]


@fixture
def bond_params():
    """Standard bond parameters."""
    return {
        "face_value": 1000.0,
        "coupon_rate": 0.05,
        "ytm": 0.04,
        "periods": 10,
        "frequency": 2
    }


@fixture
def option_params():
    """Standard option parameters."""
    return {
        "S": 100.0,
        "K": 100.0,
        "r": 0.05,
        "sigma": 0.2,
        "T": 1.0
    }


@fixture
def loan_params():
    """Standard loan parameters."""
    return {
        "principal": 200000.0,
        "annual_rate": 0.06,
        "months": 360
    }


@fixture
def fx_rates():
    """Foreign exchange rates."""
    return {
        "EUR": 1.10,
        "GBP": 1.30,
        "JPY": 0.0091,
        "CHF": 1.08,
        "CAD": 0.75
    }


@fixture
def large_returns():
    """Large return series for performance testing."""
    import random
    random.seed(42)
    return [random.gauss(0.0005, 0.02) for _ in range(1000)]


@fixture
def correlation_matrix_3x3():
    """3x3 correlation matrix."""
    return [
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ]
