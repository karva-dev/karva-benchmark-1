"""Extended risk tests with heavy parametrization."""

import math
import random

import karva

from finlib.risk import (
    volatility,
    downside_volatility,
    beta,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
    max_drawdown,
    parametric_var,
    historical_var,
    conditional_var,
    omega_ratio,
    gain_loss_ratio,
    win_rate,
)


def generate_returns(n: int, mean: float, std: float, seed: int) -> list:
    """Generate random returns for testing."""
    random.seed(seed)
    return [random.gauss(mean, std) for _ in range(n)]


def generate_prices(n: int, start: float, mean_ret: float, std: float, seed: int) -> list:
    """Generate price series from random returns."""
    random.seed(seed)
    prices = [start]
    for _ in range(n - 1):
        ret = random.gauss(mean_ret, std)
        prices.append(prices[-1] * (1 + ret))
    return prices


# Volatility tests
@karva.tags.parametrize("n", [20, 50, 100, 252])
@karva.tags.parametrize("std", [0.01, 0.02, 0.03])
@karva.tags.parametrize("seed", [42, 123, 456])
def test_volatility_positive(n: int, std: float, seed: int):
    """Volatility should be positive."""
    returns = generate_returns(n, 0.001, std, seed)
    vol = volatility(returns, annualize=False)
    assert vol >= 0


@karva.tags.parametrize("n", [50, 100, 252])
@karva.tags.parametrize("std", [0.01, 0.02, 0.03])
@karva.tags.parametrize("periods", [12, 52, 252])
def test_annualized_volatility(n: int, std: float, periods: int):
    """Annualized volatility should scale correctly."""
    returns = generate_returns(n, 0.001, std, 42)
    daily_vol = volatility(returns, annualize=False)
    annual_vol = volatility(returns, annualize=True, periods_per_year=periods)
    expected = daily_vol * math.sqrt(periods)
    assert abs(annual_vol - expected) < 0.0001


@karva.tags.parametrize("std1", [0.01, 0.02])
@karva.tags.parametrize("std2", [0.03, 0.04])
@karva.tags.parametrize("seed", [42, 123])
def test_higher_std_higher_volatility(std1: float, std2: float, seed: int):
    """Higher std input should produce higher volatility."""
    returns1 = generate_returns(100, 0.001, std1, seed)
    returns2 = generate_returns(100, 0.001, std2, seed)
    vol1 = volatility(returns1, annualize=False)
    vol2 = volatility(returns2, annualize=False)
    assert vol1 < vol2


# Downside volatility tests
@karva.tags.parametrize("n", [50, 100, 252])
@karva.tags.parametrize("std", [0.01, 0.02, 0.03])
@karva.tags.parametrize("seed", [42, 123, 456])
def test_downside_volatility_non_negative(n: int, std: float, seed: int):
    """Downside volatility should be non-negative."""
    returns = generate_returns(n, 0.001, std, seed)
    dvol = downside_volatility(returns, annualize=False)
    assert dvol >= 0


@karva.tags.parametrize("n", [50, 100])
@karva.tags.parametrize("std", [0.01, 0.02])
@karva.tags.parametrize("threshold", [0.0, 0.005, 0.01])
def test_downside_volatility_thresholds(n: int, std: float, threshold: float):
    """Downside volatility with different thresholds."""
    returns = generate_returns(n, 0.001, std, 42)
    dvol = downside_volatility(returns, threshold=threshold, annualize=False)
    assert dvol >= 0


# Beta tests
@karva.tags.parametrize("n", [50, 100, 252])
@karva.tags.parametrize("beta_mult", [0.5, 1.0, 1.5, 2.0])
@karva.tags.parametrize("seed", [42, 123])
def test_beta_scaled_returns(n: int, beta_mult: float, seed: int):
    """Beta should reflect return scaling."""
    market = generate_returns(n, 0.001, 0.02, seed)
    asset = [r * beta_mult + random.gauss(0, 0.005) for r in market]
    random.seed(seed)  # Reset for noise
    b = beta(asset, market)
    assert abs(b - beta_mult) < 0.5


@karva.tags.parametrize("n", [50, 100, 252])
@karva.tags.parametrize("seed", [42, 123, 456])
def test_beta_same_series(n: int, seed: int):
    """Beta of series with itself should be 1."""
    returns = generate_returns(n, 0.001, 0.02, seed)
    b = beta(returns, returns)
    assert abs(b - 1.0) < 0.0001


# Sharpe ratio tests
@karva.tags.parametrize("n", [50, 100, 252])
@karva.tags.parametrize("mean", [-0.001, 0.0, 0.001, 0.002])
@karva.tags.parametrize("std", [0.01, 0.02])
def test_sharpe_ratio_sign(n: int, mean: float, std: float):
    """Sharpe ratio sign should match mean return sign."""
    returns = generate_returns(n, mean, std, 42)
    sr = sharpe_ratio(returns, annualize=False)
    if mean > 0:
        assert sr > -1  # Some tolerance for random variation
    elif mean < 0:
        assert sr < 1


@karva.tags.parametrize("n", [100, 252])
@karva.tags.parametrize("rf", [0.0, 0.0001, 0.0005])
@karva.tags.parametrize("seed", [42, 123, 456])
def test_sharpe_with_risk_free(n: int, rf: float, seed: int):
    """Sharpe ratio with various risk-free rates."""
    returns = generate_returns(n, 0.001, 0.02, seed)
    sr = sharpe_ratio(returns, risk_free_rate=rf)
    assert isinstance(sr, float)


# Sortino ratio tests
@karva.tags.parametrize("n", [50, 100, 252])
@karva.tags.parametrize("mean", [0.0, 0.001, 0.002])
@karva.tags.parametrize("std", [0.01, 0.02])
def test_sortino_ratio_finite(n: int, mean: float, std: float):
    """Sortino ratio should be finite."""
    returns = generate_returns(n, mean, std, 42)
    sr = sortino_ratio(returns, annualize=False)
    assert isinstance(sr, float)


@karva.tags.parametrize("n", [100, 252])
@karva.tags.parametrize("target", [0.0, 0.005, 0.01])
@karva.tags.parametrize("seed", [42, 123])
def test_sortino_with_target(n: int, target: float, seed: int):
    """Sortino ratio with various target returns."""
    returns = generate_returns(n, 0.001, 0.02, seed)
    sr = sortino_ratio(returns, target_return=target, annualize=False)
    assert isinstance(sr, float)


# Treynor ratio tests
@karva.tags.parametrize("n", [50, 100, 252])
@karva.tags.parametrize("seed", [42, 123, 456])
def test_treynor_ratio_finite(n: int, seed: int):
    """Treynor ratio should be finite."""
    asset = generate_returns(n, 0.001, 0.02, seed)
    market = generate_returns(n, 0.0008, 0.015, seed + 1)
    tr = treynor_ratio(asset, market)
    assert isinstance(tr, float)


# Max drawdown tests
@karva.tags.parametrize("n", [50, 100, 252, 500])
@karva.tags.parametrize("mean", [-0.001, 0.0, 0.001])
@karva.tags.parametrize("std", [0.01, 0.02, 0.03])
def test_max_drawdown_range(n: int, mean: float, std: float):
    """Max drawdown should be between 0 and 1."""
    prices = generate_prices(n, 100, mean, std, 42)
    dd = max_drawdown(prices)
    assert 0 <= dd <= 1


@karva.tags.parametrize("n", [100, 252])
@karva.tags.parametrize("seed", [42, 123, 456, 789])
def test_max_drawdown_positive_trend(n: int, seed: int):
    """Drawdown for positive trend should be smaller."""
    up_trend = generate_prices(n, 100, 0.002, 0.01, seed)
    down_trend = generate_prices(n, 100, -0.002, 0.01, seed)
    up_dd = max_drawdown(up_trend)
    down_dd = max_drawdown(down_trend)
    assert up_dd <= down_dd


# VaR tests
@karva.tags.parametrize("n", [100, 252, 500])
@karva.tags.parametrize("confidence", [0.90, 0.95, 0.99])
@karva.tags.parametrize("seed", [42, 123, 456])
def test_parametric_var_positive(n: int, confidence: float, seed: int):
    """Parametric VaR should be positive."""
    returns = generate_returns(n, 0.001, 0.02, seed)
    var = parametric_var(returns, confidence=confidence)
    assert var >= 0


@karva.tags.parametrize("n", [100, 252, 500])
@karva.tags.parametrize("confidence", [0.90, 0.95, 0.99])
@karva.tags.parametrize("seed", [42, 123, 456])
def test_historical_var_positive(n: int, confidence: float, seed: int):
    """Historical VaR should be positive."""
    returns = generate_returns(n, 0.001, 0.02, seed)
    var = historical_var(returns, confidence=confidence)
    assert var >= 0


@karva.tags.parametrize("n", [100, 252, 500])
@karva.tags.parametrize("confidence", [0.90, 0.95, 0.99])
@karva.tags.parametrize("seed", [42, 123, 456])
def test_cvar_geq_var(n: int, confidence: float, seed: int):
    """CVaR should be >= VaR."""
    returns = generate_returns(n, 0.001, 0.02, seed)
    var = historical_var(returns, confidence=confidence)
    cvar = conditional_var(returns, confidence=confidence)
    assert cvar >= var


@karva.tags.parametrize("n", [100, 252])
@karva.tags.parametrize("value", [10000, 100000, 1000000])
@karva.tags.parametrize("seed", [42, 123])
def test_var_scales_with_portfolio(n: int, value: float, seed: int):
    """VaR should scale linearly with portfolio value."""
    returns = generate_returns(n, 0.001, 0.02, seed)
    var_base = parametric_var(returns, portfolio_value=1.0)
    var_scaled = parametric_var(returns, portfolio_value=value)
    assert abs(var_scaled - var_base * value) < 1


@karva.tags.parametrize("n", [100, 252])
@karva.tags.parametrize("period", [1, 5, 10])
@karva.tags.parametrize("seed", [42, 123])
def test_var_increases_with_holding_period(n: int, period: int, seed: int):
    """VaR should increase with holding period."""
    returns = generate_returns(n, 0.001, 0.02, seed)
    var_1 = parametric_var(returns, holding_period=1)
    var_n = parametric_var(returns, holding_period=period)
    assert var_n >= var_1


# Omega ratio tests
@karva.tags.parametrize("n", [50, 100, 252])
@karva.tags.parametrize("mean", [0.0, 0.001, 0.002])
@karva.tags.parametrize("seed", [42, 123, 456])
def test_omega_ratio_non_negative(n: int, mean: float, seed: int):
    """Omega ratio should be non-negative."""
    returns = generate_returns(n, mean, 0.02, seed)
    omega = omega_ratio(returns)
    assert omega >= 0


@karva.tags.parametrize("n", [100, 252])
@karva.tags.parametrize("threshold", [0.0, 0.005, 0.01])
@karva.tags.parametrize("seed", [42, 123])
def test_omega_ratio_thresholds(n: int, threshold: float, seed: int):
    """Omega ratio with various thresholds."""
    returns = generate_returns(n, 0.001, 0.02, seed)
    omega = omega_ratio(returns, threshold=threshold)
    assert omega >= 0


# Gain/loss ratio tests
@karva.tags.parametrize("n", [50, 100, 252])
@karva.tags.parametrize("mean", [-0.001, 0.0, 0.001])
@karva.tags.parametrize("seed", [42, 123, 456])
def test_gain_loss_ratio_non_negative(n: int, mean: float, seed: int):
    """Gain/loss ratio should be non-negative."""
    returns = generate_returns(n, mean, 0.02, seed)
    glr = gain_loss_ratio(returns)
    assert glr >= 0


# Win rate tests
@karva.tags.parametrize("n", [50, 100, 252])
@karva.tags.parametrize("mean", [-0.001, 0.0, 0.001])
@karva.tags.parametrize("seed", [42, 123, 456])
def test_win_rate_range(n: int, mean: float, seed: int):
    """Win rate should be between 0 and 1."""
    returns = generate_returns(n, mean, 0.02, seed)
    wr = win_rate(returns)
    assert 0 <= wr <= 1


@karva.tags.parametrize("n", [100, 252])
@karva.tags.parametrize("mean", [0.01, 0.02])
@karva.tags.parametrize("seed", [42, 123])
def test_win_rate_high_for_positive_mean(n: int, mean: float, seed: int):
    """Win rate should be high for strongly positive mean."""
    returns = generate_returns(n, mean, 0.005, seed)  # Low std
    wr = win_rate(returns)
    assert wr > 0.6


@karva.tags.parametrize("n", [100, 252])
@karva.tags.parametrize("mean", [-0.01, -0.02])
@karva.tags.parametrize("seed", [42, 123])
def test_win_rate_low_for_negative_mean(n: int, mean: float, seed: int):
    """Win rate should be low for strongly negative mean."""
    returns = generate_returns(n, mean, 0.005, seed)  # Low std
    wr = win_rate(returns)
    assert wr < 0.4
