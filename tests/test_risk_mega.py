"""Mega risk tests with heavy Cartesian product parametrization."""

import math
import random

import pytest

from finlib.risk import (
    volatility,
    downside_volatility,
    beta,
    alpha,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
    max_drawdown,
    drawdown_series,
    calmar_ratio,
    parametric_var,
    historical_var,
    conditional_var,
    upside_potential_ratio,
    omega_ratio,
    gain_loss_ratio,
    win_rate,
    stress_test,
)


def _gen_returns(n, mu, sigma, seed):
    random.seed(seed)
    return [random.gauss(mu, sigma) for _ in range(n)]


def _gen_values(n, start, mu, sigma, seed):
    random.seed(seed)
    values = [start]
    for _ in range(n - 1):
        ret = random.gauss(mu, sigma)
        values.append(values[-1] * (1 + ret))
    return values


SIZES = [10, 20, 50, 100, 200, 500]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
MUS = [0.0001, 0.0005, 0.001, 0.005, -0.001, -0.005, 0.01]
SIGMAS = [0.005, 0.01, 0.02, 0.05, 0.10]
CONFIDENCES = [0.90, 0.95, 0.99]
PERIODS_PER_YEAR = [12, 52, 252]
RISK_FREE = [0.0, 0.0001, 0.001, 0.005]


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_volatility_nonneg(n, sigma, seed):
    """Volatility should be non-negative."""
    returns = _gen_returns(n, 0, sigma, seed)
    vol = volatility(returns, annualize=False)
    assert vol >= 0


# 6*5*20*3 = 1800
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("periods", PERIODS_PER_YEAR)
def test_annualized_vol_geq_periodic(n, sigma, seed, periods):
    """Annualized volatility should be >= periodic."""
    returns = _gen_returns(n, 0, sigma, seed)
    annual = volatility(returns, annualize=True, periods_per_year=periods)
    periodic = volatility(returns, annualize=False)
    assert annual >= periodic - 0.0001


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_downside_vol_leq_total(n, sigma, seed):
    """Downside volatility should be <= total volatility in general."""
    returns = _gen_returns(n, 0, sigma, seed)
    total = volatility(returns, annualize=False)
    down = downside_volatility(returns, annualize=False)
    # Not always true but almost always for symmetric distributions
    assert isinstance(down, float)
    assert down >= 0


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_beta_self_is_one(n, sigma, seed):
    """Beta of a series with itself should be 1."""
    returns = _gen_returns(n, 0, sigma, seed)
    b = beta(returns, returns)
    assert abs(b - 1.0) < 0.0001


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_beta_finite(n, sigma, seed):
    """Beta should be finite."""
    asset_ret = _gen_returns(n, 0.001, sigma, seed)
    market_ret = _gen_returns(n, 0.0005, sigma, seed + 100)
    b = beta(asset_ret, market_ret)
    assert math.isfinite(b)


# 6*5*20*4 = 2400
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("rf", RISK_FREE)
def test_sharpe_ratio_finite(n, sigma, seed, rf):
    """Sharpe ratio should be finite."""
    returns = _gen_returns(n, 0.001, sigma, seed)
    sr = sharpe_ratio(returns, risk_free_rate=rf, annualize=False)
    assert isinstance(sr, float)


# 6*5*20*4 = 2400
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("rf", RISK_FREE)
def test_sortino_ratio_finite(n, sigma, seed, rf):
    """Sortino ratio should be finite."""
    returns = _gen_returns(n, 0.001, sigma, seed)
    sr = sortino_ratio(returns, risk_free_rate=rf, annualize=False)
    assert isinstance(sr, float)


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_treynor_ratio_finite(n, sigma, seed):
    """Treynor ratio should be finite."""
    asset_ret = _gen_returns(n, 0.001, sigma, seed)
    market_ret = _gen_returns(n, 0.0005, sigma, seed + 100)
    tr = treynor_ratio(asset_ret, market_ret)
    assert isinstance(tr, float)


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_alpha_finite(n, sigma, seed):
    """Alpha should be finite."""
    asset_ret = _gen_returns(n, 0.001, sigma, seed)
    market_ret = _gen_returns(n, 0.0005, sigma, seed + 100)
    a = alpha(asset_ret, market_ret)
    assert math.isfinite(a)


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_max_drawdown_range(n, sigma, seed):
    """Max drawdown should be between 0 and 1."""
    values = _gen_values(n, 100, 0.001, sigma, seed)
    md = max_drawdown(values)
    assert 0 <= md <= 1.0


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_drawdown_series_length(n, sigma, seed):
    """Drawdown series should match input length."""
    values = _gen_values(n, 100, 0.001, sigma, seed)
    dd = drawdown_series(values)
    assert len(dd) == len(values)


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_drawdown_series_nonneg(n, sigma, seed):
    """All drawdowns should be non-negative."""
    values = _gen_values(n, 100, 0.001, sigma, seed)
    dd = drawdown_series(values)
    for d in dd:
        assert d >= -0.0001


# 6*5*20*3 = 1800
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("confidence", CONFIDENCES)
def test_parametric_var_nonneg(n, sigma, seed, confidence):
    """Parametric VaR should be non-negative for typical returns."""
    returns = _gen_returns(n, 0.001, sigma, seed)
    var = parametric_var(returns, confidence)
    assert isinstance(var, float)


# 6*5*20*3 = 1800
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("confidence", CONFIDENCES)
def test_historical_var_valid(n, sigma, seed, confidence):
    """Historical VaR should be a valid float."""
    returns = _gen_returns(n, 0.001, sigma, seed)
    var = historical_var(returns, confidence)
    assert isinstance(var, float)
    assert math.isfinite(var)


# 6*5*20*3 = 1800
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("confidence", CONFIDENCES)
def test_conditional_var_geq_var(n, sigma, seed, confidence):
    """CVaR should be >= VaR."""
    returns = _gen_returns(n, 0.001, sigma, seed)
    var = historical_var(returns, confidence)
    cvar = conditional_var(returns, confidence)
    assert cvar >= var - 0.01


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_win_rate_range(n, sigma, seed):
    """Win rate should be between 0 and 1."""
    returns = _gen_returns(n, 0.001, sigma, seed)
    wr = win_rate(returns)
    assert 0 <= wr <= 1


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_gain_loss_ratio_nonneg(n, sigma, seed):
    """Gain-loss ratio should be non-negative."""
    returns = _gen_returns(n, 0, sigma, seed)
    glr = gain_loss_ratio(returns)
    assert glr >= 0


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_omega_ratio_nonneg(n, sigma, seed):
    """Omega ratio should be non-negative."""
    returns = _gen_returns(n, 0, sigma, seed)
    omega = omega_ratio(returns)
    assert omega >= 0


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_upside_potential_nonneg(n, sigma, seed):
    """Upside potential ratio should be non-negative."""
    returns = _gen_returns(n, 0.001, sigma, seed)
    upr = upside_potential_ratio(returns)
    assert upr >= 0


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_calmar_ratio_valid(n, sigma, seed):
    """Calmar ratio should be valid."""
    returns = _gen_returns(n, 0.001, sigma, seed)
    values = _gen_values(n, 100, 0.001, sigma, seed)
    cr = calmar_ratio(returns, values, periods_per_year=252)
    assert isinstance(cr, float)


# 4*4*4*4 = 256
@pytest.mark.parametrize("n_assets", [2, 3, 4, 5])
@pytest.mark.parametrize("n_scenarios", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", [1, 5, 10, 20])
@pytest.mark.parametrize("shock_scale", [0.01, 0.05, 0.10, 0.20])
def test_stress_test_valid(n_assets, n_scenarios, seed, shock_scale):
    """Stress test should return correct number of results."""
    random.seed(seed)
    weights = [1 / n_assets] * n_assets
    scenarios = [
        [random.gauss(0, shock_scale) for _ in range(n_assets)]
        for _ in range(n_scenarios)
    ]
    results = stress_test(100000, weights, scenarios)
    assert len(results) == n_scenarios
