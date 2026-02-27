"""Mega portfolio tests with heavy Cartesian product parametrization."""

import math
import random

import pytest

from finlib.portfolio import (
    simple_return,
    log_return,
    holding_period_return,
    annualized_return,
    cumulative_return,
    time_weighted_return,
    portfolio_weights,
    portfolio_return,
    portfolio_variance,
    portfolio_std,
    rebalance_portfolio,
    benchmark_comparison,
    geometric_mean_return,
    arithmetic_mean_return,
    expected_return,
    diversification_ratio,
    contribution_to_risk,
)


def _gen_returns(n, mu, sigma, seed):
    random.seed(seed)
    return [random.gauss(mu, sigma) for _ in range(n)]


def _gen_positive_values(n, base, spread, seed):
    random.seed(seed)
    return [base + random.uniform(0, spread) for _ in range(n)]


START_VALS = [50, 75, 100, 125, 150, 200, 300, 500]
END_VALS = [40, 60, 80, 100, 120, 150, 200, 300, 400, 500]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
SIZES = [5, 10, 20, 50, 100, 200]
MUS = [0.0001, 0.0005, 0.001, 0.005, 0.01, -0.001, -0.005]
SIGMAS = [0.005, 0.01, 0.02, 0.05, 0.10]
N_ASSETS = [2, 3, 4, 5, 6]
YEARS = [0.5, 1, 2, 3, 5, 10]


# 8*10 = 80
@pytest.mark.parametrize("start", START_VALS)
@pytest.mark.parametrize("end", END_VALS)
def test_simple_return_formula(start, end):
    """Simple return should equal (end - start) / start."""
    sr = simple_return(start, end)
    expected = (end - start) / start
    assert abs(sr - expected) < 0.0001


# 8*10 = 80
@pytest.mark.parametrize("start", START_VALS)
@pytest.mark.parametrize("end", END_VALS)
def test_log_return_formula(start, end):
    """Log return should equal ln(end/start)."""
    if end <= 0:
        return
    lr = log_return(start, end)
    expected = math.log(end / start)
    assert abs(lr - expected) < 0.0001


# 8*10 = 80
@pytest.mark.parametrize("start", START_VALS)
@pytest.mark.parametrize("end", END_VALS)
def test_simple_geq_log_return(start, end):
    """Simple return >= log return for positive values."""
    if end <= 0:
        return
    sr = simple_return(start, end)
    lr = log_return(start, end)
    assert sr >= lr - 0.0001


# 6*7*5*20 = 4200
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("mu", MUS)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_cumulative_return_valid(n, mu, sigma, seed):
    """Cumulative return should be a valid float."""
    returns = _gen_returns(n, mu, sigma, seed)
    cr = cumulative_return(returns)
    assert isinstance(cr, float)
    assert math.isfinite(cr)


# 6*7*5*20 = 4200
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("mu", MUS)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_geometric_mean_return_valid(n, mu, sigma, seed):
    """Geometric mean return should be a valid float."""
    returns = _gen_returns(n, mu, sigma, seed)
    if any(r <= -1 for r in returns):
        return
    gmr = geometric_mean_return(returns)
    assert isinstance(gmr, float)
    assert math.isfinite(gmr)


# 6*7*5*20 = 4200
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("mu", MUS)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_arithmetic_mean_return_valid(n, mu, sigma, seed):
    """Arithmetic mean return should be a valid float."""
    returns = _gen_returns(n, mu, sigma, seed)
    amr = arithmetic_mean_return(returns)
    assert isinstance(amr, float)
    assert math.isfinite(amr)


# 6*5*20 = 600
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_holding_period_return_valid(n, sigma, seed):
    """HPR should be valid."""
    values = _gen_positive_values(n, 100, 50, seed)
    hpr = holding_period_return(values)
    assert isinstance(hpr, float)
    assert math.isfinite(hpr)


# 4*6 = 24
@pytest.mark.parametrize("total_ret", [0.05, 0.10, 0.50, 1.0])
@pytest.mark.parametrize("years", YEARS)
def test_annualized_return_valid(total_ret, years):
    """Annualized return should be valid."""
    ar = annualized_return(total_ret, years)
    assert isinstance(ar, float)
    assert math.isfinite(ar)


# 5*20 = 100
@pytest.mark.parametrize("n_assets", N_ASSETS)
@pytest.mark.parametrize("seed", SEEDS)
def test_portfolio_weights_sum_one(n_assets, seed):
    """Portfolio weights should sum to 1."""
    values = _gen_positive_values(n_assets, 1000, 5000, seed)
    weights = portfolio_weights(values)
    assert abs(sum(weights) - 1.0) < 0.0001


# 5*20*5 = 500
@pytest.mark.parametrize("n_assets", N_ASSETS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("ret_sigma", SIGMAS)
def test_portfolio_return_valid(n_assets, seed, ret_sigma):
    """Portfolio return should be valid."""
    random.seed(seed)
    weights = [1 / n_assets] * n_assets
    returns = [random.gauss(0.001, ret_sigma) for _ in range(n_assets)]
    pr = portfolio_return(weights, returns)
    assert isinstance(pr, float)
    assert math.isfinite(pr)


# 5*20*5 = 500
@pytest.mark.parametrize("n_assets", N_ASSETS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("vol_scale", [0.01, 0.02, 0.05, 0.10, 0.20])
def test_portfolio_variance_nonneg(n_assets, seed, vol_scale):
    """Portfolio variance should be non-negative."""
    random.seed(seed)
    weights = [1 / n_assets] * n_assets
    cov = [[0.0] * n_assets for _ in range(n_assets)]
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                cov[i][j] = (vol_scale * (1 + random.random())) ** 2
            else:
                cov[i][j] = vol_scale**2 * 0.3 * random.random()
                cov[j][i] = cov[i][j]
    pv = portfolio_variance(weights, cov)
    assert pv >= -0.0001


# 5*20*5 = 500
@pytest.mark.parametrize("n_assets", N_ASSETS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("vol_scale", [0.01, 0.02, 0.05, 0.10, 0.20])
def test_portfolio_std_nonneg(n_assets, seed, vol_scale):
    """Portfolio std should be non-negative."""
    random.seed(seed)
    weights = [1 / n_assets] * n_assets
    cov = [[0.0] * n_assets for _ in range(n_assets)]
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                cov[i][j] = (vol_scale * (1 + random.random())) ** 2
            else:
                cov[i][j] = vol_scale**2 * 0.3 * random.random()
                cov[j][i] = cov[i][j]
    ps = portfolio_std(weights, cov)
    assert ps >= 0


# 5*20 = 100
@pytest.mark.parametrize("n_assets", N_ASSETS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rebalance_sums_zero(n_assets, seed):
    """Rebalance trades should sum to zero."""
    current = _gen_positive_values(n_assets, 1000, 5000, seed)
    target = [1 / n_assets] * n_assets
    trades = rebalance_portfolio(current, target)
    assert abs(sum(trades)) < 0.01


# 5*20*5 = 500
@pytest.mark.parametrize("n", [10, 20, 50, 100, 200])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("sigma", SIGMAS)
def test_benchmark_comparison_valid(n, seed, sigma):
    """Benchmark comparison should return valid metrics."""
    port_ret = _gen_returns(n, 0.001, sigma, seed)
    bench_ret = _gen_returns(n, 0.0008, sigma, seed + 1000)
    result = benchmark_comparison(port_ret, bench_ret)
    assert isinstance(result["portfolio_return"], float)
    assert isinstance(result["benchmark_return"], float)
    assert isinstance(result["tracking_error"], float)
    assert result["tracking_error"] >= 0


# 6*7*5*20 = 4200
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("mu", MUS)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("seed", SEEDS)
def test_time_weighted_return_valid(n, mu, sigma, seed):
    """TWR should be a valid float."""
    values = _gen_positive_values(n, 100, 50, seed)
    twr = time_weighted_return(values)
    assert isinstance(twr, float)
    assert math.isfinite(twr)


# 5*20*5 = 500
@pytest.mark.parametrize("n_assets", N_ASSETS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("vol_scale", [0.01, 0.02, 0.05, 0.10, 0.20])
def test_contribution_to_risk_sums_to_variance(n_assets, seed, vol_scale):
    """Risk contributions should sum to portfolio variance."""
    random.seed(seed)
    weights = [1 / n_assets] * n_assets
    cov = [[0.0] * n_assets for _ in range(n_assets)]
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                cov[i][j] = (vol_scale * (1 + random.random())) ** 2
            else:
                cov[i][j] = vol_scale**2 * 0.3 * random.random()
                cov[j][i] = cov[i][j]
    contributions = contribution_to_risk(weights, cov)
    pv = portfolio_variance(weights, cov)
    assert abs(sum(contributions) - pv) < 0.001


# 5*20*5 = 500
@pytest.mark.parametrize("n_assets", N_ASSETS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("ret_sigma", SIGMAS)
def test_expected_return_equals_portfolio_return(n_assets, seed, ret_sigma):
    """Expected return should equal portfolio return for same inputs."""
    random.seed(seed)
    weights = [1 / n_assets] * n_assets
    exp_rets = [random.gauss(0.001, ret_sigma) for _ in range(n_assets)]
    er = expected_return(weights, exp_rets)
    pr = portfolio_return(weights, exp_rets)
    assert abs(er - pr) < 0.0001
