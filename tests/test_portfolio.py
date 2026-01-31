"""Tests for portfolio analysis calculations."""

import math

import karva
from karva import fixture

from finlib.portfolio import (
    simple_return,
    log_return,
    holding_period_return,
    annualized_return,
    cumulative_return,
    time_weighted_return,
    money_weighted_return,
    portfolio_weights,
    portfolio_return,
    portfolio_variance,
    portfolio_std,
    rebalance_portfolio,
    benchmark_comparison,
    geometric_mean_return,
    arithmetic_mean_return,
    expected_return,
    minimum_variance_weights,
    diversification_ratio,
    contribution_to_risk,
)


# Simple Return Tests
@karva.tags.parametrize("start,end,expected", [
    (100, 110, 0.10),
    (100, 90, -0.10),
    (50, 75, 0.50),
    (200, 200, 0.00),
    (100, 150, 0.50),
])
def test_simple_return_basic(start: float, end: float, expected: float):
    result = simple_return(start, end)
    assert abs(result - expected) < 0.0001


def test_simple_return_zero_start():
    try:
        simple_return(0, 100)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


@karva.tags.parametrize("start", [10, 100, 1000, 10000])
def test_simple_return_doubling(start: float):
    result = simple_return(start, start * 2)
    assert abs(result - 1.0) < 0.0001


# Log Return Tests
@karva.tags.parametrize("start,end,expected", [
    (100, 110, 0.09531),
    (100, 90, -0.10536),
    (50, 75, 0.40546),
    (100, 100, 0.00),
])
def test_log_return_basic(start: float, end: float, expected: float):
    result = log_return(start, end)
    assert abs(result - expected) < 0.001


def test_log_return_negative_value():
    try:
        log_return(100, -50)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_log_return_vs_simple_return():
    start, end = 100, 105
    simple = simple_return(start, end)
    log = log_return(start, end)
    # Log return should be slightly less for positive returns
    assert log < simple


@karva.tags.parametrize("multiplier", [1.01, 1.05, 1.10, 1.50, 2.0])
def test_log_return_formula(multiplier: float):
    start = 100
    end = start * multiplier
    result = log_return(start, end)
    expected = math.log(multiplier)
    assert abs(result - expected) < 0.0001


# Holding Period Return Tests
def test_holding_period_return_basic(sample_portfolio_values):
    result = holding_period_return(sample_portfolio_values)
    expected = (sample_portfolio_values[-1] - sample_portfolio_values[0]) / sample_portfolio_values[0]
    assert abs(result - expected) < 0.0001


def test_holding_period_return_insufficient_values():
    try:
        holding_period_return([100])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


@karva.tags.parametrize("values", [
    [100, 110],
    [100, 105, 110],
    [100, 95, 105, 110, 120],
])
def test_holding_period_return_various_lengths(values: list):
    result = holding_period_return(values)
    expected = (values[-1] - values[0]) / values[0]
    assert abs(result - expected) < 0.0001


# Annualized Return Tests
@karva.tags.parametrize("total_return,years,expected", [
    (0.10, 1, 0.10),
    (0.21, 2, 0.10),
    (0.6105, 5, 0.10),
    (1.0, 7.27, 0.10),
])
def test_annualized_return(total_return: float, years: float, expected: float):
    result = annualized_return(total_return, years)
    assert abs(result - expected) < 0.01


def test_annualized_return_zero_years():
    try:
        annualized_return(0.10, 0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Cumulative Return Tests
@karva.tags.parametrize("returns,expected", [
    ([0.10], 0.10),
    ([0.10, 0.10], 0.21),
    ([0.10, -0.10], -0.01),
    ([0.05, 0.05, 0.05], 0.1576),
])
def test_cumulative_return(returns: list, expected: float):
    result = cumulative_return(returns)
    assert abs(result - expected) < 0.001


def test_cumulative_return_empty():
    result = cumulative_return([])
    assert result == 0.0


def test_cumulative_return_single():
    result = cumulative_return([0.15])
    assert abs(result - 0.15) < 0.0001


# Time-Weighted Return Tests
def test_twrr_basic(sample_portfolio_values):
    result = time_weighted_return(sample_portfolio_values)
    assert result > 0  # Portfolio went up


def test_twrr_with_cash_flows():
    values = [10000, 10500, 11200, 11000, 11800]
    cash_flows = [0, 500, 0, -200]
    result = time_weighted_return(values, cash_flows)
    assert isinstance(result, float)


def test_twrr_no_cash_flows_equals_hpr():
    values = [100, 105, 110, 108, 115]
    twrr = time_weighted_return(values)
    hpr = holding_period_return(values)
    assert abs(twrr - hpr) < 0.01


@karva.tags.parametrize("values", [
    [100, 110, 120],
    [1000, 1050, 1100, 1150],
    [500, 525, 550, 525, 550],
])
def test_twrr_various_series(values: list):
    result = time_weighted_return(values)
    assert isinstance(result, float)


# Money-Weighted Return Tests
def test_mwrr_basic():
    cash_flows = [-1000, -500, 200]
    values = [0, 0, 1800]
    result = money_weighted_return(cash_flows, values)
    assert isinstance(result, float)


def test_mwrr_single_investment():
    cash_flows = [-1000]
    values = [1100]
    result = money_weighted_return(cash_flows, values)
    assert abs(result - 0.10) < 0.01


# Portfolio Weights Tests
@karva.tags.parametrize("values,expected", [
    ([100, 100, 100, 100], [0.25, 0.25, 0.25, 0.25]),
    ([60, 40], [0.60, 0.40]),
    ([50, 30, 20], [0.50, 0.30, 0.20]),
])
def test_portfolio_weights(values: list, expected: list):
    result = portfolio_weights(values)
    for r, e in zip(result, expected):
        assert abs(r - e) < 0.0001


def test_portfolio_weights_sum_to_one():
    values = [1234, 5678, 9012, 3456]
    weights = portfolio_weights(values)
    assert abs(sum(weights) - 1.0) < 0.0001


def test_portfolio_weights_zero_total():
    try:
        portfolio_weights([0, 0, 0])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Portfolio Return Tests
@karva.tags.parametrize("weights,returns,expected", [
    ([0.5, 0.5], [0.10, 0.20], 0.15),
    ([0.6, 0.4], [0.10, 0.05], 0.08),
    ([0.25, 0.25, 0.25, 0.25], [0.10, 0.05, 0.15, 0.08], 0.095),
])
def test_portfolio_return(weights: list, returns: list, expected: float):
    result = portfolio_return(weights, returns)
    assert abs(result - expected) < 0.0001


def test_portfolio_return_mismatched_lengths():
    try:
        portfolio_return([0.5, 0.5], [0.10])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Portfolio Variance Tests
def test_portfolio_variance_basic(sample_weights, sample_covariance_matrix):
    result = portfolio_variance(sample_weights, sample_covariance_matrix)
    assert result > 0


def test_portfolio_variance_single_asset():
    weights = [1.0]
    cov = [[0.04]]
    result = portfolio_variance(weights, cov)
    assert abs(result - 0.04) < 0.0001


def test_portfolio_variance_equal_weights():
    weights = [0.5, 0.5]
    cov = [[0.04, 0.01], [0.01, 0.09]]
    result = portfolio_variance(weights, cov)
    # w1^2*var1 + w2^2*var2 + 2*w1*w2*cov12
    expected = 0.25 * 0.04 + 0.25 * 0.09 + 2 * 0.25 * 0.01
    assert abs(result - expected) < 0.0001


# Portfolio Std Tests
def test_portfolio_std_basic(sample_weights, sample_covariance_matrix):
    var = portfolio_variance(sample_weights, sample_covariance_matrix)
    std = portfolio_std(sample_weights, sample_covariance_matrix)
    assert abs(std - math.sqrt(var)) < 0.0001


# Rebalance Portfolio Tests
def test_rebalance_portfolio_basic():
    current = [6000, 4000]
    target = [0.5, 0.5]
    trades = rebalance_portfolio(current, target)
    assert abs(trades[0] - (-1000)) < 0.01
    assert abs(trades[1] - 1000) < 0.01


def test_rebalance_portfolio_no_change():
    current = [5000, 5000]
    target = [0.5, 0.5]
    trades = rebalance_portfolio(current, target)
    for t in trades:
        assert abs(t) < 0.01


def test_rebalance_portfolio_invalid_weights():
    try:
        rebalance_portfolio([5000, 5000], [0.5, 0.6])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


@karva.tags.parametrize("current,target", [
    ([4000, 3000, 2000, 1000], [0.25, 0.25, 0.25, 0.25]),
    ([10000, 0, 0], [0.5, 0.3, 0.2]),
])
def test_rebalance_various_portfolios(current: list, target: list):
    trades = rebalance_portfolio(current, target)
    assert abs(sum(trades)) < 0.01  # Net zero trades


# Benchmark Comparison Tests
def test_benchmark_comparison_basic(sample_returns, market_returns):
    result = benchmark_comparison(sample_returns, market_returns)
    assert "portfolio_return" in result
    assert "benchmark_return" in result
    assert "excess_return" in result
    assert "tracking_error" in result
    assert "information_ratio" in result


def test_benchmark_comparison_outperformance():
    portfolio = [0.02, 0.01, 0.03, 0.02, 0.01]
    benchmark = [0.01, 0.005, 0.02, 0.01, 0.005]
    result = benchmark_comparison(portfolio, benchmark)
    assert result["excess_return"] > 0


def test_benchmark_comparison_mismatched_lengths():
    try:
        benchmark_comparison([0.01, 0.02], [0.01])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Geometric Mean Return Tests
@karva.tags.parametrize("returns,expected", [
    ([0.10, 0.10, 0.10], 0.10),
    ([0.20, -0.10, 0.15], 0.0733),
    ([0.05, 0.05, 0.05, 0.05], 0.05),
])
def test_geometric_mean_return(returns: list, expected: float):
    result = geometric_mean_return(returns)
    assert abs(result - expected) < 0.01


def test_geometric_mean_empty():
    try:
        geometric_mean_return([])
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_geometric_mean_less_than_arithmetic():
    returns = [0.20, -0.10, 0.15, 0.05]
    geo = geometric_mean_return(returns)
    arith = arithmetic_mean_return(returns)
    assert geo <= arith


# Arithmetic Mean Return Tests
def test_arithmetic_mean_return(sample_returns):
    result = arithmetic_mean_return(sample_returns)
    expected = sum(sample_returns) / len(sample_returns)
    assert abs(result - expected) < 0.0001


# Expected Return Tests
def test_expected_return_basic():
    weights = [0.4, 0.3, 0.3]
    expected_rets = [0.12, 0.08, 0.10]
    result = expected_return(weights, expected_rets)
    expected = 0.4 * 0.12 + 0.3 * 0.08 + 0.3 * 0.10
    assert abs(result - expected) < 0.0001


# Minimum Variance Weights Tests
def test_minimum_variance_weights_basic(sample_covariance_matrix):
    weights = minimum_variance_weights(sample_covariance_matrix)
    assert abs(sum(weights) - 1.0) < 0.01
    assert all(w >= 0 for w in weights)


def test_minimum_variance_weights_2x2():
    cov = [[0.04, 0.01], [0.01, 0.09]]
    weights = minimum_variance_weights(cov)
    assert abs(sum(weights) - 1.0) < 0.01


# Diversification Ratio Tests
def test_diversification_ratio_basic():
    weights = [0.5, 0.5]
    volatilities = [0.20, 0.30]
    port_vol = 0.20  # Less than weighted avg due to diversification
    result = diversification_ratio(weights, volatilities, port_vol)
    weighted_avg = 0.5 * 0.20 + 0.5 * 0.30
    assert abs(result - weighted_avg / port_vol) < 0.01


def test_diversification_ratio_no_diversification():
    weights = [1.0]
    volatilities = [0.20]
    port_vol = 0.20
    result = diversification_ratio(weights, volatilities, port_vol)
    assert abs(result - 1.0) < 0.0001


# Contribution to Risk Tests
def test_contribution_to_risk_basic(sample_weights, sample_covariance_matrix):
    contributions = contribution_to_risk(sample_weights, sample_covariance_matrix)
    total_var = portfolio_variance(sample_weights, sample_covariance_matrix)
    assert abs(sum(contributions) - total_var) < 0.0001


def test_contribution_to_risk_equal_weights():
    weights = [0.5, 0.5]
    cov = [[0.04, 0.02], [0.02, 0.04]]  # Same variance, same correlation
    contributions = contribution_to_risk(weights, cov)
    # Equal contributions for symmetric case
    assert abs(contributions[0] - contributions[1]) < 0.0001
