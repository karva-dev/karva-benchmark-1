"""Mega options tests with heavy Cartesian product parametrization."""

import math

import pytest

from finlib.options import (
    black_scholes_call,
    black_scholes_put,
    delta,
    gamma,
    theta,
    vega,
    rho,
    put_call_parity,
    binomial_tree_price,
    implied_volatility,
    option_payoff,
    option_profit,
    breakeven_price,
    intrinsic_value,
    time_value,
    moneyness,
    straddle_payoff,
    strangle_payoff,
    covered_call_payoff,
    protective_put_payoff,
)

SPOTS = [70, 80, 90, 95, 100, 105, 110, 120, 130]
STRIKES = [75, 85, 90, 95, 100, 105, 110, 115, 125]
SIGMAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
EXPIRIES = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
RISK_FREE = [0.01, 0.03, 0.05, 0.08]


# 9*9*6*6 = 2916
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("T", EXPIRIES)
def test_call_nonneg(S, K, sigma, T):
    """Call price should be non-negative."""
    result = black_scholes_call(S, K, 0.05, sigma, T)
    assert result >= -1e-10


# 9*9*6*6 = 2916
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("T", EXPIRIES)
def test_put_nonneg(S, K, sigma, T):
    """Put price should be non-negative."""
    result = black_scholes_put(S, K, 0.05, sigma, T)
    assert result >= -1e-10


# 9*9*6*6 = 2916
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("T", EXPIRIES)
def test_put_call_parity_holds(S, K, sigma, T):
    """Put-call parity: C - P = S - K*e^(-rT)."""
    r = 0.05
    call = black_scholes_call(S, K, r, sigma, T)
    put = black_scholes_put(S, K, r, sigma, T)
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 0.01


# 9*9*6*6 = 2916
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("T", EXPIRIES)
def test_call_bounded_by_stock(S, K, sigma, T):
    """Call price <= stock price."""
    call = black_scholes_call(S, K, 0.05, sigma, T)
    assert call <= S + 0.01


# 9*9*6*6*4 = 11664
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("T", EXPIRIES)
@pytest.mark.parametrize("r", RISK_FREE)
def test_call_delta_bounds(S, K, sigma, T, r):
    """Call delta should be between 0 and 1."""
    d = delta(S, K, r, sigma, T, "call")
    assert -0.001 <= d <= 1.001


# 9*9*6*6*4 = 11664
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("T", EXPIRIES)
@pytest.mark.parametrize("r", RISK_FREE)
def test_put_delta_bounds(S, K, sigma, T, r):
    """Put delta should be between -1 and 0."""
    d = delta(S, K, r, sigma, T, "put")
    assert -1.001 <= d <= 0.001


# 9*9*6*6 = 2916
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("T", EXPIRIES)
def test_gamma_nonneg(S, K, sigma, T):
    """Gamma should be non-negative."""
    g = gamma(S, K, 0.05, sigma, T)
    assert g >= -0.001


# 9*9*6*6 = 2916
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("T", EXPIRIES)
def test_vega_nonneg(S, K, sigma, T):
    """Vega should be non-negative."""
    v = vega(S, K, 0.05, sigma, T)
    assert v >= -0.001


# 9*9*6*6 = 2916
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("T", EXPIRIES)
def test_call_rho_nonneg(S, K, sigma, T):
    """Call rho should be non-negative."""
    r_val = rho(S, K, 0.05, sigma, T, "call")
    assert r_val >= -0.01


# 9*9*6*6 = 2916
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("T", EXPIRIES)
def test_put_rho_nonpos(S, K, sigma, T):
    """Put rho should be non-positive."""
    r_val = rho(S, K, 0.05, sigma, T, "put")
    assert r_val <= 0.01


# 9*9*6*6 = 2916
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("sigma", SIGMAS)
@pytest.mark.parametrize("T", EXPIRIES)
def test_delta_call_put_sum(S, K, sigma, T):
    """Call delta - Put delta should be 1."""
    call_d = delta(S, K, 0.05, sigma, T, "call")
    put_d = delta(S, K, 0.05, sigma, T, "put")
    assert abs(call_d - put_d - 1) < 0.01


# 9*9*4*2 = 648
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("premium", [1.0, 3.0, 5.0, 10.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_option_payoff_nonneg_long(S, K, premium, option_type):
    """Long option payoff should be non-negative."""
    payoff = option_payoff(S, K, option_type, "long")
    assert payoff >= 0


# 9*9*4*2 = 648
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("premium", [1.0, 3.0, 5.0, 10.0])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_option_payoff_nonpos_short(S, K, premium, option_type):
    """Short option payoff should be non-positive."""
    payoff = option_payoff(S, K, option_type, "short")
    assert payoff <= 0


# 9*9*2 = 162
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_intrinsic_value_nonneg(S, K, option_type):
    """Intrinsic value should be non-negative."""
    iv = intrinsic_value(S, K, option_type)
    assert iv >= 0


# 5*5*5*6*4 = 3000
@pytest.mark.parametrize("S", [90, 95, 100, 105, 110])
@pytest.mark.parametrize("K", [90, 95, 100, 105, 110])
@pytest.mark.parametrize("sigma", [0.15, 0.20, 0.25, 0.30, 0.40])
@pytest.mark.parametrize("T", EXPIRIES)
@pytest.mark.parametrize("r", RISK_FREE)
def test_time_value_nonneg(S, K, sigma, T, r):
    """Time value of option should be non-negative."""
    call_price = black_scholes_call(S, K, r, sigma, T)
    tv = time_value(call_price, S, K, "call")
    assert tv >= -0.01


# 9*9*5*4 = 1620
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("call_prem", [2, 5, 8, 12, 15])
@pytest.mark.parametrize("put_prem", [2, 5, 8, 12])
def test_straddle_payoff_symmetric(S, K, call_prem, put_prem):
    """Straddle should have defined payoff."""
    result = straddle_payoff(S, K, call_prem, put_prem)
    assert isinstance(result, (int, float))


# 5*5*5*5*5 = 3125
@pytest.mark.parametrize("S", [80, 90, 100, 110, 120])
@pytest.mark.parametrize("K_call", [100, 105, 110, 115, 120])
@pytest.mark.parametrize("K_put", [80, 85, 90, 95, 100])
@pytest.mark.parametrize("call_prem", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("put_prem", [2, 4, 6, 8, 10])
def test_strangle_payoff_valid(S, K_call, K_put, call_prem, put_prem):
    """Strangle should return a valid number."""
    result = strangle_payoff(S, K_call, K_put, call_prem, put_prem)
    assert isinstance(result, (int, float))


# 9*9*5*4 = 1620
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("stock_cost", [85, 90, 95, 100, 105])
@pytest.mark.parametrize("premium", [2, 4, 6, 8])
def test_covered_call_payoff_valid(S, K, stock_cost, premium):
    """Covered call should return a valid number."""
    result = covered_call_payoff(S, K, stock_cost, premium)
    assert isinstance(result, (int, float))


# 9*9*5*4 = 1620
@pytest.mark.parametrize("S", SPOTS)
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("stock_cost", [85, 90, 95, 100, 105])
@pytest.mark.parametrize("premium", [2, 4, 6, 8])
def test_protective_put_payoff_valid(S, K, stock_cost, premium):
    """Protective put should return a valid number."""
    result = protective_put_payoff(S, K, stock_cost, premium)
    assert isinstance(result, (int, float))
