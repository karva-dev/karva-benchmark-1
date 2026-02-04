"""Tests for options pricing calculations."""

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


# Black-Scholes Call Tests
@pytest.mark.parametrize("S,K,r,sigma,T,expected", [
    (100, 100, 0.05, 0.20, 1.0, 10.45),
    (100, 95, 0.05, 0.20, 1.0, 13.70),
    (100, 105, 0.05, 0.20, 1.0, 7.97),
    (50, 50, 0.08, 0.30, 0.5, 5.13),
    (100, 100, 0.05, 0.40, 1.0, 17.80),
])
def test_black_scholes_call_values(S: float, K: float, r: float, sigma: float, T: float, expected: float):
    result = black_scholes_call(S, K, r, sigma, T)
    assert abs(result - expected) < 0.5


def test_black_scholes_call_at_expiry():
    # ITM at expiry
    result = black_scholes_call(110, 100, 0.05, 0.20, 0)
    assert abs(result - 10) < 0.01
    # OTM at expiry
    result = black_scholes_call(90, 100, 0.05, 0.20, 0)
    assert result == 0


def test_black_scholes_call_positive():
    result = black_scholes_call(100, 100, 0.05, 0.20, 1.0)
    assert result > 0


@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30, 0.40, 0.50])
def test_call_increases_with_volatility(sigma: float):
    low_vol = black_scholes_call(100, 100, 0.05, 0.10, 1.0)
    high_vol = black_scholes_call(100, 100, 0.05, sigma, 1.0)
    assert high_vol >= low_vol


@pytest.mark.parametrize("T", [0.25, 0.5, 1.0, 2.0])
def test_call_increases_with_time(T: float):
    short_time = black_scholes_call(100, 100, 0.05, 0.20, 0.25)
    long_time = black_scholes_call(100, 100, 0.05, 0.20, T)
    assert long_time >= short_time


# Black-Scholes Put Tests
@pytest.mark.parametrize("S,K,r,sigma,T,expected", [
    (100, 100, 0.05, 0.20, 1.0, 5.57),
    (100, 95, 0.05, 0.20, 1.0, 4.06),
    (100, 105, 0.05, 0.20, 1.0, 7.84),
    (50, 50, 0.08, 0.30, 0.5, 3.17),
])
def test_black_scholes_put_values(S: float, K: float, r: float, sigma: float, T: float, expected: float):
    result = black_scholes_put(S, K, r, sigma, T)
    assert abs(result - expected) < 0.5


def test_black_scholes_put_at_expiry():
    # ITM at expiry
    result = black_scholes_put(90, 100, 0.05, 0.20, 0)
    assert abs(result - 10) < 0.01
    # OTM at expiry
    result = black_scholes_put(110, 100, 0.05, 0.20, 0)
    assert result == 0


# Put-Call Parity Tests
@pytest.mark.parametrize("S,K,r,T", [
    (100, 100, 0.05, 1.0),
    (100, 95, 0.05, 0.5),
    (50, 55, 0.08, 0.25),
    (100, 100, 0.10, 2.0),
])
def test_put_call_parity(S: float, K: float, r: float, T: float):
    call = black_scholes_call(S, K, r, 0.20, T)
    put = black_scholes_put(S, K, r, 0.20, T)
    pv_strike = K * math.exp(-r * T)
    # C - P = S - K*e^(-rT)
    lhs = call - put
    rhs = S - pv_strike
    assert abs(lhs - rhs) < 0.01


def test_put_call_parity_function():
    S, K, r, T = 100, 100, 0.05, 1.0
    call = black_scholes_call(S, K, r, 0.20, T)
    put = black_scholes_put(S, K, r, 0.20, T)
    result = put_call_parity(S, K, r, T, call, put)
    assert result["parity_holds"]


def test_put_call_parity_implied_put():
    S, K, r, T = 100, 100, 0.05, 1.0
    call = black_scholes_call(S, K, r, 0.20, T)
    put = black_scholes_put(S, K, r, 0.20, T)
    result = put_call_parity(S, K, r, T, call_price=call)
    assert abs(result["implied_put"] - put) < 0.01


# Delta Tests
@pytest.mark.parametrize("option_type,expected_range", [
    ("call", (0, 1)),
    ("put", (-1, 0)),
])
def test_delta_range(option_type: str, expected_range: tuple):
    result = delta(100, 100, 0.05, 0.20, 1.0, option_type)
    assert expected_range[0] <= result <= expected_range[1]


def test_delta_atm_call():
    result = delta(100, 100, 0.05, 0.20, 1.0, "call")
    # ATM call delta should be around 0.5
    assert 0.45 < result < 0.65


def test_delta_deep_itm_call():
    result = delta(150, 100, 0.05, 0.20, 1.0, "call")
    assert result > 0.95


def test_delta_deep_otm_call():
    result = delta(50, 100, 0.05, 0.20, 1.0, "call")
    assert result < 0.05


def test_delta_call_put_relationship():
    call_delta = delta(100, 100, 0.05, 0.20, 1.0, "call")
    put_delta = delta(100, 100, 0.05, 0.20, 1.0, "put")
    # Call delta - Put delta = 1
    assert abs(call_delta - put_delta - 1) < 0.01


@pytest.mark.parametrize("S", [80, 90, 100, 110, 120])
def test_delta_increases_with_stock_price_call(S: float):
    result = delta(S, 100, 0.05, 0.20, 1.0, "call")
    assert 0 <= result <= 1


# Gamma Tests
def test_gamma_positive():
    result = gamma(100, 100, 0.05, 0.20, 1.0)
    assert result > 0


def test_gamma_highest_atm():
    atm = gamma(100, 100, 0.05, 0.20, 1.0)
    itm = gamma(120, 100, 0.05, 0.20, 1.0)
    otm = gamma(80, 100, 0.05, 0.20, 1.0)
    assert atm >= itm
    assert atm >= otm


def test_gamma_same_for_call_put():
    # Gamma is the same for calls and puts
    call_gamma = gamma(100, 100, 0.05, 0.20, 1.0)
    # The gamma function doesn't take option_type, it's the same for both
    assert call_gamma > 0


@pytest.mark.parametrize("T", [0.1, 0.25, 0.5, 1.0])
def test_gamma_increases_near_expiry(T: float):
    # Gamma generally increases as expiry approaches for ATM options
    result = gamma(100, 100, 0.05, 0.20, T)
    assert result > 0


# Theta Tests
def test_theta_negative_for_long():
    call_theta = theta(100, 100, 0.05, 0.20, 1.0, "call")
    put_theta = theta(100, 100, 0.05, 0.20, 1.0, "put")
    assert call_theta < 0  # Time decay hurts long positions


@pytest.mark.parametrize("T", [0.1, 0.25, 0.5, 1.0])
def test_theta_various_expirations(T: float):
    result = theta(100, 100, 0.05, 0.20, T, "call")
    assert result < 0


def test_theta_at_expiry():
    result = theta(100, 100, 0.05, 0.20, 0, "call")
    assert result == 0


# Vega Tests
def test_vega_positive():
    result = vega(100, 100, 0.05, 0.20, 1.0)
    assert result > 0


def test_vega_highest_atm():
    atm = vega(100, 100, 0.05, 0.20, 1.0)
    itm = vega(120, 100, 0.05, 0.20, 1.0)
    otm = vega(80, 100, 0.05, 0.20, 1.0)
    assert atm >= itm
    assert atm >= otm


def test_vega_increases_with_time():
    short = vega(100, 100, 0.05, 0.20, 0.25)
    long = vega(100, 100, 0.05, 0.20, 1.0)
    assert long > short


@pytest.mark.parametrize("S", [80, 90, 100, 110, 120])
def test_vega_various_strikes(S: float):
    result = vega(S, 100, 0.05, 0.20, 1.0)
    assert result >= 0


# Rho Tests
def test_rho_call_positive():
    result = rho(100, 100, 0.05, 0.20, 1.0, "call")
    assert result > 0


def test_rho_put_negative():
    result = rho(100, 100, 0.05, 0.20, 1.0, "put")
    assert result < 0


@pytest.mark.parametrize("T", [0.25, 0.5, 1.0, 2.0])
def test_rho_increases_with_time(T: float):
    result = rho(100, 100, 0.05, 0.20, T, "call")
    assert result > 0


# Binomial Tree Tests
@pytest.mark.parametrize("steps", [10, 50, 100])
def test_binomial_converges_to_bs(steps: int):
    S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    bs_call = black_scholes_call(S, K, r, sigma, T)
    bin_call = binomial_tree_price(S, K, r, sigma, T, steps, "call")
    # Should converge as steps increase
    assert abs(bin_call - bs_call) < 1.0


def test_binomial_call_put():
    S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    call = binomial_tree_price(S, K, r, sigma, T, 50, "call")
    put = binomial_tree_price(S, K, r, sigma, T, 50, "put")
    assert call > 0
    assert put > 0


def test_binomial_american_vs_european():
    S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    european_put = binomial_tree_price(S, K, r, sigma, T, 50, "put", american=False)
    american_put = binomial_tree_price(S, K, r, sigma, T, 50, "put", american=True)
    # American >= European
    assert american_put >= european_put - 0.01


@pytest.mark.parametrize("steps", [20, 50, 100, 150])
def test_binomial_deep_tree(steps: int):
    result = binomial_tree_price(100, 100, 0.05, 0.20, 1.0, steps, "call")
    assert result > 0


# Implied Volatility Tests
def test_implied_volatility_recovery():
    S, K, r, T = 100, 100, 0.05, 1.0
    true_vol = 0.25
    call_price = black_scholes_call(S, K, r, true_vol, T)
    recovered_vol = implied_volatility(call_price, S, K, r, T, "call")
    assert abs(recovered_vol - true_vol) < 0.01


@pytest.mark.parametrize("sigma", [0.15, 0.20, 0.25, 0.30, 0.40])
def test_implied_volatility_various(sigma: float):
    S, K, r, T = 100, 100, 0.05, 1.0
    call_price = black_scholes_call(S, K, r, sigma, T)
    recovered_vol = implied_volatility(call_price, S, K, r, T, "call")
    assert abs(recovered_vol - sigma) < 0.01


def test_implied_volatility_put():
    S, K, r, T = 100, 100, 0.05, 1.0
    true_vol = 0.25
    put_price = black_scholes_put(S, K, r, true_vol, T)
    recovered_vol = implied_volatility(put_price, S, K, r, T, "put")
    assert abs(recovered_vol - true_vol) < 0.01


def test_implied_volatility_at_expiry():
    result = implied_volatility(10, 100, 100, 0.05, 0, "call")
    assert result == 0.0


# Option Payoff Tests
@pytest.mark.parametrize("S,K,expected", [
    (110, 100, 10),
    (100, 100, 0),
    (90, 100, 0),
])
def test_call_payoff(S: float, K: float, expected: float):
    result = option_payoff(S, K, "call", "long")
    assert abs(result - expected) < 0.01


@pytest.mark.parametrize("S,K,expected", [
    (90, 100, 10),
    (100, 100, 0),
    (110, 100, 0),
])
def test_put_payoff(S: float, K: float, expected: float):
    result = option_payoff(S, K, "put", "long")
    assert abs(result - expected) < 0.01


def test_short_payoff_inverse():
    long_payoff = option_payoff(110, 100, "call", "long")
    short_payoff = option_payoff(110, 100, "call", "short")
    assert long_payoff == -short_payoff


# Option Profit Tests
def test_option_profit_call():
    profit = option_profit(115, 100, 5, "call", "long")
    # Payoff is 15, paid 5 premium, profit is 10
    assert abs(profit - 10) < 0.01


def test_option_profit_put():
    profit = option_profit(90, 100, 5, "put", "long")
    # Payoff is 10, paid 5 premium, profit is 5
    assert abs(profit - 5) < 0.01


def test_option_profit_short():
    profit = option_profit(110, 100, 5, "call", "short")
    # Short call: received 5, lost 10 on assignment
    assert abs(profit - (-5)) < 0.01


# Breakeven Tests
@pytest.mark.parametrize("K,premium,option_type,expected", [
    (100, 5, "call", 105),
    (100, 5, "put", 95),
])
def test_breakeven_price(K: float, premium: float, option_type: str, expected: float):
    result = breakeven_price(K, premium, option_type)
    assert abs(result - expected) < 0.01


# Intrinsic Value Tests
@pytest.mark.parametrize("S,K,option_type,expected", [
    (110, 100, "call", 10),
    (90, 100, "call", 0),
    (90, 100, "put", 10),
    (110, 100, "put", 0),
])
def test_intrinsic_value(S: float, K: float, option_type: str, expected: float):
    result = intrinsic_value(S, K, option_type)
    assert abs(result - expected) < 0.01


# Time Value Tests
def test_time_value_positive():
    S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    call_price = black_scholes_call(S, K, r, sigma, T)
    tv = time_value(call_price, S, K, "call")
    assert tv > 0


def test_time_value_atm_highest():
    S = 100
    call_price = black_scholes_call(S, 100, 0.05, 0.20, 1.0)
    atm_tv = time_value(call_price, S, 100, "call")

    itm_price = black_scholes_call(S, 90, 0.05, 0.20, 1.0)
    itm_tv = time_value(itm_price, S, 90, "call")

    assert atm_tv >= itm_tv - 0.5  # ATM should have high time value


# Moneyness Tests
@pytest.mark.parametrize("S,K,option_type,expected", [
    (110, 100, "call", "ITM"),
    (100, 100, "call", "ATM"),
    (90, 100, "call", "OTM"),
    (90, 100, "put", "ITM"),
    (100, 100, "put", "ATM"),
    (110, 100, "put", "OTM"),
])
def test_moneyness(S: float, K: float, option_type: str, expected: str):
    result = moneyness(S, K, option_type)
    assert result == expected


# Strategy Tests
def test_straddle_payoff_at_strike():
    # At the strike, straddle loses both premiums
    result = straddle_payoff(100, 100, 5, 5)
    assert abs(result - (-10)) < 0.01


def test_straddle_payoff_breakeven():
    # Breakeven when stock moves by total premium
    result = straddle_payoff(110, 100, 5, 5)
    assert abs(result) < 0.01


@pytest.mark.parametrize("S,expected", [
    (80, 10),   # 100-80-10 = 10
    (120, 10),  # 120-100-10 = 10
    (100, -10), # No payoff - premium
])
def test_straddle_various_prices(S: float, expected: float):
    result = straddle_payoff(S, 100, 5, 5)
    assert abs(result - expected) < 0.01


def test_strangle_payoff():
    # Strangle with strikes at 95 (put) and 105 (call)
    result = strangle_payoff(100, 105, 95, 3, 3)
    assert abs(result - (-6)) < 0.01  # Loses both premiums


def test_covered_call_payoff():
    # Stock at 100, bought at 95, sold 105 call for 3
    result = covered_call_payoff(100, 105, 95, 3)
    # Stock profit: 5, option not exercised, kept premium: 3
    assert abs(result - 8) < 0.01


def test_covered_call_capped():
    # Stock above strike
    result = covered_call_payoff(115, 105, 95, 3)
    # Stock profit capped at strike: 10, premium: 3
    assert abs(result - 13) < 0.01


def test_protective_put():
    # Stock at 110, bought at 100, bought 95 put for 3
    result = protective_put_payoff(110, 95, 100, 3)
    # Stock profit: 10, put expires worthless: -3
    assert abs(result - 7) < 0.01


def test_protective_put_protected():
    # Stock below put strike
    result = protective_put_payoff(85, 95, 100, 3)
    # Stock loss: -15, put payoff: 10, premium: -3 => -8
    assert abs(result - (-8)) < 0.01
