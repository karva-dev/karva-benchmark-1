"""Tests for currency and FX calculations."""

import pytest

from finlib.currency import (
    direct_to_indirect,
    indirect_to_direct,
    convert_currency,
    cross_rate,
    triangular_arbitrage,
    forward_rate,
    forward_points,
    swap_points_to_rate,
    currency_basket_value,
    effective_exchange_rate,
    bid_ask_spread,
    mid_rate,
    pip_value,
    position_size,
    covered_interest_arbitrage,
    forward_rate_from_parity,
    real_exchange_rate,
    purchasing_power_parity_rate,
)


# Direct/Indirect Quote Tests
@pytest.mark.parametrize(
    "direct,expected_indirect",
    [
        (1.25, 0.80),
        (1.10, 0.909),
        (0.80, 1.25),
        (2.00, 0.50),
    ],
)
def test_direct_to_indirect(direct: float, expected_indirect: float):
    result = direct_to_indirect(direct)
    assert abs(result - expected_indirect) < 0.001


def test_direct_to_indirect_zero():
    try:
        direct_to_indirect(0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


@pytest.mark.parametrize(
    "indirect,expected_direct",
    [
        (0.80, 1.25),
        (0.909, 1.10),
        (1.25, 0.80),
    ],
)
def test_indirect_to_direct(indirect: float, expected_direct: float):
    result = indirect_to_direct(indirect)
    assert abs(result - expected_direct) < 0.01


def test_roundtrip_conversion():
    direct = 1.25
    indirect = direct_to_indirect(direct)
    recovered = indirect_to_direct(indirect)
    assert abs(recovered - direct) < 0.0001


# Convert Currency Tests
@pytest.mark.parametrize(
    "amount,rate,is_direct,expected",
    [
        (100, 1.25, True, 125),
        (100, 0.80, True, 80),
        (100, 1.25, False, 80),
    ],
)
def test_convert_currency(amount: float, rate: float, is_direct: bool, expected: float):
    result = convert_currency(amount, rate, is_direct)
    assert abs(result - expected) < 0.01


# Cross Rate Tests
@pytest.mark.parametrize(
    "rate_a,rate_b,expected",
    [
        (1.10, 1.30, 0.846),  # EUR/GBP via USD
        (1.20, 1.10, 1.091),
        (110, 1.10, 100),  # JPY/EUR via USD
    ],
)
def test_cross_rate(rate_a: float, rate_b: float, expected: float):
    result = cross_rate(rate_a, rate_b)
    assert abs(result - expected) < 0.01


def test_cross_rate_zero():
    try:
        cross_rate(1.10, 0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Triangular Arbitrage Tests
def test_no_arbitrage():
    # Rates that multiply to 1 = no arbitrage
    result = triangular_arbitrage(0.90, 1.10, 1.0101)
    assert not result["has_arbitrage"]
    assert abs(result["product"] - 1) < 0.01


def test_arbitrage_exists():
    # Rates that don't multiply to 1 = arbitrage
    result = triangular_arbitrage(0.95, 1.10, 1.00)
    assert result["has_arbitrage"]
    assert result["profit_pct"] > 0


def test_arbitrage_direction():
    # Product > 1 means ABC direction is profitable
    result = triangular_arbitrage(0.95, 1.15, 1.00)
    if result["product"] > 1:
        assert result["direction"] == "ABC"
    else:
        assert result["direction"] == "CBA"


@pytest.mark.parametrize(
    "rate_ab,rate_bc,rate_ca",
    [
        (0.90, 1.10, 1.01),
        (1.20, 0.85, 0.99),
        (0.75, 1.30, 1.02),
    ],
)
def test_triangular_arbitrage_various(rate_ab: float, rate_bc: float, rate_ca: float):
    result = triangular_arbitrage(rate_ab, rate_bc, rate_ca)
    assert "product" in result
    assert "has_arbitrage" in result


# Forward Rate Tests
@pytest.mark.parametrize(
    "spot,dom,for_rate,days,expected",
    [
        (1.10, 0.05, 0.03, 90, 1.1054),
        (1.30, 0.04, 0.06, 180, 1.2871),
        (100, 0.01, 0.05, 360, 96.19),  # Corrected: (1+0.01)/(1+0.05) * 100
    ],
)
def test_forward_rate(
    spot: float, dom: float, for_rate: float, days: int, expected: float
):
    result = forward_rate(spot, dom, for_rate, days)
    assert abs(result - expected) < 0.1


def test_forward_rate_zero_days():
    result = forward_rate(1.10, 0.05, 0.03, 0)
    assert abs(result - 1.10) < 0.0001


def test_forward_rate_same_rates():
    # When rates are equal, forward = spot
    result = forward_rate(1.10, 0.05, 0.05, 90)
    assert abs(result - 1.10) < 0.0001


@pytest.mark.parametrize("days", [30, 90, 180, 365])
def test_forward_rate_various_tenors(days: int):
    result = forward_rate(1.10, 0.05, 0.03, days)
    assert result > 0


# Forward Points Tests
def test_forward_points_basic():
    spot = 1.1000
    fwd = 1.1050
    result = forward_points(spot, fwd)
    assert abs(result - 50) < 0.1


def test_forward_points_negative():
    spot = 1.3000
    fwd = 1.2950
    result = forward_points(spot, fwd)
    assert result < 0


@pytest.mark.parametrize(
    "spot,fwd",
    [
        (1.1000, 1.1100),
        (1.3000, 1.2900),
        (100.00, 99.50),
    ],
)
def test_forward_points_various(spot: float, fwd: float):
    result = forward_points(spot, fwd)
    expected = (fwd - spot) / 0.0001
    assert abs(result - expected) < 1


# Swap Points to Rate Tests
def test_swap_points_to_rate():
    spot = 1.1000
    swap_pts = 50
    result = swap_points_to_rate(spot, swap_pts)
    assert abs(result - 1.1050) < 0.0001


def test_swap_points_negative():
    spot = 1.3000
    swap_pts = -75
    result = swap_points_to_rate(spot, swap_pts)
    assert abs(result - 1.2925) < 0.0001


# Currency Basket Tests
def test_currency_basket_basic(fx_rates):
    basket = {"EUR": 100, "GBP": 50, "USD": 200}
    result = currency_basket_value(basket, fx_rates)
    expected = 100 * 1.10 + 50 * 1.30 + 200
    assert abs(result - expected) < 0.01


def test_currency_basket_single():
    basket = {"EUR": 1000}
    rates = {"EUR": 1.10}
    result = currency_basket_value(basket, rates)
    assert abs(result - 1100) < 0.01


def test_currency_basket_missing_rate():
    basket = {"EUR": 100, "XYZ": 50}
    rates = {"EUR": 1.10}
    try:
        currency_basket_value(basket, rates)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Effective Exchange Rate Tests
def test_effective_exchange_rate():
    rates = {"EUR": 1.10, "GBP": 1.30, "JPY": 0.009}
    weights = {"EUR": 0.5, "GBP": 0.3, "JPY": 0.2}
    result = effective_exchange_rate(rates, weights)
    assert result > 0


def test_effective_exchange_rate_single():
    rates = {"EUR": 1.10}
    weights = {"EUR": 1.0}
    result = effective_exchange_rate(rates, weights)
    assert abs(result - 1.10) < 0.0001


def test_effective_exchange_rate_invalid_weights():
    rates = {"EUR": 1.10, "GBP": 1.30}
    weights = {"EUR": 0.6, "GBP": 0.6}  # Sum > 1
    try:
        effective_exchange_rate(rates, weights)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Bid-Ask Spread Tests
@pytest.mark.parametrize(
    "bid,ask,expected",
    [
        (1.0990, 1.1010, 0.182),
        (1.2990, 1.3010, 0.154),
        (99.90, 100.10, 0.200),
    ],
)
def test_bid_ask_spread(bid: float, ask: float, expected: float):
    result = bid_ask_spread(bid, ask)
    assert abs(result - expected) < 0.01


def test_bid_ask_spread_invalid():
    try:
        bid_ask_spread(1.10, 1.09)  # Bid > Ask
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Mid Rate Tests
@pytest.mark.parametrize(
    "bid,ask,expected",
    [
        (1.0990, 1.1010, 1.1000),
        (1.2980, 1.3020, 1.3000),
    ],
)
def test_mid_rate(bid: float, ask: float, expected: float):
    result = mid_rate(bid, ask)
    assert abs(result - expected) < 0.0001


# Pip Value Tests
@pytest.mark.parametrize(
    "lot_size,pip_size,rate,expected",
    [
        (100000, 0.0001, 1.0, 10),
        (10000, 0.0001, 1.0, 1),
        (100000, 0.01, 1.0, 1000),  # JPY pairs
    ],
)
def test_pip_value(lot_size: float, pip_size: float, rate: float, expected: float):
    result = pip_value(lot_size, pip_size, rate)
    assert abs(result - expected) < 0.01


# Position Size Tests
def test_position_size_basic():
    result = position_size(10000, 0.02, 50, 0.0001)
    # Risk amount = 200, per pip = 50 * 0.0001 = 0.005
    # Position = 200 / 0.005 = 40000
    assert result > 0


@pytest.mark.parametrize("risk_pct", [0.01, 0.02, 0.05])
def test_position_size_risk_levels(risk_pct: float):
    result = position_size(10000, risk_pct, 50, 0.0001)
    assert result > 0


# Covered Interest Arbitrage Tests
def test_covered_interest_no_arbitrage():
    spot = 1.10
    domestic = 0.05
    foreign = 0.03
    days = 90

    theoretical_fwd = forward_rate(spot, domestic, foreign, days)
    result = covered_interest_arbitrage(spot, theoretical_fwd, domestic, foreign, days)

    assert not result["has_arbitrage"]


def test_covered_interest_arbitrage_exists():
    spot = 1.10
    market_forward = 1.15  # Mispriced forward
    domestic = 0.05
    foreign = 0.03
    days = 90

    result = covered_interest_arbitrage(spot, market_forward, domestic, foreign, days)
    assert result["has_arbitrage"]


def test_covered_interest_arbitrage_strategy():
    result = covered_interest_arbitrage(1.10, 1.15, 0.05, 0.03, 90)
    if result["has_arbitrage"]:
        assert result["strategy"] in ["borrow_domestic", "borrow_foreign"]


# Forward Rate from Parity Tests
def test_forward_rate_from_parity():
    result = forward_rate_from_parity(1.10, 0.05, 0.03, 90)
    expected = forward_rate(1.10, 0.05, 0.03, 90)
    assert abs(result - expected) < 0.0001


# Real Exchange Rate Tests
@pytest.mark.parametrize(
    "nominal,domestic,foreign,expected",
    [
        (1.10, 100, 110, 1.21),
        (1.20, 120, 100, 1.00),
    ],
)
def test_real_exchange_rate(
    nominal: float, domestic: float, foreign: float, expected: float
):
    result = real_exchange_rate(nominal, domestic, foreign)
    assert abs(result - expected) < 0.01


# PPP Rate Tests
@pytest.mark.parametrize(
    "domestic,foreign,expected",
    [
        (10, 8, 1.25),
        (100, 110, 0.909),
    ],
)
def test_ppp_rate(domestic: float, foreign: float, expected: float):
    result = purchasing_power_parity_rate(domestic, foreign)
    assert abs(result - expected) < 0.01


def test_ppp_rate_zero_foreign():
    try:
        purchasing_power_parity_rate(100, 0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Integration Tests
def test_forward_points_roundtrip():
    spot = 1.1000
    swap_pts = 75
    fwd = swap_points_to_rate(spot, swap_pts)
    recovered_pts = forward_points(spot, fwd)
    assert abs(recovered_pts - swap_pts) < 0.1


def test_arbitrage_free_triangle():
    # Create arbitrage-free rates
    usd_eur = 1.10
    usd_gbp = 1.30
    eur_gbp = usd_gbp / usd_eur  # Cross rate

    result = triangular_arbitrage(usd_eur, 1 / usd_gbp, eur_gbp)
    # Product should be approximately 1
    assert abs(result["product"] - 1) < 0.01


@pytest.mark.parametrize(
    "spot,dom,foreign",
    [
        (1.10, 0.05, 0.03),
        (1.30, 0.02, 0.06),
        (100, 0.01, 0.05),
    ],
)
def test_interest_rate_parity(spot: float, dom: float, foreign: float):
    days = 180
    fwd = forward_rate(spot, dom, foreign, days)

    # Verify the relationship
    time = days / 360
    expected = spot * (1 + dom * time) / (1 + foreign * time)
    assert abs(fwd - expected) < 0.0001
