"""Tests for interest rate calculations."""

import math

import pytest

from finlib.interest import (
    simple_interest,
    compound_interest,
    continuous_compound_interest,
    apr_to_apy,
    apy_to_apr,
    effective_interest_rate,
    future_value,
    present_value,
    real_interest_rate,
    doubling_time,
    compound_interest_earned,
)


# Simple Interest Tests
@pytest.mark.parametrize(
    "principal,rate,time,expected",
    [
        (1000, 0.05, 1, 50),
        (1000, 0.05, 2, 100),
        (5000, 0.08, 3, 1200),
        (10000, 0.10, 0.5, 500),
        (2500, 0.03, 4, 300),
    ],
)
def test_simple_interest_basic(
    principal: float, rate: float, time: float, expected: float
):
    result = simple_interest(principal, rate, time)
    assert abs(result - expected) < 0.01


@pytest.mark.parametrize(
    "principal,rate,time",
    [
        (0, 0.05, 1),
        (1000, 0, 1),
        (1000, 0.05, 0),
    ],
)
def test_simple_interest_zero_values(principal: float, rate: float, time: float):
    result = simple_interest(principal, rate, time)
    assert result == 0.0


@pytest.mark.parametrize("principal", [100, 1000, 10000, 100000, 1000000])
def test_simple_interest_scales_with_principal(principal: float):
    rate, time = 0.05, 1
    result = simple_interest(principal, rate, time)
    assert abs(result - principal * rate * time) < 0.001


@pytest.mark.parametrize("time", [0.25, 0.5, 1, 2, 5, 10])
def test_simple_interest_scales_with_time(time: float):
    principal, rate = 1000, 0.05
    result = simple_interest(principal, rate, time)
    assert abs(result - principal * rate * time) < 0.001


# Compound Interest Tests
@pytest.mark.parametrize(
    "principal,rate,time,n,expected",
    [
        (1000, 0.05, 1, 1, 1050.00),
        (1000, 0.05, 1, 12, 1051.16),
        (1000, 0.05, 1, 365, 1051.27),
        (1000, 0.10, 2, 1, 1210.00),
        (5000, 0.08, 5, 4, 7429.74),
    ],
)
def test_compound_interest_basic(
    principal: float, rate: float, time: float, n: int, expected: float
):
    result = compound_interest(principal, rate, time, n)
    assert abs(result - expected) < 0.01


@pytest.mark.parametrize("n", [1, 2, 4, 12, 52, 365])
def test_compound_interest_frequency_increases_value(n: int):
    principal, rate, time = 1000, 0.10, 1
    result = compound_interest(principal, rate, time, n)
    # Higher frequency should give higher result
    simple = compound_interest(principal, rate, time, 1)
    assert result >= simple


def test_compound_interest_equals_simple_for_one_year_annual():
    principal, rate, time = 1000, 0.05, 1
    compound = compound_interest(principal, rate, time, 1)
    simple = principal + simple_interest(principal, rate, time)
    assert abs(compound - simple) < 0.001


@pytest.mark.parametrize("years", [1, 2, 5, 10, 20, 30])
def test_compound_interest_growth_over_time(years: int):
    principal, rate = 1000, 0.08
    result = compound_interest(principal, rate, years)
    expected = principal * (1 + rate) ** years
    assert abs(result - expected) < 0.001


# Continuous Compound Interest Tests
@pytest.mark.parametrize(
    "principal,rate,time,expected",
    [
        (1000, 0.05, 1, 1051.27),
        (1000, 0.10, 1, 1105.17),
        (1000, 0.05, 2, 1105.17),
        (5000, 0.08, 5, 7459.12),
    ],
)
def test_continuous_compound_interest(
    principal: float, rate: float, time: float, expected: float
):
    result = continuous_compound_interest(principal, rate, time)
    assert abs(result - expected) < 0.01


def test_continuous_vs_daily_compound():
    principal, rate, time = 1000, 0.10, 1
    continuous = continuous_compound_interest(principal, rate, time)
    daily = compound_interest(principal, rate, time, 365)
    # Continuous should be slightly higher
    assert continuous >= daily
    assert continuous - daily < 0.1  # Relaxed tolerance


@pytest.mark.parametrize("rate", [0.01, 0.05, 0.10, 0.15, 0.20])
def test_continuous_formula(rate: float):
    principal, time = 1000, 1
    result = continuous_compound_interest(principal, rate, time)
    expected = principal * math.exp(rate * time)
    assert abs(result - expected) < 0.0001


# APR/APY Conversion Tests
@pytest.mark.parametrize(
    "apr,n,expected_apy",
    [
        (0.05, 12, 0.05116),
        (0.10, 12, 0.10471),
        (0.05, 4, 0.05095),
        (0.05, 1, 0.05000),
        (0.08, 365, 0.08328),
    ],
)
def test_apr_to_apy(apr: float, n: int, expected_apy: float):
    result = apr_to_apy(apr, n)
    assert abs(result - expected_apy) < 0.0001


@pytest.mark.parametrize(
    "apy,n",
    [
        (0.05, 12),
        (0.10, 12),
        (0.05, 4),
        (0.08, 365),
    ],
)
def test_apr_apy_roundtrip(apy: float, n: int):
    apr = apy_to_apr(apy, n)
    recovered_apy = apr_to_apy(apr, n)
    assert abs(recovered_apy - apy) < 0.00001


def test_apr_equals_apy_for_annual():
    apr = 0.05
    apy = apr_to_apy(apr, 1)
    assert abs(apy - apr) < 0.00001


@pytest.mark.parametrize("apr", [0.01, 0.05, 0.10, 0.15])
def test_apy_greater_than_apr(apr: float):
    apy = apr_to_apy(apr, 12)
    assert apy > apr


# Effective Interest Rate Tests
@pytest.mark.parametrize(
    "nominal,n,expected",
    [
        (0.05, 12, 0.05116),
        (0.10, 4, 0.10381),
        (0.08, 2, 0.08160),
        (0.12, 12, 0.12683),
    ],
)
def test_effective_interest_rate(nominal: float, n: int, expected: float):
    result = effective_interest_rate(nominal, n)
    assert abs(result - expected) < 0.0001


def test_effective_rate_equals_apy():
    nominal, n = 0.08, 12
    effective = effective_interest_rate(nominal, n)
    apy = apr_to_apy(nominal, n)
    assert abs(effective - apy) < 0.00001


# Future Value Tests
@pytest.mark.parametrize(
    "principal,rate,time,n,expected",
    [
        (1000, 0.05, 1, 1, 1050.00),
        (1000, 0.05, 10, 1, 1628.89),
        (5000, 0.08, 5, 12, 7449.23),
    ],
)
def test_future_value(
    principal: float, rate: float, time: float, n: int, expected: float
):
    result = future_value(principal, rate, time, n)
    assert abs(result - expected) < 0.01


# Present Value Tests
@pytest.mark.parametrize(
    "fv,rate,time,n,expected_pv",
    [
        (1050, 0.05, 1, 1, 1000.00),
        (1628.89, 0.05, 10, 1, 1000.00),
    ],
)
def test_present_value(fv: float, rate: float, time: float, n: int, expected_pv: float):
    result = present_value(fv, rate, time, n)
    assert abs(result - expected_pv) < 0.01


def test_pv_fv_roundtrip():
    principal, rate, time = 1000, 0.08, 5
    fv = future_value(principal, rate, time)
    pv = present_value(fv, rate, time)
    assert abs(pv - principal) < 0.001


# Real Interest Rate Tests
@pytest.mark.parametrize(
    "nominal,inflation,expected",
    [
        (0.05, 0.02, 0.0294),
        (0.10, 0.03, 0.0680),
        (0.08, 0.08, 0.0000),
        (0.05, 0.07, -0.0187),
    ],
)
def test_real_interest_rate(nominal: float, inflation: float, expected: float):
    result = real_interest_rate(nominal, inflation)
    assert abs(result - expected) < 0.001


def test_real_rate_negative_when_inflation_higher():
    nominal, inflation = 0.03, 0.05
    result = real_interest_rate(nominal, inflation)
    assert result < 0


# Doubling Time Tests
@pytest.mark.parametrize(
    "rate,expected_years",
    [
        (0.07, 10.24),
        (0.10, 7.27),
        (0.05, 14.21),
        (0.12, 6.12),
    ],
)
def test_doubling_time(rate: float, expected_years: float):
    result = doubling_time(rate)
    assert abs(result - expected_years) < 0.01


def test_doubling_time_zero_rate():
    result = doubling_time(0)
    assert result == float("inf")


def test_doubling_time_rule_of_72_approximation():
    rate = 0.08
    exact = doubling_time(rate)
    rule_of_72 = 72 / (rate * 100)
    # Rule of 72 should be close but not exact
    assert abs(exact - rule_of_72) < 0.5


# Compound Interest Earned Tests
@pytest.mark.parametrize(
    "principal,rate,time,n,expected",
    [
        (1000, 0.05, 1, 1, 50.00),
        (1000, 0.05, 1, 12, 51.16),
        (1000, 0.10, 2, 1, 210.00),
    ],
)
def test_compound_interest_earned(
    principal: float, rate: float, time: float, n: int, expected: float
):
    result = compound_interest_earned(principal, rate, time, n)
    assert abs(result - expected) < 0.01


def test_interest_earned_less_than_total():
    principal, rate, time = 1000, 0.05, 5
    total = compound_interest(principal, rate, time)
    interest = compound_interest_earned(principal, rate, time)
    assert interest == total - principal
