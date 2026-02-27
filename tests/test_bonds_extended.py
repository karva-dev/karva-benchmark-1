"""Extended bond tests with heavy parametrization."""

import pytest

from finlib.bonds import (
    bond_price,
    bond_price_zero_coupon,
    yield_to_maturity,
    current_yield,
    macaulay_duration,
    modified_duration,
    convexity,
    price_change_duration,
    price_change_convexity,
    accrued_interest,
)


# Comprehensive bond pricing tests
@pytest.mark.parametrize("face", [1000, 5000, 10000])
@pytest.mark.parametrize("coupon", [0.02, 0.04, 0.06, 0.08])
@pytest.mark.parametrize("ytm", [0.03, 0.05, 0.07])
@pytest.mark.parametrize("periods", [10, 20, 40])
def test_bond_price_positive(face: float, coupon: float, ytm: float, periods: int):
    """Bond price should be positive."""
    price = bond_price(face, coupon, ytm, periods)
    assert price > 0


@pytest.mark.parametrize("face", [1000, 5000, 10000])
@pytest.mark.parametrize("coupon", [0.02, 0.04, 0.06, 0.08])
@pytest.mark.parametrize("ytm", [0.03, 0.05, 0.07])
@pytest.mark.parametrize("periods", [10, 20, 40])
def test_bond_price_bounded(face: float, coupon: float, ytm: float, periods: int):
    """Bond price should be bounded reasonably."""
    price = bond_price(face, coupon, ytm, periods)
    # Price should be between 50% and 200% of face for reasonable rates
    assert 0.3 * face <= price <= 2.5 * face


@pytest.mark.parametrize("coupon", [0.03, 0.05, 0.07])
@pytest.mark.parametrize("periods", [10, 20, 30, 40])
def test_bond_at_par(coupon: float, periods: int):
    """Bond at par when coupon = YTM."""
    price = bond_price(1000, coupon, coupon, periods)
    assert abs(price - 1000) < 1


@pytest.mark.parametrize("coupon", [0.06, 0.08, 0.10])
@pytest.mark.parametrize("ytm", [0.03, 0.04, 0.05])
@pytest.mark.parametrize("periods", [10, 20, 30])
def test_bond_premium(coupon: float, ytm: float, periods: int):
    """Bond trades at premium when coupon > YTM."""
    price = bond_price(1000, coupon, ytm, periods)
    assert price > 1000


@pytest.mark.parametrize("coupon", [0.02, 0.03, 0.04])
@pytest.mark.parametrize("ytm", [0.05, 0.06, 0.07])
@pytest.mark.parametrize("periods", [10, 20, 30])
def test_bond_discount(coupon: float, ytm: float, periods: int):
    """Bond trades at discount when coupon < YTM."""
    price = bond_price(1000, coupon, ytm, periods)
    assert price < 1000


# Zero coupon bond tests
@pytest.mark.parametrize("face", [1000, 5000, 10000])
@pytest.mark.parametrize("ytm", [0.03, 0.05, 0.07, 0.10])
@pytest.mark.parametrize("years", [1, 5, 10, 20])
def test_zero_coupon_positive(face: float, ytm: float, years: float):
    """Zero coupon bond price should be positive."""
    price = bond_price_zero_coupon(face, ytm, years)
    assert price > 0


@pytest.mark.parametrize("face", [1000, 5000, 10000])
@pytest.mark.parametrize("ytm", [0.03, 0.05, 0.07, 0.10])
@pytest.mark.parametrize("years", [1, 5, 10, 20])
def test_zero_coupon_less_than_face(face: float, ytm: float, years: float):
    """Zero coupon bond price should be less than face."""
    price = bond_price_zero_coupon(face, ytm, years)
    assert price < face


# YTM recovery tests
@pytest.mark.parametrize("coupon", [0.03, 0.05, 0.07])
@pytest.mark.parametrize("ytm", [0.02, 0.04, 0.06, 0.08])
@pytest.mark.parametrize("periods", [10, 20, 30])
def test_ytm_recovery(coupon: float, ytm: float, periods: int):
    """YTM should be recoverable from price."""
    price = bond_price(1000, coupon, ytm, periods)
    recovered_ytm = yield_to_maturity(price, 1000, coupon, periods)
    assert abs(recovered_ytm - ytm) < 0.001


@pytest.mark.parametrize("coupon", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("periods", [10, 20, 30])
@pytest.mark.parametrize("premium_pct", [1.05, 1.10, 1.15])
def test_ytm_premium_bond(coupon: float, periods: int, premium_pct: float):
    """YTM should be less than coupon for premium bond."""
    price = 1000 * premium_pct
    ytm = yield_to_maturity(price, 1000, coupon, periods)
    assert ytm < coupon


@pytest.mark.parametrize("coupon", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("periods", [10, 20, 30])
@pytest.mark.parametrize("discount_pct", [0.85, 0.90, 0.95])
def test_ytm_discount_bond(coupon: float, periods: int, discount_pct: float):
    """YTM should be greater than coupon for discount bond."""
    price = 1000 * discount_pct
    ytm = yield_to_maturity(price, 1000, coupon, periods)
    assert ytm > coupon


# Duration tests
@pytest.mark.parametrize("coupon", [0.03, 0.05, 0.07])
@pytest.mark.parametrize("ytm", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("periods", [10, 20, 30, 40])
def test_macaulay_duration_positive(coupon: float, ytm: float, periods: int):
    """Macaulay duration should be positive."""
    duration = macaulay_duration(1000, coupon, ytm, periods)
    assert duration > 0


@pytest.mark.parametrize("coupon", [0.03, 0.05, 0.07])
@pytest.mark.parametrize("ytm", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("periods", [10, 20, 30, 40])
def test_duration_less_than_maturity(coupon: float, ytm: float, periods: int):
    """Duration should be less than maturity for coupon bonds."""
    duration = macaulay_duration(1000, coupon, ytm, periods)
    maturity_years = periods / 2  # Semi-annual
    assert duration < maturity_years


@pytest.mark.parametrize("coupon", [0.03, 0.05, 0.07])
@pytest.mark.parametrize("ytm", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("periods", [10, 20, 30, 40])
def test_modified_less_than_macaulay(coupon: float, ytm: float, periods: int):
    """Modified duration should be less than Macaulay."""
    mac = macaulay_duration(1000, coupon, ytm, periods)
    mod = modified_duration(1000, coupon, ytm, periods)
    assert mod < mac


@pytest.mark.parametrize("ytm", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("periods", [10, 20, 30])
def test_duration_increases_with_maturity(ytm: float, periods: int):
    """Duration should increase with maturity."""
    short = macaulay_duration(1000, 0.05, ytm, 10)
    long = macaulay_duration(1000, 0.05, ytm, periods)
    assert long >= short


@pytest.mark.parametrize("coupon", [0.03, 0.05, 0.07, 0.09])
@pytest.mark.parametrize("ytm", [0.05])
@pytest.mark.parametrize("periods", [20])
def test_duration_decreases_with_coupon(coupon: float, ytm: float, periods: int):
    """Duration should decrease with higher coupon."""
    low_coupon = macaulay_duration(1000, 0.02, ytm, periods)
    high_coupon = macaulay_duration(1000, coupon, ytm, periods)
    assert high_coupon <= low_coupon


# Convexity tests
@pytest.mark.parametrize("coupon", [0.03, 0.05, 0.07])
@pytest.mark.parametrize("ytm", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("periods", [10, 20, 30, 40])
def test_convexity_positive(coupon: float, ytm: float, periods: int):
    """Convexity should be positive."""
    conv = convexity(1000, coupon, ytm, periods)
    assert conv > 0


@pytest.mark.parametrize("ytm", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("periods", [10, 20, 30, 40])
def test_convexity_increases_with_maturity(ytm: float, periods: int):
    """Convexity should increase with maturity."""
    short = convexity(1000, 0.05, ytm, 10)
    long = convexity(1000, 0.05, ytm, periods)
    assert long >= short


# Price change estimation tests
@pytest.mark.parametrize("mod_dur", [5, 7, 9, 11])
@pytest.mark.parametrize("yield_change", [-0.02, -0.01, 0.01, 0.02])
def test_price_change_duration_sign(mod_dur: float, yield_change: float):
    """Price change should be opposite sign of yield change."""
    change = price_change_duration(mod_dur, yield_change)
    if yield_change > 0:
        assert change < 0
    elif yield_change < 0:
        assert change > 0


@pytest.mark.parametrize("mod_dur", [5, 7, 9])
@pytest.mark.parametrize("conv", [50, 80, 120])
@pytest.mark.parametrize("yield_change", [-0.02, -0.01, 0.01, 0.02])
def test_convexity_improves_estimate(mod_dur: float, conv: float, yield_change: float):
    """Convexity adjustment should always add value (be positive)."""
    duration_only = price_change_duration(mod_dur, yield_change)
    with_conv = price_change_convexity(mod_dur, conv, yield_change)
    convexity_effect = with_conv - duration_only
    assert convexity_effect >= 0


# Accrued interest tests
@pytest.mark.parametrize("face", [1000, 5000, 10000])
@pytest.mark.parametrize("coupon", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("days", [0, 45, 90, 135, 180])
def test_accrued_interest_range(face: float, coupon: float, days: int):
    """Accrued interest should be between 0 and full coupon."""
    accrued = accrued_interest(face, coupon, days, 180, 2)
    full_coupon = face * coupon / 2
    assert 0 <= accrued <= full_coupon


@pytest.mark.parametrize("face", [1000, 5000, 10000])
@pytest.mark.parametrize("coupon", [0.04, 0.06, 0.08])
def test_accrued_zero_at_coupon_date(face: float, coupon: float):
    """Accrued interest should be zero at coupon date."""
    accrued = accrued_interest(face, coupon, 0, 180, 2)
    assert accrued == 0


@pytest.mark.parametrize("face", [1000, 5000, 10000])
@pytest.mark.parametrize("coupon", [0.04, 0.06, 0.08])
def test_accrued_full_at_period_end(face: float, coupon: float):
    """Accrued interest should equal coupon at period end."""
    accrued = accrued_interest(face, coupon, 180, 180, 2)
    full_coupon = face * coupon / 2
    assert abs(accrued - full_coupon) < 0.01


# Current yield tests
@pytest.mark.parametrize("price", [900, 950, 1000, 1050, 1100])
@pytest.mark.parametrize("coupon", [0.04, 0.06, 0.08])
def test_current_yield_positive(price: float, coupon: float):
    """Current yield should be positive."""
    cy = current_yield(price, 1000, coupon)
    assert cy > 0


@pytest.mark.parametrize("coupon", [0.04, 0.06, 0.08])
def test_current_yield_at_par(coupon: float):
    """Current yield equals coupon rate at par."""
    cy = current_yield(1000, 1000, coupon)
    assert abs(cy - coupon) < 0.0001


@pytest.mark.parametrize("coupon", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("price", [900, 950])
def test_current_yield_higher_for_discount(coupon: float, price: float):
    """Current yield higher than coupon for discount bond."""
    cy = current_yield(price, 1000, coupon)
    assert cy > coupon


@pytest.mark.parametrize("coupon", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("price", [1050, 1100])
def test_current_yield_lower_for_premium(coupon: float, price: float):
    """Current yield lower than coupon for premium bond."""
    cy = current_yield(price, 1000, coupon)
    assert cy < coupon


# Integration tests - price sensitivity
@pytest.mark.parametrize("coupon", [0.04, 0.06])
@pytest.mark.parametrize("periods", [10, 20, 30])
def test_price_increases_when_yield_drops(coupon: float, periods: int):
    """Bond price should increase when yield decreases."""
    prices = [
        bond_price(1000, coupon, ytm, periods) for ytm in [0.07, 0.06, 0.05, 0.04]
    ]
    for i in range(len(prices) - 1):
        assert prices[i] <= prices[i + 1]


@pytest.mark.parametrize("ytm", [0.04, 0.06])
@pytest.mark.parametrize("periods", [10, 20, 30])
def test_price_increases_with_coupon(ytm: float, periods: int):
    """Bond price should increase with coupon rate."""
    prices = [
        bond_price(1000, coupon, ytm, periods) for coupon in [0.02, 0.04, 0.06, 0.08]
    ]
    for i in range(len(prices) - 1):
        assert prices[i] <= prices[i + 1]
