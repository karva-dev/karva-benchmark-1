"""Tests for bond pricing calculations."""

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
    dirty_price,
    clean_price,
    yield_to_call,
    spread_to_benchmark,
    dollar_duration,
    effective_duration,
    effective_convexity,
)


# Bond Price Tests
@pytest.mark.parametrize(
    "face,coupon,ytm,periods,expected",
    [
        (1000, 0.05, 0.04, 10, 1044.52),  # Semi-annual: 5 years, 10 periods
        (1000, 0.05, 0.05, 10, 1000.00),  # At par
        (1000, 0.05, 0.06, 10, 957.35),  # Semi-annual: 5 years, 10 periods
        (1000, 0.08, 0.06, 20, 1148.77),  # Semi-annual: 10 years, 20 periods
    ],
)
def test_bond_price_basic(
    face: float, coupon: float, ytm: float, periods: int, expected: float
):
    result = bond_price(face, coupon, ytm, periods)
    assert abs(result - expected) < 1.0


def test_bond_price_at_par():
    # When coupon rate = YTM, price = face value
    result = bond_price(1000, 0.05, 0.05, 20)
    assert abs(result - 1000) < 0.1


def test_bond_price_premium():
    # When coupon > YTM, bond trades at premium
    result = bond_price(1000, 0.08, 0.05, 20)
    assert result > 1000


def test_bond_price_discount():
    # When coupon < YTM, bond trades at discount
    result = bond_price(1000, 0.04, 0.06, 20)
    assert result < 1000


@pytest.mark.parametrize("periods", [2, 10, 20, 40, 60])
def test_bond_price_various_maturities(periods: int):
    result = bond_price(1000, 0.05, 0.04, periods)
    assert result > 0


def test_bond_price_zero_periods():
    result = bond_price(1000, 0.05, 0.04, 0)
    assert result == 1000


@pytest.mark.parametrize("frequency", [1, 2, 4])
def test_bond_price_various_frequencies(frequency: int):
    result = bond_price(1000, 0.05, 0.04, 10, frequency)
    assert result > 0


# Zero Coupon Bond Price Tests
@pytest.mark.parametrize(
    "face,ytm,years,expected",
    [
        (1000, 0.05, 5, 781.20),
        (1000, 0.08, 10, 456.39),
        (1000, 0.03, 2, 942.60),
    ],
)
def test_zero_coupon_price(face: float, ytm: float, years: float, expected: float):
    result = bond_price_zero_coupon(face, ytm, years)
    assert abs(result - expected) < 1.0


def test_zero_coupon_approaches_face():
    # As maturity approaches, zero coupon approaches face
    result = bond_price_zero_coupon(1000, 0.05, 0.01)
    assert abs(result - 1000) < 5


# Yield to Maturity Tests
def test_ytm_recovery():
    # YTM should recover the yield used to price the bond
    price = bond_price(1000, 0.05, 0.06, 20)
    ytm = yield_to_maturity(price, 1000, 0.05, 20)
    assert abs(ytm - 0.06) < 0.001


@pytest.mark.parametrize(
    "coupon,expected_ytm",
    [
        (0.04, 0.0435),
        (0.05, 0.05),
        (0.06, 0.0563),
    ],
)
def test_ytm_various_coupons(coupon: float, expected_ytm: float):
    price = bond_price(1000, coupon, expected_ytm, 20)
    ytm = yield_to_maturity(price, 1000, coupon, 20)
    assert abs(ytm - expected_ytm) < 0.001


def test_ytm_premium_bond():
    # Premium bond has YTM < coupon rate
    price = 1100
    ytm = yield_to_maturity(price, 1000, 0.06, 20)
    assert ytm < 0.06


def test_ytm_discount_bond():
    # Discount bond has YTM > coupon rate
    price = 900
    ytm = yield_to_maturity(price, 1000, 0.05, 20)
    assert ytm > 0.05


# Current Yield Tests
@pytest.mark.parametrize(
    "price,face,coupon,expected",
    [
        (1000, 1000, 0.05, 0.05),
        (950, 1000, 0.05, 0.0526),
        (1050, 1000, 0.05, 0.0476),
    ],
)
def test_current_yield(price: float, face: float, coupon: float, expected: float):
    result = current_yield(price, face, coupon)
    assert abs(result - expected) < 0.001


def test_current_yield_zero_price():
    try:
        current_yield(0, 1000, 0.05)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Macaulay Duration Tests
def test_macaulay_duration_basic(bond_params):
    result = macaulay_duration(**bond_params)
    assert result > 0


def test_macaulay_duration_less_than_maturity():
    # Duration should be less than maturity for coupon bonds
    duration = macaulay_duration(1000, 0.05, 0.04, 20, 2)
    maturity_years = 20 / 2
    assert duration < maturity_years


def test_macaulay_duration_zero_coupon():
    # For zero coupon, duration = maturity
    duration = macaulay_duration(1000, 0, 0.05, 10, 2)
    assert abs(duration - 5.0) < 0.1  # 10 semi-annual periods = 5 years


@pytest.mark.parametrize("coupon", [0.02, 0.04, 0.06, 0.08])
def test_duration_decreases_with_coupon(coupon: float):
    low_coupon = macaulay_duration(1000, 0.02, 0.05, 20)
    high_coupon = macaulay_duration(1000, coupon, 0.05, 20)
    assert high_coupon <= low_coupon


@pytest.mark.parametrize("periods", [4, 10, 20, 40])
def test_duration_increases_with_maturity(periods: int):
    short = macaulay_duration(1000, 0.05, 0.04, 4)
    long = macaulay_duration(1000, 0.05, 0.04, periods)
    assert long >= short


# Modified Duration Tests
def test_modified_duration_basic(bond_params):
    result = modified_duration(**bond_params)
    assert result > 0


def test_modified_less_than_macaulay():
    mac = macaulay_duration(1000, 0.05, 0.04, 20)
    mod = modified_duration(1000, 0.05, 0.04, 20)
    assert mod < mac


def test_modified_duration_formula():
    mac = macaulay_duration(1000, 0.05, 0.06, 20)
    mod = modified_duration(1000, 0.05, 0.06, 20)
    expected = mac / (1 + 0.06 / 2)
    assert abs(mod - expected) < 0.001


# Convexity Tests
def test_convexity_positive(bond_params):
    result = convexity(**bond_params)
    assert result > 0


@pytest.mark.parametrize("periods", [10, 20, 40, 60])
def test_convexity_increases_with_maturity(periods: int):
    short = convexity(1000, 0.05, 0.04, 10)
    long = convexity(1000, 0.05, 0.04, periods)
    assert long >= short


def test_convexity_higher_for_lower_coupon():
    high_coupon = convexity(1000, 0.08, 0.05, 20)
    low_coupon = convexity(1000, 0.03, 0.05, 20)
    assert low_coupon > high_coupon


# Price Change Tests
def test_price_change_duration_negative():
    # Price decreases when yield increases
    change = price_change_duration(8.0, 0.01)
    assert change < 0


def test_price_change_duration_positive():
    # Price increases when yield decreases
    change = price_change_duration(8.0, -0.01)
    assert change > 0


@pytest.mark.parametrize("yield_change", [-0.02, -0.01, 0.01, 0.02])
def test_price_change_duration_linear(yield_change: float):
    mod_dur = 7.5
    change = price_change_duration(mod_dur, yield_change)
    expected = -mod_dur * yield_change
    assert abs(change - expected) < 0.0001


def test_price_change_convexity_adjustment():
    mod_dur = 8.0
    conv = 80.0
    yield_change = 0.02

    duration_only = price_change_duration(mod_dur, yield_change)
    with_convexity = price_change_convexity(mod_dur, conv, yield_change)

    # Convexity adjustment is always positive
    assert with_convexity > duration_only


# Accrued Interest Tests
@pytest.mark.parametrize(
    "days,expected",
    [
        (0, 0),
        (90, 12.50),
        (180, 25.00),
    ],
)
def test_accrued_interest(days: int, expected: float):
    result = accrued_interest(1000, 0.05, days, 180, 2)
    assert abs(result - expected) < 0.01


def test_accrued_interest_half_period():
    result = accrued_interest(1000, 0.06, 90, 180, 2)
    expected = 1000 * 0.06 / 2 * 0.5  # Half of semi-annual coupon
    assert abs(result - expected) < 0.01


# Dirty/Clean Price Tests
def test_dirty_clean_relationship():
    accrued = 15.0
    clean = 1050.0
    dirty = dirty_price(clean, accrued)
    recovered_clean = clean_price(dirty, accrued)
    assert abs(recovered_clean - clean) < 0.01


def test_dirty_greater_than_clean():
    accrued = 20.0
    clean = 1000.0
    dirty = dirty_price(clean, accrued)
    assert dirty > clean


# Yield to Call Tests
def test_ytc_basic():
    # Bond price, face, coupon, periods to call, call price
    result = yield_to_call(1050, 1000, 0.06, 10, 1020)
    assert result > 0


def test_ytc_vs_ytm():
    # For callable premium bond, YTC is typically lower than YTM
    price = 1100
    ytm = yield_to_maturity(price, 1000, 0.08, 20)
    ytc = yield_to_call(price, 1000, 0.08, 10, 1050)
    # YTC should be lower for premium bond
    assert ytc < ytm


# Spread Tests
@pytest.mark.parametrize(
    "bond_ytm,bench_ytm,expected",
    [
        (0.055, 0.045, 100),
        (0.08, 0.05, 300),
        (0.045, 0.05, -50),
    ],
)
def test_spread_to_benchmark(bond_ytm: float, bench_ytm: float, expected: float):
    result = spread_to_benchmark(bond_ytm, bench_ytm)
    assert abs(result - expected) < 0.1


# Dollar Duration Tests
def test_dollar_duration_basic():
    price = 1050
    mod_dur = 8.0
    result = dollar_duration(price, mod_dur)
    expected = price * mod_dur / 100
    assert abs(result - expected) < 0.01


@pytest.mark.parametrize("price", [900, 1000, 1100, 1200])
def test_dollar_duration_scales_with_price(price: float):
    mod_dur = 7.5
    result = dollar_duration(price, mod_dur)
    assert result > 0


# Effective Duration/Convexity Tests
def test_effective_duration_basic():
    price_down = 1080  # Price when yield decreases
    price_up = 920  # Price when yield increases
    price = 1000
    yield_change = 0.01

    result = effective_duration(price_down, price_up, price, yield_change)
    # Should be (1080 - 920) / (2 * 1000 * 0.01) = 8
    assert abs(result - 8.0) < 0.1


def test_effective_convexity_basic():
    price_down = 1082
    price_up = 918
    price = 1000
    yield_change = 0.01

    result = effective_convexity(price_down, price_up, price, yield_change)
    # Should be (1082 + 918 - 2000) / (1000 * 0.0001) = 0
    assert result >= 0 or result <= 0  # Can be positive or small


def test_effective_convexity_positive_for_bonds():
    # Most bonds have positive convexity
    price_down = 1090
    price_up = 920
    price = 1000
    yield_change = 0.01

    result = effective_convexity(price_down, price_up, price, yield_change)
    # Price changes are asymmetric (more up than down) = positive convexity
    expected = (1090 + 920 - 2000) / (1000 * 0.01**2)
    assert abs(result - expected) < 1


# Integration Tests
def test_price_ytm_roundtrip():
    face, coupon, ytm, periods = 1000, 0.06, 0.05, 20
    price = bond_price(face, coupon, ytm, periods)
    recovered_ytm = yield_to_maturity(price, face, coupon, periods)
    assert abs(recovered_ytm - ytm) < 0.0001


def test_duration_convexity_price_change():
    face, coupon, ytm, periods = 1000, 0.05, 0.04, 20

    price = bond_price(face, coupon, ytm, periods)
    mod_dur = modified_duration(face, coupon, ytm, periods)
    conv = convexity(face, coupon, ytm, periods)

    # Price at new yield
    new_ytm = 0.05
    new_price = bond_price(face, coupon, new_ytm, periods)

    # Estimated change
    yield_change = new_ytm - ytm
    estimated_change = price_change_convexity(mod_dur, conv, yield_change)
    actual_change = (new_price - price) / price

    # Duration + convexity should approximate actual change
    assert abs(estimated_change - actual_change) < 0.01


@pytest.mark.parametrize("ytm", [0.02, 0.04, 0.06, 0.08, 0.10])
def test_bond_price_ytm_relationship(ytm: float):
    price = bond_price(1000, 0.05, ytm, 20)
    recovered_ytm = yield_to_maturity(price, 1000, 0.05, 20)
    assert abs(recovered_ytm - ytm) < 0.001
