"""Mega bond tests with heavy Cartesian product parametrization."""
import pytest
from finlib.bonds import bond_price, bond_price_zero_coupon, yield_to_maturity, current_yield, macaulay_duration, modified_duration, convexity, price_change_duration, price_change_convexity, accrued_interest, dirty_price, clean_price, spread_to_benchmark, dollar_duration, effective_duration, effective_convexity
FACES = [500, 10000]
COUPONS = [0.01, 0.1]
YTMS = [0.01, 0.1]
PERIODS = [2, 60]
FREQS = [1, 2]

@pytest.mark.parametrize('face', FACES)
@pytest.mark.parametrize('coupon', COUPONS)
@pytest.mark.parametrize('ytm', YTMS)
@pytest.mark.parametrize('periods', PERIODS)
def test_bond_price_positive(face, coupon, ytm, periods):
    """Bond price should always be positive."""
    price = bond_price(face, coupon, ytm, periods)
    assert price > 0

@pytest.mark.parametrize('face', FACES)
@pytest.mark.parametrize('coupon', COUPONS)
@pytest.mark.parametrize('ytm', YTMS)
@pytest.mark.parametrize('periods', PERIODS)
def test_bond_price_reasonable_range(face, coupon, ytm, periods):
    """Bond price should be within a reasonable range."""
    price = bond_price(face, coupon, ytm, periods)
    assert 0.1 * face <= price <= 5 * face

@pytest.mark.parametrize('face', FACES)
@pytest.mark.parametrize('coupon', COUPONS)
@pytest.mark.parametrize('ytm', YTMS)
@pytest.mark.parametrize('periods', PERIODS)
@pytest.mark.parametrize('freq', FREQS)
def test_duration_positive(face, coupon, ytm, periods, freq):
    """Macaulay duration should be positive."""
    dur = macaulay_duration(face, coupon, ytm, periods, freq)
    assert dur >= 0

@pytest.mark.parametrize('face', FACES)
@pytest.mark.parametrize('coupon', COUPONS)
@pytest.mark.parametrize('ytm', YTMS)
@pytest.mark.parametrize('periods', PERIODS)
@pytest.mark.parametrize('freq', FREQS)
def test_modified_leq_macaulay(face, coupon, ytm, periods, freq):
    """Modified duration should be <= Macaulay duration."""
    mac = macaulay_duration(face, coupon, ytm, periods, freq)
    mod = modified_duration(face, coupon, ytm, periods, freq)
    assert mod <= mac + 0.001

@pytest.mark.parametrize('face', FACES)
@pytest.mark.parametrize('coupon', COUPONS)
@pytest.mark.parametrize('ytm', YTMS)
@pytest.mark.parametrize('periods', PERIODS)
def test_convexity_positive(face, coupon, ytm, periods):
    """Convexity should be positive."""
    conv = convexity(face, coupon, ytm, periods)
    assert conv >= 0

@pytest.mark.parametrize('face', FACES)
@pytest.mark.parametrize('ytm', [0.03, 0.1])
@pytest.mark.parametrize('years', [1, 20])
@pytest.mark.parametrize('freq', [1, 12])
def test_zero_coupon_less_than_face(face, ytm, years, freq):
    """Zero coupon bond should be worth less than face."""
    price = bond_price_zero_coupon(face, ytm, years, freq)
    assert 0 < price < face

@pytest.mark.parametrize('coupon', COUPONS)
@pytest.mark.parametrize('ytm', YTMS)
@pytest.mark.parametrize('periods', PERIODS)
def test_par_bond_price(coupon, ytm, periods):
    """When coupon == ytm, price should be near face."""
    price = bond_price(1000, coupon, coupon, periods)
    assert abs(price - 1000) < 5

@pytest.mark.parametrize('face', FACES)
@pytest.mark.parametrize('coupon', [0.03, 0.12])
@pytest.mark.parametrize('days', [0, 180])
@pytest.mark.parametrize('freq', [1, 12])
def test_accrued_interest_range(face, coupon, days, freq):
    """Accrued interest should be between 0 and full coupon."""
    days_in_period = 360 // freq
    if days > days_in_period:
        return
    accrued = accrued_interest(face, coupon, days, days_in_period, freq)
    full_coupon = face * coupon / freq
    assert -0.01 <= accrued <= full_coupon + 0.01

@pytest.mark.parametrize('price', [800, 1200])
@pytest.mark.parametrize('coupon', [0.03, 0.12])
@pytest.mark.parametrize('face', [1000, 10000])
def test_current_yield_positive(price, coupon, face):
    """Current yield should be positive."""
    cy = current_yield(price, face, coupon)
    assert cy > 0

@pytest.mark.parametrize('mod_dur', [3, 11])
@pytest.mark.parametrize('yield_change', [-0.02, 0.02])
@pytest.mark.parametrize('conv', [30, 150])
def test_convexity_adjustment_positive(mod_dur, yield_change, conv):
    """Convexity adjustment should always add value."""
    dur_only = price_change_duration(mod_dur, yield_change)
    with_conv = price_change_convexity(mod_dur, conv, yield_change)
    assert with_conv >= dur_only - 0.0001

@pytest.mark.parametrize('clean', [800, 1200])
@pytest.mark.parametrize('accrued_val', [0, 30])
@pytest.mark.parametrize('face', FACES)
def test_dirty_clean_roundtrip(clean, accrued_val, face):
    """Dirty price - accrued = clean price."""
    dirty = dirty_price(clean, accrued_val)
    recovered = clean_price(dirty, accrued_val)
    assert abs(recovered - clean) < 0.01

@pytest.mark.parametrize('bond_ytm_val', [0.03, 0.1])
@pytest.mark.parametrize('bench_ytm', [0.01, 0.08])
def test_spread_calculation(bond_ytm_val, bench_ytm):
    """Spread should be correct in basis points."""
    spread = spread_to_benchmark(bond_ytm_val, bench_ytm)
    expected = (bond_ytm_val - bench_ytm) * 10000
    assert abs(spread - expected) < 0.01

@pytest.mark.parametrize('face', FACES)
@pytest.mark.parametrize('coupon', COUPONS)
@pytest.mark.parametrize('ytm', YTMS)
@pytest.mark.parametrize('periods', PERIODS)
def test_dollar_duration_positive(face, coupon, ytm, periods):
    """Dollar duration should be positive."""
    mod = modified_duration(face, coupon, ytm, periods)
    price = bond_price(face, coupon, ytm, periods)
    dd = dollar_duration(price, mod)
    assert dd >= 0

@pytest.mark.parametrize('face', [1000, 10000])
@pytest.mark.parametrize('coupon', [0.03, 0.1])
@pytest.mark.parametrize('periods', [10, 40])
def test_price_increases_when_yield_drops(face, coupon, periods):
    """Bond price should increase when yield decreases."""
    prices = [bond_price(face, coupon, y, periods) for y in [0.1, 0.08, 0.06, 0.04, 0.02]]
    for i in range(len(prices) - 1):
        assert prices[i] <= prices[i + 1] + 0.01

@pytest.mark.parametrize('face', [1000, 10000])
@pytest.mark.parametrize('coupon', [0.03, 0.1])
@pytest.mark.parametrize('ytm', [0.03, 0.1])
@pytest.mark.parametrize('yield_change', [0.001, 0.02])
def test_effective_duration_positive(face, coupon, ytm, yield_change):
    """Effective duration should be positive."""
    periods = 20
    p_down = bond_price(face, coupon, ytm - yield_change, periods)
    p_up = bond_price(face, coupon, ytm + yield_change, periods)
    p = bond_price(face, coupon, ytm, periods)
    eff_dur = effective_duration(p_down, p_up, p, yield_change)
    assert eff_dur >= 0

@pytest.mark.parametrize('face', [1000, 10000])
@pytest.mark.parametrize('coupon', [0.03, 0.1])
@pytest.mark.parametrize('ytm', [0.03, 0.1])
@pytest.mark.parametrize('yield_change', [0.001, 0.02])
def test_effective_convexity_positive(face, coupon, ytm, yield_change):
    """Effective convexity should be positive for standard bonds."""
    periods = 20
    p_down = bond_price(face, coupon, ytm - yield_change, periods)
    p_up = bond_price(face, coupon, ytm + yield_change, periods)
    p = bond_price(face, coupon, ytm, periods)
    eff_conv = effective_convexity(p_down, p_up, p, yield_change)
    assert eff_conv >= -0.1