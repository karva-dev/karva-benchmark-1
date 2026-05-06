"""Mega currency tests with heavy Cartesian product parametrization."""
import pytest
from finlib.currency import direct_to_indirect, indirect_to_direct, convert_currency, cross_rate, triangular_arbitrage, forward_rate, forward_points, swap_points_to_rate, currency_basket_value, bid_ask_spread, mid_rate, pip_value, position_size, real_exchange_rate, purchasing_power_parity_rate
RATES = [0.5, 1.1, 3.0]
AMOUNTS = [1, 1000, 100000]
DOM_RATES = [0.005, 0.03, 0.1]
FOR_RATES = [0.005, 0.03, 0.1]
DAYS = [1, 90, 360]
BASIS = [360, 365]

@pytest.mark.parametrize('rate', RATES)
@pytest.mark.parametrize('amount', AMOUNTS)
def test_convert_direct_positive(rate, amount):
    """Direct conversion should yield positive result."""
    result = convert_currency(amount, rate, is_direct=True)
    assert result > 0

@pytest.mark.parametrize('rate', RATES)
@pytest.mark.parametrize('amount', AMOUNTS)
def test_convert_indirect_positive(rate, amount):
    """Indirect conversion should yield positive result."""
    result = convert_currency(amount, rate, is_direct=False)
    assert result > 0

@pytest.mark.parametrize('rate', RATES)
def test_direct_indirect_roundtrip(rate):
    """Direct -> indirect -> direct should recover original."""
    indirect = direct_to_indirect(rate)
    recovered = indirect_to_direct(indirect)
    assert abs(recovered - rate) < 0.0001

@pytest.mark.parametrize('rate_a', RATES)
@pytest.mark.parametrize('rate_b', RATES)
def test_cross_rate_positive(rate_a, rate_b):
    """Cross rate should be positive."""
    cr = cross_rate(rate_a, rate_b)
    assert cr > 0

@pytest.mark.parametrize('rate_a', RATES)
@pytest.mark.parametrize('rate_b', RATES)
def test_cross_rate_transitive(rate_a, rate_b):
    """Cross rate A/B * B/A should be 1."""
    cr_ab = cross_rate(rate_a, rate_b)
    cr_ba = cross_rate(rate_b, rate_a)
    assert abs(cr_ab * cr_ba - 1.0) < 0.0001

@pytest.mark.parametrize('spot', RATES)
@pytest.mark.parametrize('dom_rate', DOM_RATES)
@pytest.mark.parametrize('for_rate', FOR_RATES)
@pytest.mark.parametrize('days', DAYS)
@pytest.mark.parametrize('basis', BASIS)
def test_forward_rate_positive(spot, dom_rate, for_rate, days, basis):
    """Forward rate should be positive."""
    fwd = forward_rate(spot, dom_rate, for_rate, days, basis)
    assert fwd > 0

@pytest.mark.parametrize('spot', RATES)
@pytest.mark.parametrize('dom_rate', DOM_RATES)
@pytest.mark.parametrize('for_rate', FOR_RATES)
@pytest.mark.parametrize('days', DAYS)
def test_forward_rate_sign_convention(spot, dom_rate, for_rate, days):
    """If dom > for, forward > spot (forward premium)."""
    fwd = forward_rate(spot, dom_rate, for_rate, days)
    if dom_rate > for_rate:
        assert fwd >= spot - 0.001
    elif dom_rate < for_rate:
        assert fwd <= spot + 0.001

@pytest.mark.parametrize('spot', RATES)
@pytest.mark.parametrize('fwd_rate', RATES)
@pytest.mark.parametrize('pip_sz', [0.0001, 0.1, 0.0005])
def test_forward_points_calculation(spot, fwd_rate, pip_sz):
    """Forward points should equal (fwd - spot) / pip_size."""
    pts = forward_points(spot, fwd_rate, pip_sz)
    expected = (fwd_rate - spot) / pip_sz
    assert abs(pts - expected) < 0.01

@pytest.mark.parametrize('spot', RATES)
@pytest.mark.parametrize('swap_pts', [-500, 10, 2000])
@pytest.mark.parametrize('pip_sz', [0.0001, 0.1, 0.0005])
def test_swap_points_roundtrip(spot, swap_pts, pip_sz):
    """Swap points to rate should be reversible."""
    fwd = swap_points_to_rate(spot, swap_pts, pip_sz)
    recovered_pts = forward_points(spot, fwd, pip_sz)
    assert abs(recovered_pts - swap_pts) < 0.01

@pytest.mark.parametrize('rate_ab', [0.8, 1.1, 1.5])
@pytest.mark.parametrize('rate_bc', [0.7, 1.0, 1.4])
@pytest.mark.parametrize('rate_ca', [0.6, 1.0, 1.3])
def test_triangular_arb_product(rate_ab, rate_bc, rate_ca):
    """Triangular arbitrage product should be computed."""
    result = triangular_arbitrage(rate_ab, rate_bc, rate_ca)
    expected_product = rate_ab * rate_bc * rate_ca
    assert abs(result['product'] - expected_product) < 0.0001

@pytest.mark.parametrize('rate_ab', [0.8, 1.1, 1.5])
@pytest.mark.parametrize('rate_bc', [0.7, 1.0, 1.4])
@pytest.mark.parametrize('rate_ca', [0.6, 1.0, 1.3])
def test_triangular_arb_direction(rate_ab, rate_bc, rate_ca):
    """Arbitrage direction should match product magnitude."""
    result = triangular_arbitrage(rate_ab, rate_bc, rate_ca)
    product = rate_ab * rate_bc * rate_ca
    if result['has_arbitrage']:
        if product > 1:
            assert result['direction'] == 'ABC'
        else:
            assert result['direction'] == 'CBA'

@pytest.mark.parametrize('bid_offset', [-0.01, -0.001, -0.1])
@pytest.mark.parametrize('ask_offset', [0.001, 0.01, 0.1])
@pytest.mark.parametrize('mid_val', RATES)
@pytest.mark.parametrize('lot', [1000, 500000, 5000000])
@pytest.mark.parametrize('pip_sz', [0.0001, 0.1, 0.0005])
def test_pip_value_positive(bid_offset, ask_offset, mid_val, lot, pip_sz):
    """Pip value should be positive."""
    pv = pip_value(lot, pip_sz, mid_val)
    assert pv > 0

@pytest.mark.parametrize('nominal_rate', RATES)
@pytest.mark.parametrize('dom_price', [80, 120, 200])
@pytest.mark.parametrize('for_price', [70, 115, 160])
def test_real_exchange_rate_positive(nominal_rate, dom_price, for_price):
    """Real exchange rate should be positive."""
    rer = real_exchange_rate(nominal_rate, dom_price, for_price)
    assert rer > 0

@pytest.mark.parametrize('dom_price', [80, 120, 200])
@pytest.mark.parametrize('for_price', [70, 115, 160])
def test_ppp_rate_positive(dom_price, for_price):
    """PPP rate should be positive."""
    ppp = purchasing_power_parity_rate(dom_price, for_price)
    assert ppp > 0

@pytest.mark.parametrize('bid', [0.99, 1.2, 1.6])
@pytest.mark.parametrize('spread_bps', [1, 20, 200])
def test_bid_ask_spread_positive(bid, spread_bps):
    """Bid-ask spread should be positive."""
    ask = bid + spread_bps * 0.0001
    spread = bid_ask_spread(bid, ask)
    assert spread >= 0

@pytest.mark.parametrize('bid', [0.99, 1.2, 1.6])
@pytest.mark.parametrize('spread_bps', [1, 20, 200])
def test_mid_rate_between_bid_ask(bid, spread_bps):
    """Mid rate should be between bid and ask."""
    ask = bid + spread_bps * 0.0001
    mid = mid_rate(bid, ask)
    assert bid <= mid <= ask

@pytest.mark.parametrize('balance', [10000, 100000, 1000000])
@pytest.mark.parametrize('risk_pct', [0.005, 0.02, 0.05])
@pytest.mark.parametrize('sl_pips', [10, 50, 200])
@pytest.mark.parametrize('pv', [0.1, 10.0, 1000.0])
def test_position_size_positive(balance, risk_pct, sl_pips, pv):
    """Position size should be positive."""
    ps = position_size(balance, risk_pct, sl_pips, pv)
    assert ps > 0