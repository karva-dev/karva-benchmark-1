"""Mega interest tests with heavy Cartesian product parametrization."""

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

PRINCIPALS = [100, 500, 1000, 5000, 10000, 50000, 100000]
RATES = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
TIMES = [0.25, 0.5, 1, 2, 5, 10, 20]
N_VALS = [1, 2, 4, 12, 52, 365]


# 7*7*7 = 343
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("time", TIMES)
def test_simple_interest_nonneg(principal, rate, time):
    """Simple interest should be non-negative for positive inputs."""
    result = simple_interest(principal, rate, time)
    assert result >= 0


# 7*7*7 = 343
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("time", TIMES)
def test_simple_interest_proportional_to_time(principal, rate, time):
    """Doubling time should double interest."""
    i1 = simple_interest(principal, rate, time)
    i2 = simple_interest(principal, rate, time * 2)
    assert abs(i2 - 2 * i1) < 0.01


# 7*7*7*6 = 2058
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("time", TIMES)
@pytest.mark.parametrize("n", N_VALS)
def test_compound_geq_principal(principal, rate, time, n):
    """Compound interest result should be >= principal."""
    result = compound_interest(principal, rate, time, n)
    assert result >= principal - 0.01


# 7*7*7*6 = 2058
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("time", TIMES)
@pytest.mark.parametrize("n", N_VALS)
def test_compound_geq_simple(principal, rate, time, n):
    """Compound interest total should be >= principal + simple interest when n*time >= 1."""
    compound = compound_interest(principal, rate, time, n)
    simple = principal + simple_interest(principal, rate, time)
    if n * time >= 1:
        assert compound >= simple - 0.01
    else:
        assert compound > 0


# 7*7*7*6 = 2058
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("time", TIMES)
@pytest.mark.parametrize("n", N_VALS)
def test_continuous_geq_compound(principal, rate, time, n):
    """Continuous compounding should be >= discrete compounding."""
    continuous = continuous_compound_interest(principal, rate, time)
    discrete = compound_interest(principal, rate, time, n)
    assert continuous >= discrete - 0.01


# 7*7*7*6 = 2058
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("time", TIMES)
@pytest.mark.parametrize("n", N_VALS)
def test_fv_pv_roundtrip(principal, rate, time, n):
    """Future value -> present value should recover principal."""
    fv = future_value(principal, rate, time, n)
    pv = present_value(fv, rate, time, n)
    assert abs(pv - principal) < 0.01 * principal + 0.01


# 7*7*7*6 = 2058
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("time", TIMES)
@pytest.mark.parametrize("n", N_VALS)
def test_compound_earned_nonneg(principal, rate, time, n):
    """Compound interest earned should be non-negative."""
    earned = compound_interest_earned(principal, rate, time, n)
    assert earned >= -0.01


# 7*7*7 = 343
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("time", TIMES)
def test_continuous_positive(principal, rate, time):
    """Continuous compounding result should be positive."""
    result = continuous_compound_interest(principal, rate, time)
    assert result > 0


# 7*7*7 = 343
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("time", TIMES)
def test_simple_interest_equals_formula(principal, rate, time):
    """Simple interest should equal P*R*T."""
    result = simple_interest(principal, rate, time)
    expected = principal * rate * time
    assert abs(result - expected) < 0.01


# 12*8 = 96
@pytest.mark.parametrize(
    "apr", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
)
@pytest.mark.parametrize("n", [1, 2, 4, 6, 12, 26, 52, 365])
def test_apr_apy_roundtrip(apr, n):
    """APR -> APY -> APR should recover original."""
    apy = apr_to_apy(apr, n)
    recovered = apy_to_apr(apy, n)
    assert abs(recovered - apr) < 0.0001


# 12*8 = 96
@pytest.mark.parametrize(
    "apr", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
)
@pytest.mark.parametrize("n", [2, 4, 6, 12, 26, 52, 365])
def test_apy_geq_apr(apr, n):
    """APY should be >= APR for n > 1."""
    apy = apr_to_apy(apr, n)
    assert apy >= apr - 0.0001


# 7*7 = 49
@pytest.mark.parametrize("nominal", RATES)
@pytest.mark.parametrize("n", [1, 2, 4, 6, 12, 52, 365])
def test_effective_rate_positive(nominal, n):
    """Effective rate should be positive."""
    eff = effective_interest_rate(nominal, n)
    assert eff > 0


# 7*7 = 49
@pytest.mark.parametrize("nominal", [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15])
@pytest.mark.parametrize("inflation", [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10])
def test_real_rate_fisher(nominal, inflation):
    """Real rate should satisfy Fisher equation."""
    real = real_interest_rate(nominal, inflation)
    recovered = (1 + real) * (1 + inflation) - 1
    assert abs(recovered - nominal) < 0.0001


# 12
@pytest.mark.parametrize(
    "rate", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
)
def test_doubling_time_positive(rate):
    """Doubling time should be positive."""
    dt = doubling_time(rate)
    assert dt > 0
    assert math.isfinite(dt)


# 4*4*4 = 64
@pytest.mark.parametrize("principal", [100, 1000, 10000, 100000])
@pytest.mark.parametrize("rate", [0.01, 0.05, 0.10, 0.15])
@pytest.mark.parametrize("n", [1, 4, 12, 365])
def test_compound_monotonic_time(principal, rate, n):
    """Compound interest should grow monotonically with time."""
    values = [compound_interest(principal, rate, t, n) for t in [1, 2, 5, 10, 20]]
    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1] + 0.01


# 4*4*4 = 64
@pytest.mark.parametrize("principal", [100, 1000, 10000, 100000])
@pytest.mark.parametrize("time", [1, 5, 10, 20])
@pytest.mark.parametrize("n", [1, 4, 12, 365])
def test_compound_monotonic_rate(principal, time, n):
    """Compound interest should grow monotonically with rate."""
    values = [
        compound_interest(principal, r, time, n) for r in [0.01, 0.03, 0.05, 0.10, 0.15]
    ]
    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1] + 0.01


# 4*4*4 = 64
@pytest.mark.parametrize("principal", [100, 1000, 10000, 100000])
@pytest.mark.parametrize("rate", [0.01, 0.05, 0.10, 0.15])
@pytest.mark.parametrize("time", [1, 5, 10, 20])
def test_compound_monotonic_n(principal, rate, time):
    """Higher compounding frequency should yield more."""
    values = [
        compound_interest(principal, rate, time, n) for n in [1, 2, 4, 12, 52, 365]
    ]
    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1] + 0.01


# 4*4*4*4 = 256
@pytest.mark.parametrize("principal", [100, 1000, 10000, 100000])
@pytest.mark.parametrize("rate", [0.01, 0.03, 0.05, 0.10])
@pytest.mark.parametrize("time", [0.5, 1, 5, 10])
@pytest.mark.parametrize("n", [1, 4, 12, 365])
def test_pv_less_than_fv(principal, rate, time, n):
    """Present value should be less than future value for positive rate."""
    fv = future_value(principal, rate, time, n)
    pv = present_value(fv, rate, time, n)
    assert pv <= fv + 0.01


# 7*7*7*6 = 2058
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("time", TIMES)
@pytest.mark.parametrize("n", N_VALS)
def test_compound_earned_equals_diff(principal, rate, time, n):
    """Interest earned should equal total minus principal."""
    total = compound_interest(principal, rate, time, n)
    earned = compound_interest_earned(principal, rate, time, n)
    assert abs(earned - (total - principal)) < 0.01
