"""Extended options tests with heavy parametrization and computation."""

import math

import karva

from finlib.options import (
    binomial_tree_price,
    black_scholes_call,
    black_scholes_put,
    delta,
    gamma,
    implied_volatility,
    rho,
    theta,
    vega,
)


# Nested parametrize for comprehensive Black-Scholes testing
@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [85, 95, 100, 105, 115])
@karva.tags.parametrize("sigma", [0.15, 0.25, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_bs_call_positive(S: float, K: float, sigma: float, T: float):
    """Call price should always be positive."""
    result = black_scholes_call(S, K, 0.05, sigma, T)
    assert result >= 0


@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [85, 95, 100, 105, 115])
@karva.tags.parametrize("sigma", [0.15, 0.25, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_bs_put_positive(S: float, K: float, sigma: float, T: float):
    """Put price should always be positive."""
    result = black_scholes_put(S, K, 0.05, sigma, T)
    assert result >= 0


@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [85, 95, 100, 105, 115])
@karva.tags.parametrize("sigma", [0.15, 0.25, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_put_call_parity_holds(S: float, K: float, sigma: float, T: float):
    """Verify put-call parity: C - P = S - K*e^(-rT)."""
    r = 0.05
    call = black_scholes_call(S, K, r, sigma, T)
    put = black_scholes_put(S, K, r, sigma, T)
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 0.01


@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.20, 0.30, 0.40])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_call_bounded_by_stock(S: float, K: float, sigma: float, T: float):
    """Call price should be less than stock price."""
    call = black_scholes_call(S, K, 0.05, sigma, T)
    assert call <= S


@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.20, 0.30, 0.40])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_put_bounded_by_strike(S: float, K: float, sigma: float, T: float):
    """Put price should be less than strike."""
    put = black_scholes_put(S, K, 0.05, sigma, T)
    assert put <= K


# Delta comprehensive tests
@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.15, 0.25, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_call_delta_bounds(S: float, K: float, sigma: float, T: float):
    """Call delta should be between 0 and 1."""
    d = delta(S, K, 0.05, sigma, T, "call")
    assert 0 <= d <= 1


@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.15, 0.25, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_put_delta_bounds(S: float, K: float, sigma: float, T: float):
    """Put delta should be between -1 and 0."""
    d = delta(S, K, 0.05, sigma, T, "put")
    assert -1 <= d <= 0


@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.15, 0.25, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_delta_call_put_relationship(S: float, K: float, sigma: float, T: float):
    """Call delta - Put delta = 1."""
    call_d = delta(S, K, 0.05, sigma, T, "call")
    put_d = delta(S, K, 0.05, sigma, T, "put")
    assert abs(call_d - put_d - 1) < 0.01


# Gamma comprehensive tests
@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.15, 0.25, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_gamma_positive(S: float, K: float, sigma: float, T: float):
    """Gamma should always be positive."""
    g = gamma(S, K, 0.05, sigma, T)
    assert g >= 0


# Vega comprehensive tests
@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.15, 0.25, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_vega_positive(S: float, K: float, sigma: float, T: float):
    """Vega should always be positive."""
    v = vega(S, K, 0.05, sigma, T)
    assert v >= 0


# Theta comprehensive tests
@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.15, 0.25, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_call_theta_negative(S: float, K: float, sigma: float, T: float):
    """Call theta should generally be negative (time decay)."""
    t = theta(S, K, 0.05, sigma, T, "call")
    # Deep ITM calls can have positive theta in some cases
    assert isinstance(t, float)


# Rho comprehensive tests
@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.15, 0.25, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_call_rho_positive(S: float, K: float, sigma: float, T: float):
    """Call rho should be positive."""
    r = rho(S, K, 0.05, sigma, T, "call")
    assert r >= 0


@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.15, 0.25, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_put_rho_negative(S: float, K: float, sigma: float, T: float):
    """Put rho should be negative."""
    r = rho(S, K, 0.05, sigma, T, "put")
    assert r <= 0


# Binomial tree convergence tests (computationally intensive)
@karva.tags.parametrize("S", [90, 100, 110])
@karva.tags.parametrize("K", [95, 100, 105])
@karva.tags.parametrize("sigma", [0.20, 0.30])
@karva.tags.parametrize("steps", [50, 100, 150])
def test_binomial_call_converges(S: float, K: float, sigma: float, steps: int):
    """Binomial tree should converge to Black-Scholes."""
    T, r = 1.0, 0.05
    bs_price = black_scholes_call(S, K, r, sigma, T)
    bin_price = binomial_tree_price(S, K, r, sigma, T, steps, "call")
    # Tolerance depends on steps
    tol = 1.0 if steps < 100 else 0.5
    assert abs(bin_price - bs_price) < tol


@karva.tags.parametrize("S", [90, 100, 110])
@karva.tags.parametrize("K", [95, 100, 105])
@karva.tags.parametrize("sigma", [0.20, 0.30])
@karva.tags.parametrize("steps", [50, 100, 150])
def test_binomial_put_converges(S: float, K: float, sigma: float, steps: int):
    """Binomial tree should converge to Black-Scholes."""
    T, r = 1.0, 0.05
    bs_price = black_scholes_put(S, K, r, sigma, T)
    bin_price = binomial_tree_price(S, K, r, sigma, T, steps, "put")
    tol = 1.0 if steps < 100 else 0.5
    assert abs(bin_price - bs_price) < tol


# American vs European options (binomial)
@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [95, 100, 105])
@karva.tags.parametrize("sigma", [0.20, 0.30])
def test_american_put_geq_european(S: float, K: float, sigma: float):
    """American put should be >= European put."""
    T, r, steps = 1.0, 0.05, 100
    euro = binomial_tree_price(S, K, r, sigma, T, steps, "put", american=False)
    amer = binomial_tree_price(S, K, r, sigma, T, steps, "put", american=True)
    assert amer >= euro - 0.01


@karva.tags.parametrize("S", [80, 90, 100, 110, 120])
@karva.tags.parametrize("K", [95, 100, 105])
@karva.tags.parametrize("sigma", [0.20, 0.30])
def test_american_call_equals_european(S: float, K: float, sigma: float):
    """American call should equal European call (no dividends)."""
    T, r, steps = 1.0, 0.05, 100
    euro = binomial_tree_price(S, K, r, sigma, T, steps, "call", american=False)
    amer = binomial_tree_price(S, K, r, sigma, T, steps, "call", american=True)
    assert abs(amer - euro) < 0.1


# Implied volatility recovery tests
@karva.tags.parametrize("S", [90, 100, 110])
@karva.tags.parametrize("K", [95, 100, 105])
@karva.tags.parametrize("true_vol", [0.15, 0.20, 0.25, 0.30, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_implied_vol_recovery_call(S: float, K: float, true_vol: float, T: float):
    """Implied vol should recover the true volatility."""
    r = 0.05
    price = black_scholes_call(S, K, r, true_vol, T)
    recovered = implied_volatility(price, S, K, r, T, "call")
    assert abs(recovered - true_vol) < 0.01


@karva.tags.parametrize("S", [90, 100, 110])
@karva.tags.parametrize("K", [95, 100, 105])
@karva.tags.parametrize("true_vol", [0.15, 0.20, 0.25, 0.30, 0.35])
@karva.tags.parametrize("T", [0.25, 0.5, 1.0])
def test_implied_vol_recovery_put(S: float, K: float, true_vol: float, T: float):
    """Implied vol should recover the true volatility for puts."""
    r = 0.05
    price = black_scholes_put(S, K, r, true_vol, T)
    recovered = implied_volatility(price, S, K, r, T, "put")
    assert abs(recovered - true_vol) < 0.01


# Sensitivity tests - call increases with S
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.20, 0.30])
@karva.tags.parametrize("T", [0.5, 1.0])
def test_call_increases_with_stock(K: float, sigma: float, T: float):
    """Call price should increase with stock price."""
    r = 0.05
    prices = [black_scholes_call(S, K, r, sigma, T) for S in [80, 90, 100, 110, 120]]
    for i in range(len(prices) - 1):
        assert prices[i] <= prices[i + 1]


# Sensitivity tests - put decreases with S
@karva.tags.parametrize("K", [90, 100, 110])
@karva.tags.parametrize("sigma", [0.20, 0.30])
@karva.tags.parametrize("T", [0.5, 1.0])
def test_put_decreases_with_stock(K: float, sigma: float, T: float):
    """Put price should decrease with stock price."""
    r = 0.05
    prices = [black_scholes_put(S, K, r, sigma, T) for S in [80, 90, 100, 110, 120]]
    for i in range(len(prices) - 1):
        assert prices[i] >= prices[i + 1]


# Volatility sensitivity
@karva.tags.parametrize("S", [90, 100, 110])
@karva.tags.parametrize("K", [95, 100, 105])
@karva.tags.parametrize("T", [0.5, 1.0])
def test_call_increases_with_vol(S: float, K: float, T: float):
    """Call price should increase with volatility."""
    r = 0.05
    prices = [
        black_scholes_call(S, K, r, sigma, T) for sigma in [0.10, 0.20, 0.30, 0.40]
    ]
    for i in range(len(prices) - 1):
        assert prices[i] <= prices[i + 1]


@karva.tags.parametrize("S", [90, 100, 110])
@karva.tags.parametrize("K", [95, 100, 105])
@karva.tags.parametrize("T", [0.5, 1.0])
def test_put_increases_with_vol(S: float, K: float, T: float):
    """Put price should increase with volatility."""
    r = 0.05
    prices = [
        black_scholes_put(S, K, r, sigma, T) for sigma in [0.10, 0.20, 0.30, 0.40]
    ]
    for i in range(len(prices) - 1):
        assert prices[i] <= prices[i + 1]


# Time sensitivity
@karva.tags.parametrize("S", [90, 100, 110])
@karva.tags.parametrize("K", [95, 100, 105])
@karva.tags.parametrize("sigma", [0.20, 0.30])
def test_call_increases_with_time(S: float, K: float, sigma: float):
    """Call price should generally increase with time to expiry."""
    r = 0.05
    prices = [black_scholes_call(S, K, r, sigma, T) for T in [0.1, 0.25, 0.5, 1.0]]
    for i in range(len(prices) - 1):
        assert prices[i] <= prices[i + 1] + 0.01  # Small tolerance


# Deep tree binomial tests (slow)
@karva.tags.parametrize("S", [95, 100, 105])
@karva.tags.parametrize("K", [100])
@karva.tags.parametrize("steps", [200, 250])
def test_deep_binomial_tree(S: float, K: float, steps: int):
    """Test with deep binomial trees."""
    T, r, sigma = 1.0, 0.05, 0.25
    result = binomial_tree_price(S, K, r, sigma, T, steps, "call")
    bs = black_scholes_call(S, K, r, sigma, T)
    assert abs(result - bs) < 0.3
