"""Extended Monte Carlo tests with heavy parametrization."""

import math

import pytest

from finlib.montecarlo import (
    geometric_brownian_motion,
    simulate_paths,
    monte_carlo_var,
    monte_carlo_option_price,
    monte_carlo_asian_option,
    monte_carlo_barrier_option,
    simulate_portfolio,
    simulate_retirement,
)
from finlib.options import black_scholes_call, black_scholes_put


# GBM path property tests
@pytest.mark.parametrize("S0", [50, 100, 200])
@pytest.mark.parametrize("mu", [-0.05, 0.0, 0.05, 0.10])
@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30])
def test_gbm_starts_correct(S0: float, mu: float, sigma: float):
    """GBM path should start at S0."""
    path = geometric_brownian_motion(S0, mu, sigma, 1.0, 0.01, seed=42)
    assert path[0] == S0


@pytest.mark.parametrize("S0", [50, 100, 200])
@pytest.mark.parametrize("mu", [-0.05, 0.0, 0.05, 0.10])
@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30])
def test_gbm_all_positive(S0: float, mu: float, sigma: float):
    """GBM path should have all positive values."""
    path = geometric_brownian_motion(S0, mu, sigma, 1.0, 0.01, seed=42)
    assert all(p > 0 for p in path)


@pytest.mark.parametrize("S0", [100])
@pytest.mark.parametrize("T", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("dt", [0.01, 0.005, 0.002])
def test_gbm_path_length(S0: float, T: float, dt: float):
    """GBM path length should be T/dt + 1."""
    path = geometric_brownian_motion(S0, 0.05, 0.20, T, dt, seed=42)
    expected_len = int(T / dt) + 1
    assert len(path) == expected_len


# Multiple path simulations
@pytest.mark.parametrize("S0", [80, 100, 120])
@pytest.mark.parametrize("num_paths", [50, 100, 200])
@pytest.mark.parametrize("sigma", [0.15, 0.25, 0.35])
def test_simulate_paths_count(S0: float, num_paths: int, sigma: float):
    """Should generate correct number of paths."""
    paths = simulate_paths(S0, 0.08, sigma, 1.0, 100, num_paths, seed=42)
    assert len(paths) == num_paths


@pytest.mark.parametrize("S0", [80, 100, 120])
@pytest.mark.parametrize("num_paths", [50, 100])
@pytest.mark.parametrize("sigma", [0.15, 0.25, 0.35])
def test_simulate_paths_start(S0: float, num_paths: int, sigma: float):
    """All paths should start at S0."""
    paths = simulate_paths(S0, 0.08, sigma, 1.0, 100, num_paths, seed=42)
    assert all(p[0] == S0 for p in paths)


@pytest.mark.parametrize("S0", [80, 100, 120])
@pytest.mark.parametrize("num_paths", [50, 100])
@pytest.mark.parametrize("sigma", [0.15, 0.25, 0.35])
def test_simulate_paths_positive(S0: float, num_paths: int, sigma: float):
    """All path values should be positive."""
    paths = simulate_paths(S0, 0.08, sigma, 1.0, 100, num_paths, seed=42)
    for path in paths:
        assert all(p > 0 for p in path)


# Monte Carlo VaR tests
@pytest.mark.parametrize("value", [10000, 100000, 1000000])
@pytest.mark.parametrize("mu", [0.05, 0.08, 0.10])
@pytest.mark.parametrize("sigma", [0.12, 0.18, 0.25])
@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_mc_var_positive(value: float, mu: float, sigma: float, confidence: float):
    """VaR should be positive."""
    result = monte_carlo_var(value, mu, sigma, 1/252, confidence, 1000, seed=42)
    assert result["var"] >= 0


@pytest.mark.parametrize("value", [10000, 100000, 1000000])
@pytest.mark.parametrize("mu", [0.05, 0.08, 0.10])
@pytest.mark.parametrize("sigma", [0.12, 0.18, 0.25])
@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_mc_cvar_geq_var(value: float, mu: float, sigma: float, confidence: float):
    """CVaR should be >= VaR."""
    result = monte_carlo_var(value, mu, sigma, 1/252, confidence, 1000, seed=42)
    assert result["cvar"] >= result["var"]


@pytest.mark.parametrize("value", [100000])
@pytest.mark.parametrize("sigma", [0.15, 0.20, 0.25])
@pytest.mark.parametrize("T", [1/252, 5/252, 10/252])
@pytest.mark.parametrize("num_sims", [500, 1000, 2000])
def test_mc_var_simulations(value: float, sigma: float, T: float, num_sims: int):
    """Test VaR with various simulation counts."""
    result = monte_carlo_var(value, 0.08, sigma, T, 0.95, num_sims, seed=42)
    assert result["var"] > 0


# Monte Carlo option pricing
@pytest.mark.parametrize("S", [90, 100, 110])
@pytest.mark.parametrize("K", [95, 100, 105])
@pytest.mark.parametrize("sigma", [0.20, 0.30])
@pytest.mark.parametrize("T", [0.5, 1.0])
def test_mc_call_positive(S: float, K: float, sigma: float, T: float):
    """MC call price should be positive."""
    result = monte_carlo_option_price(S, K, 0.05, sigma, T, "call", 2000, seed=42)
    assert result["price"] >= 0


@pytest.mark.parametrize("S", [90, 100, 110])
@pytest.mark.parametrize("K", [95, 100, 105])
@pytest.mark.parametrize("sigma", [0.20, 0.30])
@pytest.mark.parametrize("T", [0.5, 1.0])
def test_mc_put_positive(S: float, K: float, sigma: float, T: float):
    """MC put price should be positive."""
    result = monte_carlo_option_price(S, K, 0.05, sigma, T, "put", 2000, seed=42)
    assert result["price"] >= 0


@pytest.mark.parametrize("S", [90, 100, 110])
@pytest.mark.parametrize("K", [95, 100, 105])
@pytest.mark.parametrize("sigma", [0.20, 0.30])
def test_mc_call_convergence(S: float, K: float, sigma: float):
    """MC call should converge to BS with enough simulations."""
    T, r = 1.0, 0.05
    bs_price = black_scholes_call(S, K, r, sigma, T)
    mc_result = monte_carlo_option_price(S, K, r, sigma, T, "call", 5000, seed=42)
    assert abs(mc_result["price"] - bs_price) < bs_price * 0.1


@pytest.mark.parametrize("S", [90, 100, 110])
@pytest.mark.parametrize("K", [95, 100, 105])
@pytest.mark.parametrize("sigma", [0.20, 0.30])
def test_mc_put_convergence(S: float, K: float, sigma: float):
    """MC put should converge to BS with enough simulations."""
    T, r = 1.0, 0.05
    bs_price = black_scholes_put(S, K, r, sigma, T)
    mc_result = monte_carlo_option_price(S, K, r, sigma, T, "put", 5000, seed=42)
    assert abs(mc_result["price"] - bs_price) < bs_price * 0.15


# Asian option tests
@pytest.mark.parametrize("S", [90, 100, 110])
@pytest.mark.parametrize("K", [95, 100, 105])
@pytest.mark.parametrize("sigma", [0.20, 0.30])
@pytest.mark.parametrize("steps", [12, 52])
def test_asian_call_positive(S: float, K: float, sigma: float, steps: int):
    """Asian call should be positive."""
    result = monte_carlo_asian_option(S, K, 0.05, sigma, 1.0, steps, "call", 1000, seed=42)
    assert result["price"] >= 0


@pytest.mark.parametrize("S", [90, 100, 110])
@pytest.mark.parametrize("K", [95, 100, 105])
@pytest.mark.parametrize("sigma", [0.20, 0.30])
@pytest.mark.parametrize("steps", [12, 52])
def test_asian_put_positive(S: float, K: float, sigma: float, steps: int):
    """Asian put should be positive."""
    result = monte_carlo_asian_option(S, K, 0.05, sigma, 1.0, steps, "put", 1000, seed=42)
    assert result["price"] >= 0


@pytest.mark.parametrize("S", [90, 100, 110])
@pytest.mark.parametrize("K", [100])
@pytest.mark.parametrize("sigma", [0.25, 0.35])
def test_asian_cheaper_than_european(S: float, K: float, sigma: float):
    """Asian options are typically cheaper due to averaging."""
    asian = monte_carlo_asian_option(S, K, 0.05, sigma, 1.0, 50, "call", 3000, seed=42)
    euro = monte_carlo_option_price(S, K, 0.05, sigma, 1.0, "call", 3000, seed=42)
    # Asian should be cheaper or similar
    assert asian["price"] < euro["price"] * 1.2


# Barrier option tests
@pytest.mark.parametrize("S", [95, 100, 105])
@pytest.mark.parametrize("barrier", [115, 120, 125])
@pytest.mark.parametrize("sigma", [0.20, 0.30])
def test_up_out_call_positive(S: float, barrier: float, sigma: float):
    """Up-and-out call should be positive."""
    result = monte_carlo_barrier_option(
        S, 100, barrier, 0.05, sigma, 1.0, 100, "call", "up-and-out", 1000, seed=42
    )
    assert result["price"] >= 0


@pytest.mark.parametrize("S", [95, 100, 105])
@pytest.mark.parametrize("barrier", [75, 80, 85])
@pytest.mark.parametrize("sigma", [0.20, 0.30])
def test_down_out_put_positive(S: float, barrier: float, sigma: float):
    """Down-and-out put should be positive."""
    result = monte_carlo_barrier_option(
        S, 100, barrier, 0.05, sigma, 1.0, 100, "put", "down-and-out", 1000, seed=42
    )
    assert result["price"] >= 0


@pytest.mark.parametrize("S", [95, 100, 105])
@pytest.mark.parametrize("barrier", [115, 120])
@pytest.mark.parametrize("sigma", [0.20, 0.30])
def test_barrier_cheaper_than_vanilla(S: float, barrier: float, sigma: float):
    """Knock-out options should be cheaper than vanilla."""
    barrier_result = monte_carlo_barrier_option(
        S, 100, barrier, 0.05, sigma, 1.0, 100, "call", "up-and-out", 2000, seed=42
    )
    vanilla_result = monte_carlo_option_price(S, 100, 0.05, sigma, 1.0, "call", 2000, seed=42)
    assert barrier_result["price"] <= vanilla_result["price"] + 0.1


# Portfolio simulation tests
@pytest.mark.parametrize("num_paths", [100, 200, 500])
@pytest.mark.parametrize("T", [0.5, 1.0, 2.0])
def test_portfolio_simulation_count(num_paths: int, T: float):
    """Portfolio simulation should return correct number of values."""
    values = [5000, 3000, 2000]
    returns = [0.08, 0.06, 0.10]
    vols = [0.15, 0.10, 0.20]
    corr = [[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]]
    result = simulate_portfolio(values, returns, vols, corr, T, num_paths, seed=42)
    assert len(result) == num_paths


@pytest.mark.parametrize("returns", [[0.05, 0.05], [0.08, 0.08], [0.10, 0.10]])
@pytest.mark.parametrize("vols", [[0.10, 0.10], [0.15, 0.15], [0.20, 0.20]])
def test_portfolio_simulation_positive(returns: list, vols: list):
    """Most portfolio values should be positive."""
    values = [5000, 5000]
    corr = [[1.0, 0.5], [0.5, 1.0]]
    result = simulate_portfolio(values, returns, vols, corr, 1.0, 200, seed=42)
    positive_count = sum(1 for v in result if v > 0)
    assert positive_count > 180  # >90% positive


# Retirement simulation tests
@pytest.mark.parametrize("initial", [100000, 250000, 500000])
@pytest.mark.parametrize("contribution", [10000, 20000, 30000])
@pytest.mark.parametrize("return_rate", [0.05, 0.07, 0.09])
def test_retirement_mean_positive(initial: float, contribution: float, return_rate: float):
    """Mean final value should be positive."""
    result = simulate_retirement(
        initial, contribution, return_rate, 0.15, 20, 25, 40000, 100, seed=42
    )
    assert result["mean_final_value"] >= 0


@pytest.mark.parametrize("initial", [100000, 250000])
@pytest.mark.parametrize("withdrawal", [30000, 50000, 80000])
@pytest.mark.parametrize("volatility", [0.10, 0.15, 0.20])
def test_retirement_ruin_probability_range(initial: float, withdrawal: float, volatility: float):
    """Ruin probability should be between 0 and 1."""
    result = simulate_retirement(
        initial, 20000, 0.07, volatility, 15, 25, withdrawal, 100, seed=42
    )
    assert 0 <= result["probability_of_ruin"] <= 1


@pytest.mark.parametrize("years_saving", [10, 20, 30])
@pytest.mark.parametrize("years_retirement", [20, 25, 30])
def test_retirement_percentiles_ordered(years_saving: int, years_retirement: int):
    """Percentiles should be in order."""
    result = simulate_retirement(
        200000, 25000, 0.07, 0.15, years_saving, years_retirement, 50000, 100, seed=42
    )
    assert result["percentile_10"] <= result["median_final_value"]
    assert result["median_final_value"] <= result["percentile_90"]


# Large simulation tests (slow)
@pytest.mark.parametrize("S", [100])
@pytest.mark.parametrize("K", [95, 100, 105])
@pytest.mark.parametrize("num_sims", [5000, 10000])
def test_large_mc_option_call(S: float, K: float, num_sims: int):
    """Test with large number of simulations."""
    result = monte_carlo_option_price(S, K, 0.05, 0.25, 1.0, "call", num_sims, seed=42)
    bs = black_scholes_call(S, K, 0.05, 0.25, 1.0)
    assert abs(result["price"] - bs) < bs * 0.05


@pytest.mark.parametrize("S", [100])
@pytest.mark.parametrize("K", [95, 100, 105])
@pytest.mark.parametrize("num_sims", [5000, 10000])
def test_large_mc_option_put(S: float, K: float, num_sims: int):
    """Test with large number of simulations."""
    result = monte_carlo_option_price(S, K, 0.05, 0.25, 1.0, "put", num_sims, seed=42)
    bs = black_scholes_put(S, K, 0.05, 0.25, 1.0)
    assert abs(result["price"] - bs) < bs * 0.1


@pytest.mark.parametrize("num_paths", [500, 1000])
@pytest.mark.parametrize("steps", [252, 504])
def test_large_path_simulation(num_paths: int, steps: int):
    """Test simulating many paths with many steps."""
    paths = simulate_paths(100, 0.08, 0.20, 1.0, steps, num_paths, seed=42)
    assert len(paths) == num_paths
    assert all(len(p) == steps + 1 for p in paths)
