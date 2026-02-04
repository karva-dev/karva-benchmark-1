"""Tests for Monte Carlo simulations."""

import math

import pytest

from finlib.montecarlo import (
    geometric_brownian_motion,
    simulate_paths,
    simulate_portfolio,
    monte_carlo_var,
    monte_carlo_option_price,
    monte_carlo_asian_option,
    monte_carlo_barrier_option,
    convergence_test,
    simulate_retirement,
)
from finlib.options import black_scholes_call, black_scholes_put


# GBM Tests
def test_gbm_starts_at_s0():
    path = geometric_brownian_motion(100, 0.10, 0.20, 1.0, 0.01, seed=42)
    assert path[0] == 100


def test_gbm_path_length():
    path = geometric_brownian_motion(100, 0.10, 0.20, 1.0, 0.01, seed=42)
    expected_length = int(1.0 / 0.01) + 1
    assert len(path) == expected_length


def test_gbm_positive_prices():
    path = geometric_brownian_motion(100, 0.10, 0.20, 1.0, 0.01, seed=42)
    assert all(p > 0 for p in path)


def test_gbm_reproducible():
    path1 = geometric_brownian_motion(100, 0.10, 0.20, 1.0, 0.01, seed=42)
    path2 = geometric_brownian_motion(100, 0.10, 0.20, 1.0, 0.01, seed=42)
    assert path1 == path2


@pytest.mark.parametrize("mu", [-0.10, 0.0, 0.10, 0.20])
def test_gbm_various_drifts(mu: float):
    path = geometric_brownian_motion(100, mu, 0.20, 1.0, 0.01, seed=42)
    assert len(path) > 0


@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30, 0.50])
def test_gbm_various_volatilities(sigma: float):
    path = geometric_brownian_motion(100, 0.10, sigma, 1.0, 0.01, seed=42)
    assert all(p > 0 for p in path)


# Simulate Paths Tests
def test_simulate_paths_count():
    paths = simulate_paths(100, 0.10, 0.20, 1.0, 100, 50, seed=42)
    assert len(paths) == 50


def test_simulate_paths_length():
    paths = simulate_paths(100, 0.10, 0.20, 1.0, 100, 10, seed=42)
    assert all(len(p) == 101 for p in paths)


def test_simulate_paths_start_values():
    paths = simulate_paths(100, 0.10, 0.20, 1.0, 100, 10, seed=42)
    assert all(p[0] == 100 for p in paths)


def test_simulate_paths_positive():
    paths = simulate_paths(100, 0.10, 0.20, 1.0, 100, 50, seed=42)
    for path in paths:
        assert all(p > 0 for p in path)


# Simulate Portfolio Tests
def test_simulate_portfolio_count():
    values = [5000, 3000, 2000]
    returns = [0.08, 0.06, 0.10]
    vols = [0.15, 0.10, 0.20]
    corr = [[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]]
    result = simulate_portfolio(values, returns, vols, corr, 1.0, 100, seed=42)
    assert len(result) == 100


def test_simulate_portfolio_positive():
    values = [5000, 5000]
    returns = [0.08, 0.06]
    vols = [0.15, 0.10]
    corr = [[1.0, 0.5], [0.5, 1.0]]
    result = simulate_portfolio(values, returns, vols, corr, 1.0, 100, seed=42)
    # Most values should be positive
    positive_count = sum(1 for v in result if v > 0)
    assert positive_count > 90


def test_simulate_portfolio_reproducible():
    values = [5000, 5000]
    returns = [0.08, 0.06]
    vols = [0.15, 0.10]
    corr = [[1.0, 0.5], [0.5, 1.0]]
    result1 = simulate_portfolio(values, returns, vols, corr, 1.0, 50, seed=42)
    result2 = simulate_portfolio(values, returns, vols, corr, 1.0, 50, seed=42)
    assert result1 == result2


# Monte Carlo VaR Tests
def test_mc_var_basic():
    result = monte_carlo_var(100000, 0.08, 0.15, 1/252, 0.95, 1000, seed=42)
    assert "var" in result
    assert "cvar" in result
    assert result["var"] > 0


def test_mc_var_cvar_greater():
    result = monte_carlo_var(100000, 0.08, 0.15, 1/252, 0.95, 1000, seed=42)
    assert result["cvar"] >= result["var"]


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_mc_var_confidence_levels(confidence: float):
    result = monte_carlo_var(100000, 0.08, 0.15, 1/252, confidence, 1000, seed=42)
    assert result["var"] > 0


def test_mc_var_higher_confidence_higher_var():
    var_90 = monte_carlo_var(100000, 0.08, 0.15, 1/252, 0.90, 1000, seed=42)["var"]
    var_95 = monte_carlo_var(100000, 0.08, 0.15, 1/252, 0.95, 1000, seed=42)["var"]
    var_99 = monte_carlo_var(100000, 0.08, 0.15, 1/252, 0.99, 1000, seed=42)["var"]
    assert var_90 < var_95 < var_99


def test_mc_var_more_simulations():
    result1 = monte_carlo_var(100000, 0.08, 0.15, 1/252, 0.95, 100, seed=42)
    result2 = monte_carlo_var(100000, 0.08, 0.15, 1/252, 0.95, 5000, seed=42)
    # Both should give reasonable results
    assert result1["var"] > 0
    assert result2["var"] > 0


# Monte Carlo Option Pricing Tests
def test_mc_option_call_convergence():
    S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    bs_price = black_scholes_call(S, K, r, sigma, T)
    mc_result = monte_carlo_option_price(S, K, r, sigma, T, "call", 10000, seed=42)
    # Should be within a few percent
    assert abs(mc_result["price"] - bs_price) < bs_price * 0.1


def test_mc_option_put_convergence():
    S, K, r, sigma, T = 100, 100, 0.05, 0.20, 1.0
    bs_price = black_scholes_put(S, K, r, sigma, T)
    mc_result = monte_carlo_option_price(S, K, r, sigma, T, "put", 10000, seed=42)
    assert abs(mc_result["price"] - bs_price) < bs_price * 0.15


def test_mc_option_has_std_error():
    result = monte_carlo_option_price(100, 100, 0.05, 0.20, 1.0, "call", 1000, seed=42)
    assert "std_error" in result
    assert result["std_error"] > 0


@pytest.mark.parametrize("K", [90, 95, 100, 105, 110])
def test_mc_option_various_strikes(K: float):
    result = monte_carlo_option_price(100, K, 0.05, 0.20, 1.0, "call", 1000, seed=42)
    assert result["price"] >= 0


def test_mc_option_std_error_decreases():
    result1 = monte_carlo_option_price(100, 100, 0.05, 0.20, 1.0, "call", 100, seed=42)
    result2 = monte_carlo_option_price(100, 100, 0.05, 0.20, 1.0, "call", 10000, seed=42)
    # Std error should be lower with more simulations
    assert result2["std_error"] < result1["std_error"]


# Asian Option Tests
def test_mc_asian_option_basic():
    result = monte_carlo_asian_option(100, 100, 0.05, 0.20, 1.0, 50, "call", 1000, seed=42)
    assert "price" in result
    assert result["price"] >= 0


def test_mc_asian_cheaper_than_european():
    asian = monte_carlo_asian_option(100, 100, 0.05, 0.30, 1.0, 50, "call", 5000, seed=42)
    european = monte_carlo_option_price(100, 100, 0.05, 0.30, 1.0, "call", 5000, seed=42)
    # Asian options are typically cheaper due to averaging
    assert asian["price"] < european["price"] * 1.1


@pytest.mark.parametrize("steps", [12, 52, 252])
def test_mc_asian_various_averaging(steps: int):
    result = monte_carlo_asian_option(100, 100, 0.05, 0.20, 1.0, steps, "call", 500, seed=42)
    assert result["price"] >= 0


# Barrier Option Tests
def test_mc_barrier_up_and_out():
    result = monte_carlo_barrier_option(
        100, 100, 120, 0.05, 0.20, 1.0, 100, "call", "up-and-out", 1000, seed=42
    )
    assert "price" in result
    assert result["price"] >= 0


def test_mc_barrier_down_and_out():
    result = monte_carlo_barrier_option(
        100, 100, 80, 0.05, 0.20, 1.0, 100, "put", "down-and-out", 1000, seed=42
    )
    assert result["price"] >= 0


def test_mc_barrier_up_and_in():
    result = monte_carlo_barrier_option(
        100, 100, 120, 0.05, 0.20, 1.0, 100, "call", "up-and-in", 1000, seed=42
    )
    assert result["price"] >= 0


def test_mc_barrier_has_knock_prob():
    result = monte_carlo_barrier_option(
        100, 100, 120, 0.05, 0.20, 1.0, 100, "call", "up-and-out", 1000, seed=42
    )
    assert "knock_probability" in result


def test_mc_barrier_out_cheaper_than_vanilla():
    barrier = monte_carlo_barrier_option(
        100, 100, 120, 0.05, 0.20, 1.0, 100, "call", "up-and-out", 5000, seed=42
    )
    vanilla = monte_carlo_option_price(100, 100, 0.05, 0.20, 1.0, "call", 5000, seed=42)
    # Knockout should be cheaper than vanilla
    assert barrier["price"] <= vanilla["price"]


# Convergence Test
def test_convergence_basic():
    def estimate_mean(n):
        import random
        random.seed(42)
        return sum(random.random() for _ in range(n)) / n

    results = convergence_test(estimate_mean, 0.5, [100, 1000, 10000], seed=42)
    assert len(results) == 3
    # Error should decrease with more samples
    assert results[2]["error"] < results[0]["error"]


# Retirement Simulation Tests
def test_retirement_simulation_basic():
    result = simulate_retirement(
        initial_savings=100000,
        annual_contribution=20000,
        annual_return=0.07,
        return_volatility=0.15,
        years_saving=20,
        years_retirement=25,
        annual_withdrawal=50000,
        num_simulations=100,
        seed=42
    )
    assert "mean_final_value" in result
    assert "probability_of_ruin" in result


def test_retirement_simulation_ruin_probability():
    result = simulate_retirement(
        initial_savings=100000,
        annual_contribution=10000,
        annual_return=0.05,
        return_volatility=0.20,
        years_saving=15,
        years_retirement=30,
        annual_withdrawal=80000,  # High withdrawal
        num_simulations=200,
        seed=42
    )
    # High withdrawal should increase ruin probability
    assert result["probability_of_ruin"] > 0


def test_retirement_simulation_no_ruin():
    result = simulate_retirement(
        initial_savings=1000000,
        annual_contribution=50000,
        annual_return=0.08,
        return_volatility=0.10,
        years_saving=20,
        years_retirement=20,
        annual_withdrawal=30000,  # Low withdrawal
        num_simulations=100,
        seed=42
    )
    # Low withdrawal should have low ruin probability
    assert result["probability_of_ruin"] < 0.2


def test_retirement_simulation_percentiles():
    result = simulate_retirement(
        initial_savings=200000,
        annual_contribution=25000,
        annual_return=0.07,
        return_volatility=0.15,
        years_saving=20,
        years_retirement=25,
        annual_withdrawal=50000,
        num_simulations=200,
        seed=42
    )
    assert result["percentile_10"] <= result["median_final_value"]
    assert result["median_final_value"] <= result["percentile_90"]


@pytest.mark.parametrize("num_sims", [50, 100, 200])
def test_retirement_various_simulations(num_sims: int):
    result = simulate_retirement(
        initial_savings=100000,
        annual_contribution=20000,
        annual_return=0.07,
        return_volatility=0.15,
        years_saving=20,
        years_retirement=25,
        annual_withdrawal=50000,
        num_simulations=num_sims,
        seed=42
    )
    assert result["mean_final_value"] >= 0


# Integration Tests - Slower simulations
def test_mc_var_large_simulation():
    result = monte_carlo_var(1000000, 0.08, 0.18, 10/252, 0.99, 5000, seed=42)
    assert result["var"] > 0
    assert result["cvar"] > result["var"]


def test_mc_option_large_simulation():
    result = monte_carlo_option_price(100, 100, 0.05, 0.25, 1.0, "call", 10000, seed=42)
    bs_price = black_scholes_call(100, 100, 0.05, 0.25, 1.0)
    # Should be very close with many simulations
    assert abs(result["price"] - bs_price) < bs_price * 0.05


def test_simulate_paths_large():
    paths = simulate_paths(100, 0.10, 0.20, 1.0, 252, 100, seed=42)
    assert len(paths) == 100
    assert all(len(p) == 253 for p in paths)
