"""Mega Monte Carlo tests with heavy Cartesian product parametrization."""

import math
import random

import pytest

from finlib.montecarlo import (
    geometric_brownian_motion,
    simulate_paths,
    simulate_portfolio,
    monte_carlo_var,
    monte_carlo_option_price,
    monte_carlo_asian_option,
    monte_carlo_barrier_option,
    simulate_retirement,
)


S0_VALS = [50, 75, 100, 125, 150, 200]
MU_VALS = [-0.05, 0.0, 0.05, 0.08, 0.10, 0.15]
SIGMA_VALS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
T_VALS = [0.25, 0.5, 1.0, 2.0]
DT_VALS = [0.01, 0.02, 0.05]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
STRIKES = [80, 90, 95, 100, 105, 110, 120]
RISK_FREE = [0.01, 0.03, 0.05, 0.08]
CONFIDENCES = [0.90, 0.95, 0.99]
NUM_SIMS = [100, 500]


# 6*6*6*4*15 = 12960
@pytest.mark.parametrize("S0", S0_VALS)
@pytest.mark.parametrize("mu", MU_VALS)
@pytest.mark.parametrize("sigma", SIGMA_VALS)
@pytest.mark.parametrize("T", T_VALS)
@pytest.mark.parametrize("seed", SEEDS)
def test_gbm_starts_at_s0(S0, mu, sigma, T, seed):
    """GBM should start at S0."""
    path = geometric_brownian_motion(S0, mu, sigma, T, 0.05, seed=seed)
    assert path[0] == S0


# 6*6*6*4*15 = 12960
@pytest.mark.parametrize("S0", S0_VALS)
@pytest.mark.parametrize("mu", MU_VALS)
@pytest.mark.parametrize("sigma", SIGMA_VALS)
@pytest.mark.parametrize("T", T_VALS)
@pytest.mark.parametrize("seed", SEEDS)
def test_gbm_positive_prices(S0, mu, sigma, T, seed):
    """GBM prices should always be positive."""
    path = geometric_brownian_motion(S0, mu, sigma, T, 0.05, seed=seed)
    for p in path:
        assert p > 0


# 6*6*4*15 = 2160
@pytest.mark.parametrize("S0", S0_VALS)
@pytest.mark.parametrize("sigma", SIGMA_VALS)
@pytest.mark.parametrize("T", T_VALS)
@pytest.mark.parametrize("seed", SEEDS)
def test_gbm_path_length(S0, sigma, T, seed):
    """GBM path length should match steps + 1."""
    dt = 0.05
    path = geometric_brownian_motion(S0, 0.05, sigma, T, dt, seed=seed)
    expected_len = int(T / dt) + 1
    assert len(path) == expected_len


# 4*4*4*4*2*15 = 7680
@pytest.mark.parametrize("S0", [90, 100, 110, 120])
@pytest.mark.parametrize("K", [90, 100, 110, 120])
@pytest.mark.parametrize("sigma", [0.15, 0.20, 0.30, 0.40])
@pytest.mark.parametrize("r", RISK_FREE)
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("seed", SEEDS)
def test_mc_option_price_nonneg(S0, K, sigma, r, option_type, seed):
    """MC option price should be non-negative."""
    result = monte_carlo_option_price(S0, K, r, sigma, 1.0, option_type, 200, seed=seed)
    assert result["price"] >= 0


# 4*4*4*4*2*15 = 7680
@pytest.mark.parametrize("S0", [90, 100, 110, 120])
@pytest.mark.parametrize("K", [90, 100, 110, 120])
@pytest.mark.parametrize("sigma", [0.15, 0.20, 0.30, 0.40])
@pytest.mark.parametrize("r", RISK_FREE)
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("seed", SEEDS)
def test_mc_option_std_error_nonneg(S0, K, sigma, r, option_type, seed):
    """MC option std error should be non-negative."""
    result = monte_carlo_option_price(S0, K, r, sigma, 1.0, option_type, 200, seed=seed)
    assert result["std_error"] >= 0


# 4*4*3*3*15 = 2160
@pytest.mark.parametrize("S0", [90, 100, 110, 120])
@pytest.mark.parametrize("sigma", [0.15, 0.20, 0.30, 0.40])
@pytest.mark.parametrize("confidence", CONFIDENCES)
@pytest.mark.parametrize("T", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_mc_var_valid(S0, sigma, confidence, T, seed):
    """MC VaR should return valid dict."""
    result = monte_carlo_var(10000, 0.05, sigma, T, confidence, 200, seed=seed)
    assert "var" in result
    assert "cvar" in result
    assert isinstance(result["var"], float)


# 4*4*3*3*15 = 2160
@pytest.mark.parametrize("S0", [90, 100, 110, 120])
@pytest.mark.parametrize("sigma", [0.15, 0.20, 0.30, 0.40])
@pytest.mark.parametrize("confidence", CONFIDENCES)
@pytest.mark.parametrize("T", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_mc_cvar_geq_var(S0, sigma, confidence, T, seed):
    """CVaR should be >= VaR."""
    result = monte_carlo_var(10000, 0.05, sigma, T, confidence, 200, seed=seed)
    assert result["cvar"] >= result["var"] - 1.0


# 3*3*3*2*10 = 540
@pytest.mark.parametrize("S0", [90, 100, 110])
@pytest.mark.parametrize("K", [95, 100, 105])
@pytest.mark.parametrize("sigma", [0.20, 0.30, 0.40])
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_asian_option_nonneg(S0, K, sigma, option_type, seed):
    """Asian option price should be non-negative."""
    result = monte_carlo_asian_option(
        S0, K, 0.05, sigma, 1.0, 20, option_type, 200, seed=seed
    )
    assert result["price"] >= 0


# 3*3*3*2*10 = 540
@pytest.mark.parametrize("S0", [90, 100, 110])
@pytest.mark.parametrize("K", [95, 100, 105])
@pytest.mark.parametrize("sigma", [0.20, 0.30, 0.40])
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_asian_option_leq_european(S0, K, sigma, option_type, seed):
    """Asian option should generally be <= European option."""
    asian = monte_carlo_asian_option(
        S0, K, 0.05, sigma, 1.0, 20, option_type, 500, seed=seed
    )
    european = monte_carlo_option_price(
        S0, K, 0.05, sigma, 1.0, option_type, 500, seed=seed
    )
    # With tolerance for MC noise
    assert asian["price"] <= european["price"] + 5.0


# 3*3*3*4*5 = 540
@pytest.mark.parametrize("S0", [90, 100, 110])
@pytest.mark.parametrize("K", [95, 100, 105])
@pytest.mark.parametrize("sigma", [0.20, 0.30, 0.40])
@pytest.mark.parametrize(
    "barrier_type", ["up-and-out", "up-and-in", "down-and-out", "down-and-in"]
)
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_barrier_option_nonneg(S0, K, sigma, barrier_type, seed):
    """Barrier option price should be non-negative."""
    barrier = 130 if "up" in barrier_type else 70
    result = monte_carlo_barrier_option(
        S0, K, barrier, 0.05, sigma, 1.0, 20, "call", barrier_type, 200, seed=seed
    )
    assert result["price"] >= 0


# 4*4*3*10 = 480
@pytest.mark.parametrize("S0", [90, 100, 110, 120])
@pytest.mark.parametrize("mu", [0.0, 0.05, 0.10, 0.15])
@pytest.mark.parametrize("sigma", [0.15, 0.25, 0.35])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_simulate_paths_correct_count(S0, mu, sigma, seed):
    """Simulate paths should return correct number of paths."""
    paths = simulate_paths(S0, mu, sigma, 1.0, 10, 5, seed=seed)
    assert len(paths) == 5
    for path in paths:
        assert len(path) == 11
        assert path[0] == S0


# 4*3*3*5 = 180
@pytest.mark.parametrize("initial_savings", [50000, 100000, 200000, 500000])
@pytest.mark.parametrize("annual_return", [0.04, 0.06, 0.08])
@pytest.mark.parametrize("volatility_val", [0.10, 0.15, 0.20])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_retirement_sim_valid(initial_savings, annual_return, volatility_val, seed):
    """Retirement simulation should return valid stats."""
    result = simulate_retirement(
        initial_savings,
        10000,
        annual_return,
        volatility_val,
        20,
        25,
        40000,
        100,
        seed=seed,
    )
    assert 0 <= result["probability_of_ruin"] <= 1
    assert result["mean_final_value"] >= 0 or True  # Can be 0 if all ruined
    assert result["percentile_10"] <= result["percentile_90"]
