"""Monte Carlo simulation methods for finance."""

import math
import random


def _box_muller() -> tuple[float, float]:
    """Generate two standard normal random variables using Box-Muller transform."""
    u1 = random.random()
    u2 = random.random()

    # Avoid log(0)
    while u1 == 0:
        u1 = random.random()

    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    return z0, z1


def _standard_normal() -> float:
    """Generate a single standard normal random variable."""
    return _box_muller()[0]


def _normal(mean: float, std: float) -> float:
    """Generate a normal random variable with given mean and std."""
    return mean + std * _standard_normal()


def geometric_brownian_motion(
    S0: float, mu: float, sigma: float, T: float, dt: float, seed: int | None = None
) -> list[float]:
    """Simulate stock price path using Geometric Brownian Motion.

    dS = mu * S * dt + sigma * S * dW

    Args:
        S0: Initial stock price
        mu: Drift (expected return)
        sigma: Volatility
        T: Time horizon
        dt: Time step
        seed: Random seed for reproducibility

    Returns:
        List of stock prices
    """
    if seed is not None:
        random.seed(seed)

    steps = int(T / dt)
    prices = [S0]
    S = S0

    for _ in range(steps):
        dW = _standard_normal() * math.sqrt(dt)
        S = S * math.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        prices.append(S)

    return prices


def simulate_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    num_paths: int,
    seed: int | None = None,
) -> list[list[float]]:
    """Simulate multiple GBM paths.

    Args:
        S0: Initial price
        mu: Drift
        sigma: Volatility
        T: Time horizon
        steps: Number of time steps
        num_paths: Number of simulation paths
        seed: Random seed

    Returns:
        List of price paths
    """
    if seed is not None:
        random.seed(seed)

    dt = T / steps
    paths = []

    for _ in range(num_paths):
        path = [S0]
        S = S0
        for _ in range(steps):
            dW = _standard_normal() * math.sqrt(dt)
            S = S * math.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            path.append(S)
        paths.append(path)

    return paths


def simulate_portfolio(
    initial_values: list[float],
    returns: list[float],
    volatilities: list[float],
    correlations: list[list[float]],
    T: float,
    num_paths: int,
    seed: int | None = None,
) -> list[float]:
    """Simulate portfolio value distribution.

    Args:
        initial_values: Initial value of each asset
        returns: Expected annual returns for each asset
        volatilities: Annual volatilities for each asset
        correlations: Correlation matrix
        T: Time horizon (years)
        num_paths: Number of simulations
        seed: Random seed

    Returns:
        List of final portfolio values
    """
    if seed is not None:
        random.seed(seed)

    n_assets = len(initial_values)
    total_initial = sum(initial_values)
    weights = [v / total_initial for v in initial_values]

    # Cholesky decomposition for correlated normals
    L = _cholesky(correlations)

    final_values = []
    for _ in range(num_paths):
        # Generate correlated random variables
        z = [_standard_normal() for _ in range(n_assets)]
        corr_z = [sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(n_assets)]

        # Calculate portfolio return
        portfolio_return = 0.0
        for i in range(n_assets):
            asset_return = (returns[i] - 0.5 * volatilities[i] ** 2) * T + volatilities[
                i
            ] * math.sqrt(T) * corr_z[i]
            portfolio_return += weights[i] * (math.exp(asset_return) - 1)

        final_values.append(total_initial * (1 + portfolio_return))

    return final_values


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    """Perform Cholesky decomposition."""
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                val = matrix[i][i] - sum(L[i][k] ** 2 for k in range(j))
                L[i][j] = math.sqrt(max(val, 0))
            else:
                if L[j][j] != 0:
                    L[i][j] = (
                        matrix[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
                    ) / L[j][j]

    return L


def monte_carlo_var(
    initial_value: float,
    mu: float,
    sigma: float,
    T: float,
    confidence: float,
    num_simulations: int,
    seed: int | None = None,
) -> dict[str, float]:
    """Calculate Value at Risk using Monte Carlo simulation.

    Args:
        initial_value: Portfolio value
        mu: Expected annual return
        sigma: Annual volatility
        T: Time horizon (years)
        confidence: Confidence level (e.g., 0.95)
        num_simulations: Number of simulations
        seed: Random seed

    Returns:
        Dict with VaR and CVaR
    """
    if seed is not None:
        random.seed(seed)

    final_values = []
    for _ in range(num_simulations):
        z = _standard_normal()
        final = initial_value * math.exp(
            (mu - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z
        )
        final_values.append(final)

    returns = sorted([(v - initial_value) / initial_value for v in final_values])

    var_index = int((1 - confidence) * num_simulations)
    var = -returns[var_index] * initial_value

    # CVaR (Expected Shortfall)
    tail = returns[: var_index + 1]
    cvar = -sum(tail) / len(tail) * initial_value if tail else var

    return {
        "var": var,
        "cvar": cvar,
        "mean_return": sum(returns) / len(returns),
        "std_return": math.sqrt(
            sum((r - sum(returns) / len(returns)) ** 2 for r in returns) / len(returns)
        ),
    }


def monte_carlo_option_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str,
    num_simulations: int,
    seed: int | None = None,
) -> dict[str, float]:
    """Price European option using Monte Carlo simulation.

    Args:
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to expiration
        option_type: "call" or "put"
        num_simulations: Number of simulations
        seed: Random seed

    Returns:
        Dict with price and standard error
    """
    if seed is not None:
        random.seed(seed)

    payoffs = []
    for _ in range(num_simulations):
        z = _standard_normal()
        ST = S0 * math.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)

        if option_type == "call":
            payoff = max(ST - K, 0)
        else:
            payoff = max(K - ST, 0)
        payoffs.append(payoff)

    mean_payoff = sum(payoffs) / len(payoffs)
    price = math.exp(-r * T) * mean_payoff

    # Standard error
    variance = sum((p - mean_payoff) ** 2 for p in payoffs) / (len(payoffs) - 1)
    std_error = math.exp(-r * T) * math.sqrt(variance / num_simulations)

    return {"price": price, "std_error": std_error, "mean_payoff": mean_payoff}


def monte_carlo_asian_option(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    option_type: str,
    num_simulations: int,
    seed: int | None = None,
) -> dict[str, float]:
    """Price Asian option using Monte Carlo simulation.

    Args:
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to expiration
        steps: Number of averaging points
        option_type: "call" or "put"
        num_simulations: Number of simulations
        seed: Random seed

    Returns:
        Dict with price and standard error
    """
    if seed is not None:
        random.seed(seed)

    dt = T / steps
    payoffs = []

    for _ in range(num_simulations):
        path = [S0]
        S = S0
        for _ in range(steps):
            z = _standard_normal()
            S = S * math.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)
            path.append(S)

        avg_price = sum(path) / len(path)

        if option_type == "call":
            payoff = max(avg_price - K, 0)
        else:
            payoff = max(K - avg_price, 0)
        payoffs.append(payoff)

    mean_payoff = sum(payoffs) / len(payoffs)
    price = math.exp(-r * T) * mean_payoff

    variance = sum((p - mean_payoff) ** 2 for p in payoffs) / (len(payoffs) - 1)
    std_error = math.exp(-r * T) * math.sqrt(variance / num_simulations)

    return {"price": price, "std_error": std_error}


def monte_carlo_barrier_option(
    S0: float,
    K: float,
    barrier: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    option_type: str,
    barrier_type: str,
    num_simulations: int,
    seed: int | None = None,
) -> dict[str, float]:
    """Price barrier option using Monte Carlo simulation.

    Args:
        S0: Initial stock price
        K: Strike price
        barrier: Barrier level
        r: Risk-free rate
        sigma: Volatility
        T: Time to expiration
        steps: Number of monitoring points
        option_type: "call" or "put"
        barrier_type: "up-and-out", "up-and-in", "down-and-out", "down-and-in"
        num_simulations: Number of simulations
        seed: Random seed

    Returns:
        Dict with price and standard error
    """
    if seed is not None:
        random.seed(seed)

    dt = T / steps
    payoffs = []

    for _ in range(num_simulations):
        S = S0
        crossed = False

        for _ in range(steps):
            z = _standard_normal()
            S = S * math.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)

            if "up" in barrier_type and S >= barrier:
                crossed = True
            if "down" in barrier_type and S <= barrier:
                crossed = True

        # Determine payoff based on barrier type
        if "out" in barrier_type:
            active = not crossed
        else:  # "in"
            active = crossed

        if active:
            if option_type == "call":
                payoff = max(S - K, 0)
            else:
                payoff = max(K - S, 0)
        else:
            payoff = 0

        payoffs.append(payoff)

    mean_payoff = sum(payoffs) / len(payoffs)
    price = math.exp(-r * T) * mean_payoff

    variance = (
        sum((p - mean_payoff) ** 2 for p in payoffs) / (len(payoffs) - 1)
        if len(payoffs) > 1
        else 0
    )
    std_error = (
        math.exp(-r * T) * math.sqrt(variance / num_simulations)
        if num_simulations > 0
        else 0
    )

    return {
        "price": price,
        "std_error": std_error,
        "knock_probability": sum(1 for p in payoffs if p == 0) / len(payoffs)
        if "out" in barrier_type
        else sum(1 for p in payoffs if p > 0) / len(payoffs),
    }


def convergence_test(
    target_function,
    true_value: float,
    simulation_sizes: list[int],
    seed: int | None = None,
) -> list[dict[str, float]]:
    """Test Monte Carlo convergence.

    Args:
        target_function: Function that takes num_simulations and returns estimated value
        true_value: Known true value for comparison
        simulation_sizes: List of simulation sizes to test
        seed: Random seed

    Returns:
        List of convergence metrics
    """
    results = []
    for n in simulation_sizes:
        if seed is not None:
            random.seed(seed)
        estimate = target_function(n)
        error = abs(estimate - true_value)
        results.append(
            {
                "n": n,
                "estimate": estimate,
                "error": error,
                "relative_error": error / abs(true_value) if true_value != 0 else error,
            }
        )
    return results


def simulate_retirement(
    initial_savings: float,
    annual_contribution: float,
    annual_return: float,
    return_volatility: float,
    years_saving: int,
    years_retirement: int,
    annual_withdrawal: float,
    num_simulations: int,
    seed: int | None = None,
) -> dict[str, float]:
    """Simulate retirement outcomes.

    Args:
        initial_savings: Current savings
        annual_contribution: Annual contribution during saving years
        annual_return: Expected annual return
        return_volatility: Annual return volatility
        years_saving: Years until retirement
        years_retirement: Years in retirement
        annual_withdrawal: Annual withdrawal in retirement
        num_simulations: Number of simulations
        seed: Random seed

    Returns:
        Retirement simulation statistics
    """
    if seed is not None:
        random.seed(seed)

    final_values = []
    ruin_count = 0

    for _ in range(num_simulations):
        value = initial_savings

        # Accumulation phase
        for _ in range(years_saving):
            ret = _normal(annual_return, return_volatility)
            value = value * (1 + ret) + annual_contribution

        # Retirement phase
        ruined = False
        for _ in range(years_retirement):
            ret = _normal(annual_return, return_volatility)
            value = value * (1 + ret) - annual_withdrawal
            if value <= 0:
                ruined = True
                value = 0
                break

        final_values.append(value)
        if ruined:
            ruin_count += 1

    mean_final = sum(final_values) / len(final_values)
    median_final = sorted(final_values)[len(final_values) // 2]

    return {
        "mean_final_value": mean_final,
        "median_final_value": median_final,
        "min_final_value": min(final_values),
        "max_final_value": max(final_values),
        "probability_of_ruin": ruin_count / num_simulations,
        "percentile_10": sorted(final_values)[int(0.1 * len(final_values))],
        "percentile_90": sorted(final_values)[int(0.9 * len(final_values))],
    }
