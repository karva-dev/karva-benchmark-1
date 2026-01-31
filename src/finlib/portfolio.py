"""Portfolio analysis and return calculations."""

import math


def simple_return(start_value: float, end_value: float) -> float:
    """Calculate simple (arithmetic) return.

    Args:
        start_value: Starting value
        end_value: Ending value

    Returns:
        Simple return (as decimal)
    """
    if start_value <= 0:
        raise ValueError("Start value must be positive")
    return (end_value - start_value) / start_value


def log_return(start_value: float, end_value: float) -> float:
    """Calculate logarithmic (continuously compounded) return.

    Args:
        start_value: Starting value
        end_value: Ending value

    Returns:
        Log return (as decimal)
    """
    if start_value <= 0 or end_value <= 0:
        raise ValueError("Values must be positive")
    return math.log(end_value / start_value)


def holding_period_return(values: list[float]) -> float:
    """Calculate holding period return from a series of values.

    Args:
        values: List of portfolio values over time

    Returns:
        Total holding period return
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 values")
    return simple_return(values[0], values[-1])


def annualized_return(total_return: float, years: float) -> float:
    """Convert total return to annualized return.

    Args:
        total_return: Total return (as decimal)
        years: Number of years

    Returns:
        Annualized return (as decimal)
    """
    if years <= 0:
        raise ValueError("Years must be positive")
    return (1 + total_return) ** (1 / years) - 1


def cumulative_return(returns: list[float]) -> float:
    """Calculate cumulative return from a series of periodic returns.

    Args:
        returns: List of periodic returns (as decimals)

    Returns:
        Cumulative return
    """
    result = 1.0
    for r in returns:
        result *= 1 + r
    return result - 1


def time_weighted_return(
    values: list[float], cash_flows: list[float] | None = None
) -> float:
    """Calculate time-weighted rate of return (TWRR).

    TWRR eliminates the effect of cash flows, measuring manager performance.

    Args:
        values: Portfolio values at each period (including start and end)
        cash_flows: Cash flows at each period (positive = inflow, None = no flows)

    Returns:
        Time-weighted return
    """
    if len(values) < 2:
        raise ValueError("Need at least 2 values")

    if cash_flows is None:
        cash_flows = [0.0] * (len(values) - 1)

    if len(cash_flows) != len(values) - 1:
        raise ValueError("Cash flows should have length of values - 1")

    result = 1.0
    for i in range(len(values) - 1):
        # Value before cash flow
        start_val = values[i]
        # Value after period (before next cash flow)
        end_val = values[i + 1]
        # Adjust for cash flow that occurred during period
        adjusted_start = start_val + cash_flows[i]
        if adjusted_start > 0:
            period_return = end_val / adjusted_start
            result *= period_return

    return result - 1


def money_weighted_return(
    cash_flows: list[float],
    values: list[float],
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Calculate money-weighted rate of return (MWRR/IRR).

    MWRR considers the timing and size of cash flows.

    Args:
        cash_flows: Cash flows (negative = outflow/investment, positive = inflow)
        values: Ending values (last value is final portfolio value)
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations

    Returns:
        Money-weighted return (periodic rate)
    """
    # Combine cash flows and final value
    flows = list(cash_flows)
    flows.append(values[-1])

    # Newton-Raphson to find IRR
    r = 0.1  # Initial guess

    for _ in range(max_iterations):
        npv = sum(cf / (1 + r) ** t for t, cf in enumerate(flows))
        d_npv = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(flows))

        if abs(npv) < tolerance:
            return r

        if d_npv != 0:
            r = r - npv / d_npv
        else:
            break

    return r


def portfolio_weights(values: list[float]) -> list[float]:
    """Calculate portfolio weights from values.

    Args:
        values: List of asset values

    Returns:
        List of weights (sum to 1)
    """
    total = sum(values)
    if total == 0:
        raise ValueError("Total portfolio value is zero")
    return [v / total for v in values]


def portfolio_return(weights: list[float], returns: list[float]) -> float:
    """Calculate portfolio return from asset weights and returns.

    Args:
        weights: Asset weights (should sum to 1)
        returns: Asset returns

    Returns:
        Portfolio return
    """
    if len(weights) != len(returns):
        raise ValueError("Weights and returns must have same length")
    return sum(w * r for w, r in zip(weights, returns))


def portfolio_variance(weights: list[float], cov_matrix: list[list[float]]) -> float:
    """Calculate portfolio variance from weights and covariance matrix.

    Args:
        weights: Asset weights
        cov_matrix: Covariance matrix

    Returns:
        Portfolio variance
    """
    n = len(weights)
    if len(cov_matrix) != n or any(len(row) != n for row in cov_matrix):
        raise ValueError("Dimensions mismatch")

    variance = 0.0
    for i in range(n):
        for j in range(n):
            variance += weights[i] * weights[j] * cov_matrix[i][j]
    return variance


def portfolio_std(weights: list[float], cov_matrix: list[list[float]]) -> float:
    """Calculate portfolio standard deviation."""
    return math.sqrt(portfolio_variance(weights, cov_matrix))


def rebalance_portfolio(
    current_values: list[float], target_weights: list[float]
) -> list[float]:
    """Calculate trades needed to rebalance portfolio.

    Args:
        current_values: Current asset values
        target_weights: Target weights (sum to 1)

    Returns:
        List of value changes needed (positive = buy, negative = sell)
    """
    if len(current_values) != len(target_weights):
        raise ValueError("Lengths must match")
    if abs(sum(target_weights) - 1) > 0.0001:
        raise ValueError("Target weights must sum to 1")

    total = sum(current_values)
    target_values = [w * total for w in target_weights]
    return [target - current for target, current in zip(target_values, current_values)]


def benchmark_comparison(
    portfolio_returns: list[float], benchmark_returns: list[float]
) -> dict[str, float]:
    """Compare portfolio performance to benchmark.

    Args:
        portfolio_returns: Portfolio periodic returns
        benchmark_returns: Benchmark periodic returns

    Returns:
        Dict with comparison metrics
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Return series must have same length")

    port_cum = cumulative_return(portfolio_returns)
    bench_cum = cumulative_return(benchmark_returns)

    # Active returns
    active_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
    avg_active = sum(active_returns) / len(active_returns) if active_returns else 0

    # Tracking error (std of active returns)
    if len(active_returns) > 1:
        mean_active = sum(active_returns) / len(active_returns)
        tracking_error = math.sqrt(
            sum((r - mean_active) ** 2 for r in active_returns)
            / (len(active_returns) - 1)
        )
    else:
        tracking_error = 0

    # Information ratio
    info_ratio = avg_active / tracking_error if tracking_error > 0 else 0

    return {
        "portfolio_return": port_cum,
        "benchmark_return": bench_cum,
        "excess_return": port_cum - bench_cum,
        "average_active_return": avg_active,
        "tracking_error": tracking_error,
        "information_ratio": info_ratio,
    }


def geometric_mean_return(returns: list[float]) -> float:
    """Calculate geometric mean return.

    Args:
        returns: List of periodic returns

    Returns:
        Geometric mean return
    """
    if not returns:
        raise ValueError("Returns list is empty")

    product = 1.0
    for r in returns:
        product *= 1 + r

    return product ** (1 / len(returns)) - 1


def arithmetic_mean_return(returns: list[float]) -> float:
    """Calculate arithmetic mean return."""
    if not returns:
        raise ValueError("Returns list is empty")
    return sum(returns) / len(returns)


def expected_return(weights: list[float], expected_returns: list[float]) -> float:
    """Calculate expected portfolio return.

    Args:
        weights: Asset weights
        expected_returns: Expected returns for each asset

    Returns:
        Expected portfolio return
    """
    return portfolio_return(weights, expected_returns)


def minimum_variance_weights(cov_matrix: list[list[float]]) -> list[float]:
    """Calculate minimum variance portfolio weights (no short selling).

    Simple iterative solver for small portfolios.

    Args:
        cov_matrix: Covariance matrix

    Returns:
        Optimal weights for minimum variance
    """
    n = len(cov_matrix)

    # Start with equal weights
    weights = [1.0 / n] * n

    # Gradient descent
    learning_rate = 0.01
    for _ in range(1000):
        # Calculate gradient
        gradient = [0.0] * n
        for i in range(n):
            for j in range(n):
                gradient[i] += 2 * weights[j] * cov_matrix[i][j]

        # Update weights
        for i in range(n):
            weights[i] -= learning_rate * gradient[i]

        # Project to simplex (ensure non-negative and sum to 1)
        weights = [max(0, w) for w in weights]
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / n] * n

    return weights


def diversification_ratio(
    weights: list[float], volatilities: list[float], portfolio_volatility: float
) -> float:
    """Calculate diversification ratio.

    Args:
        weights: Asset weights
        volatilities: Individual asset volatilities
        portfolio_volatility: Portfolio volatility

    Returns:
        Diversification ratio (>1 indicates diversification benefit)
    """
    weighted_vol = sum(w * v for w, v in zip(weights, volatilities))
    if portfolio_volatility == 0:
        return 1.0
    return weighted_vol / portfolio_volatility


def contribution_to_risk(
    weights: list[float], cov_matrix: list[list[float]]
) -> list[float]:
    """Calculate each asset's contribution to portfolio risk.

    Args:
        weights: Asset weights
        cov_matrix: Covariance matrix

    Returns:
        List of risk contributions (sum equals portfolio variance)
    """
    n = len(weights)
    port_var = portfolio_variance(weights, cov_matrix)

    contributions = []
    for i in range(n):
        marginal = sum(weights[j] * cov_matrix[i][j] for j in range(n))
        contributions.append(weights[i] * marginal)

    return contributions
