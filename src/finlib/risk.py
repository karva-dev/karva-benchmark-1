"""Risk metrics and calculations."""

import math


def volatility(
    returns: list[float], annualize: bool = True, periods_per_year: int = 252
) -> float:
    """Calculate volatility (standard deviation of returns).

    Args:
        returns: List of periodic returns
        annualize: Whether to annualize the result
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Volatility (annualized if specified)
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 returns")

    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)

    if annualize:
        return std * math.sqrt(periods_per_year)
    return std


def downside_volatility(
    returns: list[float],
    threshold: float = 0.0,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> float:
    """Calculate downside volatility (semi-deviation).

    Args:
        returns: List of periodic returns
        threshold: Return threshold (typically 0 or risk-free rate)
        annualize: Whether to annualize
        periods_per_year: Periods per year

    Returns:
        Downside volatility
    """
    downside = [r for r in returns if r < threshold]

    if len(downside) < 2:
        return 0.0

    mean_downside = sum(downside) / len(downside)
    variance = sum((r - mean_downside) ** 2 for r in downside) / (len(downside) - 1)
    std = math.sqrt(variance)

    if annualize:
        return std * math.sqrt(periods_per_year)
    return std


def beta(asset_returns: list[float], market_returns: list[float]) -> float:
    """Calculate beta (systematic risk).

    Args:
        asset_returns: Asset returns
        market_returns: Market/benchmark returns

    Returns:
        Beta coefficient
    """
    if len(asset_returns) != len(market_returns):
        raise ValueError("Return series must have same length")
    if len(asset_returns) < 2:
        raise ValueError("Need at least 2 returns")

    mean_asset = sum(asset_returns) / len(asset_returns)
    mean_market = sum(market_returns) / len(market_returns)

    covariance = sum(
        (a - mean_asset) * (m - mean_market)
        for a, m in zip(asset_returns, market_returns)
    ) / (len(asset_returns) - 1)

    market_variance = sum((m - mean_market) ** 2 for m in market_returns) / (
        len(market_returns) - 1
    )

    if market_variance == 0:
        return 0.0
    return covariance / market_variance


def alpha(
    asset_returns: list[float], market_returns: list[float], risk_free_rate: float = 0.0
) -> float:
    """Calculate Jensen's alpha.

    Args:
        asset_returns: Asset returns
        market_returns: Market returns
        risk_free_rate: Risk-free rate (periodic)

    Returns:
        Alpha (periodic)
    """
    b = beta(asset_returns, market_returns)
    mean_asset = sum(asset_returns) / len(asset_returns)
    mean_market = sum(market_returns) / len(market_returns)

    return mean_asset - (risk_free_rate + b * (mean_market - risk_free_rate))


def sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe ratio.

    Args:
        returns: Periodic returns
        risk_free_rate: Risk-free rate (periodic)
        annualize: Whether to annualize
        periods_per_year: Periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 returns")

    excess_returns = [r - risk_free_rate for r in returns]
    mean_excess = sum(excess_returns) / len(excess_returns)
    vol = volatility(returns, annualize=False)

    if vol == 0:
        return 0.0

    ratio = mean_excess / vol

    if annualize:
        return ratio * math.sqrt(periods_per_year)
    return ratio


def sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sortino ratio.

    Args:
        returns: Periodic returns
        risk_free_rate: Risk-free rate (periodic)
        target_return: Target return for downside calculation
        annualize: Whether to annualize
        periods_per_year: Periods per year

    Returns:
        Sortino ratio
    """
    excess_returns = [r - risk_free_rate for r in returns]
    mean_excess = sum(excess_returns) / len(excess_returns)

    downside_vol = downside_volatility(
        returns, threshold=target_return, annualize=False
    )

    if downside_vol == 0:
        return 0.0

    ratio = mean_excess / downside_vol

    if annualize:
        return ratio * math.sqrt(periods_per_year)
    return ratio


def treynor_ratio(
    returns: list[float], market_returns: list[float], risk_free_rate: float = 0.0
) -> float:
    """Calculate Treynor ratio.

    Args:
        returns: Asset returns
        market_returns: Market returns
        risk_free_rate: Risk-free rate (periodic)

    Returns:
        Treynor ratio
    """
    b = beta(returns, market_returns)
    if b == 0:
        return 0.0

    mean_return = sum(returns) / len(returns)
    return (mean_return - risk_free_rate) / b


def max_drawdown(values: list[float]) -> float:
    """Calculate maximum drawdown.

    Args:
        values: Portfolio values over time

    Returns:
        Maximum drawdown (as positive decimal)
    """
    if not values:
        return 0.0

    max_dd = 0.0
    peak = values[0]

    for value in values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)

    return max_dd


def drawdown_series(values: list[float]) -> list[float]:
    """Calculate drawdown series.

    Args:
        values: Portfolio values over time

    Returns:
        List of drawdowns at each point
    """
    if not values:
        return []

    drawdowns = []
    peak = values[0]

    for value in values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        drawdowns.append(drawdown)

    return drawdowns


def calmar_ratio(
    returns: list[float], values: list[float], periods_per_year: int = 252
) -> float:
    """Calculate Calmar ratio (return / max drawdown).

    Args:
        returns: Periodic returns
        values: Portfolio values
        periods_per_year: Periods per year

    Returns:
        Calmar ratio
    """
    max_dd = max_drawdown(values)
    if max_dd == 0:
        return 0.0

    annual_return = sum(returns) / len(returns) * periods_per_year
    return annual_return / max_dd


def parametric_var(
    returns: list[float],
    confidence: float = 0.95,
    holding_period: int = 1,
    portfolio_value: float = 1.0,
) -> float:
    """Calculate parametric Value at Risk (assuming normal distribution).

    Args:
        returns: Historical returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        holding_period: Holding period in days
        portfolio_value: Portfolio value

    Returns:
        VaR (positive number representing potential loss)
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 returns")

    mean_return = sum(returns) / len(returns)
    std_return = math.sqrt(
        sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    )

    # Z-score for confidence level (approximation)
    z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326, 0.999: 3.090}
    z = z_scores.get(confidence, 1.645)

    # Scale for holding period
    var_return = (
        z * std_return * math.sqrt(holding_period) - mean_return * holding_period
    )

    return var_return * portfolio_value


def historical_var(
    returns: list[float], confidence: float = 0.95, portfolio_value: float = 1.0
) -> float:
    """Calculate historical Value at Risk.

    Args:
        returns: Historical returns
        confidence: Confidence level
        portfolio_value: Portfolio value

    Returns:
        VaR (positive number representing potential loss)
    """
    if not returns:
        raise ValueError("Returns list is empty")

    sorted_returns = sorted(returns)
    index = int((1 - confidence) * len(sorted_returns))
    var_return = -sorted_returns[index]

    return var_return * portfolio_value


def conditional_var(
    returns: list[float], confidence: float = 0.95, portfolio_value: float = 1.0
) -> float:
    """Calculate Conditional VaR (Expected Shortfall).

    Args:
        returns: Historical returns
        confidence: Confidence level
        portfolio_value: Portfolio value

    Returns:
        CVaR (positive number representing average loss beyond VaR)
    """
    if not returns:
        raise ValueError("Returns list is empty")

    sorted_returns = sorted(returns)
    cutoff_index = int((1 - confidence) * len(sorted_returns))

    if cutoff_index == 0:
        cutoff_index = 1

    tail_returns = sorted_returns[:cutoff_index]
    avg_tail = sum(tail_returns) / len(tail_returns)

    return -avg_tail * portfolio_value


def marginal_var(
    portfolio_returns: list[float], asset_returns: list[float], confidence: float = 0.95
) -> float:
    """Calculate marginal VaR contribution of an asset.

    Args:
        portfolio_returns: Portfolio returns
        asset_returns: Individual asset returns
        confidence: Confidence level

    Returns:
        Marginal VaR
    """
    b = beta(asset_returns, portfolio_returns)
    port_var = parametric_var(portfolio_returns, confidence)

    return b * port_var


def component_var(
    weights: list[float], returns_matrix: list[list[float]], confidence: float = 0.95
) -> list[float]:
    """Calculate component VaR for each asset.

    Args:
        weights: Asset weights
        returns_matrix: Matrix of returns (each row is an asset's returns)
        confidence: Confidence level

    Returns:
        Component VaR for each asset
    """
    n = len(weights)
    if len(returns_matrix) != n:
        raise ValueError("Number of return series must match weights")

    # Calculate portfolio returns
    num_periods = len(returns_matrix[0])
    portfolio_returns = []
    for t in range(num_periods):
        port_ret = sum(weights[i] * returns_matrix[i][t] for i in range(n))
        portfolio_returns.append(port_ret)

    total_var = parametric_var(portfolio_returns, confidence)

    # Component VaR
    component_vars = []
    for i in range(n):
        b = beta(returns_matrix[i], portfolio_returns)
        component_vars.append(weights[i] * b * total_var)

    return component_vars


def incremental_var(
    portfolio_returns: list[float],
    new_asset_returns: list[float],
    new_weight: float,
    confidence: float = 0.95,
) -> float:
    """Calculate incremental VaR from adding a new asset.

    Args:
        portfolio_returns: Current portfolio returns
        new_asset_returns: New asset returns
        new_weight: Weight of new asset
        confidence: Confidence level

    Returns:
        Change in VaR
    """
    old_weight = 1 - new_weight

    # New portfolio returns
    new_portfolio = [
        old_weight * p + new_weight * a
        for p, a in zip(portfolio_returns, new_asset_returns)
    ]

    old_var = parametric_var(portfolio_returns, confidence)
    new_var = parametric_var(new_portfolio, confidence)

    return new_var - old_var


def stress_test(
    portfolio_value: float,
    asset_weights: list[float],
    shock_scenarios: list[list[float]],
) -> list[float]:
    """Run stress test scenarios.

    Args:
        portfolio_value: Portfolio value
        asset_weights: Asset weights
        shock_scenarios: List of scenarios, each being returns for each asset

    Returns:
        Portfolio value change under each scenario
    """
    results = []
    for scenario in shock_scenarios:
        if len(scenario) != len(asset_weights):
            raise ValueError("Scenario must have same length as weights")
        portfolio_return = sum(w * r for w, r in zip(asset_weights, scenario))
        results.append(portfolio_value * portfolio_return)
    return results


def upside_potential_ratio(returns: list[float], target: float = 0.0) -> float:
    """Calculate upside potential ratio.

    Args:
        returns: Returns
        target: Target return

    Returns:
        Upside potential ratio
    """
    upside = [r - target for r in returns if r > target]
    downside = [(target - r) ** 2 for r in returns if r < target]

    if not downside:
        return float("inf") if upside else 0.0

    upside_mean = sum(upside) / len(returns)  # Probability weighted
    downside_dev = math.sqrt(sum(downside) / len(returns))

    if downside_dev == 0:
        return float("inf") if upside_mean > 0 else 0.0

    return upside_mean / downside_dev


def omega_ratio(returns: list[float], threshold: float = 0.0) -> float:
    """Calculate Omega ratio.

    Args:
        returns: Returns
        threshold: Threshold return

    Returns:
        Omega ratio
    """
    gains = sum(r - threshold for r in returns if r > threshold)
    losses = sum(threshold - r for r in returns if r <= threshold)

    if losses == 0:
        return float("inf") if gains > 0 else 1.0

    return gains / losses


def gain_loss_ratio(returns: list[float]) -> float:
    """Calculate gain-to-loss ratio.

    Args:
        returns: Returns

    Returns:
        Ratio of average gain to average loss
    """
    gains = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]

    if not gains:
        return 0.0
    if not losses:
        return float("inf")

    avg_gain = sum(gains) / len(gains)
    avg_loss = -sum(losses) / len(losses)

    return avg_gain / avg_loss


def win_rate(returns: list[float]) -> float:
    """Calculate win rate (percentage of positive returns).

    Args:
        returns: Returns

    Returns:
        Win rate (0 to 1)
    """
    if not returns:
        return 0.0
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)
