"""Options pricing and Greeks calculations."""

import math


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    """Probability density function for standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate d1 parameter for Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate d2 parameter for Black-Scholes."""
    return _d1(S, K, r, sigma, T) - sigma * math.sqrt(T)


def black_scholes_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate Black-Scholes call option price.

    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)
        T: Time to expiration (years)

    Returns:
        Call option price
    """
    if T <= 0:
        return max(S - K, 0)

    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(S, K, r, sigma, T)

    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def black_scholes_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate Black-Scholes put option price.

    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)
        T: Time to expiration (years)

    Returns:
        Put option price
    """
    if T <= 0:
        return max(K - S, 0)

    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(S, K, r, sigma, T)

    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def delta(
    S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call"
) -> float:
    """Calculate option delta.

    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to expiration
        option_type: "call" or "put"

    Returns:
        Delta
    """
    if T <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = _d1(S, K, r, sigma, T)

    if option_type == "call":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1


def gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate option gamma (same for calls and puts).

    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to expiration

    Returns:
        Gamma
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = _d1(S, K, r, sigma, T)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def theta(
    S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call"
) -> float:
    """Calculate option theta (time decay per year).

    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to expiration
        option_type: "call" or "put"

    Returns:
        Theta (per year, usually negative)
    """
    if T <= 0:
        return 0.0

    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(S, K, r, sigma, T)

    term1 = -S * _norm_pdf(d1) * sigma / (2 * math.sqrt(T))

    if option_type == "call":
        term2 = -r * K * math.exp(-r * T) * _norm_cdf(d2)
        return term1 + term2
    else:
        term2 = r * K * math.exp(-r * T) * _norm_cdf(-d2)
        return term1 + term2


def vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """Calculate option vega (same for calls and puts).

    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to expiration

    Returns:
        Vega (price change per 1 unit change in volatility)
    """
    if T <= 0:
        return 0.0

    d1 = _d1(S, K, r, sigma, T)
    return S * math.sqrt(T) * _norm_pdf(d1)


def rho(
    S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call"
) -> float:
    """Calculate option rho.

    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to expiration
        option_type: "call" or "put"

    Returns:
        Rho (price change per 1 unit change in rate)
    """
    if T <= 0:
        return 0.0

    d2 = _d2(S, K, r, sigma, T)

    if option_type == "call":
        return K * T * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return -K * T * math.exp(-r * T) * _norm_cdf(-d2)


def put_call_parity(
    S: float,
    K: float,
    r: float,
    T: float,
    call_price: float = None,
    put_price: float = None,
) -> dict[str, float]:
    """Verify or calculate using put-call parity.

    C - P = S - K*e^(-rT)

    Args:
        S: Stock price
        K: Strike price
        r: Risk-free rate
        T: Time to expiration
        call_price: Call price (optional)
        put_price: Put price (optional)

    Returns:
        Dict with calculated/verified values
    """
    pv_strike = K * math.exp(-r * T)
    theoretical_diff = S - pv_strike

    result = {"pv_strike": pv_strike, "theoretical_diff": theoretical_diff}

    if call_price is not None and put_price is not None:
        actual_diff = call_price - put_price
        result["actual_diff"] = actual_diff
        result["parity_holds"] = abs(actual_diff - theoretical_diff) < 0.01
    elif call_price is not None:
        result["implied_put"] = call_price - theoretical_diff
    elif put_price is not None:
        result["implied_call"] = put_price + theoretical_diff

    return result


def binomial_tree_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    steps: int = 100,
    option_type: str = "call",
    american: bool = False,
) -> float:
    """Calculate option price using binomial tree model.

    Args:
        S: Current stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to expiration
        steps: Number of time steps
        option_type: "call" or "put"
        american: True for American option (allows early exercise)

    Returns:
        Option price
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    discount = math.exp(-r * dt)

    # Build stock price tree at expiration
    stock_prices = [S * (u ** (steps - i)) * (d**i) for i in range(steps + 1)]

    # Calculate option values at expiration
    if option_type == "call":
        option_values = [max(price - K, 0) for price in stock_prices]
    else:
        option_values = [max(K - price, 0) for price in stock_prices]

    # Work backwards through tree
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            option_values[i] = discount * (
                p * option_values[i] + (1 - p) * option_values[i + 1]
            )

            if american:
                stock_price = S * (u ** (step - i)) * (d**i)
                if option_type == "call":
                    intrinsic = max(stock_price - K, 0)
                else:
                    intrinsic = max(K - stock_price, 0)
                option_values[i] = max(option_values[i], intrinsic)

    return option_values[0]


def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float:
    """Calculate implied volatility using Newton-Raphson method.

    Args:
        option_price: Market price of option
        S: Stock price
        K: Strike price
        r: Risk-free rate
        T: Time to expiration
        option_type: "call" or "put"
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations

    Returns:
        Implied volatility
    """
    if T <= 0:
        return 0.0

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2 * math.pi / T) * option_price / S

    if sigma <= 0:
        sigma = 0.2  # Default guess

    for _ in range(max_iterations):
        if option_type == "call":
            price = black_scholes_call(S, K, r, sigma, T)
        else:
            price = black_scholes_put(S, K, r, sigma, T)

        diff = price - option_price

        if abs(diff) < tolerance:
            return sigma

        v = vega(S, K, r, sigma, T)
        if v == 0:
            break

        sigma = sigma - diff / v

        if sigma <= 0:
            sigma = 0.001

    return sigma


def option_payoff(
    S: float, K: float, option_type: str = "call", position: str = "long"
) -> float:
    """Calculate option payoff at expiration.

    Args:
        S: Stock price at expiration
        K: Strike price
        option_type: "call" or "put"
        position: "long" or "short"

    Returns:
        Payoff (can be negative for short positions)
    """
    if option_type == "call":
        payoff = max(S - K, 0)
    else:
        payoff = max(K - S, 0)

    if position == "short":
        payoff = -payoff

    return payoff


def option_profit(
    S: float,
    K: float,
    premium: float,
    option_type: str = "call",
    position: str = "long",
) -> float:
    """Calculate option profit/loss at expiration.

    Args:
        S: Stock price at expiration
        K: Strike price
        premium: Option premium paid/received
        option_type: "call" or "put"
        position: "long" or "short"

    Returns:
        Profit/loss
    """
    payoff = option_payoff(S, K, option_type, position)

    if position == "long":
        return payoff - premium
    else:
        return payoff + premium


def breakeven_price(
    K: float, premium: float, option_type: str = "call", position: str = "long"
) -> float:
    """Calculate breakeven stock price.

    Args:
        K: Strike price
        premium: Option premium
        option_type: "call" or "put"
        position: "long" or "short"

    Returns:
        Breakeven stock price
    """
    if position == "long":
        if option_type == "call":
            return K + premium
        else:
            return K - premium
    else:
        if option_type == "call":
            return K + premium
        else:
            return K - premium


def intrinsic_value(S: float, K: float, option_type: str = "call") -> float:
    """Calculate intrinsic value of option.

    Args:
        S: Current stock price
        K: Strike price
        option_type: "call" or "put"

    Returns:
        Intrinsic value (>=0)
    """
    if option_type == "call":
        return max(S - K, 0)
    else:
        return max(K - S, 0)


def time_value(
    option_price: float, S: float, K: float, option_type: str = "call"
) -> float:
    """Calculate time value of option.

    Args:
        option_price: Current option price
        S: Current stock price
        K: Strike price
        option_type: "call" or "put"

    Returns:
        Time value
    """
    return option_price - intrinsic_value(S, K, option_type)


def moneyness(S: float, K: float, option_type: str = "call") -> str:
    """Determine moneyness of option.

    Args:
        S: Current stock price
        K: Strike price
        option_type: "call" or "put"

    Returns:
        "ITM", "ATM", or "OTM"
    """
    if option_type == "call":
        if S > K * 1.02:
            return "ITM"
        elif S < K * 0.98:
            return "OTM"
        else:
            return "ATM"
    else:
        if S < K * 0.98:
            return "ITM"
        elif S > K * 1.02:
            return "OTM"
        else:
            return "ATM"


def straddle_payoff(
    S: float, K: float, call_premium: float, put_premium: float
) -> float:
    """Calculate straddle payoff at expiration.

    Args:
        S: Stock price at expiration
        K: Strike price (same for both)
        call_premium: Call premium paid
        put_premium: Put premium paid

    Returns:
        Net profit/loss
    """
    call_payoff = max(S - K, 0)
    put_payoff = max(K - S, 0)
    return call_payoff + put_payoff - call_premium - put_premium


def strangle_payoff(
    S: float, K_call: float, K_put: float, call_premium: float, put_premium: float
) -> float:
    """Calculate strangle payoff at expiration.

    Args:
        S: Stock price at expiration
        K_call: Call strike price
        K_put: Put strike price
        call_premium: Call premium paid
        put_premium: Put premium paid

    Returns:
        Net profit/loss
    """
    call_payoff = max(S - K_call, 0)
    put_payoff = max(K_put - S, 0)
    return call_payoff + put_payoff - call_premium - put_premium


def covered_call_payoff(
    S: float, K: float, stock_cost: float, premium_received: float
) -> float:
    """Calculate covered call payoff.

    Args:
        S: Stock price at expiration
        K: Strike price
        stock_cost: Cost of stock
        premium_received: Premium received for selling call

    Returns:
        Net profit/loss
    """
    stock_pnl = S - stock_cost
    option_pnl = -max(S - K, 0) + premium_received
    return stock_pnl + option_pnl


def protective_put_payoff(
    S: float, K: float, stock_cost: float, premium_paid: float
) -> float:
    """Calculate protective put payoff.

    Args:
        S: Stock price at expiration
        K: Strike price
        stock_cost: Cost of stock
        premium_paid: Premium paid for put

    Returns:
        Net profit/loss
    """
    stock_pnl = S - stock_cost
    option_pnl = max(K - S, 0) - premium_paid
    return stock_pnl + option_pnl
