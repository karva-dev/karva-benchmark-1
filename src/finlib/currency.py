"""Currency and FX calculations."""


def direct_to_indirect(direct_quote: float) -> float:
    """Convert direct quote to indirect quote.

    Direct quote: domestic currency per unit of foreign (e.g., 1.25 USD/EUR)
    Indirect quote: foreign currency per unit of domestic (e.g., 0.80 EUR/USD)

    Args:
        direct_quote: Direct exchange rate

    Returns:
        Indirect exchange rate
    """
    if direct_quote <= 0:
        raise ValueError("Quote must be positive")
    return 1 / direct_quote


def indirect_to_direct(indirect_quote: float) -> float:
    """Convert indirect quote to direct quote.

    Args:
        indirect_quote: Indirect exchange rate

    Returns:
        Direct exchange rate
    """
    if indirect_quote <= 0:
        raise ValueError("Quote must be positive")
    return 1 / indirect_quote


def convert_currency(amount: float, rate: float, is_direct: bool = True) -> float:
    """Convert amount from one currency to another.

    Args:
        amount: Amount to convert
        rate: Exchange rate
        is_direct: If True, rate is domestic/foreign; multiply to get domestic

    Returns:
        Converted amount
    """
    if is_direct:
        return amount * rate
    return amount / rate


def cross_rate(rate_a_base: float, rate_b_base: float) -> float:
    """Calculate cross rate between two currencies via a common base.

    If USD/EUR = 1.10 and USD/GBP = 1.30
    Then GBP/EUR = 1.10 / 1.30 = 0.846

    Args:
        rate_a_base: Rate of currency A vs base (base/A)
        rate_b_base: Rate of currency B vs base (base/B)

    Returns:
        Cross rate (B/A)
    """
    if rate_b_base <= 0:
        raise ValueError("Rates must be positive")
    return rate_a_base / rate_b_base


def triangular_arbitrage(
    rate_ab: float, rate_bc: float, rate_ca: float, tolerance: float = 0.0001
) -> dict[str, float]:
    """Detect triangular arbitrage opportunity.

    For currencies A, B, C:
    - rate_ab: A per B (B/A)
    - rate_bc: B per C (C/B)
    - rate_ca: C per A (A/C)

    No arbitrage if: rate_ab * rate_bc * rate_ca = 1

    Args:
        rate_ab: Exchange rate A/B
        rate_bc: Exchange rate B/C
        rate_ca: Exchange rate C/A
        tolerance: Tolerance for detecting arbitrage

    Returns:
        Dict with:
        - product: Product of rates
        - has_arbitrage: Whether arbitrage exists
        - profit_pct: Profit percentage if arbitrage exists
        - direction: "ABC" or "CBA" for profitable direction
    """
    product = rate_ab * rate_bc * rate_ca

    has_arbitrage = abs(product - 1) > tolerance
    profit_pct = 0.0
    direction = ""

    if has_arbitrage:
        if product > 1:
            # A -> B -> C -> A is profitable
            profit_pct = (product - 1) * 100
            direction = "ABC"
        else:
            # A -> C -> B -> A is profitable
            profit_pct = (1 / product - 1) * 100
            direction = "CBA"

    return {
        "product": product,
        "has_arbitrage": has_arbitrage,
        "profit_pct": profit_pct,
        "direction": direction,
    }


def forward_rate(
    spot_rate: float,
    domestic_rate: float,
    foreign_rate: float,
    days: int,
    day_basis: int = 360,
) -> float:
    """Calculate forward exchange rate using interest rate parity.

    Args:
        spot_rate: Current spot rate (domestic/foreign)
        domestic_rate: Domestic interest rate (annualized, as decimal)
        foreign_rate: Foreign interest rate (annualized, as decimal)
        days: Forward contract period in days
        day_basis: Day count basis (360 or 365)

    Returns:
        Forward exchange rate
    """
    time_factor = days / day_basis
    return (
        spot_rate * (1 + domestic_rate * time_factor) / (1 + foreign_rate * time_factor)
    )


def forward_points(
    spot_rate: float, forward_rate: float, pip_size: float = 0.0001
) -> float:
    """Calculate forward points (pips).

    Args:
        spot_rate: Spot exchange rate
        forward_rate: Forward exchange rate
        pip_size: Size of one pip (typically 0.0001 for most pairs)

    Returns:
        Forward points in pips
    """
    return (forward_rate - spot_rate) / pip_size


def swap_points_to_rate(
    spot_rate: float, swap_points: float, pip_size: float = 0.0001
) -> float:
    """Convert swap points to forward rate.

    Args:
        spot_rate: Spot exchange rate
        swap_points: Swap points in pips
        pip_size: Size of one pip

    Returns:
        Forward exchange rate
    """
    return spot_rate + swap_points * pip_size


def currency_basket_value(
    basket: dict[str, float], rates: dict[str, float], base_currency: str = "USD"
) -> float:
    """Calculate value of a currency basket.

    Args:
        basket: Dict mapping currency codes to weights/amounts
        rates: Dict mapping currency codes to rates vs base
        base_currency: Base currency for valuation

    Returns:
        Total basket value in base currency
    """
    total = 0.0
    for currency, amount in basket.items():
        if currency == base_currency:
            total += amount
        elif currency in rates:
            total += amount * rates[currency]
        else:
            raise ValueError(f"Missing rate for {currency}")
    return total


def effective_exchange_rate(
    rates: dict[str, float], weights: dict[str, float]
) -> float:
    """Calculate trade-weighted effective exchange rate.

    Args:
        rates: Dict mapping currency codes to bilateral rates
        weights: Dict mapping currency codes to trade weights (should sum to 1)

    Returns:
        Effective exchange rate index
    """
    if abs(sum(weights.values()) - 1) > 0.0001:
        raise ValueError("Weights must sum to 1")

    result = 1.0
    for currency, weight in weights.items():
        if currency not in rates:
            raise ValueError(f"Missing rate for {currency}")
        result *= rates[currency] ** weight

    return result


def bid_ask_spread(bid: float, ask: float) -> float:
    """Calculate bid-ask spread as percentage.

    Args:
        bid: Bid price
        ask: Ask price

    Returns:
        Spread as percentage of mid price
    """
    if bid <= 0 or ask <= 0:
        raise ValueError("Prices must be positive")
    if bid > ask:
        raise ValueError("Bid cannot exceed ask")

    mid = (bid + ask) / 2
    return (ask - bid) / mid * 100


def mid_rate(bid: float, ask: float) -> float:
    """Calculate mid rate from bid and ask.

    Args:
        bid: Bid price
        ask: Ask price

    Returns:
        Mid rate
    """
    return (bid + ask) / 2


def pip_value(lot_size: float, pip_size: float = 0.0001, rate: float = 1.0) -> float:
    """Calculate the value of one pip.

    Args:
        lot_size: Size of position (e.g., 100000 for standard lot)
        pip_size: Size of one pip (0.0001 for most pairs, 0.01 for JPY)
        rate: Exchange rate for conversion (if pip currency differs from account)

    Returns:
        Value of one pip in account currency
    """
    return lot_size * pip_size / rate


def position_size(
    account_balance: float, risk_percent: float, stop_loss_pips: float, pip_value: float
) -> float:
    """Calculate position size based on risk management.

    Args:
        account_balance: Account balance
        risk_percent: Percentage of account to risk (as decimal)
        stop_loss_pips: Stop loss distance in pips
        pip_value: Value of one pip per unit

    Returns:
        Position size in units
    """
    risk_amount = account_balance * risk_percent
    return risk_amount / (stop_loss_pips * pip_value)


def covered_interest_arbitrage(
    spot_rate: float,
    forward_rate: float,
    domestic_rate: float,
    foreign_rate: float,
    days: int,
    day_basis: int = 360,
) -> dict[str, float]:
    """Check for covered interest arbitrage opportunity.

    Args:
        spot_rate: Spot exchange rate
        forward_rate: Market forward rate
        domestic_rate: Domestic interest rate (annualized)
        foreign_rate: Foreign interest rate (annualized)
        days: Period in days
        day_basis: Day count basis

    Returns:
        Dict with arbitrage analysis
    """
    # Theoretical forward rate from interest rate parity
    theoretical_forward = forward_rate_calc = forward_rate_from_parity(
        spot_rate, domestic_rate, foreign_rate, days, day_basis
    )

    diff = forward_rate - theoretical_forward
    has_arbitrage = abs(diff) > 0.0001

    return {
        "theoretical_forward": theoretical_forward,
        "market_forward": forward_rate,
        "difference": diff,
        "has_arbitrage": has_arbitrage,
        "strategy": "borrow_domestic"
        if diff > 0
        else "borrow_foreign"
        if diff < 0
        else "none",
    }


def forward_rate_from_parity(
    spot_rate: float,
    domestic_rate: float,
    foreign_rate: float,
    days: int,
    day_basis: int = 360,
) -> float:
    """Alias for forward_rate using interest rate parity."""
    return forward_rate(spot_rate, domestic_rate, foreign_rate, days, day_basis)


def real_exchange_rate(
    nominal_rate: float, domestic_price_level: float, foreign_price_level: float
) -> float:
    """Calculate real exchange rate.

    Args:
        nominal_rate: Nominal exchange rate (domestic/foreign)
        domestic_price_level: Domestic price index
        foreign_price_level: Foreign price index

    Returns:
        Real exchange rate
    """
    return nominal_rate * foreign_price_level / domestic_price_level


def purchasing_power_parity_rate(domestic_price: float, foreign_price: float) -> float:
    """Calculate PPP exchange rate.

    Args:
        domestic_price: Price of basket in domestic currency
        foreign_price: Price of same basket in foreign currency

    Returns:
        PPP exchange rate (domestic/foreign)
    """
    if foreign_price <= 0:
        raise ValueError("Foreign price must be positive")
    return domestic_price / foreign_price
