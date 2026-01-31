"""Bond pricing and yield calculations."""


def bond_price(
    face_value: float, coupon_rate: float, ytm: float, periods: int, frequency: int = 2
) -> float:
    """Calculate bond price using present value of cash flows.

    Args:
        face_value: Face/par value of bond
        coupon_rate: Annual coupon rate (as decimal)
        ytm: Yield to maturity (annual, as decimal)
        periods: Number of periods until maturity
        frequency: Coupon payments per year (1=annual, 2=semi-annual)

    Returns:
        Bond price
    """
    if periods <= 0:
        return face_value

    coupon = face_value * coupon_rate / frequency
    periodic_yield = ytm / frequency

    if periodic_yield == 0:
        return coupon * periods + face_value

    # PV of coupons (annuity formula)
    pv_coupons = coupon * (1 - (1 + periodic_yield) ** -periods) / periodic_yield

    # PV of face value
    pv_face = face_value / (1 + periodic_yield) ** periods

    return pv_coupons + pv_face


def bond_price_zero_coupon(
    face_value: float, ytm: float, years: float, frequency: int = 2
) -> float:
    """Calculate zero-coupon bond price.

    Args:
        face_value: Face value at maturity
        ytm: Yield to maturity (annual, as decimal)
        years: Years to maturity
        frequency: Compounding frequency per year

    Returns:
        Bond price
    """
    periods = years * frequency
    periodic_yield = ytm / frequency
    return face_value / (1 + periodic_yield) ** periods


def yield_to_maturity(
    price: float,
    face_value: float,
    coupon_rate: float,
    periods: int,
    frequency: int = 2,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Calculate yield to maturity using Newton-Raphson method.

    Args:
        price: Current bond price
        face_value: Face value
        coupon_rate: Annual coupon rate (as decimal)
        periods: Periods to maturity
        frequency: Payments per year
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations

    Returns:
        Yield to maturity (annual)
    """
    coupon = face_value * coupon_rate / frequency

    # Initial guess using current yield approximation
    if price > 0:
        ytm_guess = (coupon * frequency + (face_value - price) / periods) / (
            (face_value + price) / 2
        )
    else:
        ytm_guess = coupon_rate

    for _ in range(max_iterations):
        r = ytm_guess / frequency

        # Calculate price at current YTM guess
        if r == 0:
            calc_price = coupon * periods + face_value
            derivative = (
                -sum(t * coupon for t in range(1, periods + 1)) - periods * face_value
            )
        else:
            calc_price = (
                coupon * (1 - (1 + r) ** -periods) / r + face_value / (1 + r) ** periods
            )

            # Derivative of price with respect to yield
            derivative = 0
            for t in range(1, periods + 1):
                derivative -= t * coupon / (1 + r) ** (t + 1)
            derivative -= periods * face_value / (1 + r) ** (periods + 1)
            derivative /= frequency

        diff = calc_price - price

        if abs(diff) < tolerance:
            return ytm_guess

        # Newton-Raphson update
        if derivative != 0:
            ytm_guess -= diff / derivative
        else:
            break

    return ytm_guess


def current_yield(price: float, face_value: float, coupon_rate: float) -> float:
    """Calculate current yield.

    Args:
        price: Current bond price
        face_value: Face value
        coupon_rate: Annual coupon rate (as decimal)

    Returns:
        Current yield (annual)
    """
    if price <= 0:
        raise ValueError("Price must be positive")
    annual_coupon = face_value * coupon_rate
    return annual_coupon / price


def macaulay_duration(
    face_value: float, coupon_rate: float, ytm: float, periods: int, frequency: int = 2
) -> float:
    """Calculate Macaulay duration.

    Args:
        face_value: Face value
        coupon_rate: Annual coupon rate (as decimal)
        ytm: Yield to maturity (annual, as decimal)
        periods: Periods to maturity
        frequency: Payments per year

    Returns:
        Macaulay duration in years
    """
    if periods <= 0:
        return 0.0

    coupon = face_value * coupon_rate / frequency
    periodic_yield = ytm / frequency
    price = bond_price(face_value, coupon_rate, ytm, periods, frequency)

    if price == 0:
        return 0.0

    weighted_sum = 0.0
    for t in range(1, periods + 1):
        pv = coupon / (1 + periodic_yield) ** t
        weighted_sum += t * pv

    # Add weighted PV of face value
    pv_face = face_value / (1 + periodic_yield) ** periods
    weighted_sum += periods * pv_face

    # Convert to years
    return weighted_sum / (price * frequency)


def modified_duration(
    face_value: float, coupon_rate: float, ytm: float, periods: int, frequency: int = 2
) -> float:
    """Calculate modified duration.

    Args:
        face_value: Face value
        coupon_rate: Annual coupon rate (as decimal)
        ytm: Yield to maturity (annual, as decimal)
        periods: Periods to maturity
        frequency: Payments per year

    Returns:
        Modified duration
    """
    mac_duration = macaulay_duration(face_value, coupon_rate, ytm, periods, frequency)
    return mac_duration / (1 + ytm / frequency)


def convexity(
    face_value: float, coupon_rate: float, ytm: float, periods: int, frequency: int = 2
) -> float:
    """Calculate bond convexity.

    Args:
        face_value: Face value
        coupon_rate: Annual coupon rate (as decimal)
        ytm: Yield to maturity (annual, as decimal)
        periods: Periods to maturity
        frequency: Payments per year

    Returns:
        Convexity
    """
    if periods <= 0:
        return 0.0

    coupon = face_value * coupon_rate / frequency
    periodic_yield = ytm / frequency
    price = bond_price(face_value, coupon_rate, ytm, periods, frequency)

    if price == 0:
        return 0.0

    conv_sum = 0.0
    for t in range(1, periods + 1):
        pv = coupon / (1 + periodic_yield) ** t
        conv_sum += t * (t + 1) * pv

    # Add face value term
    pv_face = face_value / (1 + periodic_yield) ** periods
    conv_sum += periods * (periods + 1) * pv_face

    return conv_sum / (price * (1 + periodic_yield) ** 2 * frequency**2)


def price_change_duration(modified_duration: float, yield_change: float) -> float:
    """Estimate price change using duration.

    Args:
        modified_duration: Modified duration
        yield_change: Change in yield (as decimal, e.g., 0.01 for 1%)

    Returns:
        Estimated percentage price change
    """
    return -modified_duration * yield_change


def price_change_convexity(
    modified_duration: float, convexity: float, yield_change: float
) -> float:
    """Estimate price change using duration and convexity.

    Args:
        modified_duration: Modified duration
        convexity: Convexity
        yield_change: Change in yield (as decimal)

    Returns:
        Estimated percentage price change
    """
    duration_effect = -modified_duration * yield_change
    convexity_effect = 0.5 * convexity * yield_change**2
    return duration_effect + convexity_effect


def accrued_interest(
    face_value: float,
    coupon_rate: float,
    days_since_coupon: int,
    days_in_period: int = 180,
    frequency: int = 2,
) -> float:
    """Calculate accrued interest.

    Args:
        face_value: Face value
        coupon_rate: Annual coupon rate (as decimal)
        days_since_coupon: Days since last coupon payment
        days_in_period: Days in coupon period
        frequency: Coupon payments per year

    Returns:
        Accrued interest
    """
    coupon = face_value * coupon_rate / frequency
    return coupon * days_since_coupon / days_in_period


def dirty_price(clean_price: float, accrued: float) -> float:
    """Calculate dirty (full) price from clean price.

    Args:
        clean_price: Clean price (quoted)
        accrued: Accrued interest

    Returns:
        Dirty price
    """
    return clean_price + accrued


def clean_price(dirty_price: float, accrued: float) -> float:
    """Calculate clean price from dirty price.

    Args:
        dirty_price: Dirty (full) price
        accrued: Accrued interest

    Returns:
        Clean price
    """
    return dirty_price - accrued


def yield_to_call(
    price: float,
    face_value: float,
    coupon_rate: float,
    periods_to_call: int,
    call_price: float,
    frequency: int = 2,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Calculate yield to call.

    Args:
        price: Current bond price
        face_value: Face value
        coupon_rate: Annual coupon rate
        periods_to_call: Periods until call date
        call_price: Call price
        frequency: Payments per year
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations

    Returns:
        Yield to call (annual)
    """
    # Similar to YTM but with call price instead of face value
    coupon = face_value * coupon_rate / frequency
    ytc_guess = coupon_rate

    for _ in range(max_iterations):
        r = ytc_guess / frequency

        if r == 0:
            calc_price = coupon * periods_to_call + call_price
        else:
            calc_price = (
                coupon * (1 - (1 + r) ** -periods_to_call) / r
                + call_price / (1 + r) ** periods_to_call
            )

        diff = calc_price - price

        if abs(diff) < tolerance:
            return ytc_guess

        # Numerical derivative
        delta = 0.0001
        r2 = (ytc_guess + delta) / frequency
        if r2 == 0:
            calc_price2 = coupon * periods_to_call + call_price
        else:
            calc_price2 = (
                coupon * (1 - (1 + r2) ** -periods_to_call) / r2
                + call_price / (1 + r2) ** periods_to_call
            )

        derivative = (calc_price2 - calc_price) / delta

        if derivative != 0:
            ytc_guess -= diff / derivative

    return ytc_guess


def spread_to_benchmark(bond_ytm: float, benchmark_ytm: float) -> float:
    """Calculate spread to benchmark yield.

    Args:
        bond_ytm: Bond yield to maturity
        benchmark_ytm: Benchmark (e.g., Treasury) yield

    Returns:
        Spread in basis points
    """
    return (bond_ytm - benchmark_ytm) * 10000


def dollar_duration(price: float, modified_duration: float) -> float:
    """Calculate dollar duration (DV01 approximation).

    Args:
        price: Bond price
        modified_duration: Modified duration

    Returns:
        Dollar duration (price change per 100bp yield change)
    """
    return price * modified_duration / 100


def effective_duration(
    price_down: float, price_up: float, price: float, yield_change: float
) -> float:
    """Calculate effective duration using prices.

    Args:
        price_down: Price when yield decreases
        price_up: Price when yield increases
        price: Current price
        yield_change: Yield change used (as decimal)

    Returns:
        Effective duration
    """
    return (price_down - price_up) / (2 * price * yield_change)


def effective_convexity(
    price_down: float, price_up: float, price: float, yield_change: float
) -> float:
    """Calculate effective convexity using prices.

    Args:
        price_down: Price when yield decreases
        price_up: Price when yield increases
        price: Current price
        yield_change: Yield change used (as decimal)

    Returns:
        Effective convexity
    """
    return (price_down + price_up - 2 * price) / (price * yield_change**2)
