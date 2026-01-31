"""Interest rate calculations."""

import math


def simple_interest(principal: float, rate: float, time: float) -> float:
    """Calculate simple interest.

    Args:
        principal: Initial amount
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time: Time period in years

    Returns:
        Interest earned
    """
    return principal * rate * time


def compound_interest(principal: float, rate: float, time: float, n: int = 1) -> float:
    """Calculate compound interest.

    Args:
        principal: Initial amount
        rate: Annual interest rate (as decimal)
        time: Time period in years
        n: Compounding frequency per year (1=annual, 4=quarterly, 12=monthly, 365=daily)

    Returns:
        Total amount after compounding (principal + interest)
    """
    return principal * (1 + rate / n) ** (n * time)


def continuous_compound_interest(principal: float, rate: float, time: float) -> float:
    """Calculate continuously compounded interest.

    Args:
        principal: Initial amount
        rate: Annual interest rate (as decimal)
        time: Time period in years

    Returns:
        Total amount after continuous compounding
    """
    return principal * math.exp(rate * time)


def apr_to_apy(apr: float, n: int = 12) -> float:
    """Convert Annual Percentage Rate to Annual Percentage Yield.

    Args:
        apr: Annual percentage rate (as decimal)
        n: Compounding periods per year

    Returns:
        Annual percentage yield (as decimal)
    """
    return (1 + apr / n) ** n - 1


def apy_to_apr(apy: float, n: int = 12) -> float:
    """Convert Annual Percentage Yield to Annual Percentage Rate.

    Args:
        apy: Annual percentage yield (as decimal)
        n: Compounding periods per year

    Returns:
        Annual percentage rate (as decimal)
    """
    return n * ((1 + apy) ** (1 / n) - 1)


def effective_interest_rate(nominal_rate: float, n: int) -> float:
    """Calculate effective annual interest rate from nominal rate.

    Args:
        nominal_rate: Nominal annual interest rate (as decimal)
        n: Number of compounding periods per year

    Returns:
        Effective annual interest rate
    """
    return (1 + nominal_rate / n) ** n - 1


def future_value(principal: float, rate: float, time: float, n: int = 1) -> float:
    """Calculate future value of an investment.

    Args:
        principal: Initial investment
        rate: Annual interest rate (as decimal)
        time: Time period in years
        n: Compounding frequency per year

    Returns:
        Future value
    """
    return compound_interest(principal, rate, time, n)


def present_value(future_value: float, rate: float, time: float, n: int = 1) -> float:
    """Calculate present value given future value.

    Args:
        future_value: Future amount
        rate: Annual discount rate (as decimal)
        time: Time period in years
        n: Compounding frequency per year

    Returns:
        Present value
    """
    return future_value / (1 + rate / n) ** (n * time)


def real_interest_rate(nominal_rate: float, inflation_rate: float) -> float:
    """Calculate real interest rate using Fisher equation.

    Args:
        nominal_rate: Nominal interest rate (as decimal)
        inflation_rate: Inflation rate (as decimal)

    Returns:
        Real interest rate
    """
    return (1 + nominal_rate) / (1 + inflation_rate) - 1


def doubling_time(rate: float) -> float:
    """Calculate time to double investment (Rule of 72 exact).

    Args:
        rate: Annual interest rate (as decimal)

    Returns:
        Years to double
    """
    if rate <= 0:
        return float("inf")
    return math.log(2) / math.log(1 + rate)


def compound_interest_earned(
    principal: float, rate: float, time: float, n: int = 1
) -> float:
    """Calculate only the interest earned from compound interest.

    Args:
        principal: Initial amount
        rate: Annual interest rate (as decimal)
        time: Time period in years
        n: Compounding frequency per year

    Returns:
        Interest earned (not including principal)
    """
    return compound_interest(principal, rate, time, n) - principal
