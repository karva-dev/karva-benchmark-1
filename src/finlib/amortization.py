"""Amortization and loan calculations."""


def monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    """Calculate monthly payment for a loan.

    Args:
        principal: Loan amount
        annual_rate: Annual interest rate (as decimal)
        months: Number of monthly payments

    Returns:
        Monthly payment amount
    """
    if months <= 0:
        raise ValueError("Months must be positive")
    if annual_rate < 0:
        raise ValueError("Rate cannot be negative")

    if annual_rate == 0:
        return principal / months

    monthly_rate = annual_rate / 12
    return (
        principal
        * (monthly_rate * (1 + monthly_rate) ** months)
        / ((1 + monthly_rate) ** months - 1)
    )


def amortization_schedule(
    principal: float, annual_rate: float, months: int, extra_payment: float = 0.0
) -> list[dict[str, float]]:
    """Generate full amortization schedule.

    Args:
        principal: Loan amount
        annual_rate: Annual interest rate (as decimal)
        months: Number of monthly payments
        extra_payment: Additional monthly payment towards principal

    Returns:
        List of payment records with keys:
        - period: Payment number
        - payment: Total payment
        - principal: Principal portion
        - interest: Interest portion
        - extra: Extra payment applied
        - balance: Remaining balance
    """
    schedule = []
    balance = principal
    base_payment = monthly_payment(principal, annual_rate, months)
    monthly_rate = annual_rate / 12

    period = 1
    while balance > 0.001 and period <= months * 2:  # Safety limit
        interest_portion = balance * monthly_rate
        principal_portion = min(base_payment - interest_portion, balance)
        extra_applied = min(extra_payment, balance - principal_portion)
        total_payment = principal_portion + interest_portion + extra_applied

        balance -= principal_portion + extra_applied
        if balance < 0.001:
            balance = 0

        schedule.append(
            {
                "period": period,
                "payment": total_payment,
                "principal": principal_portion,
                "interest": interest_portion,
                "extra": extra_applied,
                "balance": balance,
            }
        )

        if balance == 0:
            break
        period += 1

    return schedule


def principal_payment(
    principal: float, annual_rate: float, months: int, period: int
) -> float:
    """Calculate principal portion of a specific payment.

    Args:
        principal: Original loan amount
        annual_rate: Annual interest rate (as decimal)
        months: Total number of payments
        period: Payment number (1-indexed)

    Returns:
        Principal portion of payment
    """
    if period < 1 or period > months:
        raise ValueError("Period out of range")

    payment = monthly_payment(principal, annual_rate, months)
    balance = principal
    monthly_rate = annual_rate / 12

    for i in range(1, period + 1):
        interest = balance * monthly_rate
        principal_portion = payment - interest
        if i == period:
            return principal_portion
        balance -= principal_portion

    return 0.0


def interest_payment(
    principal: float, annual_rate: float, months: int, period: int
) -> float:
    """Calculate interest portion of a specific payment.

    Args:
        principal: Original loan amount
        annual_rate: Annual interest rate (as decimal)
        months: Total number of payments
        period: Payment number (1-indexed)

    Returns:
        Interest portion of payment
    """
    if period < 1 or period > months:
        raise ValueError("Period out of range")

    payment = monthly_payment(principal, annual_rate, months)
    balance = principal
    monthly_rate = annual_rate / 12

    for i in range(1, period + 1):
        interest = balance * monthly_rate
        if i == period:
            return interest
        principal_portion = payment - interest
        balance -= principal_portion

    return 0.0


def remaining_balance(
    principal: float, annual_rate: float, months: int, payments_made: int
) -> float:
    """Calculate remaining balance after n payments.

    Args:
        principal: Original loan amount
        annual_rate: Annual interest rate (as decimal)
        months: Total number of payments
        payments_made: Number of payments already made

    Returns:
        Remaining balance
    """
    if payments_made < 0:
        raise ValueError("Payments made cannot be negative")
    if payments_made >= months:
        return 0.0

    payment = monthly_payment(principal, annual_rate, months)
    balance = principal
    monthly_rate = annual_rate / 12

    for _ in range(payments_made):
        interest = balance * monthly_rate
        principal_portion = payment - interest
        balance -= principal_portion

    return max(0, balance)


def total_interest(principal: float, annual_rate: float, months: int) -> float:
    """Calculate total interest paid over loan lifetime.

    Args:
        principal: Loan amount
        annual_rate: Annual interest rate (as decimal)
        months: Number of payments

    Returns:
        Total interest paid
    """
    payment = monthly_payment(principal, annual_rate, months)
    return payment * months - principal


def early_payoff_savings(
    principal: float, annual_rate: float, months: int, extra_monthly: float
) -> dict[str, float]:
    """Calculate savings from extra monthly payments.

    Args:
        principal: Loan amount
        annual_rate: Annual interest rate (as decimal)
        months: Original term in months
        extra_monthly: Additional monthly payment

    Returns:
        Dict with:
        - months_saved: Months saved
        - interest_saved: Interest saved
        - new_term: New term in months
    """
    original_schedule = amortization_schedule(principal, annual_rate, months)
    new_schedule = amortization_schedule(principal, annual_rate, months, extra_monthly)

    original_interest = sum(p["interest"] for p in original_schedule)
    new_interest = sum(p["interest"] for p in new_schedule)

    return {
        "months_saved": len(original_schedule) - len(new_schedule),
        "interest_saved": original_interest - new_interest,
        "new_term": len(new_schedule),
    }


def balloon_payment(
    principal: float, annual_rate: float, months: int, balloon_month: int
) -> dict[str, float]:
    """Calculate loan with balloon payment.

    Args:
        principal: Loan amount
        annual_rate: Annual interest rate (as decimal)
        months: Amortization period (for calculating monthly payment)
        balloon_month: When balloon payment is due

    Returns:
        Dict with:
        - monthly_payment: Regular monthly payment
        - balloon_amount: Final balloon payment
        - total_regular_payments: Sum of regular payments
    """
    if balloon_month > months:
        raise ValueError("Balloon month cannot exceed amortization period")

    payment = monthly_payment(principal, annual_rate, months)
    balance = remaining_balance(principal, annual_rate, months, balloon_month - 1)

    # Last payment is balloon
    monthly_rate = annual_rate / 12
    final_interest = balance * monthly_rate
    balloon = balance + final_interest

    return {
        "monthly_payment": payment,
        "balloon_amount": balloon,
        "total_regular_payments": payment * (balloon_month - 1),
    }


def loan_to_value(loan_amount: float, property_value: float) -> float:
    """Calculate Loan-to-Value ratio.

    Args:
        loan_amount: Loan amount
        property_value: Property value

    Returns:
        LTV ratio (as decimal)
    """
    if property_value <= 0:
        raise ValueError("Property value must be positive")
    return loan_amount / property_value


def debt_to_income(monthly_debt: float, monthly_income: float) -> float:
    """Calculate Debt-to-Income ratio.

    Args:
        monthly_debt: Total monthly debt payments
        monthly_income: Gross monthly income

    Returns:
        DTI ratio (as decimal)
    """
    if monthly_income <= 0:
        raise ValueError("Monthly income must be positive")
    return monthly_debt / monthly_income


def max_loan_amount(
    monthly_income: float,
    max_dti: float,
    other_debt: float,
    annual_rate: float,
    months: int,
) -> float:
    """Calculate maximum loan amount based on DTI.

    Args:
        monthly_income: Gross monthly income
        max_dti: Maximum DTI ratio (as decimal)
        other_debt: Other monthly debt payments
        annual_rate: Annual interest rate (as decimal)
        months: Loan term in months

    Returns:
        Maximum loan amount
    """
    max_payment = monthly_income * max_dti - other_debt
    if max_payment <= 0:
        return 0.0

    # Solve for principal from payment formula
    if annual_rate == 0:
        return max_payment * months

    monthly_rate = annual_rate / 12
    return (
        max_payment
        * ((1 + monthly_rate) ** months - 1)
        / (monthly_rate * (1 + monthly_rate) ** months)
    )
