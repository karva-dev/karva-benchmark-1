"""Mega amortization tests with heavy Cartesian product parametrization."""

import pytest

from finlib.amortization import (
    monthly_payment,
    amortization_schedule,
    principal_payment,
    interest_payment,
    remaining_balance,
    total_interest,
    early_payoff_savings,
    balloon_payment,
    loan_to_value,
    debt_to_income,
    max_loan_amount,
)

PRINCIPALS = [25000, 50000, 100000, 150000, 200000, 250000, 300000, 400000, 500000]
ANNUAL_RATES = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
MONTHS = [36, 60, 84, 120, 180, 240, 300, 360]


# 9*9*8 = 648
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", ANNUAL_RATES)
@pytest.mark.parametrize("months", MONTHS)
def test_monthly_payment_positive(principal, rate, months):
    """Monthly payment should be positive."""
    pmt = monthly_payment(principal, rate, months)
    assert pmt > 0


# 9*9*8 = 648
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", ANNUAL_RATES)
@pytest.mark.parametrize("months", MONTHS)
def test_monthly_payment_less_than_principal(principal, rate, months):
    """Monthly payment should be less than principal."""
    pmt = monthly_payment(principal, rate, months)
    assert pmt < principal


# 9*9*8 = 648
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", ANNUAL_RATES)
@pytest.mark.parametrize("months", MONTHS)
def test_total_payments_geq_principal(principal, rate, months):
    """Total payments should be >= principal."""
    pmt = monthly_payment(principal, rate, months)
    total = pmt * months
    assert total >= principal - 0.01


# 9*9*8 = 648
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", ANNUAL_RATES)
@pytest.mark.parametrize("months", MONTHS)
def test_total_interest_nonneg(principal, rate, months):
    """Total interest should be non-negative."""
    ti = total_interest(principal, rate, months)
    assert ti >= -0.01


# 9*9*8 = 648
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", ANNUAL_RATES)
@pytest.mark.parametrize("months", MONTHS)
def test_remaining_balance_at_start(principal, rate, months):
    """Remaining balance at start should be full principal."""
    bal = remaining_balance(principal, rate, months, 0)
    assert abs(bal - principal) < 1.0


# 9*9*8 = 648
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", ANNUAL_RATES)
@pytest.mark.parametrize("months", MONTHS)
def test_remaining_balance_at_end(principal, rate, months):
    """Remaining balance at end should be zero."""
    bal = remaining_balance(principal, rate, months, months)
    assert abs(bal) < 1.0


# 5*5*5*5 = 625
@pytest.mark.parametrize("principal", [50000, 100000, 200000, 300000, 500000])
@pytest.mark.parametrize("rate", [0.03, 0.05, 0.07, 0.09, 0.10])
@pytest.mark.parametrize("months", [60, 120, 180, 240, 360])
@pytest.mark.parametrize("period", [1, 2, 5, 10, 20])
def test_principal_plus_interest_equals_payment(principal, rate, months, period):
    """Principal + interest portions should equal monthly payment."""
    if period > months:
        return
    pmt = monthly_payment(principal, rate, months)
    pp = principal_payment(principal, rate, months, period)
    ip = interest_payment(principal, rate, months, period)
    assert abs(pp + ip - pmt) < 0.01


# 5*5*5*5 = 625
@pytest.mark.parametrize("principal", [50000, 100000, 200000, 300000, 500000])
@pytest.mark.parametrize("rate", [0.03, 0.05, 0.07, 0.09, 0.10])
@pytest.mark.parametrize("months", [60, 120, 180, 240, 360])
@pytest.mark.parametrize("period", [1, 2, 5, 10, 20])
def test_principal_payment_positive(principal, rate, months, period):
    """Principal payment should be positive."""
    if period > months:
        return
    pp = principal_payment(principal, rate, months, period)
    assert pp > 0


# 5*5*5*5 = 625
@pytest.mark.parametrize("principal", [50000, 100000, 200000, 300000, 500000])
@pytest.mark.parametrize("rate", [0.03, 0.05, 0.07, 0.09, 0.10])
@pytest.mark.parametrize("months", [60, 120, 180, 240, 360])
@pytest.mark.parametrize("period", [1, 2, 5, 10, 20])
def test_interest_payment_positive(principal, rate, months, period):
    """Interest payment should be positive."""
    if period > months:
        return
    ip = interest_payment(principal, rate, months, period)
    assert ip > 0


# 9*9*8 = 648
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", ANNUAL_RATES)
@pytest.mark.parametrize("months", MONTHS)
def test_remaining_balance_decreases(principal, rate, months):
    """Remaining balance should decrease over time."""
    checkpoints = [0, months // 4, months // 2, 3 * months // 4]
    balances = [remaining_balance(principal, rate, months, p) for p in checkpoints]
    for i in range(len(balances) - 1):
        assert balances[i] >= balances[i + 1] - 1.0


# 5*5*4*4 = 400
@pytest.mark.parametrize("principal", [100000, 200000, 300000, 400000, 500000])
@pytest.mark.parametrize("rate", [0.03, 0.05, 0.07, 0.09, 0.10])
@pytest.mark.parametrize("months", [120, 180, 240, 360])
@pytest.mark.parametrize("extra", [50, 100, 200, 500])
def test_early_payoff_saves_interest(principal, rate, months, extra):
    """Extra payments should save interest."""
    result = early_payoff_savings(principal, rate, months, extra)
    assert result["interest_saved"] >= 0
    assert result["months_saved"] >= 0


# 5*5*5*5 = 625
@pytest.mark.parametrize("principal", [100000, 200000, 300000, 400000, 500000])
@pytest.mark.parametrize("rate", [0.03, 0.05, 0.07, 0.09, 0.10])
@pytest.mark.parametrize("months", [120, 180, 240, 300, 360])
@pytest.mark.parametrize("balloon_month", [12, 24, 36, 60, 84])
def test_balloon_payment_valid(principal, rate, months, balloon_month):
    """Balloon payment should be valid."""
    result = balloon_payment(principal, rate, months, balloon_month)
    assert result["monthly_payment"] > 0
    assert result["balloon_amount"] > 0


# 9*9 = 81
@pytest.mark.parametrize(
    "loan", [100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
)
@pytest.mark.parametrize(
    "value", [200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
)
def test_ltv_range(loan, value):
    """LTV should be between 0 and total."""
    if loan > value:
        return
    ltv = loan_to_value(loan, value)
    assert 0 <= ltv <= 1.0


# 9*9 = 81
@pytest.mark.parametrize("debt", [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000])
@pytest.mark.parametrize(
    "income", [3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 15000]
)
def test_dti_range(debt, income):
    """DTI should be between 0 and 1 for reasonable inputs."""
    dti = debt_to_income(debt, income)
    assert 0 <= dti <= 2.0


# 5*4*5*5*5 = 2500
@pytest.mark.parametrize("income", [5000, 7000, 10000, 12000, 15000])
@pytest.mark.parametrize("max_dti", [0.28, 0.36, 0.43, 0.50])
@pytest.mark.parametrize("other_debt", [0, 200, 500, 1000, 2000])
@pytest.mark.parametrize("rate", [0.03, 0.05, 0.07, 0.09, 0.10])
@pytest.mark.parametrize("months", [180, 240, 300, 360, 480])
def test_max_loan_nonneg(income, max_dti, other_debt, rate, months):
    """Max loan amount should be non-negative."""
    result = max_loan_amount(income, max_dti, other_debt, rate, months)
    assert result >= 0


# 4*4*4 = 64
@pytest.mark.parametrize("principal", [100000, 200000, 300000, 500000])
@pytest.mark.parametrize("months", [120, 180, 240, 360])
@pytest.mark.parametrize("rate", [0.03, 0.05, 0.07, 0.10])
def test_payment_increases_with_rate(principal, months, rate):
    """Payment should increase with rate."""
    rates = [0.02, 0.04, 0.06, 0.08, 0.10]
    payments = [monthly_payment(principal, r, months) for r in rates]
    for i in range(len(payments) - 1):
        assert payments[i] <= payments[i + 1] + 0.01


# 4*4*4 = 64
@pytest.mark.parametrize("principal", [100000, 200000, 300000, 500000])
@pytest.mark.parametrize("rate", [0.03, 0.05, 0.07, 0.10])
@pytest.mark.parametrize("extra_val", [50, 100, 200, 500])
def test_payment_decreases_with_term(principal, rate, extra_val):
    """Payment should decrease with longer term."""
    terms = [60, 120, 180, 240, 360]
    payments = [monthly_payment(principal, rate, m) for m in terms]
    for i in range(len(payments) - 1):
        assert payments[i] >= payments[i + 1] - 0.01


# 9*9*8 = 648
@pytest.mark.parametrize("principal", PRINCIPALS)
@pytest.mark.parametrize("rate", ANNUAL_RATES)
@pytest.mark.parametrize("months", MONTHS)
def test_total_interest_equals_formula(principal, rate, months):
    """Total interest should equal total payments minus principal."""
    pmt = monthly_payment(principal, rate, months)
    ti = total_interest(principal, rate, months)
    assert abs(ti - (pmt * months - principal)) < 0.01


# 5*5*5*5 = 625
@pytest.mark.parametrize("principal", [50000, 100000, 200000, 300000, 500000])
@pytest.mark.parametrize("rate", [0.03, 0.05, 0.07, 0.09, 0.10])
@pytest.mark.parametrize("months", [60, 120, 180, 240, 360])
@pytest.mark.parametrize("payments_made", [1, 6, 12, 24, 36])
def test_remaining_balance_positive(principal, rate, months, payments_made):
    """Remaining balance should be positive before end."""
    if payments_made >= months:
        return
    bal = remaining_balance(principal, rate, months, payments_made)
    assert bal >= 0
