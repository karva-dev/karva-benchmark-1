"""Tests for amortization calculations."""

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


# Monthly Payment Tests
@pytest.mark.parametrize(
    "principal,rate,months,expected",
    [
        (200000, 0.06, 360, 1199.10),
        (100000, 0.05, 180, 790.79),
        (300000, 0.045, 360, 1520.06),
        (150000, 0.07, 240, 1162.95),
        (250000, 0.055, 360, 1419.47),
    ],
)
def test_monthly_payment_basic(
    principal: float, rate: float, months: int, expected: float
):
    result = monthly_payment(principal, rate, months)
    assert abs(result - expected) < 0.1


def test_monthly_payment_zero_rate():
    result = monthly_payment(12000, 0, 12)
    assert abs(result - 1000) < 0.01


@pytest.mark.parametrize("months", [60, 120, 180, 240, 360])
def test_payment_decreases_with_longer_term(months: int):
    principal, rate = 200000, 0.06
    short_payment = monthly_payment(principal, rate, 60)
    long_payment = monthly_payment(principal, rate, months)
    assert long_payment <= short_payment


@pytest.mark.parametrize("rate", [0.03, 0.05, 0.07, 0.09])
def test_payment_increases_with_higher_rate(rate: float):
    principal, months = 200000, 360
    low_payment = monthly_payment(principal, 0.03, months)
    high_payment = monthly_payment(principal, rate, months)
    assert high_payment >= low_payment


def test_monthly_payment_invalid_months():
    try:
        monthly_payment(100000, 0.05, 0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_monthly_payment_negative_rate():
    try:
        monthly_payment(100000, -0.05, 360)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Amortization Schedule Tests
def test_amortization_schedule_length():
    schedule = amortization_schedule(200000, 0.06, 360)
    assert len(schedule) == 360


def test_amortization_schedule_final_balance_zero():
    schedule = amortization_schedule(200000, 0.06, 360)
    assert schedule[-1]["balance"] < 0.01


def test_amortization_schedule_payment_consistency():
    principal, rate, months = 100000, 0.05, 180
    schedule = amortization_schedule(principal, rate, months)
    expected_payment = monthly_payment(principal, rate, months)
    for entry in schedule[:-1]:  # Last payment might differ slightly
        assert abs(entry["payment"] - expected_payment) < 0.01


def test_amortization_schedule_principal_interest_sum():
    schedule = amortization_schedule(100000, 0.05, 120)
    for entry in schedule:
        expected_sum = entry["principal"] + entry["interest"] + entry["extra"]
        assert abs(entry["payment"] - expected_sum) < 0.01


@pytest.mark.parametrize("extra", [100, 200, 500, 1000])
def test_amortization_with_extra_payment(extra: float):
    base_schedule = amortization_schedule(200000, 0.06, 360)
    extra_schedule = amortization_schedule(200000, 0.06, 360, extra)
    assert len(extra_schedule) < len(base_schedule)


def test_amortization_extra_payment_saves_interest():
    base_schedule = amortization_schedule(200000, 0.06, 360)
    extra_schedule = amortization_schedule(200000, 0.06, 360, 200)
    base_interest = sum(p["interest"] for p in base_schedule)
    extra_interest = sum(p["interest"] for p in extra_schedule)
    assert extra_interest < base_interest


# Principal/Interest Payment Tests
@pytest.mark.parametrize("period", [1, 12, 60, 120, 180])
def test_principal_payment_period(period: int):
    principal, rate, months = 200000, 0.06, 360
    p_payment = principal_payment(principal, rate, months, period)
    assert p_payment > 0


@pytest.mark.parametrize("period", [1, 12, 60, 120, 180])
def test_interest_payment_period(period: int):
    principal, rate, months = 200000, 0.06, 360
    i_payment = interest_payment(principal, rate, months, period)
    assert i_payment > 0


def test_principal_interest_sum_equals_payment():
    principal, rate, months = 200000, 0.06, 360
    total_payment = monthly_payment(principal, rate, months)
    for period in [1, 12, 60, 180, 360]:
        p = principal_payment(principal, rate, months, period)
        i = interest_payment(principal, rate, months, period)
        assert abs(p + i - total_payment) < 0.01


def test_principal_portion_increases_over_time():
    principal, rate, months = 200000, 0.06, 360
    p1 = principal_payment(principal, rate, months, 1)
    p180 = principal_payment(principal, rate, months, 180)
    p360 = principal_payment(principal, rate, months, 360)
    assert p1 < p180 < p360


def test_interest_portion_decreases_over_time():
    principal, rate, months = 200000, 0.06, 360
    i1 = interest_payment(principal, rate, months, 1)
    i180 = interest_payment(principal, rate, months, 180)
    i360 = interest_payment(principal, rate, months, 360)
    assert i1 > i180 > i360


def test_principal_payment_invalid_period():
    try:
        principal_payment(200000, 0.06, 360, 0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_interest_payment_invalid_period():
    try:
        interest_payment(200000, 0.06, 360, 400)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Remaining Balance Tests
@pytest.mark.parametrize(
    "payments_made,expected_approx",
    [
        (0, 200000),
        (60, 186109),
        (120, 167371),
        (180, 142098),
        (240, 108007),
        (300, 62024),
    ],
)
def test_remaining_balance(payments_made: int, expected_approx: float):
    balance = remaining_balance(200000, 0.06, 360, payments_made)
    assert abs(balance - expected_approx) < 200


def test_remaining_balance_zero_at_end():
    balance = remaining_balance(200000, 0.06, 360, 360)
    assert balance == 0


def test_remaining_balance_negative_payments():
    try:
        remaining_balance(200000, 0.06, 360, -1)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Total Interest Tests
@pytest.mark.parametrize(
    "principal,rate,months,expected_approx",
    [
        (200000, 0.06, 360, 231676),
        (100000, 0.05, 180, 42343),
        (300000, 0.045, 360, 247221),
    ],
)
def test_total_interest(
    principal: float, rate: float, months: int, expected_approx: float
):
    result = total_interest(principal, rate, months)
    assert abs(result - expected_approx) < 100


def test_total_interest_increases_with_rate():
    principal, months = 200000, 360
    low = total_interest(principal, 0.04, months)
    high = total_interest(principal, 0.06, months)
    assert high > low


def test_total_interest_increases_with_term():
    principal, rate = 200000, 0.06
    short = total_interest(principal, rate, 180)
    long = total_interest(principal, rate, 360)
    assert long > short


# Early Payoff Savings Tests
def test_early_payoff_saves_months():
    savings = early_payoff_savings(200000, 0.06, 360, 200)
    assert savings["months_saved"] > 0


def test_early_payoff_saves_interest():
    savings = early_payoff_savings(200000, 0.06, 360, 200)
    assert savings["interest_saved"] > 0


@pytest.mark.parametrize("extra", [100, 200, 500, 1000])
def test_more_extra_saves_more(extra: float):
    small = early_payoff_savings(200000, 0.06, 360, extra)
    large = early_payoff_savings(200000, 0.06, 360, extra + 100)
    assert large["interest_saved"] > small["interest_saved"]
    assert large["months_saved"] >= small["months_saved"]


# Balloon Payment Tests
def test_balloon_payment_basic():
    result = balloon_payment(200000, 0.06, 360, 60)
    assert result["monthly_payment"] > 0
    assert result["balloon_amount"] > 0


def test_balloon_amount_larger_than_remaining():
    result = balloon_payment(200000, 0.06, 360, 60)
    remaining = remaining_balance(200000, 0.06, 360, 59)
    # Balloon includes final interest
    assert result["balloon_amount"] > remaining


def test_balloon_invalid_month():
    try:
        balloon_payment(200000, 0.06, 360, 400)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


@pytest.mark.parametrize("balloon_month", [12, 36, 60, 84])
def test_balloon_at_various_months(balloon_month: int):
    result = balloon_payment(200000, 0.06, 360, balloon_month)
    assert result["balloon_amount"] > 0
    assert result["total_regular_payments"] > 0


# Loan-to-Value Tests
@pytest.mark.parametrize(
    "loan,value,expected",
    [
        (160000, 200000, 0.80),
        (180000, 200000, 0.90),
        (100000, 250000, 0.40),
    ],
)
def test_loan_to_value(loan: float, value: float, expected: float):
    result = loan_to_value(loan, value)
    assert abs(result - expected) < 0.001


def test_ltv_invalid_property_value():
    try:
        loan_to_value(100000, 0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Debt-to-Income Tests
@pytest.mark.parametrize(
    "debt,income,expected",
    [
        (2000, 5000, 0.40),
        (1500, 6000, 0.25),
        (3000, 10000, 0.30),
    ],
)
def test_debt_to_income(debt: float, income: float, expected: float):
    result = debt_to_income(debt, income)
    assert abs(result - expected) < 0.001


def test_dti_invalid_income():
    try:
        debt_to_income(2000, 0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# Max Loan Amount Tests
def test_max_loan_amount_basic():
    max_loan = max_loan_amount(8000, 0.43, 500, 0.06, 360)
    assert max_loan > 0


def test_max_loan_amount_with_high_other_debt():
    low_debt = max_loan_amount(8000, 0.43, 500, 0.06, 360)
    high_debt = max_loan_amount(8000, 0.43, 2500, 0.06, 360)
    assert high_debt < low_debt


def test_max_loan_zero_when_dti_exceeded():
    result = max_loan_amount(5000, 0.30, 2000, 0.06, 360)
    assert result <= max_loan_amount(5000, 0.30, 1000, 0.06, 360)


@pytest.mark.parametrize("max_dti", [0.28, 0.36, 0.43])
def test_max_loan_increases_with_dti(max_dti: float):
    low_dti = max_loan_amount(8000, 0.28, 500, 0.06, 360)
    high_dti = max_loan_amount(8000, max_dti, 500, 0.06, 360)
    assert high_dti >= low_dti
