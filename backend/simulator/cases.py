from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
from decimal import (
    Decimal,
    ROUND_HALF_UP,
)
from hashlib import sha256
from random import Random

from pydantic import BaseModel

from app.domain.customer import Customer
from app.domain.enums import (
    CaseType,
    FailureClass,
    InvoiceStatus,
    PaymentMethod,
    PaymentStatus,
    RecoveryCaseStatus,
    SubscriptionStatus,
)
from app.domain.invoice import Invoice
from app.domain.payment_attempt import PaymentAttempt
from app.domain.recovery_case import RecoveryCase
from app.domain.subscription import Subscription

from simulator.case_assumptions import (
    BANKS_BY_METHOD,
    FAILURE_TEMPLATES,
    PAYMENT_FAILURE_WEIGHTS,
    SUBSCRIPTION_FAILURE_WEIGHTS,
    FailureTemplate,
)
from simulator.customers import (
    DEFAULT_REFERENCE_TIME,
    SyntheticPopulation,
)
from simulator.merchants import (
    SyntheticMerchant,
)
from simulator.random_utils import (
    weighted_choice,
)


class SyntheticRecoveryScenario(BaseModel):
    """
    One generated RecoveryCase plus the source domain entities
    that caused the recovery problem.

    expected_failure_class is simulator ground truth.

    It deliberately stays OUTSIDE RecoveryCase so the Diagnosis
    Engine must infer the normalized failure class from raw facts.
    """

    case: RecoveryCase

    expected_failure_class: FailureClass

    payment: PaymentAttempt | None = None

    subscription: Subscription | None = None

    invoice: Invoice | None = None


class SyntheticCaseBatch(BaseModel):
    """
    Reproducible batch of synthetic recovery scenarios.
    """

    seed: int

    reference_time: datetime

    scenarios: list[
        SyntheticRecoveryScenario
    ]


# ============================================================
# DETERMINISTIC IDS
# ============================================================


def _stable_token(
    *,
    seed: int,
    case_index: int,
    customer_id: str,
    case_type: CaseType,
) -> str:
    """
    Create deterministic IDs without UUID4.

    Re-running the same generation parameters produces the
    same case/payment/subscription/invoice IDs.
    """

    raw = (
        f"{seed}:"
        f"{case_index}:"
        f"{customer_id}:"
        f"{case_type.value}"
    ).encode(
        "utf-8"
    )

    return sha256(
        raw
    ).hexdigest()[:18]


# ============================================================
# VALIDATION
# ============================================================


def _validate_reference_time(
    reference_time: datetime,
) -> None:

    if reference_time.tzinfo is None:
        raise ValueError(
            "reference_time must be timezone-aware."
        )


def _validate_relationship(
    merchant: SyntheticMerchant,
    customer: Customer,
) -> None:

    if (
        customer.merchant_id
        != merchant.id
    ):
        raise ValueError(
            (
                "Customer does not belong to "
                "the supplied merchant."
            )
        )

    if (
        merchant.policy.merchant_id
        != merchant.id
    ):
        raise ValueError(
            (
                "MerchantPolicy does not belong "
                "to the supplied merchant."
            )
        )


# ============================================================
# AMOUNT GENERATION
# ============================================================


def _generate_amount(
    rng: Random,
    merchant: SyntheticMerchant,
    *,
    invoice: bool = False,
) -> Decimal:
    """
    Generate an amount around the merchant's average order value.

    Invoices have a wider range because B2B receivables are
    naturally more variable.
    """

    if invoice:

        multiplier = rng.uniform(
            0.75,
            2.50,
        )

    else:

        multiplier = (
            rng.lognormvariate(
                0.0,
                0.35,
            )
        )

        multiplier = max(
            0.35,
            min(
                multiplier,
                3.0,
            ),
        )

    amount = (
        merchant.average_order_value
        * Decimal(
            str(multiplier)
        )
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )

    return max(
        amount,
        Decimal("50.00"),
    )


# ============================================================
# PAYMENT METHOD
# ============================================================


def _choose_payment_method(
    rng: Random,
    merchant: SyntheticMerchant,
    customer: Customer,
    *,
    compatible_methods: (
        frozenset[PaymentMethod]
        | None
    ) = None,
) -> PaymentMethod:
    """
    Choose a method from the merchant's real method distribution.

    When a failure only makes sense for certain methods, the
    merchant distribution is filtered first.

    Customer preference is used 65% of the time when compatible.
    """

    weights = {
        method: weight
        for method, weight
        in merchant.payment_method_weights.items()
        if (
            compatible_methods is None
            or method
            in compatible_methods
        )
    }

    if not weights:
        raise ValueError(
            (
                "Merchant has no payment method "
                "compatible with the generated scenario."
            )
        )

    preferred = (
        customer.preferred_payment_method
    )

    if (
        preferred is not None
        and preferred in weights
        and rng.random() < 0.65
    ):
        return preferred

    return weighted_choice(
        rng,
        weights,
    )


def _choose_bank(
    rng: Random,
    payment_method: PaymentMethod,
) -> str | None:

    banks = BANKS_BY_METHOD.get(
        payment_method
    )

    if not banks:
        return None

    return rng.choice(
        banks
    )


# ============================================================
# FAILURE SELECTION
# ============================================================


def _choose_failure_template(
    rng: Random,
    case_type: CaseType,
) -> FailureTemplate:

    if (
        case_type
        == CaseType.PAYMENT_FAILURE
    ):

        failure_class = (
            weighted_choice(
                rng,
                PAYMENT_FAILURE_WEIGHTS,
            )
        )

    elif (
        case_type
        == CaseType.SUBSCRIPTION_FAILURE
    ):

        failure_class = (
            weighted_choice(
                rng,
                SUBSCRIPTION_FAILURE_WEIGHTS,
            )
        )

    else:

        raise ValueError(
            (
                "Failure templates are only used "
                "for payment or subscription failures."
            )
        )

    templates = FAILURE_TEMPLATES[
        failure_class
    ]

    return rng.choice(
        templates
    )


# ============================================================
# EVENT TIME
# ============================================================


def _recent_event_time(
    rng: Random,
    merchant: SyntheticMerchant,
    reference_time: datetime,
) -> datetime:
    """
    Generate recent failures inside the merchant's currently
    active recovery window.
    """

    max_age_hours = min(
        48,
        max(
            1,
            (
                merchant.policy
                .max_recovery_window_days
                * 24
                - 1
            ),
        ),
    )

    age_minutes = rng.randint(
        5,
        max_age_hours * 60,
    )

    return (
        reference_time
        - timedelta(
            minutes=age_minutes
        )
    )


# ============================================================
# PAYMENT + SUBSCRIPTION FAILURE BUILDER
# ============================================================


def _build_payment_failure(
    *,
    merchant: SyntheticMerchant,
    customer: Customer,
    rng: Random,
    reference_time: datetime,
    seed: int,
    case_index: int,
    case_type: CaseType,
) -> SyntheticRecoveryScenario:

    template = (
        _choose_failure_template(
            rng,
            case_type,
        )
    )

    payment_method = (
        _choose_payment_method(
            rng,
            merchant,
            customer,
            compatible_methods=(
                template.compatible_methods
            ),
        )
    )

    amount = _generate_amount(
        rng,
        merchant,
    )

    event_at = _recent_event_time(
        rng,
        merchant,
        reference_time,
    )

    token = _stable_token(
        seed=seed,
        case_index=case_index,
        customer_id=customer.id,
        case_type=case_type,
    )

    case_id = (
        f"rc_sim_{token}"
    )

    payment_id = (
        f"pay_sim_{token}"
    )

    attempt_number = rng.randint(
        1,
        3,
    )

    bank = _choose_bank(
        rng,
        payment_method,
    )

    # --------------------------------------------------------
    # Failed Payment entity
    # --------------------------------------------------------

    payment = PaymentAttempt(
        id=payment_id,
        merchant_id=merchant.id,
        customer_id=customer.id,
        order_id=(
            f"order_sim_{token}"
        ),
        amount=amount,
        currency="INR",
        status=PaymentStatus.FAILED,
        method=payment_method,
        bank=bank,
        attempt_number=attempt_number,
        error_code=template.error_code,
        error_source=(
            template.error_source
        ),
        error_step=template.error_step,
        error_reason=(
            template.error_reason
        ),
        error_description=(
            template.error_description
        ),
        created_at=event_at,
    )

    # --------------------------------------------------------
    # Subscription entity when this is a subscription failure
    # --------------------------------------------------------

    subscription: (
        Subscription
        | None
    ) = None

    subscription_id: (
        str
        | None
    ) = None

    if (
        case_type
        == CaseType.SUBSCRIPTION_FAILURE
    ):

        subscription_id = (
            f"sub_sim_{token}"
        )

        mandate_active = (
            template.failure_class
            not in {
                FailureClass.MANDATE_FAILURE,
                FailureClass.MANDATE_CANCELLED,
            }
        )

        subscription_status = (
            SubscriptionStatus.HALTED
            if (
                template.failure_class
                == FailureClass.MANDATE_CANCELLED
            )
            else SubscriptionStatus.PENDING
        )

        subscription = Subscription(
            id=subscription_id,
            merchant_id=merchant.id,
            customer_id=customer.id,
            amount=amount,
            currency="INR",
            status=subscription_status,
            preferred_payment_method=(
                payment_method
            ),
            retry_count=max(
                0,
                attempt_number - 1,
            ),
            mandate_active=(
                mandate_active
            ),
            current_period_start=(
                event_at
                - timedelta(
                    days=30
                )
            ),
            current_period_end=(
                event_at
            ),
            next_charge_at=(
                event_at
                + timedelta(
                    days=1
                )
            ),
            created_at=(
                customer.created_at
            ),
            raw_payload={
                "synthetic": True,
                "scenario_token": token,
            },
        )

    # --------------------------------------------------------
    # RecoveryCase
    # --------------------------------------------------------

    case = RecoveryCase(
        id=case_id,

        merchant_id=merchant.id,

        customer_id=customer.id,

        case_type=case_type,

        status=RecoveryCaseStatus.OPEN,

        amount_at_risk=amount,

        currency="INR",

        payment_id=payment_id,

        subscription_id=(
            subscription_id
        ),

        payment_method=(
            payment_method
        ),

        # IMPORTANT:
        #
        # The synthetic generator does not give the Diagnosis
        # Engine its answer.
        failure_class=(
            FailureClass.UNKNOWN
        ),

        error_code=template.error_code,

        error_source=(
            template.error_source
        ),

        error_step=(
            template.error_step
        ),

        error_reason=(
            template.error_reason
        ),

        error_description=(
            template.error_description
        ),

        attempt_count=(
            attempt_number
        ),

        # No RecoverAI actions have happened yet.
        recovery_retry_count=0,

        previous_contacts=0,

        recovered_amount=(
            Decimal("0")
        ),

        metadata={
            "synthetic": True,
            "merchant_archetype": (
                merchant.archetype.value
            ),
            "scenario_token": token,
        },

        created_at=event_at,

        updated_at=event_at,
    )

    return SyntheticRecoveryScenario(
        case=case,

        expected_failure_class=(
            template.failure_class
        ),

        payment=payment,

        subscription=subscription,
    )


# ============================================================
# CHECKOUT ABANDONMENT
# ============================================================


def _build_checkout_abandonment(
    *,
    merchant: SyntheticMerchant,
    customer: Customer,
    rng: Random,
    reference_time: datetime,
    seed: int,
    case_index: int,
) -> SyntheticRecoveryScenario:

    case_type = (
        CaseType.CHECKOUT_ABANDONMENT
    )

    token = _stable_token(
        seed=seed,
        case_index=case_index,
        customer_id=customer.id,
        case_type=case_type,
    )

    amount = _generate_amount(
        rng,
        merchant,
    )

    payment_method = (
        _choose_payment_method(
            rng,
            merchant,
            customer,
        )
    )

    event_at = _recent_event_time(
        rng,
        merchant,
        reference_time,
    )

    case = RecoveryCase(
        id=f"rc_sim_{token}",

        merchant_id=merchant.id,

        customer_id=customer.id,

        case_type=case_type,

        status=RecoveryCaseStatus.OPEN,

        amount_at_risk=amount,

        currency="INR",

        checkout_id=(
            f"checkout_sim_{token}"
        ),

        payment_method=(
            payment_method
        ),

        failure_class=(
            FailureClass.UNKNOWN
        ),

        attempt_count=0,

        recovery_retry_count=0,

        previous_contacts=0,

        recovered_amount=(
            Decimal("0")
        ),

        metadata={
            "synthetic": True,
            "merchant_archetype": (
                merchant.archetype.value
            ),
            "scenario_token": token,
        },

        created_at=event_at,

        updated_at=event_at,
    )

    return SyntheticRecoveryScenario(
        case=case,

        expected_failure_class=(
            FailureClass.CHECKOUT_ABANDONMENT
        ),
    )


# ============================================================
# OVERDUE INVOICE
# ============================================================


def _build_overdue_invoice(
    *,
    merchant: SyntheticMerchant,
    customer: Customer,
    rng: Random,
    reference_time: datetime,
    seed: int,
    case_index: int,
) -> SyntheticRecoveryScenario:

    case_type = (
        CaseType.OVERDUE_INVOICE
    )

    token = _stable_token(
        seed=seed,
        case_index=case_index,
        customer_id=customer.id,
        case_type=case_type,
    )

    amount_due = _generate_amount(
        rng,
        merchant,
        invoice=True,
    )

    # Keep generated invoice cases inside the merchant's
    # active recovery window.
    max_days_overdue = min(
        30,
        max(
            1,
            (
                merchant.policy
                .max_recovery_window_days
                - 1
            ),
        ),
    )

    days_overdue = rng.randint(
        1,
        max_days_overdue,
    )

    due_at = (
        reference_time
        - timedelta(
            days=days_overdue
        )
    )

    issued_at = (
        due_at
        - timedelta(
            days=30
        )
    )

    amount_paid = Decimal("0")

    # Some overdue invoices have already been partially paid.
    if rng.random() < 0.20:

        paid_fraction = Decimal(
            str(
                rng.uniform(
                    0.10,
                    0.60,
                )
            )
        )

        amount_paid = (
            amount_due
            * paid_fraction
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

    amount_at_risk = (
        amount_due
        - amount_paid
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )

    invoice_status = (
        InvoiceStatus.PARTIALLY_PAID
        if amount_paid > 0
        else InvoiceStatus.OVERDUE
    )

    invoice_id = (
        f"inv_sim_{token}"
    )

    invoice = Invoice(
        id=invoice_id,

        merchant_id=merchant.id,

        customer_id=customer.id,

        amount_due=amount_due,

        amount_paid=amount_paid,

        currency="INR",

        status=invoice_status,

        issued_at=issued_at,

        due_at=due_at,

        days_overdue=(
            days_overdue
        ),

        created_at=issued_at,

        raw_payload={
            "synthetic": True,
            "scenario_token": token,
        },
    )

    payment_method = (
        _choose_payment_method(
            rng,
            merchant,
            customer,
        )
    )

    case = RecoveryCase(
        id=f"rc_sim_{token}",

        merchant_id=merchant.id,

        customer_id=customer.id,

        case_type=case_type,

        status=RecoveryCaseStatus.OPEN,

        amount_at_risk=(
            amount_at_risk
        ),

        currency="INR",

        invoice_id=invoice_id,

        payment_method=(
            payment_method
        ),

        failure_class=(
            FailureClass.UNKNOWN
        ),

        attempt_count=0,

        recovery_retry_count=0,

        previous_contacts=0,

        recovered_amount=(
            Decimal("0")
        ),

        metadata={
            "synthetic": True,
            "merchant_archetype": (
                merchant.archetype.value
            ),
            "scenario_token": token,
            "days_overdue": (
                days_overdue
            ),
        },

        # The case becomes recoverable once the invoice
        # crosses its due date.
        created_at=due_at,

        updated_at=(
            reference_time
        ),
    )

    return SyntheticRecoveryScenario(
        case=case,

        expected_failure_class=(
            FailureClass.OVERDUE_RECEIVABLE
        ),

        invoice=invoice,
    )


# ============================================================
# INTERNAL CASE DISPATCH
# ============================================================


def _generate_case_with_rng(
    *,
    merchant: SyntheticMerchant,
    customer: Customer,
    rng: Random,
    reference_time: datetime,
    seed: int,
    case_index: int,
    case_type: CaseType | None = None,
) -> SyntheticRecoveryScenario:

    _validate_relationship(
        merchant,
        customer,
    )

    _validate_reference_time(
        reference_time
    )

    if (
        customer.created_at
        > reference_time
    ):
        raise ValueError(
            (
                "Customer cannot be created "
                "after reference_time."
            )
        )

    selected_case_type = (
        case_type
        if case_type is not None
        else weighted_choice(
            rng,
            merchant.case_type_weights,
        )
    )

    if selected_case_type in {
        CaseType.PAYMENT_FAILURE,
        CaseType.SUBSCRIPTION_FAILURE,
    }:

        return _build_payment_failure(
            merchant=merchant,
            customer=customer,
            rng=rng,
            reference_time=(
                reference_time
            ),
            seed=seed,
            case_index=case_index,
            case_type=(
                selected_case_type
            ),
        )

    if (
        selected_case_type
        == CaseType.CHECKOUT_ABANDONMENT
    ):

        return (
            _build_checkout_abandonment(
                merchant=merchant,
                customer=customer,
                rng=rng,
                reference_time=(
                    reference_time
                ),
                seed=seed,
                case_index=case_index,
            )
        )

    if (
        selected_case_type
        == CaseType.OVERDUE_INVOICE
    ):

        return _build_overdue_invoice(
            merchant=merchant,
            customer=customer,
            rng=rng,
            reference_time=(
                reference_time
            ),
            seed=seed,
            case_index=case_index,
        )

    raise ValueError(
        (
            "Unsupported case type: "
            f"{selected_case_type}"
        )
    )


# ============================================================
# PUBLIC: ONE CASE
# ============================================================


def generate_case_for_customer(
    merchant: SyntheticMerchant,
    customer: Customer,
    *,
    seed: int = 42,
    case_index: int = 1,
    reference_time: datetime = (
        DEFAULT_REFERENCE_TIME
    ),
    case_type: CaseType | None = None,
) -> SyntheticRecoveryScenario:
    """
    Generate one deterministic synthetic recovery scenario.

    case_type can be supplied explicitly for tests/demo scenarios.
    If omitted, it is sampled from the merchant's case-type mix.
    """

    if case_index <= 0:
        raise ValueError(
            (
                "case_index must be "
                "greater than zero."
            )
        )

    return _generate_case_with_rng(
        merchant=merchant,
        customer=customer,
        rng=Random(seed),
        reference_time=reference_time,
        seed=seed,
        case_index=case_index,
        case_type=case_type,
    )


# ============================================================
# PUBLIC: BATCH
# ============================================================


def generate_recovery_cases(
    population: SyntheticPopulation,
    count: int,
    *,
    seed: int = 42,
    reference_time: datetime | None = None,
) -> SyntheticCaseBatch:
    """
    Generate a deterministic batch of RecoveryCase scenarios
    from an existing SyntheticPopulation.
    """

    if count <= 0:
        raise ValueError(
            (
                "count must be "
                "greater than zero."
            )
        )

    if not population.merchants:
        raise ValueError(
            (
                "population must contain "
                "at least one merchant."
            )
        )

    if not population.customers:
        raise ValueError(
            (
                "population must contain "
                "at least one customer."
            )
        )

    resolved_reference_time = (
        reference_time
        if reference_time is not None
        else population.reference_time
    )

    _validate_reference_time(
        resolved_reference_time
    )

    merchants_by_id = {
        merchant.id: merchant
        for merchant
        in population.merchants
    }

    valid_customers = [
        customer
        for customer
        in population.customers
        if (
            customer.merchant_id
            in merchants_by_id
        )
    ]

    if not valid_customers:
        raise ValueError(
            (
                "population contains no customers "
                "linked to generated merchants."
            )
        )

    rng = Random(seed)

    scenarios: list[
        SyntheticRecoveryScenario
    ] = []

    for case_index in range(
        1,
        count + 1,
    ):

        customer = rng.choice(
            valid_customers
        )

        merchant = merchants_by_id[
            customer.merchant_id
        ]

        scenario = (
            _generate_case_with_rng(
                merchant=merchant,
                customer=customer,
                rng=rng,
                reference_time=(
                    resolved_reference_time
                ),
                seed=seed,
                case_index=case_index,
            )
        )

        scenarios.append(
            scenario
        )

    return SyntheticCaseBatch(
        seed=seed,

        reference_time=(
            resolved_reference_time
        ),

        scenarios=scenarios,
    )
