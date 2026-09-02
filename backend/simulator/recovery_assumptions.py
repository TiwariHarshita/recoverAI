from enum import Enum
from math import log

from pydantic import BaseModel, ConfigDict

from app.domain.enums import (
    FailureClass,
    PaymentMethod,
    RecoveryActionType,
)


class RecoverySensitivity(str, Enum):
    """Named synthetic recovery conditions used for sensitivity analysis."""

    CONSERVATIVE = "conservative"
    NEUTRAL = "neutral"
    OPTIMISTIC = "optimistic"


class RecoverySimulationConfig(BaseModel):
    """Serializable selection of the simulator's hidden assumption regime."""

    model_config = ConfigDict(frozen=True)

    sensitivity: RecoverySensitivity = RecoverySensitivity.NEUTRAL

    @property
    def log_odds_shift(self) -> float:
        return SENSITIVITY_LOG_ODDS_SHIFTS[self.sensitivity]


# Sensitivity modifies every non-terminal recovery action's odds uniformly.
# Conservative multiplies neutral odds by 0.8; optimistic is its reciprocal
# and multiplies them by 1.25. The symmetric log shifts avoid probability
# clipping and make the scenario meaning explicit rather than action-specific.
SENSITIVITY_LOG_ODDS_SHIFTS: dict[RecoverySensitivity, float] = {
    RecoverySensitivity.CONSERVATIVE: log(0.8),
    RecoverySensitivity.NEUTRAL: 0.0,
    RecoverySensitivity.OPTIMISTIC: log(1.25),
}


# ============================================================
# BASE RECOVERY PROBABILITIES
# ============================================================

"""
These are synthetic-world assumptions.

They are NOT claims about Razorpay's real recovery rates.

Their job is to create a structured environment in which:
- the correct action matters
- customer history matters
- amount matters
- channel matters
- timing matters
- bank/payment method can have a small effect

The ML model we train later must learn these patterns from generated
historical data without being given these probabilities directly.
"""


BASE_RECOVERY_PROBABILITIES: dict[
    FailureClass,
    dict[
        RecoveryActionType,
        float,
    ],
] = {

    # ========================================================
    # AUTHENTICATION FAILURE
    # ========================================================

    FailureClass.AUTHENTICATION_FAILURE: {
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.62,
        RecoveryActionType.IMMEDIATE_RETRY: 0.42,
        RecoveryActionType.SEND_REMINDER: 0.34,
        RecoveryActionType.WAIT: 0.12,
    },

    # ========================================================
    # INSUFFICIENT FUNDS
    # ========================================================

    FailureClass.INSUFFICIENT_FUNDS: {
        RecoveryActionType.DELAYED_RETRY: 0.46,
        RecoveryActionType.SEND_REMINDER: 0.40,
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.38,
        RecoveryActionType.WAIT: 0.18,
    },

    # ========================================================
    # EXPIRED INSTRUMENT
    # ========================================================

    FailureClass.EXPIRED_INSTRUMENT: {
        RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: 0.72,
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.58,
        RecoveryActionType.SEND_REMINDER: 0.32,
    },

    # ========================================================
    # BLOCKED INSTRUMENT
    # ========================================================

    FailureClass.BLOCKED_INSTRUMENT: {
        RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: 0.65,
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.53,
        RecoveryActionType.SEND_REMINDER: 0.28,
    },

    # ========================================================
    # INACTIVE INSTRUMENT
    # ========================================================

    FailureClass.INACTIVE_INSTRUMENT: {
        RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: 0.63,
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.50,
    },

    # ========================================================
    # TRANSACTION LIMIT
    # ========================================================

    FailureClass.TRANSACTION_LIMIT: {
        RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: 0.60,
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.48,
        RecoveryActionType.DELAYED_RETRY: 0.26,
    },

    # ========================================================
    # BANK DECLINE
    # ========================================================

    FailureClass.BANK_DECLINE: {
        RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: 0.58,
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.46,
        RecoveryActionType.DELAYED_RETRY: 0.20,
    },

    # ========================================================
    # RISK DECLINE
    # ========================================================

    FailureClass.RISK_DECLINE: {
        RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: 0.30,
        RecoveryActionType.ESCALATE_TO_HUMAN: 0.44,
    },

    # ========================================================
    # CUSTOMER CANCELLED
    # ========================================================

    FailureClass.CUSTOMER_CANCELLED: {
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.52,
        RecoveryActionType.SEND_REMINDER: 0.44,
        RecoveryActionType.WAIT: 0.20,
    },

    # ========================================================
    # PAYMENT TIMEOUT
    # ========================================================

    FailureClass.PAYMENT_TIMEOUT: {
        RecoveryActionType.IMMEDIATE_RETRY: 0.56,
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.50,
        RecoveryActionType.DELAYED_RETRY: 0.58,
    },

    # ========================================================
    # NETWORK / GATEWAY
    # ========================================================

    FailureClass.NETWORK_OR_GATEWAY: {
        RecoveryActionType.IMMEDIATE_RETRY: 0.60,
        RecoveryActionType.DELAYED_RETRY: 0.66,
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.48,
        RecoveryActionType.WAIT: 0.24,
    },

    # ========================================================
    # MANDATE FAILURE
    # ========================================================

    FailureClass.MANDATE_FAILURE: {
        RecoveryActionType.DELAYED_RETRY: 0.42,
        RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: 0.55,
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.50,
    },

    # ========================================================
    # MANDATE CANCELLED
    # ========================================================

    FailureClass.MANDATE_CANCELLED: {
        RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: 0.72,
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.56,
        RecoveryActionType.SEND_REMINDER: 0.34,
    },

    # ========================================================
    # BUSINESS CONFIGURATION
    # ========================================================

    FailureClass.BUSINESS_CONFIGURATION: {
        RecoveryActionType.ESCALATE_TO_HUMAN: 0.58,
        RecoveryActionType.STOP: 0.0,
    },

    # ========================================================
    # CHECKOUT ABANDONMENT
    # ========================================================

    FailureClass.CHECKOUT_ABANDONMENT: {
        RecoveryActionType.CREATE_PAYMENT_LINK: 0.57,
        RecoveryActionType.SEND_REMINDER: 0.49,
        RecoveryActionType.WAIT: 0.16,
    },

    # ========================================================
    # OVERDUE RECEIVABLE
    # ========================================================

    FailureClass.OVERDUE_RECEIVABLE: {
        RecoveryActionType.SEND_REMINDER: 0.42,
        RecoveryActionType.REQUEST_PROMISE_TO_PAY: 0.48,
        RecoveryActionType.OFFER_PARTIAL_PAYMENT: 0.62,
        RecoveryActionType.ESCALATE_TO_HUMAN: 0.52,
    },

    # ========================================================
    # UNKNOWN
    # ========================================================

    FailureClass.UNKNOWN: {
        RecoveryActionType.ESCALATE_TO_HUMAN: 0.25,
        RecoveryActionType.STOP: 0.0,
    },
}


# ============================================================
# FALLBACK PROBABILITIES
# ============================================================

"""
Normally only Candidate Action Generator outputs should reach
the environment.

These fallback values make the simulator robust when a test or future
experiment intentionally applies an unusual action.
"""


DEFAULT_ACTION_PROBABILITIES: dict[
    RecoveryActionType,
    float,
] = {
    RecoveryActionType.IMMEDIATE_RETRY: 0.15,
    RecoveryActionType.DELAYED_RETRY: 0.18,
    RecoveryActionType.CREATE_PAYMENT_LINK: 0.25,
    RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: 0.30,
    RecoveryActionType.SEND_REMINDER: 0.22,
    RecoveryActionType.OFFER_PARTIAL_PAYMENT: 0.40,
    RecoveryActionType.REQUEST_PROMISE_TO_PAY: 0.30,
    RecoveryActionType.WAIT: 0.08,
    RecoveryActionType.ESCALATE_TO_HUMAN: 0.25,
    RecoveryActionType.STOP: 0.0,
}


# ============================================================
# SYNTHETIC PAYMENT-METHOD LOG-ODDS EFFECTS
# ============================================================

"""
Small synthetic log-odds effects only.

These exist so payment_method can carry real predictive signal in the
future ML dataset.

They are NOT real-world claims about method performance.
"""


PAYMENT_METHOD_RECOVERY_LOG_ODDS_EFFECTS: dict[
    PaymentMethod,
    float,
] = {
    PaymentMethod.CARD: 0.00,
    PaymentMethod.UPI: 0.02,
    PaymentMethod.NETBANKING: -0.01,
    PaymentMethod.WALLET: 0.01,
    PaymentMethod.EMI: -0.02,
    PaymentMethod.BANK_TRANSFER: -0.01,
    PaymentMethod.UNKNOWN: -0.02,
}


# ============================================================
# SYNTHETIC BANK LOG-ODDS EFFECTS
# ============================================================

"""
Again, these are arbitrary synthetic-world effects.

Do not interpret them as claims about actual banks.
"""


BANK_RECOVERY_LOG_ODDS_EFFECTS: dict[
    str,
    float,
] = {
    "HDFC": 0.020,
    "ICICI": 0.010,
    "SBI": -0.010,
    "AXIS": 0.000,
    "KOTAK": 0.005,
    "YES_BANK": -0.020,
}


# ============================================================
# RECOVERY DELAY WINDOWS
# ============================================================

"""
Range of hours between executing an action and observing recovery.

Used only after the action successfully causes recovery.
"""


ACTION_RECOVERY_DELAY_HOURS: dict[
    RecoveryActionType,
    tuple[float, float],
] = {
    RecoveryActionType.IMMEDIATE_RETRY: (
        0.05,
        1.0,
    ),

    RecoveryActionType.DELAYED_RETRY: (
        2.0,
        24.0,
    ),

    RecoveryActionType.CREATE_PAYMENT_LINK: (
        0.25,
        72.0,
    ),

    RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD: (
        2.0,
        96.0,
    ),

    RecoveryActionType.SEND_REMINDER: (
        1.0,
        96.0,
    ),

    RecoveryActionType.OFFER_PARTIAL_PAYMENT: (
        2.0,
        120.0,
    ),

    RecoveryActionType.REQUEST_PROMISE_TO_PAY: (
        24.0,
        168.0,
    ),

    RecoveryActionType.WAIT: (
        12.0,
        168.0,
    ),

    RecoveryActionType.ESCALATE_TO_HUMAN: (
        24.0,
        240.0,
    ),

    RecoveryActionType.STOP: (
        0.0,
        0.0,
    ),
}
