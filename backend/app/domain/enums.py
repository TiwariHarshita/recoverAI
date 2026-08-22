from enum import Enum


class CaseType(str, Enum):
    PAYMENT_FAILURE = "payment_failure"
    SUBSCRIPTION_FAILURE = "subscription_failure"
    CHECKOUT_ABANDONMENT = "checkout_abandonment"
    OVERDUE_INVOICE = "overdue_invoice"


class RecoveryCaseStatus(str, Enum):
    OPEN = "open"
    DIAGNOSED = "diagnosed"
    PLANNED = "planned"

    WAITING_APPROVAL = "waiting_approval"
    ACTION_SCHEDULED = "action_scheduled"
    ACTION_EXECUTED = "action_executed"
    WAITING_CUSTOMER = "waiting_customer"

    RECOVERED = "recovered"
    ESCALATED = "escalated"
    STOPPED = "stopped"


class PaymentMethod(str, Enum):
    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMI = "emi"
    BANK_TRANSFER = "bank_transfer"
    UNKNOWN = "unknown"


class PaymentStatus(str, Enum):
    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"


class SubscriptionStatus(str, Enum):
    CREATED = "created"
    AUTHENTICATED = "authenticated"
    ACTIVE = "active"
    PENDING = "pending"
    HALTED = "halted"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    EXPIRED = "expired"


class InvoiceStatus(str, Enum):
    ISSUED = "issued"
    PARTIALLY_PAID = "partially_paid"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class FailureClass(str, Enum):
    AUTHENTICATION_FAILURE = "authentication_failure"

    INSUFFICIENT_FUNDS = "insufficient_funds"

    EXPIRED_INSTRUMENT = "expired_instrument"

    BLOCKED_INSTRUMENT = "blocked_instrument"

    INACTIVE_INSTRUMENT = "inactive_instrument"

    TRANSACTION_LIMIT = "transaction_limit"

    BANK_DECLINE = "bank_decline"

    RISK_DECLINE = "risk_decline"

    CUSTOMER_CANCELLED = "customer_cancelled"

    PAYMENT_TIMEOUT = "payment_timeout"

    NETWORK_OR_GATEWAY = "network_or_gateway"

    MANDATE_FAILURE = "mandate_failure"

    MANDATE_CANCELLED = "mandate_cancelled"

    BUSINESS_CONFIGURATION = "business_configuration"

    CHECKOUT_ABANDONMENT = "checkout_abandonment"

    OVERDUE_RECEIVABLE = "overdue_receivable"

    UNKNOWN = "unknown"


class DiagnosisCertainty(str, Enum):
    EXACT = "exact"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class RecoveryActionType(str, Enum):
    IMMEDIATE_RETRY = "immediate_retry"

    DELAYED_RETRY = "delayed_retry"

    CREATE_PAYMENT_LINK = "create_payment_link"

    REQUEST_NEW_PAYMENT_METHOD = "request_new_payment_method"

    SEND_REMINDER = "send_reminder"

    OFFER_PARTIAL_PAYMENT = "offer_partial_payment"

    REQUEST_PROMISE_TO_PAY = "request_promise_to_pay"

    WAIT = "wait"

    ESCALATE_TO_HUMAN = "escalate_to_human"

    STOP = "stop"


class CommunicationChannel(str, Enum):
    SMS = "sms"
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    VOICE = "voice"
    IN_APP = "in_app"
    NONE = "none"


class ActionStatus(str, Enum):
    PROPOSED = "proposed"

    REQUIRES_APPROVAL = "requires_approval"

    APPROVED = "approved"

    BLOCKED = "blocked"

    SCHEDULED = "scheduled"

    EXECUTED = "executed"

    FAILED = "failed"

    CANCELLED = "cancelled"


class PolicyDecision(str, Enum):
    ALLOWED = "allowed"

    BLOCKED = "blocked"

    REQUIRES_APPROVAL = "requires_approval"

    DEFERRED = "deferred"


class AuditActor(str, Enum):
    SYSTEM = "system"
    RAZORPAY = "razorpay"
    ML_MODEL = "ml_model"
    LLM = "llm"
    POLICY_ENGINE = "policy_engine"
    HUMAN = "human"


class AuditEventType(str, Enum):
    CASE_CREATED = "case_created"

    CASE_DIAGNOSED = "case_diagnosed"

    ACTION_SCORED = "action_scored"

    ACTION_SELECTED = "action_selected"

    POLICY_CHECKED = "policy_checked"

    APPROVAL_REQUESTED = "approval_requested"

    APPROVAL_RECEIVED = "approval_received"

    ACTION_EXECUTED = "action_executed"

    ACTION_FAILED = "action_failed"

    PAYMENT_RECEIVED = "payment_received"

    CASE_RECOVERED = "case_recovered"

    CASE_ESCALATED = "case_escalated"

    CASE_STOPPED = "case_stopped"

    ERROR = "error"