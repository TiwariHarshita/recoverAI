from datetime import datetime, time, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import inspect

from app.db import (
    AuditEventRepository,
    CustomerRepository,
    InvoiceRepository,
    MerchantPolicyRepository,
    PaymentRepository,
    RecoveryActionRepository,
    RecoveryCaseRepository,
    SubscriptionRepository,
    build_engine,
    build_session_factory,
    create_schema,
    session_scope,
)
from app.domain.actions import RecoveryAction
from app.domain.audit import AuditEvent
from app.domain.customer import Customer
from app.domain.enums import (
    ActionStatus,
    AuditActor,
    AuditEventType,
    CaseType,
    CommunicationChannel,
    FailureClass,
    InvoiceStatus,
    PaymentMethod,
    PaymentStatus,
    RecoveryActionType,
    RecoveryCaseStatus,
    SubscriptionStatus,
)
from app.domain.invoice import Invoice
from app.domain.payment import Payment
from app.domain.policies import MerchantPolicy
from app.domain.recovery_case import RecoveryCase
from app.domain.subscription import Subscription


NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def db():
    engine = build_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    factory = build_session_factory(engine)
    try:
        yield engine, factory
    finally:
        engine.dispose()


def make_case(**overrides):
    values = dict(
        id="rc_test_1",
        merchant_id="merchant_1",
        customer_id="cust_1",
        case_type=CaseType.PAYMENT_FAILURE,
        status=RecoveryCaseStatus.PLANNED,
        amount_at_risk=Decimal("12500.50"),
        payment_id="pay_1",
        payment_method=PaymentMethod.UPI,
        failure_class=FailureClass.INSUFFICIENT_FUNDS,
        attempt_count=2,
        recovery_retry_count=1,
        previous_contacts=1,
        predicted_recovery_probability=0.7345678,
        expected_recovery_value=Decimal("9179.10"),
        selected_action_id="act_1",
        metadata={"source": "test", "nested": {"ok": True}},
        created_at=NOW,
        updated_at=NOW,
    )
    values.update(overrides)
    return RecoveryCase(**values)


def test_schema_contains_all_layer18_tables(db):
    engine, _ = db
    tables = set(inspect(engine).get_table_names())
    assert {
        "customers",
        "payments",
        "subscriptions",
        "invoices",
        "recovery_cases",
        "recovery_actions",
        "audit_events",
        "merchant_policies",
    }.issubset(tables)


def test_recovery_case_round_trip_and_upsert(db):
    _, factory = db
    case = make_case()

    with session_scope(factory) as session:
        repo = RecoveryCaseRepository(session)
        repo.save(case)

    with session_scope(factory) as session:
        repo = RecoveryCaseRepository(session)
        loaded = repo.get(case.id)
        assert loaded is not None
        assert loaded.id == case.id
        assert loaded.case_type == CaseType.PAYMENT_FAILURE
        assert loaded.failure_class == FailureClass.INSUFFICIENT_FUNDS
        assert loaded.amount_at_risk == Decimal("12500.50")
        assert loaded.predicted_recovery_probability == pytest.approx(0.7345678)
        assert loaded.metadata["nested"]["ok"] is True

        loaded.status = RecoveryCaseStatus.RECOVERED
        loaded.recovered_amount = Decimal("12500.50")
        loaded.closed_at = NOW + timedelta(hours=2)
        repo.save(loaded)

    with session_scope(factory) as session:
        loaded = RecoveryCaseRepository(session).get(case.id)
        assert loaded.status == RecoveryCaseStatus.RECOVERED
        assert loaded.recovered_amount == Decimal("12500.50")
        assert loaded.closed_at is not None


def test_case_listing_is_merchant_scoped_and_filterable(db):
    _, factory = db
    with session_scope(factory) as session:
        repo = RecoveryCaseRepository(session)
        repo.save(make_case(id="rc_open", status=RecoveryCaseStatus.OPEN))
        repo.save(make_case(id="rc_recovered", status=RecoveryCaseStatus.RECOVERED))
        repo.save(make_case(id="rc_other", merchant_id="merchant_2"))

    with session_scope(factory) as session:
        repo = RecoveryCaseRepository(session)
        assert {case.id for case in repo.list_for_merchant("merchant_1")} == {"rc_open", "rc_recovered"}
        assert [case.id for case in repo.list_for_merchant("merchant_1", status=RecoveryCaseStatus.OPEN)] == ["rc_open"]


def test_recovery_actions_round_trip_and_case_history(db):
    _, factory = db
    case = make_case(selected_action_id=None)
    a1 = RecoveryAction(
        id="act_a", case_id=case.id, action_type=RecoveryActionType.DELAYED_RETRY,
        channel=CommunicationChannel.NONE, status=ActionStatus.SCHEDULED,
        scheduled_for=NOW + timedelta(hours=4), amount=Decimal("12500.50"),
        predicted_recovery_probability=0.61, expected_recovery_value=Decimal("7600.25"),
        reason="best ERV", metadata={"policy": "allowed"}, created_at=NOW,
    )
    a2 = RecoveryAction(
        id="act_b", case_id=case.id, action_type=RecoveryActionType.SEND_REMINDER,
        channel=CommunicationChannel.WHATSAPP, created_at=NOW + timedelta(minutes=1),
    )

    with session_scope(factory) as session:
        RecoveryCaseRepository(session).save(case)
        repo = RecoveryActionRepository(session)
        repo.save(a1)
        repo.save(a2)

    with session_scope(factory) as session:
        repo = RecoveryActionRepository(session)
        loaded = repo.get("act_a")
        assert loaded.action_type == RecoveryActionType.DELAYED_RETRY
        assert loaded.status == ActionStatus.SCHEDULED
        assert loaded.expected_recovery_value == Decimal("7600.25")
        assert loaded.predicted_recovery_probability == pytest.approx(0.61)
        assert [a.id for a in repo.list_for_case(case.id)] == ["act_a", "act_b"]


def test_audit_events_are_append_only_history_in_order(db):
    _, factory = db
    case = make_case()
    events = [
        AuditEvent(
            id="audit_1", case_id=case.id, event_type=AuditEventType.CASE_CREATED,
            actor=AuditActor.SYSTEM, message="created", data={"source": "webhook"}, created_at=NOW,
        ),
        AuditEvent(
            id="audit_2", case_id=case.id, event_type=AuditEventType.ACTION_SELECTED,
            actor=AuditActor.ML_MODEL, message="selected", data={"action_id": "act_1"},
            created_at=NOW + timedelta(seconds=1),
        ),
    ]

    with session_scope(factory) as session:
        RecoveryCaseRepository(session).save(case)
        repo = AuditEventRepository(session)
        for event in events:
            repo.append(event)

    with session_scope(factory) as session:
        loaded = AuditEventRepository(session).list_for_case(case.id)
        assert [e.id for e in loaded] == ["audit_1", "audit_2"]
        assert loaded[1].actor == AuditActor.ML_MODEL
        assert loaded[1].data["action_id"] == "act_1"


def test_customer_round_trip(db):
    _, factory = db
    customer = Customer(
        id="cust_1", merchant_id="merchant_1", email="x@example.com", phone="9999999999",
        created_at=NOW, lifetime_value=Decimal("50000.25"), successful_payments=8,
        failed_payments=2, historical_payment_success_rate=0.8,
        previous_recovery_attempts=3, previous_recovery_successes=2,
        preferred_payment_method=PaymentMethod.CARD,
        preferred_channel=CommunicationChannel.EMAIL, language_preference="en",
        do_not_contact=True, timezone="Asia/Kolkata",
    )
    with session_scope(factory) as session:
        CustomerRepository(session).save(customer)
    with session_scope(factory) as session:
        loaded = CustomerRepository(session).get(customer.id)
        assert loaded.lifetime_value == Decimal("50000.25")
        assert loaded.preferred_payment_method == PaymentMethod.CARD
        assert loaded.preferred_channel == CommunicationChannel.EMAIL
        assert loaded.do_not_contact is True


def test_payment_subscription_and_invoice_round_trip(db):
    _, factory = db
    payment = Payment(
        id="pay_1", merchant_id="merchant_1", customer_id="cust_1", order_id="order_1",
        amount=Decimal("1200.00"), status=PaymentStatus.FAILED, method=PaymentMethod.UPI,
        error_code="BAD_REQUEST_ERROR", error_reason="insufficient_funds", created_at=NOW,
        raw_payload={"provider": "razorpay"},
    )
    subscription = Subscription(
        id="sub_1", merchant_id="merchant_1", customer_id="cust_1", amount=Decimal("999.00"),
        status=SubscriptionStatus.HALTED, preferred_payment_method=PaymentMethod.CARD,
        retry_count=2, mandate_active=True, next_charge_at=NOW + timedelta(days=1),
        created_at=NOW, raw_payload={"kind": "subscription"},
    )
    invoice = Invoice(
        id="inv_1", merchant_id="merchant_1", customer_id="cust_1",
        amount_due=Decimal("5000.00"), amount_paid=Decimal("1000.00"),
        status=InvoiceStatus.OVERDUE, issued_at=NOW - timedelta(days=10),
        due_at=NOW - timedelta(days=3), days_overdue=3, created_at=NOW - timedelta(days=10),
        raw_payload={"kind": "invoice"},
    )

    with session_scope(factory) as session:
        PaymentRepository(session).save(payment)
        SubscriptionRepository(session).save(subscription)
        InvoiceRepository(session).save(invoice)

    with session_scope(factory) as session:
        p = PaymentRepository(session).get("pay_1")
        s = SubscriptionRepository(session).get("sub_1")
        i = InvoiceRepository(session).get("inv_1")
        assert p.status == PaymentStatus.FAILED and p.raw_payload["provider"] == "razorpay"
        assert s.status == SubscriptionStatus.HALTED and s.retry_count == 2
        assert i.status == InvoiceStatus.OVERDUE and i.amount_paid == Decimal("1000.00")


def test_merchant_policy_round_trip_preserves_sets_and_times(db):
    _, factory = db
    policy = MerchantPolicy(
        merchant_id="merchant_1", max_contacts_per_case=5, contact_window_days=10,
        max_payment_retries=3, max_recovery_window_days=30,
        human_approval_threshold=Decimal("15000"), allow_partial_payments=False,
        allow_voice_calls=True, quiet_hours_start=time(22, 30), quiet_hours_end=time(7, 15),
        timezone="Asia/Kolkata",
        allowed_channels={CommunicationChannel.EMAIL, CommunicationChannel.WHATSAPP},
        allowed_actions={RecoveryActionType.DELAYED_RETRY, RecoveryActionType.SEND_REMINDER},
    )
    with session_scope(factory) as session:
        MerchantPolicyRepository(session).save(policy)
    with session_scope(factory) as session:
        loaded = MerchantPolicyRepository(session).get("merchant_1")
        assert loaded.max_recovery_window_days == 30
        assert loaded.human_approval_threshold == Decimal("15000.00")
        assert loaded.quiet_hours_start == time(22, 30)
        assert loaded.allowed_channels == {CommunicationChannel.EMAIL, CommunicationChannel.WHATSAPP}
        assert loaded.allowed_actions == {RecoveryActionType.DELAYED_RETRY, RecoveryActionType.SEND_REMINDER}


def test_session_scope_rolls_back_on_exception(db):
    _, factory = db
    with pytest.raises(RuntimeError):
        with session_scope(factory) as session:
            RecoveryCaseRepository(session).save(make_case(id="rc_rollback"))
            raise RuntimeError("boom")

    with session_scope(factory) as session:
        assert RecoveryCaseRepository(session).get("rc_rollback") is None
