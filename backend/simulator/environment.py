from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
)
from decimal import (
    Decimal,
    ROUND_HALF_UP,
)
from enum import Enum
from hashlib import sha256
from math import exp, log, log2

from pydantic import BaseModel, Field

from app.domain.actions import RecoveryAction
from app.domain.customer import Customer
from app.domain.enums import (
    ActionStatus,
    CommunicationChannel,
    FailureClass,
    RecoveryActionType,
    RecoveryCaseStatus,
)
from app.domain.recovery_case import RecoveryCase

from simulator.cases import (
    SyntheticRecoveryScenario,
)
from simulator.merchants import (
    SyntheticMerchant,
)
from simulator.recovery_assumptions import (
    ACTION_RECOVERY_DELAY_HOURS,
    BANK_RECOVERY_LOG_ODDS_EFFECTS,
    BASE_RECOVERY_PROBABILITIES,
    DEFAULT_ACTION_PROBABILITIES,
    PAYMENT_METHOD_RECOVERY_LOG_ODDS_EFFECTS,
    RecoverySensitivity,
    RecoverySimulationConfig,
)


class RecoveryOutcomeType(str, Enum):
    """
    Simulator-only result types.

    They do not belong in app.domain because these are synthetic
    environment outcomes rather than production workflow states.
    """

    FULL_RECOVERY = "full_recovery"

    PARTIAL_RECOVERY = "partial_recovery"

    NO_RECOVERY = "no_recovery"

    ESCALATED = "escalated"

    STOPPED = "stopped"


class SimulationResult(BaseModel):
    """
    Result of one simulated action.

    latent_recovery_probability is simulator ground truth and MUST NOT
    later be included as an ML feature.
    """

    case_id: str

    action_id: str

    action_type: RecoveryActionType

    channel: CommunicationChannel

    outcome: RecoveryOutcomeType

    success: bool

    fully_recovered: bool

    recovered_amount_this_action: Decimal = Decimal("0")

    remaining_amount_after: Decimal = Decimal("0")

    recovered_at: datetime | None = None

    recovery_delay_hours: float | None = Field(
        default=None,
        ge=0,
    )

    latent_recovery_probability: float = Field(
        ge=0,
        le=1,
    )

    random_draw: float = Field(
        ge=0,
        le=1,
    )

    sensitivity: RecoverySensitivity

    case_after: RecoveryCase

    action_after: RecoveryAction

    notes: list[str] = Field(
        default_factory=list
    )


class RecoveryEnvironment:
    """
    Synthetic recovery world.

    This is NOT our ML model.

    It provides hidden synthetic ground truth for:
        P(recovery | case, customer, action)

    Historical dataset generation will later interact with this
    environment and create observable training examples.

    The ML model must learn recovery behavior from those examples.
    """

    TERMINAL_CASE_STATUSES = {
        RecoveryCaseStatus.RECOVERED,
        RecoveryCaseStatus.STOPPED,
    }

    NON_EXECUTABLE_ACTION_STATUSES = {
        ActionStatus.REQUIRES_APPROVAL,
        ActionStatus.BLOCKED,
        ActionStatus.EXECUTED,
        ActionStatus.FAILED,
        ActionStatus.CANCELLED,
    }

    RETRY_ACTIONS = {
        RecoveryActionType.IMMEDIATE_RETRY,
        RecoveryActionType.DELAYED_RETRY,
    }

    def __init__(
        self,
        *,
        seed: int = 42,
        configuration: RecoverySimulationConfig | None = None,
    ) -> None:
        self.seed = seed
        self.configuration = configuration or RecoverySimulationConfig()

    # ========================================================
    # PUBLIC: LATENT PROBABILITY
    # ========================================================

    def recovery_probability(
        self,
        scenario: SyntheticRecoveryScenario,
        merchant: SyntheticMerchant,
        customer: Customer,
        action: RecoveryAction,
        *,
        now: datetime,
    ) -> float:
        """
        Calculate the hidden probability that the supplied action
        produces recovery in the synthetic world.

        This value is useful for simulator validation only.

        It MUST NOT become an ML input feature later.
        """

        self._validate_inputs(
            scenario=scenario,
            merchant=merchant,
            customer=customer,
            action=action,
            now=now,
        )

        failure_class = (
            scenario.expected_failure_class
        )

        action_type = (
            action.action_type
        )

        base_probability = (
            BASE_RECOVERY_PROBABILITIES
            .get(
                failure_class,
                {},
            )
            .get(
                action_type,
                DEFAULT_ACTION_PROBABILITIES[
                    action_type
                ],
            )
        )

        # STOP is deterministic.
        if (
            action_type
            == RecoveryActionType.STOP
        ):
            return 0.0

        log_odds = self._logit(base_probability)

        # The named sensitivity regime is a documented global odds shift.
        log_odds += self.configuration.log_odds_shift

        # ----------------------------------------------------
        # Customer's historical payment reliability
        # ----------------------------------------------------

        payment_success_signal = (
            customer.historical_payment_success_rate
            - 0.75
        )

        log_odds += (
            0.20
            * payment_success_signal
        )

        # ----------------------------------------------------
        # Customer's prior recovery history
        # ----------------------------------------------------

        if (
            customer.previous_recovery_attempts
            > 0
        ):
            recovery_history_rate = (
                customer.previous_recovery_successes
                / customer.previous_recovery_attempts
            )

            log_odds += (
                0.12
                * (
                    recovery_history_rate
                    - 0.50
                )
            )

        # ----------------------------------------------------
        # Channel preference
        # ----------------------------------------------------

        if (
            action.channel
            != CommunicationChannel.NONE
        ):
            if (
                customer.preferred_channel
                == action.channel
            ):
                log_odds += 0.04

            elif (
                customer.preferred_channel
                is not None
                and customer.preferred_channel
                != CommunicationChannel.NONE
            ):
                log_odds -= 0.02

        # ----------------------------------------------------
        # Preferred payment method
        # ----------------------------------------------------

        if (
            scenario.case.payment_method
            is not None
            and customer.preferred_payment_method
            is not None
            and action_type
            != RecoveryActionType.REQUEST_NEW_PAYMENT_METHOD
        ):
            if (
                scenario.case.payment_method
                == customer.preferred_payment_method
            ):
                log_odds += 0.02

        # ----------------------------------------------------
        # Payment method synthetic effect
        # ----------------------------------------------------

        if (
            scenario.case.payment_method
            is not None
        ):
            log_odds += (
                PAYMENT_METHOD_RECOVERY_LOG_ODDS_EFFECTS.get(
                    scenario.case.payment_method,
                    0.0,
                )
            )

        # ----------------------------------------------------
        # Bank synthetic effect
        # ----------------------------------------------------

        if (
            scenario.payment is not None
            and scenario.payment.bank is not None
        ):
            log_odds += (
                BANK_RECOVERY_LOG_ODDS_EFFECTS.get(
                    scenario.payment.bank,
                    0.0,
                )
            )

        # ----------------------------------------------------
        # Amount pressure
        # ----------------------------------------------------

        log_odds += (
            self._amount_log_odds_effect(
                scenario=scenario,
                merchant=merchant,
                action=action,
            )
        )

        # ----------------------------------------------------
        # Case age
        # ----------------------------------------------------

        log_odds += (
            self._age_log_odds_effect(
                scenario=scenario,
                now=now,
            )
        )

        return self._sigmoid(log_odds)

    # ========================================================
    # PUBLIC: SIMULATE ACTION
    # ========================================================

    def simulate_action(
        self,
        scenario: SyntheticRecoveryScenario,
        merchant: SyntheticMerchant,
        customer: Customer,
        action: RecoveryAction,
        *,
        now: datetime,
        rollout_index: int = 0,
    ) -> SimulationResult:
        """
        Simulate execution and eventual result of one recovery action.

        rollout_index lets us generate multiple independent outcomes
        for the same case/action without changing the environment seed.
        """

        if rollout_index < 0:
            raise ValueError(
                "rollout_index must be greater than or equal to zero."
            )

        probability = (
            self.recovery_probability(
                scenario=scenario,
                merchant=merchant,
                customer=customer,
                action=action,
                now=now,
            )
        )

        draw = self._stable_uniform(
            scenario=scenario,
            action=action,
            rollout_index=rollout_index,
            stream="recovery",
        )

        # ----------------------------------------------------
        # STOP is deterministic
        # ----------------------------------------------------

        if (
            action.action_type
            == RecoveryActionType.STOP
        ):
            return self._build_stop_result(
                scenario=scenario,
                action=action,
                now=now,
                probability=probability,
                draw=draw,
            )

        recovered = (
            draw < probability
        )

        if not recovered:
            return self._build_no_recovery_result(
                scenario=scenario,
                action=action,
                now=now,
                probability=probability,
                draw=draw,
            )

        return self._build_recovery_result(
            scenario=scenario,
            action=action,
            now=now,
            probability=probability,
            draw=draw,
            rollout_index=rollout_index,
        )

    # ========================================================
    # AMOUNT ADJUSTMENT
    # ========================================================

    @staticmethod
    def _amount_log_odds_effect(
        scenario: SyntheticRecoveryScenario,
        merchant: SyntheticMerchant,
        action: RecoveryAction,
    ) -> float:
        """
        Synthetic log-odds effect for amount relative to merchant average.

        Partial payment and human escalation are less sensitive to
        amount because they exist partly to handle difficult/high-value
        cases.
        """

        average = (
            merchant.average_order_value
        )

        if average <= 0:
            return 0.0

        ratio = float(
            scenario.case.amount_at_risk
            / average
        )

        if ratio <= 0:
            return 0.0

        if ratio < 0.75:
            return 0.03

        if ratio <= 1.0:
            return 0.0

        penalty = min(
            0.16,
            0.05 * log2(
                max(
                    ratio,
                    1.0,
                )
            ),
        )

        if (
            action.action_type
            == RecoveryActionType.OFFER_PARTIAL_PAYMENT
        ):
            penalty *= 0.40

        elif (
            action.action_type
            == RecoveryActionType.ESCALATE_TO_HUMAN
        ):
            penalty *= 0.50

        elif (
            action.action_type
            == RecoveryActionType.REQUEST_PROMISE_TO_PAY
        ):
            penalty *= 0.70

        return -penalty

    # ========================================================
    # AGE ADJUSTMENT
    # ========================================================

    @staticmethod
    def _age_log_odds_effect(
        scenario: SyntheticRecoveryScenario,
        now: datetime,
    ) -> float:
        age_seconds = (
            now
            - scenario.case.created_at
        ).total_seconds()

        age_days = max(
            0.0,
            age_seconds / 86400.0,
        )

        if (
            scenario.expected_failure_class
            == FailureClass.OVERDUE_RECEIVABLE
        ):
            penalty = min(
                0.12,
                age_days * 0.003,
            )

        else:
            penalty = min(
                0.12,
                age_days * 0.015,
            )

        return -penalty

    # ========================================================
    # SUCCESS RESULT
    # ========================================================

    def _build_recovery_result(
        self,
        *,
        scenario: SyntheticRecoveryScenario,
        action: RecoveryAction,
        now: datetime,
        probability: float,
        draw: float,
        rollout_index: int,
    ) -> SimulationResult:

        remaining_before = (
            self._remaining_amount(
                scenario.case
            )
        )

        delay_hours = (
            self._recovery_delay_hours(
                scenario=scenario,
                action=action,
                rollout_index=rollout_index,
            )
        )

        recovered_at = (
            now
            + timedelta(
                hours=delay_hours
            )
        )

        # ----------------------------------------------------
        # Partial payment
        # ----------------------------------------------------

        if (
            action.action_type
            == RecoveryActionType.OFFER_PARTIAL_PAYMENT
        ):
            recovered_amount = (
                self._partial_recovery_amount(
                    scenario=scenario,
                    action=action,
                    remaining_before=remaining_before,
                    rollout_index=rollout_index,
                )
            )

        else:
            recovered_amount = (
                remaining_before
            )

        total_recovered = (
            scenario.case.recovered_amount
            + recovered_amount
        )

        total_recovered = min(
            total_recovered,
            scenario.case.amount_at_risk,
        )

        remaining_after = max(
            Decimal("0"),
            (
                scenario.case.amount_at_risk
                - total_recovered
            ),
        )

        fully_recovered = (
            remaining_after
            <= Decimal("0")
        )

        if fully_recovered:
            outcome = (
                RecoveryOutcomeType.FULL_RECOVERY
            )

            case_status = (
                RecoveryCaseStatus.RECOVERED
            )

            closed_at = (
                recovered_at
            )

        else:
            outcome = (
                RecoveryOutcomeType.PARTIAL_RECOVERY
            )

            case_status = (
                RecoveryCaseStatus.WAITING_CUSTOMER
            )

            closed_at = None

        case_after = (
            self._updated_case(
                case=scenario.case,
                action=action,
                status=case_status,
                recovered_amount=total_recovered,
                updated_at=recovered_at,
                closed_at=closed_at,
            )
        )

        action_after = (
            self._executed_action(
                action=action,
                now=now,
            )
        )

        return SimulationResult(
            case_id=scenario.case.id,

            action_id=action.id,

            action_type=(
                action.action_type
            ),

            channel=action.channel,

            outcome=outcome,

            success=True,

            fully_recovered=(
                fully_recovered
            ),

            recovered_amount_this_action=(
                recovered_amount
            ),

            remaining_amount_after=(
                remaining_after
            ),

            recovered_at=recovered_at,

            recovery_delay_hours=(
                delay_hours
            ),

            latent_recovery_probability=(
                probability
            ),

            random_draw=draw,

            sensitivity=self.configuration.sensitivity,

            case_after=case_after,

            action_after=action_after,

            notes=[
                (
                    "Synthetic environment generated "
                    "a successful recovery outcome."
                )
            ],
        )

    # ========================================================
    # NO-RECOVERY RESULT
    # ========================================================

    def _build_no_recovery_result(
        self,
        *,
        scenario: SyntheticRecoveryScenario,
        action: RecoveryAction,
        now: datetime,
        probability: float,
        draw: float,
    ) -> SimulationResult:

        remaining = (
            self._remaining_amount(
                scenario.case
            )
        )

        # Human escalation changes ownership even if it has not
        # yet recovered money.
        if (
            action.action_type
            == RecoveryActionType.ESCALATE_TO_HUMAN
        ):
            outcome = (
                RecoveryOutcomeType.ESCALATED
            )

            case_status = (
                RecoveryCaseStatus.ESCALATED
            )

        elif (
            action.channel
            != CommunicationChannel.NONE
            or action.action_type
            == RecoveryActionType.WAIT
        ):
            outcome = (
                RecoveryOutcomeType.NO_RECOVERY
            )

            case_status = (
                RecoveryCaseStatus.WAITING_CUSTOMER
            )

        else:
            outcome = (
                RecoveryOutcomeType.NO_RECOVERY
            )

            case_status = (
                RecoveryCaseStatus.ACTION_EXECUTED
            )

        case_after = (
            self._updated_case(
                case=scenario.case,
                action=action,
                status=case_status,
                recovered_amount=(
                    scenario.case.recovered_amount
                ),
                updated_at=now,
                closed_at=(
                    scenario.case.closed_at
                ),
            )
        )

        action_after = (
            self._executed_action(
                action=action,
                now=now,
            )
        )

        return SimulationResult(
            case_id=scenario.case.id,

            action_id=action.id,

            action_type=(
                action.action_type
            ),

            channel=action.channel,

            outcome=outcome,

            success=False,

            fully_recovered=False,

            recovered_amount_this_action=(
                Decimal("0")
            ),

            remaining_amount_after=(
                remaining
            ),

            recovered_at=None,

            recovery_delay_hours=None,

            latent_recovery_probability=(
                probability
            ),

            random_draw=draw,

            sensitivity=self.configuration.sensitivity,

            case_after=case_after,

            action_after=action_after,

            notes=[
                (
                    "Synthetic environment generated "
                    "no monetary recovery for this action."
                )
            ],
        )

    # ========================================================
    # STOP RESULT
    # ========================================================

    def _build_stop_result(
        self,
        *,
        scenario: SyntheticRecoveryScenario,
        action: RecoveryAction,
        now: datetime,
        probability: float,
        draw: float,
    ) -> SimulationResult:

        remaining = (
            self._remaining_amount(
                scenario.case
            )
        )

        case_after = (
            self._updated_case(
                case=scenario.case,
                action=action,
                status=(
                    RecoveryCaseStatus.STOPPED
                ),
                recovered_amount=(
                    scenario.case.recovered_amount
                ),
                updated_at=now,
                closed_at=now,
            )
        )

        action_after = (
            self._executed_action(
                action=action,
                now=now,
            )
        )

        return SimulationResult(
            case_id=scenario.case.id,

            action_id=action.id,

            action_type=(
                action.action_type
            ),

            channel=action.channel,

            outcome=(
                RecoveryOutcomeType.STOPPED
            ),

            success=False,

            fully_recovered=False,

            recovered_amount_this_action=(
                Decimal("0")
            ),

            remaining_amount_after=(
                remaining
            ),

            recovered_at=None,

            recovery_delay_hours=None,

            latent_recovery_probability=(
                probability
            ),

            random_draw=draw,

            sensitivity=self.configuration.sensitivity,

            case_after=case_after,

            action_after=action_after,

            notes=[
                (
                    "Recovery was intentionally stopped."
                )
            ],
        )

    # ========================================================
    # PARTIAL RECOVERY
    # ========================================================

    def _partial_recovery_amount(
        self,
        *,
        scenario: SyntheticRecoveryScenario,
        action: RecoveryAction,
        remaining_before: Decimal,
        rollout_index: int,
    ) -> Decimal:
        """
        If a specific action.amount below the outstanding balance is
        supplied later, treat that as the requested partial amount.

        Current candidate generation uses the full at-risk amount, so
        the simulator otherwise samples a realistic partial fraction.
        """

        if (
            action.amount is not None
            and action.amount > 0
            and action.amount < remaining_before
        ):
            return min(
                action.amount,
                remaining_before,
            ).quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            )

        fraction_draw = (
            self._stable_uniform(
                scenario=scenario,
                action=action,
                rollout_index=rollout_index,
                stream="partial_amount",
            )
        )

        fraction = (
            0.30
            + (
                0.40
                * fraction_draw
            )
        )

        amount = (
            remaining_before
            * Decimal(
                str(fraction)
            )
        ).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )

        amount = max(
            Decimal("0.01"),
            amount,
        )

        return min(
            amount,
            remaining_before,
        )

    # ========================================================
    # RECOVERY DELAY
    # ========================================================

    def _recovery_delay_hours(
        self,
        *,
        scenario: SyntheticRecoveryScenario,
        action: RecoveryAction,
        rollout_index: int,
    ) -> float:

        minimum, maximum = (
            ACTION_RECOVERY_DELAY_HOURS[
                action.action_type
            ]
        )

        if maximum <= minimum:
            return minimum

        draw = (
            self._stable_uniform(
                scenario=scenario,
                action=action,
                rollout_index=rollout_index,
                stream="recovery_delay",
            )
        )

        return (
            minimum
            + (
                maximum
                - minimum
            )
            * draw
        )

    # ========================================================
    # CASE UPDATE
    # ========================================================

    def _updated_case(
        self,
        *,
        case: RecoveryCase,
        action: RecoveryAction,
        status: RecoveryCaseStatus,
        recovered_amount: Decimal,
        updated_at: datetime,
        closed_at: datetime | None,
    ) -> RecoveryCase:

        recovery_retry_count = (
            case.recovery_retry_count
        )

        if (
            action.action_type
            in self.RETRY_ACTIONS
        ):
            recovery_retry_count += 1

        previous_contacts = (
            case.previous_contacts
        )

        if (
            action.channel
            != CommunicationChannel.NONE
        ):
            previous_contacts += 1

        return case.model_copy(
            update={
                "status": status,

                "recovery_retry_count": (
                    recovery_retry_count
                ),

                "previous_contacts": (
                    previous_contacts
                ),

                "recovered_amount": (
                    recovered_amount
                ),

                "selected_action_id": (
                    action.id
                ),

                "updated_at": (
                    updated_at
                ),

                "closed_at": (
                    closed_at
                ),
            }
        )

    # ========================================================
    # ACTION UPDATE
    # ========================================================

    @staticmethod
    def _executed_action(
        *,
        action: RecoveryAction,
        now: datetime,
    ) -> RecoveryAction:

        return action.model_copy(
            update={
                "status": (
                    ActionStatus.EXECUTED
                ),

                "executed_at": now,
            }
        )

    # ========================================================
    # REMAINING AMOUNT
    # ========================================================

    @staticmethod
    def _remaining_amount(
        case: RecoveryCase,
    ) -> Decimal:

        return max(
            Decimal("0"),
            (
                case.amount_at_risk
                - case.recovered_amount
            ),
        )

    # ========================================================
    # DETERMINISTIC RANDOMNESS
    # ========================================================

    def _stable_uniform(
        self,
        *,
        scenario: SyntheticRecoveryScenario,
        action: RecoveryAction,
        rollout_index: int,
        stream: str,
    ) -> float:
        """
        Deterministic pseudo-random number in [0, 1).

        Importantly, action.id is NOT used because RecoveryAction IDs
        currently use UUID4 and would make re-generated candidate sets
        produce different simulation results.

        We use the stable synthetic case ID + action characteristics.
        """

        raw = (
            f"{self.seed}|"
            f"{scenario.case.id}|"
            f"{action.action_type.value}|"
            f"{action.channel.value}|"
            f"{rollout_index}|"
            f"{stream}"
        ).encode(
            "utf-8"
        )

        digest = sha256(
            raw
        ).digest()

        integer = int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )

        return (
            integer
            / float(
                2 ** 64
            )
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_inputs(
        self,
        *,
        scenario: SyntheticRecoveryScenario,
        merchant: SyntheticMerchant,
        customer: Customer,
        action: RecoveryAction,
        now: datetime,
    ) -> None:

        case = scenario.case

        if now.tzinfo is None:
            raise ValueError(
                "now must be timezone-aware."
            )

        if case.created_at.tzinfo is None:
            raise ValueError(
                (
                    "RecoveryCase.created_at must "
                    "be timezone-aware."
                )
            )

        if now < case.created_at:
            raise ValueError(
                (
                    "Environment cannot execute an action "
                    "before the RecoveryCase was created."
                )
            )

        if (
            case.merchant_id
            != merchant.id
        ):
            raise ValueError(
                (
                    "RecoveryCase does not belong to "
                    "the supplied merchant."
                )
            )

        if (
            merchant.policy.merchant_id
            != merchant.id
        ):
            raise ValueError(
                (
                    "MerchantPolicy does not belong to "
                    "the supplied merchant."
                )
            )

        if (
            case.customer_id
            != customer.id
        ):
            raise ValueError(
                (
                    "RecoveryCase does not belong to "
                    "the supplied customer."
                )
            )

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
            action.case_id
            != case.id
        ):
            raise ValueError(
                (
                    "RecoveryAction does not belong "
                    "to the supplied RecoveryCase."
                )
            )

        if (
            case.status
            in self.TERMINAL_CASE_STATUSES
        ):
            raise ValueError(
                (
                    "Cannot simulate an action for "
                    "a terminal RecoveryCase."
                )
            )

        if (
            action.status
            in self.NON_EXECUTABLE_ACTION_STATUSES
        ):
            raise ValueError(
                (
                    "RecoveryAction is not executable "
                    f"from status {action.status.value}."
                )
            )

        if (
            self._remaining_amount(
                case
            )
            <= Decimal("0")
        ):
            raise ValueError(
                (
                    "RecoveryCase has no outstanding "
                    "amount remaining."
                )
            )

    # ========================================================
    # NUMERICALLY STABLE LOG-ODDS HELPERS
    # ========================================================

    @staticmethod
    def _logit(
        probability: float,
    ) -> float:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Base recovery probability must be in [0, 1].")

        # Finite bounds make exact/near-exact edge assumptions safe while
        # retaining their intended ordering under additional log-odds effects.
        epsilon = 1e-12
        bounded = min(1.0 - epsilon, max(epsilon, probability))
        return log(bounded / (1.0 - bounded))

    @staticmethod
    def _sigmoid(log_odds: float) -> float:
        if log_odds >= 0.0:
            return 1.0 / (1.0 + exp(-log_odds))

        exponent = exp(log_odds)
        return exponent / (1.0 + exponent)
