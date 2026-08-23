from __future__ import annotations

from app.domain.action_scoring import (
    MerchantScoringProfile,
    RecoverySourceContext,
)
from app.ml.catboost_model import (
    CatBoostRecoveryModel,
)
from app.policy.models import (
    PolicyContext,
)
from app.services.action_selector import (
    select_best_recovery_action,
)
from app.services.candidate_actions import (
    generate_candidate_actions,
)
from app.services.diagnosis import (
    diagnose_case,
)
from simulator.cases import (
    generate_recovery_cases,
)
from simulator.customers import (
    generate_synthetic_population,
)


MODEL_PATH = (
    "artifacts/models/"
    "catboost_recovery.cbm"
)


def main() -> None:

    population = (
        generate_synthetic_population(

            merchant_count=4,

            customers_per_merchant=25,

            seed=1200,
        )
    )

    batch = (
        generate_recovery_cases(

            population,

            25,

            seed=1201,

            reference_time=(
                population.reference_time
            ),
        )
    )

    merchants = {
        merchant.id: merchant
        for merchant
        in population.merchants
    }

    customers = {
        customer.id: customer
        for customer
        in population.customers
    }

    scenario = next(

        item

        for item
        in batch.scenarios

        if (
            item.case.customer_id
            is not None
        )
    )

    case = (
        scenario.case
    )

    merchant = (
        merchants[
            case.merchant_id
        ]
    )

    customer = (
        customers[
            case.customer_id
        ]
    )

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    diagnosis = (
        diagnose_case(
            case
        )
    )

    # ========================================================
    # CANDIDATE ACTIONS
    # ========================================================

    candidates = (
        generate_candidate_actions(
            case,
            diagnosis,
        )
    )

    # ========================================================
    # LIVE MODEL CONTEXT
    # ========================================================

    merchant_profile = (
        MerchantScoringProfile(

            merchant_id=(
                merchant.id
            ),

            archetype=(
                merchant
                .archetype
                .value
            ),

            average_order_value=(
                merchant
                .average_order_value
            ),
        )
    )

    source_context = (
        RecoverySourceContext(

            bank=(
                scenario.payment.bank
                if (
                    scenario.payment
                    is not None
                )
                else None
            ),

            payment_attempt_number=(
                scenario.payment
                .attempt_number
                if (
                    scenario.payment
                    is not None
                )
                else None
            ),

            subscription_retry_count=(
                scenario.subscription
                .retry_count
                if (
                    scenario.subscription
                    is not None
                )
                else None
            ),

            mandate_active=(
                scenario.subscription
                .mandate_active
                if (
                    scenario.subscription
                    is not None
                )
                else None
            ),

            invoice_days_overdue=(
                scenario.invoice
                .days_overdue
                if (
                    scenario.invoice
                    is not None
                )
                else None
            ),
        )
    )

    policy_context = (
        PolicyContext(

            now=(
                population.reference_time
            ),

            customer_do_not_contact=(
                customer.do_not_contact
            ),

            action_history=[],
        )
    )

    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    model = (
        CatBoostRecoveryModel.load(
            MODEL_PATH
        )
    )

    # ========================================================
    # SCORE + SELECT
    # ========================================================

    result = (
        select_best_recovery_action(

            recovery_case=(
                case
            ),

            customer=(
                customer
            ),

            diagnosis=(
                diagnosis
            ),

            candidate_actions=(
                candidates.actions
            ),

            merchant=(
                merchant_profile
            ),

            merchant_policy=(
                merchant.policy
            ),

            policy_context=(
                policy_context
            ),

            source_context=(
                source_context
            ),

            probability_model=(
                model
            ),
        )
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        f"Case: {case.id}"
    )

    print(
        "Failure: "
        f"{diagnosis.failure_class.value}"
    )

    print(
        "Amount at risk: "
        f"{case.amount_at_risk} "
        f"{case.currency}"
    )

    print(
        f"Outcome: {result.outcome.value}"
    )

    print()

    print(
        "Ranked actions:"
    )

    for score in (
        result.scored_actions
    ):

        print(
            f"  {score.rank}. "
            f"{score.action_type.value:<28} "
            f"p="
            f"{score.predicted_recovery_probability:.6f} "
            f"ERV="
            f"{score.expected_recovery_value} "
            f"policy="
            f"{score.policy_decision_at_selection.value}"
        )

    if (
        result.excluded_actions
    ):

        print()

        print(
            "Policy-excluded actions:"
        )

        for excluded in (
            result.excluded_actions
        ):

            print(
                f"  - "
                f"{excluded.action_type.value}: "
                f"{excluded.policy_reason}"
            )

    print()

    print(
        result.explanation
    )


if __name__ == "__main__":
    main()