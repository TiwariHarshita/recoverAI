from __future__ import annotations

import argparse
from pathlib import Path

from app.ml.catboost_model import CatBoostRecoveryModel
from app.ml.logistic_baseline import LogisticRecoveryBaseline
from simulator.cases import generate_recovery_cases
from simulator.customers import generate_synthetic_population
from simulator.evaluation import (
    EvaluationPolicy,
    evaluate_policy_batch,
    write_batch_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RecoverAI CatBoost+ERV against logistic+ERV "
            "and a deterministic rules-first baseline."
        )
    )

    parser.add_argument(
        "--catboost-model",
        type=Path,
        default=Path(
            "artifacts/models/catboost_recovery.cbm"
        ),
    )

    parser.add_argument(
        "--logistic-model",
        type=Path,
        default=Path(
            "artifacts/models/logistic_baseline.joblib"
        ),
    )

    parser.add_argument(
        "--merchants",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--customers-per-merchant",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--cases",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--rollouts",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2000,
        help=(
            "Base seed for fresh evaluation data. "
            "Population, cases and environment use separate streams."
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "artifacts/evaluation/batch_evaluation.json"
        ),
    )

    parser.add_argument(
        "--cases-output",
        type=Path,
        default=Path(
            "artifacts/evaluation/batch_evaluation_cases.csv"
        ),
    )

    return parser.parse_args()


def _money(value) -> str:
    return f"{value:.2f}"


def main() -> None:
    args = parse_args()

    if args.merchants <= 0:
        raise ValueError("--merchants must be greater than zero.")

    if args.customers_per_merchant <= 0:
        raise ValueError(
            "--customers-per-merchant must be greater than zero."
        )

    if args.cases <= 0:
        raise ValueError("--cases must be greater than zero.")

    if args.rollouts <= 0:
        raise ValueError("--rollouts must be greater than zero.")

    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be greater than zero.")

    population_seed = args.seed
    case_seed = args.seed + 1
    environment_seed = args.seed + 2

    print("Loading trained models...")

    catboost_model = CatBoostRecoveryModel.load(
        args.catboost_model
    )

    logistic_model = LogisticRecoveryBaseline.load(
        args.logistic_model
    )

    print("Generating fresh evaluation population/cases...")

    population = generate_synthetic_population(
        merchant_count=args.merchants,
        customers_per_merchant=args.customers_per_merchant,
        seed=population_seed,
    )

    case_batch = generate_recovery_cases(
        population,
        args.cases,
        seed=case_seed,
        reference_time=population.reference_time,
    )

    print("Evaluating policies...")

    report = evaluate_policy_batch(
        population=population,
        case_batch=case_batch,
        catboost_model=catboost_model,
        logistic_model=logistic_model,
        environment_seed=environment_seed,
        rollouts_per_case=args.rollouts,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=environment_seed + 1,
    )

    summary_path, cases_path = write_batch_evaluation(
        report,
        summary_path=args.summary_output,
        cases_path=args.cases_output,
    )

    print()
    print("Batch evaluation complete")
    print(f"Summary: {summary_path}")
    print(f"Case records: {cases_path}")
    print(f"Fresh cases: {report.evaluated_case_count}")
    print(f"Rollouts per selected action: {report.rollouts_per_case}")
    print()

    for policy in EvaluationPolicy:
        summary = report.summaries[policy.value]

        print(policy.value)
        print(
            "  expected recovered amount: "
            f"{_money(summary.expected_recovered_amount)}"
        )
        print(
            "  recovered amount rate:     "
            f"{summary.recovered_amount_rate:.4f}"
        )
        print(
            "  mean recovery success:     "
            f"{summary.mean_recovery_success_rate:.4f}"
        )
        print(
            "  expected net value:        "
            f"{_money(summary.expected_net_recovery_value)}"
        )
        print(
            "  selected latent p:         "
            f"{summary.mean_selected_latent_probability:.4f}"
        )
        print(
            "  execute/schedule/approval: "
            f"{summary.execute_rate:.3f}/"
            f"{summary.schedule_rate:.3f}/"
            f"{summary.approval_rate:.3f}"
        )
        print()

    for name, comparison in report.comparisons.items():
        print(name)
        print(
            "  recovered amount uplift:   "
            f"{_money(comparison.recovered_amount_uplift)} "
            f"({comparison.relative_recovered_amount_uplift_pct:+.2f}%)"
        )
        print(
            "  net value uplift:          "
            f"{_money(comparison.net_recovery_value_uplift)} "
            f"({comparison.relative_net_value_uplift_pct:+.2f}%)"
        )
        print(
            "  recovery-rate delta:       "
            f"{comparison.recovered_amount_rate_delta_pp:+.2f} pp"
        )
        print(
            "  action disagreement:       "
            f"{comparison.action_disagreement_rate:.3f}"
        )
        print(
            "  mean net uplift/case:      "
            f"{_money(comparison.mean_net_value_uplift_per_case)} "
            f"[95% CI "
            f"{_money(comparison.net_value_uplift_ci95_low_per_case)}, "
            f"{_money(comparison.net_value_uplift_ci95_high_per_case)}]"
        )
        print(
            "  signal:                    "
            f"{comparison.comparison_signal.value}"
        )
        print(
            "  case wins/losses/ties:     "
            f"{comparison.primary_case_wins}/"
            f"{comparison.baseline_case_wins}/"
            f"{comparison.case_ties}"
        )
        print()


if __name__ == "__main__":
    main()
