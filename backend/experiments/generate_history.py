from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from simulator.customers import (
    generate_synthetic_population,
)
from simulator.historical import (
    generate_historical_dataset,
    write_historical_dataset_csv,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate the RecoverAI synthetic "
            "historical training dataset."
        )
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
        default=10_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/synthetic/"
            "recovery_history.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = (
        parse_args()
    )

    # Separate deterministic random streams.
    population_seed = (
        args.seed
    )

    case_seed = (
        args.seed
        + 1
    )

    environment_seed = (
        args.seed
        + 2
    )

    # ========================================================
    # SYNTHETIC POPULATION
    # ========================================================

    population = (
        generate_synthetic_population(
            merchant_count=(
                args.merchants
            ),

            customers_per_merchant=(
                args
                .customers_per_merchant
            ),

            seed=(
                population_seed
            ),
        )
    )

    # ========================================================
    # HISTORICAL CASES + EXECUTED ACTIONS
    # ========================================================

    dataset = (
        generate_historical_dataset(
            population,
            args.cases,

            case_seed=(
                case_seed
            ),

            environment_seed=(
                environment_seed
            ),
        )
    )

    # ========================================================
    # WRITE CSV
    # ========================================================

    output_path = (
        write_historical_dataset_csv(
            dataset,
            args.output,
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    action_counts = Counter(
        record.action_type.value
        for record
        in dataset.records
    )

    recovered_count = sum(
        record.recovered
        for record
        in dataset.records
    )

    recovery_rate = (
        recovered_count
        / dataset.record_count
        if (
            dataset.record_count
            > 0
        )
        else 0.0
    )

    print(
        f"CSV: {output_path}"
    )

    print(
        "Metadata: "
        f"{output_path.with_suffix('.meta.json')}"
    )

    print(
        "Requested cases: "
        f"{dataset.requested_case_count}"
    )

    print(
        "Written records: "
        f"{dataset.record_count}"
    )

    print(
        "Skipped cases: "
        f"{dataset.skipped_case_count}"
    )

    print(
        "Recovery rate: "
        f"{recovery_rate:.4f}"
    )

    print(
        "Action counts:"
    )

    for (
        action_type,
        count,
    ) in sorted(
        action_counts.items()
    ):

        print(
            f"  {action_type}: {count}"
        )


if __name__ == "__main__":
    main()