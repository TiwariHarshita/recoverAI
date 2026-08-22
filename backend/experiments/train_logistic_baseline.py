from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ml.dataset import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    HIDDEN_SIMULATOR_COLUMNS,
    IDENTIFIER_COLUMNS,
    LOGGING_ONLY_COLUMNS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    POST_OUTCOME_COLUMNS,
    TARGET_COLUMN,
    load_historical_csv,
)
from app.ml.logistic_baseline import (
    DEFAULT_THRESHOLD,
    MODEL_KIND,
    MODEL_FORMAT_VERSION,
    train_logistic_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the RecoverAI logistic-regression recovery baseline."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/synthetic/recovery_history.csv"
        ),
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path(
            "artifacts/models/logistic_baseline.joblib"
        ),
    )

    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path(
            "artifacts/models/logistic_baseline.metrics.json"
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    return parser.parse_args()


def _top_coefficients(
    model,
    *,
    limit: int = 15,
) -> dict[str, list[dict[str, float | str]]]:
    table = model.coefficient_table()

    positive = (
        table.sort_values(
            "coefficient",
            ascending=False,
        )
        .head(limit)
    )

    negative = (
        table.sort_values(
            "coefficient",
            ascending=True,
        )
        .head(limit)
    )

    def rows(frame):
        return [
            {
                "feature": str(
                    row.feature
                ),

                "coefficient": float(
                    row.coefficient
                ),
            }

            for row in frame.itertuples()
        ]

    return {
        "positive": rows(
            positive
        ),

        "negative": rows(
            negative
        ),
    }


def main() -> None:
    args = parse_args()

    dataframe = load_historical_csv(
        args.data
    )

    result = train_logistic_baseline(
        dataframe,
        test_size=args.test_size,
        random_state=args.seed,
        threshold=args.threshold,
    )

    model_path = result.model.save(
        args.model_output
    )

    metrics_path = (
        args.metrics_output
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model_kind": MODEL_KIND,

        "model_format_version": (
            MODEL_FORMAT_VERSION
        ),

        "data_path": str(
            args.data
        ),

        "rows": int(
            len(dataframe)
        ),

        "train_rows": int(
            result.train_rows
        ),

        "test_rows": int(
            result.test_rows
        ),

        "train_positive_rate": float(
            result.train_positive_rate
        ),

        "test_positive_rate": float(
            result.test_positive_rate
        ),

        "random_state": int(
            args.seed
        ),

        "test_size": float(
            args.test_size
        ),

        "target_column": (
            TARGET_COLUMN
        ),

        "feature_count_before_encoding": len(
            MODEL_FEATURES
        ),

        "categorical_features": list(
            CATEGORICAL_FEATURES
        ),

        "numeric_features": list(
            NUMERIC_FEATURES
        ),

        "boolean_features": list(
            BOOLEAN_FEATURES
        ),

        "excluded_identifiers": list(
            IDENTIFIER_COLUMNS
        ),

        "excluded_post_outcome": list(
            POST_OUTCOME_COLUMNS
        ),

        "excluded_logging_only": list(
            LOGGING_ONLY_COLUMNS
        ),

        "excluded_hidden_simulator": list(
            HIDDEN_SIMULATOR_COLUMNS
        ),

        "metrics": (
            result.metrics
        ),

        "top_coefficients": (
            _top_coefficients(
                result.model
            )
        ),
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
        )

        handle.write(
            "\n"
        )

    print(
        f"Model: {model_path}"
    )

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        "Train/Test: "
        f"{result.train_rows}/"
        f"{result.test_rows}"
    )

    print(
        "Test recovery rate: "
        f"{result.test_positive_rate:.4f}"
    )

    print(
        "ROC-AUC: "
        f"{result.metrics['roc_auc']:.4f}"
    )

    print(
        "Log loss: "
        f"{result.metrics['log_loss']:.4f}"
    )

    print(
        "Brier score: "
        f"{result.metrics['brier_score']:.4f}"
    )

    print(
        "Accuracy: "
        f"{result.metrics['accuracy']:.4f}"
    )

    print(
        "F1: "
        f"{result.metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()