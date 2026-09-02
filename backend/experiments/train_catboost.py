from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ml.catboost_model import (
    DEFAULT_THRESHOLD,
    MODEL_FORMAT_VERSION,
    MODEL_KIND,
    train_catboost_model,
)

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
    GroupingStrategy,
    load_historical_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the RecoverAI CatBoost "
            "recovery-probability model."
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
            "artifacts/models/catboost_recovery.cbm"
        ),
    )

    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path(
            "artifacts/models/catboost_recovery.metrics.json"
        ),
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--grouping-strategy",
        choices=[strategy.value for strategy in GroupingStrategy],
        default=GroupingStrategy.CUSTOMER.value,
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

    parser.add_argument(
        "--iterations",
        type=int,
        default=890,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--l2-leaf-reg",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--one-hot-max-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--random-strength",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--boosting-type",
        choices=[
            "Ordered",
            "Plain",
        ],
        default="Ordered",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )

    return parser.parse_args()


def _top_feature_importance(
    model,
    *,
    limit: int = 20,
) -> list[
    dict[
        str,
        float | str,
    ]
]:
    table = (
        model
        .feature_importance_table()
        .head(limit)
    )

    return [
        {
            "feature": str(
                row.feature
            ),

            "importance": float(
                row.importance
            ),
        }

        for row in table.itertuples()
    ]


def main() -> None:
    args = parse_args()

    dataframe = load_historical_csv(
        args.data
    )

    result = train_catboost_model(
        dataframe,

        validation_size=args.validation_size,

        test_size=(
            args.test_size
        ),

        random_state=(
            args.seed
        ),

        grouping_strategy=args.grouping_strategy,

        data_generation_reference=str(args.data),

        threshold=(
            args.threshold
        ),

        iterations=(
            args.iterations
        ),

        depth=(
            args.depth
        ),

        learning_rate=(
            args.learning_rate
        ),

        l2_leaf_reg=(
            args.l2_leaf_reg
        ),

        one_hot_max_size=(
            args.one_hot_max_size
        ),

        random_strength=(
            args.random_strength
        ),

        boosting_type=(
            args.boosting_type
        ),

        thread_count=(
            args.threads
        ),
    )

    model_path = (
        result.model.save(
            args.model_output
        )
    )

    metrics_path = (
        args.metrics_output
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model_kind": (
            MODEL_KIND
        ),

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

        "validation_rows": int(result.validation_rows),

        "test_rows": int(
            result.test_rows
        ),

        "train_positive_rate": float(
            result.train_positive_rate
        ),

        "validation_positive_rate": float(result.validation_positive_rate),

        "test_positive_rate": float(
            result.test_positive_rate
        ),

        "random_state": int(
            args.seed
        ),

        "test_size": float(
            args.test_size
        ),

        "validation_size": float(args.validation_size),

        "grouping_strategy": args.grouping_strategy,

        "split_metadata": result.split_metadata,

        "target_column": (
            TARGET_COLUMN
        ),

        "feature_count": len(
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

        "hyperparameters": {
            "iterations": int(
                args.iterations
            ),

            "depth": int(
                args.depth
            ),

            "learning_rate": float(
                args.learning_rate
            ),

            "l2_leaf_reg": float(
                args.l2_leaf_reg
            ),

            "one_hot_max_size": int(
                args.one_hot_max_size
            ),

            "random_strength": float(
                args.random_strength
            ),

            "boosting_type": str(
                args.boosting_type
            ),

            "threads": int(
                args.threads
            ),
        },

        "calibration_method": result.model.training_metadata["calibration_method"],

        "raw_metrics": result.raw_metrics,

        "calibrated_metrics": result.calibrated_metrics,

        "metrics": result.calibrated_metrics,

        "top_feature_importance": (
            _top_feature_importance(
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
        "Model metadata: "
        f"{model_path.with_suffix('.meta.json')}"
    )

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        "Train/Validation/Test: "
        f"{result.train_rows}/"
        f"{result.validation_rows}/"
        f"{result.test_rows}"
    )

    print(
        "Test recovery rate: "
        f"{result.test_positive_rate:.4f}"
    )

    print(
        "Raw/Calibrated ROC-AUC: "
        f"{result.raw_metrics['roc_auc']:.4f}/"
        f"{result.calibrated_metrics['roc_auc']:.4f}"
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
