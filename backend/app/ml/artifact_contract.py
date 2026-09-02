from __future__ import annotations

import math
import re
from typing import Any

from app.ml.calibration import CALIBRATION_METHOD, PlattProbabilityCalibrator
from app.ml.dataset import GROUPING_COLUMNS, GroupingStrategy


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PARTITIONS = ("train", "validation", "test")
OVERLAP_PAIRS = ("train_validation", "train_test", "validation_test")
REQUIRED_METRICS = (
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
    "calibration_mean_absolute_error",
    "calibration_max_error",
)


class ModelArtifactValidationError(ValueError):
    """A model artifact is unsafe or incompatible with canonical inference."""


def _require_dictionary(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelArtifactValidationError(f"{name} must be a dictionary.")
    return value


def _require_nonempty_string(
    metadata: dict[str, Any],
    key: str,
    *,
    label: str | None = None,
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelArtifactValidationError(
            f"Missing or invalid {label or key.replace('_', ' ')} metadata."
        )
    return value


def _require_integer(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelArtifactValidationError(
            f"Missing or invalid {key.replace('_', ' ')} metadata."
        )
    return value


def _validate_partition_mapping(
    training_metadata: dict[str, Any],
    key: str,
    *,
    value_kind: str,
) -> dict[str, Any]:
    values = _require_dictionary(training_metadata.get(key), name=f"{key} metadata")
    if any(partition not in values for partition in PARTITIONS):
        raise ModelArtifactValidationError(f"Incomplete {key} split metadata.")

    for partition in PARTITIONS:
        value = values[partition]
        if value_kind == "positive_integer":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelArtifactValidationError(f"Invalid {key} split metadata.")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ModelArtifactValidationError(f"Invalid {key} split metadata.")

    return values


def _validate_identity_overlaps(
    training_metadata: dict[str, Any],
    *,
    grouping_column: str,
) -> None:
    overlaps = _require_dictionary(
        training_metadata.get("identity_overlaps"),
        name="identity overlaps metadata",
    )
    required_identities = {"history_id", "case_id", "customer_id", grouping_column}

    for identity in required_identities:
        pairs = _require_dictionary(
            overlaps.get(identity),
            name=f"{identity} overlap metadata",
        )
        for pair in OVERLAP_PAIRS:
            value = pairs.get(pair)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelArtifactValidationError(
                    f"Invalid {identity} overlap metadata."
                )
            if value != 0:
                raise ModelArtifactValidationError(
                    f"Artifact split metadata reports forbidden {identity} overlap."
                )


def _validate_test_metrics(training_metadata: dict[str, Any], key: str) -> None:
    metrics = _require_dictionary(training_metadata.get(key), name=f"{key} metadata")
    for metric in REQUIRED_METRICS:
        value = metrics.get(metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ModelArtifactValidationError(
                f"Missing or invalid {metric} in {key} metadata."
            )


def validate_canonical_artifact_metadata(
    payload: object,
    *,
    expected_model_kind: str,
    expected_format_version: int,
    expected_features: tuple[str, ...],
    expected_target_column: str,
    expected_categorical_features: tuple[str, ...] | None = None,
    expected_missing_category_token: str | None = None,
) -> tuple[PlattProbabilityCalibrator, dict[str, Any]]:
    """Validate the common production artifact contract and build calibration."""

    metadata = _require_dictionary(payload, name="Model artifact metadata")

    if metadata.get("model_kind") != expected_model_kind:
        raise ModelArtifactValidationError("Wrong model kind for artifact loader.")
    if metadata.get("model_format_version") != expected_format_version:
        raise ModelArtifactValidationError("Unsupported artifact format version.")
    if metadata.get("target_column") != expected_target_column:
        raise ModelArtifactValidationError("Target-column metadata mismatch.")

    model_features = metadata.get("model_features")
    if not isinstance(model_features, list) or tuple(model_features) != expected_features:
        raise ModelArtifactValidationError("Model artifact feature contract mismatch.")

    if expected_categorical_features is not None:
        categorical = metadata.get("categorical_features")
        if not isinstance(categorical, list) or tuple(categorical) != expected_categorical_features:
            raise ModelArtifactValidationError(
                "Model artifact categorical feature metadata mismatch."
            )
    if (
        expected_missing_category_token is not None
        and metadata.get("missing_category_token") != expected_missing_category_token
    ):
        raise ModelArtifactValidationError(
            "Model artifact missing-category metadata mismatch."
        )

    calibration_metadata = _require_dictionary(
        metadata.get("calibration"),
        name="Calibration metadata",
    )
    if calibration_metadata.get("method") != CALIBRATION_METHOD:
        raise ModelArtifactValidationError("Calibration method metadata mismatch.")
    try:
        calibrator = PlattProbabilityCalibrator.from_dict(calibration_metadata)
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelArtifactValidationError("Malformed calibration metadata.") from exc
    if not calibrator.is_fitted:
        raise ModelArtifactValidationError("Artifact calibrator is missing or unfitted.")
    if not all(
        math.isfinite(value)
        for value in (calibrator.coefficient, calibrator.intercept)
    ):
        raise ModelArtifactValidationError("Calibration parameters must be finite.")

    training_metadata = _require_dictionary(
        metadata.get("training_metadata"),
        name="Training metadata",
    )
    data_reference = _require_nonempty_string(
        training_metadata, "data_generation_reference"
    )
    if not data_reference:
        raise ModelArtifactValidationError("Missing data generation reference metadata.")
    fingerprint = _require_nonempty_string(
        training_metadata,
        "data_fingerprint_sha256",
        label="training dataset fingerprint",
    )
    if SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ModelArtifactValidationError("Invalid training dataset fingerprint metadata.")

    feature_contract = training_metadata.get("feature_contract")
    if not isinstance(feature_contract, list) or tuple(feature_contract) != expected_features:
        raise ModelArtifactValidationError("Training feature contract mismatch.")

    _require_integer(training_metadata, "split_seed")
    strategy_value = _require_nonempty_string(training_metadata, "grouping_strategy")
    try:
        grouping_strategy = GroupingStrategy(strategy_value)
    except ValueError as exc:
        raise ModelArtifactValidationError("Unsupported split grouping strategy.") from exc

    grouping_column = _require_nonempty_string(training_metadata, "grouping_column")
    if grouping_column != GROUPING_COLUMNS[grouping_strategy]:
        raise ModelArtifactValidationError("Split grouping strategy/column mismatch.")

    _validate_partition_mapping(
        training_metadata, "partition_rows", value_kind="positive_integer"
    )
    _validate_partition_mapping(
        training_metadata, "partition_groups", value_kind="positive_integer"
    )
    _validate_partition_mapping(
        training_metadata, "partition_positive_rates", value_kind="probability"
    )
    _validate_identity_overlaps(training_metadata, grouping_column=grouping_column)

    _validate_test_metrics(training_metadata, "raw_test_metrics")
    _validate_test_metrics(training_metadata, "calibrated_test_metrics")

    if training_metadata.get("calibration_method") != CALIBRATION_METHOD:
        raise ModelArtifactValidationError("Training calibration method metadata mismatch.")
    if training_metadata["calibration_method"] != calibration_metadata["method"]:
        raise ModelArtifactValidationError("Inconsistent calibration metadata.")

    return calibrator, training_metadata
