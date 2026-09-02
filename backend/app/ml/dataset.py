from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


TARGET_COLUMN = "recovered"
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAIN_SIZE = 0.70
DEFAULT_VALIDATION_SIZE = 0.15
DEFAULT_TEST_SIZE = 0.15


class GroupingStrategy(str, Enum):
    """Entity boundary used to keep related histories in one partition."""

    CUSTOMER = "customer"
    MERCHANT = "merchant"


GROUPING_COLUMNS = {
    GroupingStrategy.CUSTOMER: "customer_id",
    GroupingStrategy.MERCHANT: "merchant_id",
}


@dataclass(frozen=True)
class HistoricalDataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    grouping_strategy: GroupingStrategy
    split_seed: int

    def identity_overlaps(self) -> dict[str, dict[str, int]]:
        """Return pairwise identifier overlaps for split-audit metadata."""

        partitions = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        overlaps: dict[str, dict[str, int]] = {}

        for column in IDENTIFIER_COLUMNS:
            if not all(column in frame.columns for frame in partitions.values()):
                continue
            values = {
                name: set(frame[column].dropna().astype(str))
                for name, frame in partitions.items()
            }
            overlaps[column] = {
                "train_validation": len(values["train"] & values["validation"]),
                "train_test": len(values["train"] & values["test"]),
                "validation_test": len(values["validation"] & values["test"]),
            }

        return overlaps


# Audit identifiers are useful for tracing rows, but must never be
# learned by a recovery model.
IDENTIFIER_COLUMNS = (
    "history_id",
    "case_id",
    "merchant_id",
    "customer_id",
)

# These values are only known after the selected recovery action runs.
# Using them as model inputs would leak the answer into training.
POST_OUTCOME_COLUMNS = (
    "fully_recovered",
    "outcome",
    "recovered_amount",
    "recovered_fraction",
    "recovery_delay_hours",
    "remaining_amount_after",
)

# Useful for later off-policy evaluation, but it describes the historical
# action-selection mechanism rather than the recovery outcome itself.
LOGGING_ONLY_COLUMNS = (
    "behavior_selection_probability",
)

# These fields belong to the simulator's hidden ground truth and must
# never appear in a production feature contract.
HIDDEN_SIMULATOR_COLUMNS = (
    "expected_failure_class",
    "latent_recovery_probability",
    "random_draw",
)

CATEGORICAL_FEATURES = (
    "merchant_archetype",
    "case_type",
    "currency",
    "payment_method",
    "bank",
    "error_code",
    "error_source",
    "error_step",
    "error_reason",
    "failure_class",
    "diagnosis_certainty",
    "preferred_payment_method",
    "preferred_channel",
    "language_preference",
    "action_type",
    "channel",
    "policy_decision_at_selection",
)

NUMERIC_FEATURES = (
    "merchant_average_order_value",
    "amount_to_average_order_ratio",
    "amount_at_risk",
    "payment_attempt_number",
    "subscription_retry_count",
    "invoice_days_overdue",
    "attempt_count",
    "recovery_retry_count",
    "previous_contacts",
    "case_age_hours",
    "customer_lifetime_value",
    "historical_payment_success_rate",
    "successful_payments",
    "failed_payments",
    "previous_recovery_attempts",
    "previous_recovery_successes",
    "previous_recovery_success_rate",
    "customer_tenure_days",
    "action_amount",
    "eligible_action_count",
    "action_delay_hours",
)

BOOLEAN_FEATURES = (
    "mandate_active",
    "temporary_failure",
    "retry_same_method_reasonable",
    "requires_new_payment_method",
    "customer_action_required",
    "merchant_action_required",
    "customer_do_not_contact",
    "was_deferred",
)

MODEL_FEATURES = (
    *CATEGORICAL_FEATURES,
    *NUMERIC_FEATURES,
    *BOOLEAN_FEATURES,
)

FORBIDDEN_MODEL_COLUMNS = frozenset(
    (
        *IDENTIFIER_COLUMNS,
        TARGET_COLUMN,
        *POST_OUTCOME_COLUMNS,
        *LOGGING_ONLY_COLUMNS,
        *HIDDEN_SIMULATOR_COLUMNS,
    )
)


def validate_feature_contract() -> None:
    """Fail fast if a future edit accidentally introduces leakage."""

    features = set(MODEL_FEATURES)

    duplicates = (
        len(MODEL_FEATURES)
        != len(features)
    )

    if duplicates:
        raise ValueError(
            "MODEL_FEATURES contains duplicate columns."
        )

    leaked = (
        features
        & FORBIDDEN_MODEL_COLUMNS
    )

    if leaked:
        raise ValueError(
            "Model feature contract contains forbidden columns: "
            f"{sorted(leaked)}"
        )


def validate_historical_dataframe(
    dataframe: pd.DataFrame,
    *,
    require_target: bool = True,
) -> None:
    """Validate that a historical dataframe satisfies the ML contract."""

    validate_feature_contract()

    required = set(
        MODEL_FEATURES
    )

    if require_target:
        required.add(
            TARGET_COLUMN
        )

    missing = sorted(
        required
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Historical dataset is missing required columns: "
            f"{missing}"
        )

    if require_target:
        target = pd.to_numeric(
            dataframe[TARGET_COLUMN],
            errors="coerce",
        )

        if target.isna().any():
            raise ValueError(
                "Target column 'recovered' contains non-numeric or missing values."
            )

        unique = set(
            target.unique()
        )

        if not unique.issubset(
            {0.0, 1.0}
        ):
            raise ValueError(
                "Target column 'recovered' must contain only 0 or 1."
            )


def load_historical_csv(
    path: str | Path,
) -> pd.DataFrame:
    """Load and validate a generated RecoverAI historical CSV."""

    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {csv_path}"
        )

    dataframe = pd.read_csv(
        csv_path
    )

    if dataframe.empty:
        raise ValueError(
            "Historical dataset is empty."
        )

    validate_historical_dataframe(
        dataframe,
        require_target=True,
    )

    return dataframe


def _coerce_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """Convert CSV boolean representations to 0/1 while preserving missingness."""

    mapping = {
        True: 1.0,
        False: 0.0,
        1: 1.0,
        0: 0.0,
        1.0: 1.0,
        0.0: 0.0,
        "true": 1.0,
        "false": 0.0,
        "True": 1.0,
        "False": 0.0,
        "TRUE": 1.0,
        "FALSE": 0.0,
        "1": 1.0,
        "0": 0.0,
    }

    def convert(value):
        if pd.isna(value):
            return np.nan

        if value in mapping:
            return mapping[value]

        if isinstance(value, str):
            normalized = value.strip()

            if normalized in mapping:
                return mapping[normalized]

        return np.nan

    return series.map(
        convert
    ).astype(float)


def prepare_model_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return only approved pre-outcome model features with stable dtypes.

    Explicit allow-listing is intentional. Extra CSV columns are ignored,
    which prevents future outcome/audit fields from silently entering ML.
    """

    validate_historical_dataframe(
        dataframe,
        require_target=False,
    )

    features = dataframe.loc[
        :,
        list(MODEL_FEATURES),
    ].copy()

    for column in NUMERIC_FEATURES:
        features[column] = (
            pd.to_numeric(
                features[column],
                errors="coerce",
            )
        )

    for column in BOOLEAN_FEATURES:
        features[column] = (
            _coerce_boolean_series(
                features[column]
            )
        )

    def normalize_categorical_value(value):
        # Domain enums inherit from both str and Enum. Calling str(enum)
        # produces values such as "RecoveryActionType.WAIT", while the
        # historical CSV contains the canonical value "wait". Normalize
        # enums explicitly so training and live inference share one contract.
        if isinstance(value, Enum):
            return str(value.value)

        if pd.isna(value):
            return np.nan

        return str(value)

    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].map(
            normalize_categorical_value
        )

    return features


def prepare_target(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """Return the binary recovery target as integer 0/1."""

    validate_historical_dataframe(
        dataframe,
        require_target=True,
    )

    target = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="raise",
    ).astype(int)

    return target


def split_historical_dataframe(
    dataframe: pd.DataFrame,
    *,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_SPLIT_SEED,
    grouping_strategy: GroupingStrategy | str = GroupingStrategy.CUSTOMER,
) -> HistoricalDataSplit:
    """
    Create the shared identity-grouped train/validation/test split.

    Logistic regression and CatBoost must call this same function with the
    same dataset and seed for an apples-to-apples benchmark. Candidate grouped
    splits are scored for row-ratio and outcome-rate similarity, while entity
    isolation remains an absolute constraint.
    """

    if not 0.0 < validation_size < 1.0:
        raise ValueError(
            "validation_size must be between 0 and 1."
        )

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    if validation_size + test_size >= 1.0:
        raise ValueError("validation_size + test_size must be less than 1.")

    try:
        strategy = GroupingStrategy(grouping_strategy)
    except ValueError as exc:
        raise ValueError(
            "grouping_strategy must be 'customer' or 'merchant'."
        ) from exc

    validate_historical_dataframe(
        dataframe,
        require_target=True,
    )

    group_column = GROUPING_COLUMNS[strategy]
    missing_identity_columns = sorted(
        {"history_id", "case_id", "customer_id", group_column}
        - set(dataframe.columns)
    )
    if missing_identity_columns:
        raise ValueError(
            "Grouped splitting requires identity columns: "
            f"{missing_identity_columns}"
        )

    if dataframe[group_column].isna().any():
        raise ValueError(f"Grouping column '{group_column}' contains missing values.")

    target = prepare_target(
        dataframe
    )

    if target.nunique() < 2:
        raise ValueError(
            "Historical dataset must contain both recovery labels 0 and 1."
        )

    holdout_size = validation_size + test_size
    train, holdout = _best_group_holdout(
        dataframe,
        group_column=group_column,
        holdout_fraction=holdout_size,
        random_state=random_state,
    )

    test_fraction_of_holdout = test_size / holdout_size
    validation, test = _best_group_holdout(
        holdout,
        group_column=group_column,
        holdout_fraction=test_fraction_of_holdout,
        random_state=random_state + 1,
    )

    split = HistoricalDataSplit(
        train=train.copy(),
        validation=validation.copy(),
        test=test.copy(),
        grouping_strategy=strategy,
        split_seed=random_state,
    )

    assert_no_forbidden_identity_overlap(split)

    for name, frame in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if prepare_target(frame).nunique() < 2:
            raise ValueError(f"{name} split must contain both recovery labels 0 and 1.")

    return split


def _best_group_holdout(
    dataframe: pd.DataFrame,
    *,
    group_column: str,
    holdout_fraction: float,
    random_state: int,
    candidate_count: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose a deterministic group-isolated holdout with reasonable balance."""

    group_count = dataframe[group_column].nunique()
    if group_count < 3:
        raise ValueError(
            f"Grouped splitting requires at least 3 distinct {group_column} values."
        )

    splitter = GroupShuffleSplit(
        n_splits=candidate_count,
        test_size=holdout_fraction,
        random_state=random_state,
    )
    overall_rate = float(prepare_target(dataframe).mean())
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for retained_indexes, held_out_indexes in splitter.split(
        dataframe,
        groups=dataframe[group_column],
    ):
        retained = dataframe.iloc[retained_indexes]
        held_out = dataframe.iloc[held_out_indexes]
        retained_target = prepare_target(retained)
        held_out_target = prepare_target(held_out)

        # Prefer candidates containing both classes, then minimize deviations
        # in row share and outcome rate. Group isolation is never relaxed.
        missing_class_penalty = float(
            retained_target.nunique() < 2 or held_out_target.nunique() < 2
        )
        score = (
            missing_class_penalty * 10.0
            + abs(len(held_out) / len(dataframe) - holdout_fraction)
            + abs(float(retained_target.mean()) - overall_rate)
            + abs(float(held_out_target.mean()) - overall_rate)
        )
        candidate = (score, retained_indexes, held_out_indexes)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:  # pragma: no cover - GroupShuffleSplit always yields.
        raise RuntimeError("Unable to construct a grouped data split.")

    return dataframe.iloc[best[1]], dataframe.iloc[best[2]]


def assert_no_forbidden_identity_overlap(split: HistoricalDataSplit) -> None:
    """Fail if an identity forbidden by the selected strategy crosses splits."""

    required_zero_overlap = {"history_id", "case_id", "customer_id"}
    if split.grouping_strategy is GroupingStrategy.MERCHANT:
        required_zero_overlap.add("merchant_id")

    overlaps = split.identity_overlaps()
    violations = {
        column: pairs
        for column, pairs in overlaps.items()
        if column in required_zero_overlap and any(pairs.values())
    }
    if violations:
        raise AssertionError(f"Forbidden identity overlap detected: {violations}")


def historical_dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    """Stable content fingerprint recorded with trained model metadata."""

    row_hashes = pd.util.hash_pandas_object(dataframe, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def split_audit_metadata(split: HistoricalDataSplit) -> dict[str, object]:
    """Build serializable partition metadata shared by both model families."""

    partitions = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    grouping_column = GROUPING_COLUMNS[split.grouping_strategy]

    return {
        "split_seed": split.split_seed,
        "grouping_strategy": split.grouping_strategy.value,
        "grouping_column": grouping_column,
        "partition_rows": {name: len(frame) for name, frame in partitions.items()},
        "partition_groups": {
            name: int(frame[grouping_column].nunique())
            for name, frame in partitions.items()
        },
        "partition_positive_rates": {
            name: float(prepare_target(frame).mean())
            for name, frame in partitions.items()
        },
        "identity_overlaps": split.identity_overlaps(),
    }
