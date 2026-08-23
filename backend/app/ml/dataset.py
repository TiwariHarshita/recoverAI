from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "recovered"


@dataclass(frozen=True)
class HistoricalDataSplit:
    train: pd.DataFrame
    test: pd.DataFrame


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
    test_size: float = 0.20,
    random_state: int = 42,
) -> HistoricalDataSplit:
    """
    Create the shared stratified train/test split used by every ML model.

    Logistic regression and CatBoost must call this same function with the
    same dataset, test_size, and seed for an apples-to-apples benchmark.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    validate_historical_dataframe(
        dataframe,
        require_target=True,
    )

    target = prepare_target(
        dataframe
    )

    if target.nunique() < 2:
        raise ValueError(
            "Historical dataset must contain both recovery labels 0 and 1."
        )

    train, test = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return HistoricalDataSplit(
        train=train.copy(),
        test=test.copy(),
    )
