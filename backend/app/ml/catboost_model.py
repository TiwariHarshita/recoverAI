from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from app.ml.artifact_contract import (
    ModelArtifactValidationError,
    validate_canonical_artifact_metadata,
)
from app.ml.calibration import CALIBRATION_METHOD, PlattProbabilityCalibrator
from app.ml.dataset import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    GroupingStrategy,
    historical_dataframe_fingerprint,
    prepare_model_features,
    prepare_target,
    split_audit_metadata,
    split_historical_dataframe,
    validate_historical_dataframe,
)
from app.ml.evaluation import evaluate_probabilities


MODEL_KIND = "catboost_recovery_model"
MODEL_FORMAT_VERSION = 3
DEFAULT_THRESHOLD = 0.5
MISSING_CATEGORY_TOKEN = "__MISSING__"


@dataclass(frozen=True)
class CatBoostTrainingResult:
    model: "CatBoostRecoveryModel"
    metrics: dict[str, float | int]
    raw_metrics: dict[str, float | int]
    calibrated_metrics: dict[str, float | int]
    train_rows: int
    validation_rows: int
    test_rows: int
    train_positive_rate: float
    validation_positive_rate: float
    test_positive_rate: float
    split_metadata: dict[str, object]


def prepare_catboost_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the shared RecoverAI feature contract for native CatBoost.

    CatBoost consumes categorical columns directly rather than one-hot
    encoding them.

    Missing categorical values are converted to one stable sentinel
    string. Numeric missing values may remain NaN.
    """

    features = prepare_model_features(
        dataframe
    )

    for column in CATEGORICAL_FEATURES:
        features[column] = (
            features[column]
            .fillna(MISSING_CATEGORY_TOKEN)
            .astype(str)
        )

    return features


class CatBoostRecoveryModel:
    """
    Nonlinear recovery-probability model.

    Estimates:

        P(recovery | case, customer, merchant, action)

    It deliberately consumes exactly the same observable feature
    contract as the logistic-regression baseline.
    """

    def __init__(
        self,
        *,
        random_state: int = 42,
        iterations: int = 890,
        depth: int = 2,
        learning_rate: float = 0.03,
        l2_leaf_reg: float = 10.0,
        one_hot_max_size: int = 5,
        random_strength: float = 0.5,
        boosting_type: str = "Ordered",
        thread_count: int = 1,
    ) -> None:
        if iterations <= 0:
            raise ValueError(
                "iterations must be greater than zero."
            )

        if depth <= 0:
            raise ValueError(
                "depth must be greater than zero."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if l2_leaf_reg < 0:
            raise ValueError(
                "l2_leaf_reg cannot be negative."
            )

        if one_hot_max_size < 0:
            raise ValueError(
                "one_hot_max_size cannot be negative."
            )

        if random_strength < 0:
            raise ValueError(
                "random_strength cannot be negative."
            )

        if thread_count == 0:
            raise ValueError(
                "thread_count cannot be zero."
            )

        if boosting_type not in {
            "Ordered",
            "Plain",
        }:
            raise ValueError(
                "boosting_type must be 'Ordered' or 'Plain'."
            )

        self.random_state = random_state
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.one_hot_max_size = one_hot_max_size
        self.random_strength = random_strength
        self.boosting_type = boosting_type
        self.thread_count = thread_count

        self.model = self._build_model()
        self.calibrator: PlattProbabilityCalibrator | None = None
        self.training_metadata: dict[str, Any] = {}
        self._is_fitted = False

    def _build_model(
        self,
    ) -> CatBoostClassifier:
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            one_hot_max_size=self.one_hot_max_size,
            random_strength=self.random_strength,
            boosting_type=self.boosting_type,
            random_seed=self.random_state,
            thread_count=self.thread_count,
            verbose=False,

            # Prevent CatBoost from creating catboost_info/
            # inside the repository.
            allow_writing_files=False,
        )

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "CatBoostRecoveryModel":
        validate_historical_dataframe(
            dataframe,
            require_target=True,
        )

        target = prepare_target(
            dataframe
        )

        if target.nunique() < 2:
            raise ValueError(
                "CatBoost training requires both recovery labels 0 and 1."
            )

        features = prepare_catboost_features(
            dataframe
        )

        self.model.fit(
            features,
            target,
            cat_features=list(
                CATEGORICAL_FEATURES
            ),
            verbose=False,
        )

        self._is_fitted = True

        return self

    def _require_fitted(
        self,
    ) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "CatBoostRecoveryModel has not been fitted yet."
            )

    def predict_raw_recovery_probability(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return P(recovered=1) for every row.
        """

        self._require_fitted()

        features = prepare_catboost_features(
            dataframe
        )

        probabilities = self.model.predict_proba(
            features
        )[:, 1]

        return np.asarray(
            probabilities,
            dtype=float,
        )

    def fit_calibration(self, dataframe: pd.DataFrame) -> None:
        """Fit Platt calibration using a validation partition only."""

        target = prepare_target(dataframe).to_numpy()
        probabilities = self.predict_raw_recovery_probability(dataframe)
        self.calibrator = PlattProbabilityCalibrator().fit(probabilities, target)

    def predict_recovery_probability(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """Return canonical calibrated P(recovery | case, action)."""

        if self.calibrator is None:
            raise ModelArtifactValidationError(
                "Canonical prediction requires a fitted probability calibrator."
            )
        probabilities = self.predict_raw_recovery_probability(dataframe)
        return self.calibrator.predict(probabilities)

    def predict(
        self,
        dataframe: pd.DataFrame,
        *,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> np.ndarray:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        probabilities = (
            self.predict_recovery_probability(
                dataframe
            )
        )

        return (
            probabilities >= threshold
        ).astype(int)

    def feature_importance_table(
        self,
    ) -> pd.DataFrame:
        """
        Return CatBoost feature importance using the original
        shared RecoverAI feature names.
        """

        self._require_fitted()

        importances = (
            self.model.get_feature_importance()
        )

        table = pd.DataFrame(
            {
                "feature": list(
                    MODEL_FEATURES
                ),
                "importance": importances,
            }
        )

        return (
            table
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    def save(
        self,
        path: str | Path,
    ) -> Path:
        """
        Save the model in CatBoost's native format plus
        a RecoverAI metadata sidecar.
        """

        self._require_fitted()

        output_path = Path(
            path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path = (
            output_path.with_suffix(
                ".meta.json"
            )
        )

        metadata: dict[str, Any] = {
            "model_kind": (
                MODEL_KIND
            ),

            "model_format_version": (
                MODEL_FORMAT_VERSION
            ),

            "target_column": (
                TARGET_COLUMN
            ),

            "model_features": list(
                MODEL_FEATURES
            ),

            "categorical_features": list(
                CATEGORICAL_FEATURES
            ),

            "missing_category_token": (
                MISSING_CATEGORY_TOKEN
            ),

            "random_state": (
                self.random_state
            ),

            "iterations": (
                self.iterations
            ),

            "depth": (
                self.depth
            ),

            "learning_rate": (
                self.learning_rate
            ),

            "l2_leaf_reg": (
                self.l2_leaf_reg
            ),

            "one_hot_max_size": (
                self.one_hot_max_size
            ),

            "random_strength": (
                self.random_strength
            ),

            "boosting_type": (
                self.boosting_type
            ),

            "thread_count": (
                self.thread_count
            ),

            "calibration": (
                self.calibrator.to_dict() if self.calibrator is not None else None
            ),

            "training_metadata": self.training_metadata,
        }

        validate_canonical_artifact_metadata(
            metadata,
            expected_model_kind=MODEL_KIND,
            expected_format_version=MODEL_FORMAT_VERSION,
            expected_features=tuple(MODEL_FEATURES),
            expected_target_column=TARGET_COLUMN,
            expected_categorical_features=tuple(CATEGORICAL_FEATURES),
            expected_missing_category_token=MISSING_CATEGORY_TOKEN,
        )

        self.model.save_model(
            str(output_path),
            format="cbm",
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                metadata,
                handle,
                indent=2,
            )

            handle.write(
                "\n"
            )

        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "CatBoostRecoveryModel":
        input_path = Path(
            path
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {input_path}"
            )

        metadata_path = (
            input_path.with_suffix(
                ".meta.json"
            )
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {metadata_path}"
            )

        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelArtifactValidationError(
                "Unable to read valid CatBoost artifact metadata."
            ) from exc

        calibrator, training_metadata = validate_canonical_artifact_metadata(
            metadata,
            expected_model_kind=MODEL_KIND,
            expected_format_version=MODEL_FORMAT_VERSION,
            expected_features=tuple(MODEL_FEATURES),
            expected_target_column=TARGET_COLUMN,
            expected_categorical_features=tuple(CATEGORICAL_FEATURES),
            expected_missing_category_token=MISSING_CATEGORY_TOKEN,
        )

        required_model_fields = (
            "random_state",
            "iterations",
            "depth",
            "learning_rate",
            "l2_leaf_reg",
            "one_hot_max_size",
            "random_strength",
            "boosting_type",
            "thread_count",
        )
        missing_model_fields = [key for key in required_model_fields if key not in metadata]
        if missing_model_fields:
            raise ModelArtifactValidationError(
                "Missing CatBoost model metadata: "
                f"{', '.join(missing_model_fields)}."
            )

        try:
            instance = cls(
                random_state=int(metadata["random_state"]),
                iterations=int(metadata["iterations"]),
                depth=int(metadata["depth"]),
                learning_rate=float(metadata["learning_rate"]),
                l2_leaf_reg=float(metadata["l2_leaf_reg"]),
                one_hot_max_size=int(metadata["one_hot_max_size"]),
                random_strength=float(metadata["random_strength"]),
                boosting_type=str(metadata["boosting_type"]),
                thread_count=int(metadata["thread_count"]),
            )
        except (TypeError, ValueError) as exc:
            raise ModelArtifactValidationError(
                "Malformed CatBoost model metadata."
            ) from exc

        try:
            instance.model.load_model(str(input_path), format="cbm")
        except Exception as exc:
            raise ModelArtifactValidationError(
                "CatBoost model binary is invalid or inconsistent with its metadata."
            ) from exc

        instance.calibrator = calibrator
        instance.training_metadata = training_metadata

        instance._is_fitted = True

        return instance


def _evaluate_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    return evaluate_probabilities(y_true, probabilities, threshold=threshold)


def train_catboost_model(
    dataframe: pd.DataFrame,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    threshold: float = DEFAULT_THRESHOLD,
    grouping_strategy: GroupingStrategy | str = GroupingStrategy.CUSTOMER,
    data_generation_reference: str = "in-memory historical dataframe",
    iterations: int = 890,
    depth: int = 2,
    learning_rate: float = 0.03,
    l2_leaf_reg: float = 10.0,
    one_hot_max_size: int = 5,
    random_strength: float = 0.5,
    boosting_type: str = "Ordered",
    thread_count: int = 1,
) -> CatBoostTrainingResult:
    """
    Train and evaluate CatBoost using the exact same shared
    historical split as the logistic-regression baseline.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
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

    split = split_historical_dataframe(
        dataframe,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
        grouping_strategy=grouping_strategy,
    )

    train_frame = (
        split.train
    )

    validation_frame = split.validation

    test_frame = (
        split.test
    )

    model = CatBoostRecoveryModel(
        random_state=random_state,
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        one_hot_max_size=one_hot_max_size,
        random_strength=random_strength,
        boosting_type=boosting_type,
        thread_count=thread_count,
    )

    model.fit(
        train_frame
    )

    # Fit calibration on validation data after the raw model is frozen. The
    # final test partition is first accessed only after this point.
    model.fit_calibration(validation_frame)

    test_target = prepare_target(
        test_frame
    )

    raw_probabilities = model.predict_raw_recovery_probability(test_frame)
    calibrated_probabilities = model.predict_recovery_probability(test_frame)

    raw_metrics = _evaluate_probabilities(
        test_target,
        raw_probabilities,
        threshold=threshold,
    )
    calibrated_metrics = _evaluate_probabilities(
        test_target,
        calibrated_probabilities,
        threshold=threshold,
    )
    audit_metadata = split_audit_metadata(split)
    model.training_metadata = {
        "data_generation_reference": data_generation_reference,
        "data_fingerprint_sha256": historical_dataframe_fingerprint(dataframe),
        "feature_contract": list(MODEL_FEATURES),
        **audit_metadata,
        "calibration_method": CALIBRATION_METHOD,
        "raw_test_metrics": raw_metrics,
        "calibrated_test_metrics": calibrated_metrics,
    }

    return CatBoostTrainingResult(
        model=model,
        metrics=calibrated_metrics,
        raw_metrics=raw_metrics,
        calibrated_metrics=calibrated_metrics,

        train_rows=len(
            train_frame
        ),

        validation_rows=len(validation_frame),

        test_rows=len(
            test_frame
        ),

        train_positive_rate=float(
            prepare_target(
                train_frame
            ).mean()
        ),

        validation_positive_rate=float(prepare_target(validation_frame).mean()),

        test_positive_rate=float(
            test_target.mean()
        ),

        split_metadata=audit_metadata,
    )
