from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.artifact_contract import (
    ModelArtifactValidationError,
    validate_canonical_artifact_metadata,
)
from app.ml.calibration import CALIBRATION_METHOD, PlattProbabilityCalibrator
from app.ml.dataset import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
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


MODEL_KIND = "logistic_regression_baseline"
MODEL_FORMAT_VERSION = 2
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class BaselineTrainingResult:
    model: "LogisticRecoveryBaseline"
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


class LogisticRecoveryBaseline:
    """
    Interpretable baseline for P(recovery | case, action).

    The model consumes only the explicit observable feature allow-list
    defined in app.ml.dataset.
    """

    def __init__(
        self,
        *,
        random_state: int = 42,
        max_iter: int = 3000,
    ) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.pipeline = self._build_pipeline()
        self.calibrator: PlattProbabilityCalibrator | None = None
        self.training_metadata: dict[str, Any] = {}
        self._is_fitted = False

    def _build_pipeline(self) -> Pipeline:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        boolean_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "one_hot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    list(NUMERIC_FEATURES),
                ),
                (
                    "boolean",
                    boolean_pipeline,
                    list(BOOLEAN_FEATURES),
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    list(CATEGORICAL_FEATURES),
                ),
            ],
            remainder="drop",
        )

        classifier = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

    def fit(
        self,
        dataframe: pd.DataFrame,
    ) -> "LogisticRecoveryBaseline":
        validate_historical_dataframe(
            dataframe,
            require_target=True,
        )

        features = prepare_model_features(
            dataframe
        )
        target = prepare_target(
            dataframe
        )

        if target.nunique() < 2:
            raise ValueError(
                "Logistic regression training requires both recovery labels 0 and 1."
            )

        self.pipeline.fit(
            features,
            target,
        )
        self._is_fitted = True

        return self

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "LogisticRecoveryBaseline has not been fitted yet."
            )

    def predict_raw_recovery_probability(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """Return the uncalibrated model probability for each row."""

        self._require_fitted()

        features = prepare_model_features(
            dataframe
        )

        probabilities = self.pipeline.predict_proba(
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
            probabilities
            >= threshold
        ).astype(int)

    def coefficient_table(self) -> pd.DataFrame:
        """Return transformed feature names and learned coefficients."""

        self._require_fitted()

        preprocessor = self.pipeline.named_steps[
            "preprocessor"
        ]
        classifier = self.pipeline.named_steps[
            "classifier"
        ]

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        coefficients = (
            classifier.coef_[0]
        )

        table = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
            }
        )

        table["abs_coefficient"] = (
            table["coefficient"].abs()
        )

        return table.sort_values(
            "abs_coefficient",
            ascending=False,
        ).reset_index(
            drop=True
        )

    def save(
        self,
        path: str | Path,
    ) -> Path:
        self._require_fitted()

        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload: dict[str, Any] = {
            "model_kind": MODEL_KIND,
            "model_format_version": MODEL_FORMAT_VERSION,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "model_features": list(MODEL_FEATURES),
            "target_column": TARGET_COLUMN,
            "calibration": (
                self.calibrator.to_dict() if self.calibrator is not None else None
            ),
            "training_metadata": self.training_metadata,
            "pipeline": self.pipeline,
        }

        validate_canonical_artifact_metadata(
            payload,
            expected_model_kind=MODEL_KIND,
            expected_format_version=MODEL_FORMAT_VERSION,
            expected_features=tuple(MODEL_FEATURES),
            expected_target_column=TARGET_COLUMN,
        )

        joblib.dump(
            payload,
            output_path,
        )

        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "LogisticRecoveryBaseline":
        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {input_path}"
            )

        try:
            payload = joblib.load(input_path)
        except Exception as exc:
            raise ModelArtifactValidationError(
                "Unable to read valid Logistic model artifact metadata."
            ) from exc
        calibrator, training_metadata = validate_canonical_artifact_metadata(
            payload,
            expected_model_kind=MODEL_KIND,
            expected_format_version=MODEL_FORMAT_VERSION,
            expected_features=tuple(MODEL_FEATURES),
            expected_target_column=TARGET_COLUMN,
        )

        missing_model_fields = [
            key for key in ("random_state", "max_iter", "pipeline") if key not in payload
        ]
        if missing_model_fields:
            raise ModelArtifactValidationError(
                "Missing Logistic model metadata: "
                f"{', '.join(missing_model_fields)}."
            )

        try:
            model = cls(
                random_state=int(payload["random_state"]),
                max_iter=int(payload["max_iter"]),
            )
        except (TypeError, ValueError) as exc:
            raise ModelArtifactValidationError(
                "Malformed Logistic model metadata."
            ) from exc

        model.pipeline = payload["pipeline"]
        model.calibrator = calibrator
        model.training_metadata = training_metadata
        model._is_fitted = True

        return model


def train_logistic_baseline(
    dataframe: pd.DataFrame,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    threshold: float = DEFAULT_THRESHOLD,
    grouping_strategy: GroupingStrategy | str = GroupingStrategy.CUSTOMER,
    data_generation_reference: str = "in-memory historical dataframe",
) -> BaselineTrainingResult:
    """Train and evaluate a deterministic logistic-regression baseline."""

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

    train_frame = split.train
    validation_frame = split.validation
    test_frame = split.test

    model = LogisticRecoveryBaseline(
        random_state=random_state,
    )

    model.fit(
        train_frame
    )

    # Calibration is the only operation after fitting that may learn from the
    # held-out validation data. The final test partition is not read until both
    # the raw model and calibrator are frozen.
    model.fit_calibration(validation_frame)

    test_target = prepare_target(
        test_frame
    )

    raw_probabilities = model.predict_raw_recovery_probability(test_frame)
    calibrated_probabilities = model.predict_recovery_probability(test_frame)

    raw_metrics = evaluate_probabilities(
        test_target,
        raw_probabilities,
        threshold=threshold,
    )
    calibrated_metrics = evaluate_probabilities(
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

    return BaselineTrainingResult(
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
