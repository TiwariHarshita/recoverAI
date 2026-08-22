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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.dataset import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    prepare_model_features,
    prepare_target,
    split_historical_dataframe,
    validate_historical_dataframe,
)


MODEL_KIND = "logistic_regression_baseline"
MODEL_FORMAT_VERSION = 1
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class BaselineTrainingResult:
    model: "LogisticRecoveryBaseline"
    metrics: dict[str, float | int]
    train_rows: int
    test_rows: int
    train_positive_rate: float
    test_positive_rate: float


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

    def predict_recovery_probability(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        """Return P(recovered=1) for each row."""

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
            "pipeline": self.pipeline,
        }

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

        payload = joblib.load(
            input_path
        )

        if (
            payload.get("model_kind")
            != MODEL_KIND
        ):
            raise ValueError(
                "Artifact is not a RecoverAI logistic baseline model."
            )

        if (
            payload.get("model_format_version")
            != MODEL_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported logistic baseline artifact version."
            )

        if tuple(
            payload.get(
                "model_features",
                [],
            )
        ) != tuple(MODEL_FEATURES):
            raise ValueError(
                "Model artifact feature contract does not match current code."
            )

        model = cls(
            random_state=int(
                payload["random_state"]
            ),
            max_iter=int(
                payload["max_iter"]
            ),
        )

        model.pipeline = payload[
            "pipeline"
        ]
        model._is_fitted = True

        return model


def _evaluate_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    majority_accuracy = max(
        float((y_true == 0).mean()),
        float((y_true == 1).mean()),
    )

    return {
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),

        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),

        "majority_class_accuracy": (
            majority_accuracy
        ),

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def train_logistic_baseline(
    dataframe: pd.DataFrame,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
    threshold: float = DEFAULT_THRESHOLD,
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
        test_size=test_size,
        random_state=random_state,
    )

    train_frame = split.train
    test_frame = split.test

    model = LogisticRecoveryBaseline(
        random_state=random_state,
    )

    model.fit(
        train_frame
    )

    test_target = prepare_target(
        test_frame
    )

    probabilities = (
        model.predict_recovery_probability(
            test_frame
        )
    )

    metrics = _evaluate_probabilities(
        test_target,
        probabilities,
        threshold=threshold,
    )

    return BaselineTrainingResult(
        model=model,

        metrics=metrics,

        train_rows=len(
            train_frame
        ),

        test_rows=len(
            test_frame
        ),

        train_positive_rate=float(
            prepare_target(
                train_frame
            ).mean()
        ),

        test_positive_rate=float(
            test_target.mean()
        ),
    )