from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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

from app.ml.dataset import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    prepare_model_features,
    prepare_target,
    split_historical_dataframe,
    validate_historical_dataframe,
)


MODEL_KIND = "catboost_recovery_model"
MODEL_FORMAT_VERSION = 2
DEFAULT_THRESHOLD = 0.5
MISSING_CATEGORY_TOKEN = "__MISSING__"


@dataclass(frozen=True)
class CatBoostTrainingResult:
    model: "CatBoostRecoveryModel"
    metrics: dict[str, float | int]
    train_rows: int
    test_rows: int
    train_positive_rate: float
    test_positive_rate: float


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

    def predict_recovery_probability(
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

        self.model.save_model(
            str(output_path),
            format="cbm",
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
        }

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

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            metadata = json.load(
                handle
            )

        if (
            metadata.get(
                "model_kind"
            )
            != MODEL_KIND
        ):
            raise ValueError(
                "Artifact is not a RecoverAI CatBoost recovery model."
            )

        if (
            metadata.get(
                "model_format_version"
            )
            != MODEL_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported CatBoost model artifact version."
            )

        if tuple(
            metadata.get(
                "model_features",
                [],
            )
        ) != tuple(
            MODEL_FEATURES
        ):
            raise ValueError(
                "Model artifact feature contract does not match current code."
            )

        if tuple(
            metadata.get(
                "categorical_features",
                [],
            )
        ) != tuple(
            CATEGORICAL_FEATURES
        ):
            raise ValueError(
                "Model artifact categorical feature contract does not match current code."
            )

        if (
            metadata.get(
                "missing_category_token"
            )
            != MISSING_CATEGORY_TOKEN
        ):
            raise ValueError(
                "Model artifact missing-category contract does not match current code."
            )

        instance = cls(
            random_state=int(
                metadata[
                    "random_state"
                ]
            ),

            iterations=int(
                metadata[
                    "iterations"
                ]
            ),

            depth=int(
                metadata[
                    "depth"
                ]
            ),

            learning_rate=float(
                metadata[
                    "learning_rate"
                ]
            ),

            l2_leaf_reg=float(
                metadata[
                    "l2_leaf_reg"
                ]
            ),

            one_hot_max_size=int(
                metadata[
                    "one_hot_max_size"
                ]
            ),

            random_strength=float(
                metadata[
                    "random_strength"
                ]
            ),

            boosting_type=str(
                metadata[
                    "boosting_type"
                ]
            ),

            thread_count=int(
                metadata.get(
                    "thread_count",
                    1,
                )
            ),
        )

        instance.model.load_model(
            str(input_path),
            format="cbm",
        )

        instance._is_fitted = True

        return instance


def _evaluate_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    (
        tn,
        fp,
        fn,
        tp,
    ) = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1,
        ],
    ).ravel()

    majority_accuracy = max(
        float(
            (
                y_true == 0
            ).mean()
        ),
        float(
            (
                y_true == 1
            ).mean()
        ),
    )

    return {
        "threshold": float(
            threshold
        ),

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
                labels=[
                    0,
                    1,
                ],
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

        "true_negative": int(
            tn
        ),

        "false_positive": int(
            fp
        ),

        "false_negative": int(
            fn
        ),

        "true_positive": int(
            tp
        ),
    }


def train_catboost_model(
    dataframe: pd.DataFrame,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
    threshold: float = DEFAULT_THRESHOLD,
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
        test_size=test_size,
        random_state=random_state,
    )

    train_frame = (
        split.train
    )

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

    return CatBoostTrainingResult(
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