from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

import numpy as np
import pandas as pd
import pytest

from app.ml.catboost_model import (
    MISSING_CATEGORY_TOKEN,
    CatBoostRecoveryModel,
    prepare_catboost_features,
    train_catboost_model,
)

from app.ml.dataset import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    split_historical_dataframe,
)

from simulator.customers import (
    generate_synthetic_population,
)

from simulator.historical import (
    generate_historical_dataset,
)


REFERENCE_TIME = datetime(
    2026,
    8,
    23,
    0,
    0,
    tzinfo=timezone.utc,
)


@pytest.fixture(
    scope="module"
)
def historical_frame() -> pd.DataFrame:
    population = (
        generate_synthetic_population(
            merchant_count=8,

            customers_per_merchant=40,

            seed=1000,

            reference_time=(
                REFERENCE_TIME
            ),
        )
    )

    dataset = (
        generate_historical_dataset(
            population,

            900,

            case_seed=1001,

            environment_seed=1002,
        )
    )

    return pd.DataFrame(
        [
            record.model_dump(
                mode="json"
            )

            for record
            in dataset.records
        ]
    )


def _train_fast(
    dataframe: pd.DataFrame,
    *,
    seed: int = 42,
):
    """
    Faster CatBoost settings for unit tests.

    These intentionally differ from production training defaults
    to keep the test suite fast.
    """

    return train_catboost_model(
        dataframe,

        test_size=0.20,

        random_state=seed,

        iterations=80,

        depth=5,

        learning_rate=0.08,

        one_hot_max_size=5,

        random_strength=0.5,

        boosting_type="Ordered",

        thread_count=1,
    )


def test_catboost_uses_same_feature_contract(
    historical_frame,
):
    prepared = prepare_catboost_features(
        historical_frame
    )

    assert list(
        prepared.columns
    ) == list(
        MODEL_FEATURES
    )

    assert (
        "recovered"
        not in prepared.columns
    )

    assert (
        "case_id"
        not in prepared.columns
    )

    assert (
        "recovered_amount"
        not in prepared.columns
    )

    assert (
        "behavior_selection_probability"
        not in prepared.columns
    )

    assert (
        "latent_recovery_probability"
        not in prepared.columns
    )


def test_categorical_missing_values_are_normalized(
    historical_frame,
):
    sample = (
        historical_frame
        .head(5)
        .copy()
    )

    sample.loc[
        sample.index[0],
        "bank",
    ] = None

    prepared = (
        prepare_catboost_features(
            sample
        )
    )

    assert (
        prepared.loc[
            sample.index[0],
            "bank",
        ]
        == MISSING_CATEGORY_TOKEN
    )

    for column in (
        CATEGORICAL_FEATURES
    ):
        assert (
            prepared[
                column
            ].isna().sum()
            == 0
        )


def test_catboost_uses_same_deterministic_split(
    historical_frame,
):
    expected = (
        split_historical_dataframe(
            historical_frame,

            test_size=0.20,

            random_state=42,
        )
    )

    result = _train_fast(
        historical_frame,
        seed=42,
    )

    assert (
        result.train_rows
        == len(
            expected.train
        )
    )

    assert (
        result.test_rows
        == len(
            expected.test
        )
    )

    assert (
        result.test_positive_rate
        == pytest.approx(
            expected.test[
                "recovered"
            ].mean(),

            abs=1e-12,
        )
    )


def test_training_produces_valid_probability_metrics(
    historical_frame,
):
    result = _train_fast(
        historical_frame,
        seed=51,
    )

    assert (
        0.0
        <= result.metrics[
            "roc_auc"
        ]
        <= 1.0
    )

    assert (
        0.0
        <= result.metrics[
            "accuracy"
        ]
        <= 1.0
    )

    assert (
        0.0
        <= result.metrics[
            "brier_score"
        ]
        <= 1.0
    )

    assert (
        result.metrics[
            "log_loss"
        ]
        >= 0.0
    )

    probabilities = (
        result.model
        .predict_recovery_probability(
            historical_frame
            .head(25)
        )
    )

    assert (
        probabilities.shape
        == (25,)
    )

    assert np.all(
        np.isfinite(
            probabilities
        )
    )

    assert np.all(
        probabilities >= 0.0
    )

    assert np.all(
        probabilities <= 1.0
    )


def test_model_handles_unseen_categories(
    historical_frame,
):
    result = _train_fast(
        historical_frame,
        seed=52,
    )

    unseen = (
        historical_frame
        .head(4)
        .copy()
    )

    unseen[
        "bank"
    ] = (
        "NEW_BANK_NEVER_SEEN"
    )

    unseen[
        "error_code"
    ] = (
        "NEW_PROVIDER_ERROR"
    )

    probabilities = (
        result.model
        .predict_recovery_probability(
            unseen
        )
    )

    assert (
        probabilities.shape
        == (4,)
    )

    assert np.all(
        np.isfinite(
            probabilities
        )
    )


def test_save_and_load_preserve_predictions(
    historical_frame,
    tmp_path,
):
    result = _train_fast(
        historical_frame,
        seed=53,
    )

    sample = (
        historical_frame
        .tail(12)
    )

    before = (
        result.model
        .predict_recovery_probability(
            sample
        )
    )

    artifact_path = (
        tmp_path
        / "catboost_recovery.cbm"
    )

    result.model.save(
        artifact_path
    )

    assert (
        artifact_path.exists()
    )

    assert (
        artifact_path
        .with_suffix(
            ".meta.json"
        )
        .exists()
    )

    loaded = (
        CatBoostRecoveryModel.load(
            artifact_path
        )
    )

    after = (
        loaded
        .predict_recovery_probability(
            sample
        )
    )

    np.testing.assert_allclose(
        before,
        after,
        rtol=0,
        atol=1e-12,
    )


def test_training_is_deterministic_for_same_seed(
    historical_frame,
):
    first = _train_fast(
        historical_frame,
        seed=54,
    )

    second = _train_fast(
        historical_frame,
        seed=54,
    )

    sample = (
        historical_frame
        .sample(
            n=20,
            random_state=999,
        )
    )

    np.testing.assert_allclose(
        first.model
        .predict_recovery_probability(
            sample
        ),

        second.model
        .predict_recovery_probability(
            sample
        ),

        rtol=0,
        atol=1e-12,
    )

    assert (
        first.metrics
        == second.metrics
    )


def test_feature_importance_covers_shared_features(
    historical_frame,
):
    result = _train_fast(
        historical_frame,
        seed=55,
    )

    table = (
        result.model
        .feature_importance_table()
    )

    assert (
        set(
            table[
                "feature"
            ]
        )
        == set(
            MODEL_FEATURES
        )
    )

    assert (
        len(table)
        == len(
            MODEL_FEATURES
        )
    )

    assert np.all(
        table[
            "importance"
        ]
        >= 0.0
    )


def test_training_rejects_single_class_target(
    historical_frame,
):
    single_class = (
        historical_frame
        .copy()
    )

    single_class[
        "recovered"
    ] = 0

    with pytest.raises(
        ValueError,
        match="both recovery labels",
    ):
        _train_fast(
            single_class,
            seed=56,
        )


def test_invalid_boosting_type_is_rejected():
    with pytest.raises(
        ValueError,
        match="boosting_type",
    ):
        CatBoostRecoveryModel(
            boosting_type="INVALID"
        )