from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.ml.dataset import (
    FORBIDDEN_MODEL_COLUMNS,
    MODEL_FEATURES,
    GroupingStrategy,
    assert_no_forbidden_identity_overlap,
    prepare_model_features,
    split_historical_dataframe,
)
from app.ml.logistic_baseline import (
    LogisticRecoveryBaseline,
    train_logistic_baseline,
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


@pytest.fixture(scope="module")
def historical_frame() -> pd.DataFrame:
    population = (
        generate_synthetic_population(
            merchant_count=8,
            customers_per_merchant=40,
            seed=900,
            reference_time=REFERENCE_TIME,
        )
    )

    dataset = (
        generate_historical_dataset(
            population,
            900,
            case_seed=901,
            environment_seed=902,
        )
    )

    return pd.DataFrame(
        [
            record.model_dump(
                mode="json"
            )

            for record in dataset.records
        ]
    )


def test_feature_contract_excludes_leakage():
    assert set(
        MODEL_FEATURES
    ).isdisjoint(
        FORBIDDEN_MODEL_COLUMNS
    )


def test_prepare_model_features_uses_only_allow_list(
    historical_frame,
):
    prepared = prepare_model_features(
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


def test_shared_split_is_deterministic_grouped_and_balanced(
    historical_frame,
):
    first = split_historical_dataframe(
        historical_frame,
        test_size=0.20,
        random_state=42,
    )

    second = split_historical_dataframe(
        historical_frame,
        test_size=0.20,
        random_state=42,
    )

    assert list(
        first.train.index
    ) == list(
        second.train.index
    )

    assert list(
        first.validation.index
    ) == list(
        second.validation.index
    )

    assert list(
        first.test.index
    ) == list(
        second.test.index
    )

    overall_rate = (
        historical_frame[
            "recovered"
        ].mean()
    )

    test_rate = (
        first.test[
            "recovered"
        ].mean()
    )

    assert test_rate == pytest.approx(
        overall_rate,
        abs=0.02,
    )

    assert_no_forbidden_identity_overlap(first)
    partition_customers = [
        set(frame["customer_id"])
        for frame in (first.train, first.validation, first.test)
    ]
    assert partition_customers[0].isdisjoint(partition_customers[1])
    assert partition_customers[0].isdisjoint(partition_customers[2])
    assert partition_customers[1].isdisjoint(partition_customers[2])


def test_merchant_grouping_prevents_merchant_and_customer_leakage(
    historical_frame,
):
    split = split_historical_dataframe(
        historical_frame,
        random_state=43,
        grouping_strategy=GroupingStrategy.MERCHANT,
    )

    assert_no_forbidden_identity_overlap(split)
    for identity in ("merchant_id", "customer_id"):
        train = set(split.train[identity])
        validation = set(split.validation[identity])
        test = set(split.test[identity])
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)


def test_training_produces_probability_metrics(
    historical_frame,
):
    result = train_logistic_baseline(
        historical_frame,
        test_size=0.20,
        random_state=42,
    )

    assert (
        result.train_rows
        + result.validation_rows
        + result.test_rows
        == len(historical_frame)
    )

    assert (
        0.0
        <= result.metrics["roc_auc"]
        <= 1.0
    )

    assert (
        0.0
        <= result.metrics["accuracy"]
        <= 1.0
    )

    assert (
        0.0
        <= result.metrics["brier_score"]
        <= 1.0
    )

    assert (
        result.metrics["log_loss"]
        >= 0.0
    )

    probabilities = (
        result.model
        .predict_recovery_probability(
            historical_frame.head(
                25
            )
        )
    )

    assert (
        probabilities.shape
        == (25,)
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
    result = train_logistic_baseline(
        historical_frame,
        random_state=66,
    )

    unseen = (
        historical_frame
        .head(4)
        .copy()
    )

    unseen["bank"] = (
        "NEW_BANK_NEVER_SEEN_IN_TRAINING"
    )

    unseen["error_code"] = (
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
    result = train_logistic_baseline(
        historical_frame,
        random_state=77,
    )

    sample = (
        historical_frame.tail(
            12
        )
    )

    before = (
        result.model
        .predict_recovery_probability(
            sample
        )
    )

    raw_before = result.model.predict_raw_recovery_probability(sample)

    artifact_path = (
        tmp_path
        / "logistic.joblib"
    )

    result.model.save(
        artifact_path
    )

    loaded = (
        LogisticRecoveryBaseline.load(
            artifact_path
        )
    )

    after = (
        loaded
        .predict_recovery_probability(
            sample
        )
    )

    raw_after = loaded.predict_raw_recovery_probability(sample)

    np.testing.assert_allclose(
        before,
        after,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(raw_before, raw_after, rtol=0, atol=1e-12)
    assert loaded.calibrator is not None
    assert loaded.training_metadata["grouping_strategy"] == "customer"


def test_training_is_deterministic_for_same_seed(
    historical_frame,
):
    first = train_logistic_baseline(
        historical_frame,
        random_state=88,
    )

    second = train_logistic_baseline(
        historical_frame,
        random_state=88,
    )

    sample = (
        historical_frame.sample(
            n=20,
            random_state=999,
        )
    )

    np.testing.assert_allclose(
        (
            first.model
            .predict_recovery_probability(
                sample
            )
        ),

        (
            second.model
            .predict_recovery_probability(
                sample
            )
        ),

        rtol=0,
        atol=1e-12,
    )

    assert (
        first.metrics
        == second.metrics
    )


def test_training_rejects_invalid_targets(
    historical_frame,
):
    fractional = (
        historical_frame.copy()
    )

    fractional["recovered"] = (
        fractional[
            "recovered"
        ].astype(float)
    )

    fractional.loc[
        fractional.index[0],
        "recovered",
    ] = 0.5

    with pytest.raises(
        ValueError,
        match="must contain only 0 or 1",
    ):
        train_logistic_baseline(
            fractional
        )

    single_class = (
        historical_frame.copy()
    )

    single_class["recovered"] = 0

    with pytest.raises(
        ValueError,
        match="both recovery labels",
    ):
        train_logistic_baseline(
            single_class
        )
