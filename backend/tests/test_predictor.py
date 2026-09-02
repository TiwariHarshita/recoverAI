from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.domain.action_scoring import RecoverySourceContext
from app.domain.merchant import Merchant
from app.ml.artifact_contract import ModelArtifactValidationError
from app.ml.calibration import CALIBRATION_METHOD
from app.ml.catboost_model import CatBoostRecoveryModel
from app.ml.dataset import MODEL_FEATURES, load_historical_csv
from app.ml.logistic_baseline import LogisticRecoveryBaseline
from app.policy.models import PolicyContext
from app.services.action_selector import select_best_recovery_action
from app.services.candidate_actions import generate_candidate_actions
from app.services.diagnosis import diagnose_case
from simulator.cases import generate_recovery_cases
from simulator.customers import generate_synthetic_population


BACKEND_ROOT = Path(__file__).resolve().parents[1]
LOGISTIC_ARTIFACT = BACKEND_ROOT / "artifacts/models/logistic_baseline.joblib"
CATBOOST_ARTIFACT = BACKEND_ROOT / "artifacts/models/catboost_recovery.cbm"
HISTORICAL_DATA = BACKEND_ROOT / "data/synthetic/recovery_history.csv"
FAMILIES = ("logistic", "catboost")


def _load(family: str, path: Path):
    if family == "logistic":
        return LogisticRecoveryBaseline.load(path)
    return CatBoostRecoveryModel.load(path)


def _mutated_artifact(tmp_path: Path, family: str, mutate) -> Path:
    if family == "logistic":
        payload = joblib.load(LOGISTIC_ARTIFACT)
        payload = copy.deepcopy(payload)
        mutate(payload)
        path = tmp_path / "logistic.joblib"
        joblib.dump(payload, path)
        return path

    path = tmp_path / "catboost.cbm"
    shutil.copy2(CATBOOST_ARTIFACT, path)
    metadata = json.loads(
        CATBOOST_ARTIFACT.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    mutate(metadata)
    path.with_suffix(".meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def inference_sample() -> pd.DataFrame:
    return load_historical_csv(HISTORICAL_DATA).head(8)


@pytest.mark.parametrize("family", FAMILIES)
def test_valid_canonical_artifact_loads_and_predicts_calibrated(
    family,
    inference_sample,
):
    path = LOGISTIC_ARTIFACT if family == "logistic" else CATBOOST_ARTIFACT
    model = _load(family, path)

    raw = model.predict_raw_recovery_probability(inference_sample)
    calibrated = model.predict_recovery_probability(inference_sample)

    assert model.calibrator is not None
    assert model.training_metadata["calibration_method"] == CALIBRATION_METHOD
    assert raw.shape == calibrated.shape == (8,)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    assert not np.allclose(raw, calibrated)


@pytest.mark.parametrize("family", FAMILIES)
def test_prediction_enforces_feature_contract(family, inference_sample):
    path = LOGISTIC_ARTIFACT if family == "logistic" else CATBOOST_ARTIFACT
    model = _load(family, path)
    incomplete = inference_sample.drop(columns=[MODEL_FEATURES[0]])

    with pytest.raises(ValueError, match="missing required columns"):
        model.predict_recovery_probability(incomplete)


@pytest.mark.parametrize("family", FAMILIES)
def test_missing_calibration_metadata_is_rejected(tmp_path, family):
    path = _mutated_artifact(tmp_path, family, lambda data: data.pop("calibration"))
    with pytest.raises(ModelArtifactValidationError, match="Calibration metadata"):
        _load(family, path)


@pytest.mark.parametrize("family", FAMILIES)
def test_missing_calibrator_is_rejected(tmp_path, family):
    path = _mutated_artifact(
        tmp_path,
        family,
        lambda data: data.__setitem__("calibration", None),
    )
    with pytest.raises(ModelArtifactValidationError, match="Calibration metadata"):
        _load(family, path)


@pytest.mark.parametrize("family", FAMILIES)
def test_missing_calibrator_parameters_are_rejected(tmp_path, family):
    def mutate(data):
        data["calibration"].pop("coefficient")

    path = _mutated_artifact(tmp_path, family, mutate)
    with pytest.raises(ModelArtifactValidationError, match="Malformed calibration"):
        _load(family, path)


@pytest.mark.parametrize("family", FAMILIES)
def test_inconsistent_calibration_metadata_is_rejected(tmp_path, family):
    def mutate(data):
        data["training_metadata"]["calibration_method"] = "isotonic"

    path = _mutated_artifact(tmp_path, family, mutate)
    with pytest.raises(ModelArtifactValidationError, match="calibration method"):
        _load(family, path)


@pytest.mark.parametrize("model_class", (LogisticRecoveryBaseline, CatBoostRecoveryModel))
def test_canonical_prediction_never_falls_back_to_raw(model_class, inference_sample):
    model = model_class()
    model._is_fitted = True
    model.calibrator = None

    with pytest.raises(ModelArtifactValidationError, match="requires a fitted.*calibrator"):
        model.predict_recovery_probability(inference_sample)


@pytest.mark.parametrize("family", FAMILIES)
def test_missing_training_metadata_is_rejected(tmp_path, family):
    path = _mutated_artifact(
        tmp_path, family, lambda data: data.pop("training_metadata")
    )
    with pytest.raises(ModelArtifactValidationError, match="Training metadata"):
        _load(family, path)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    ("key", "message"),
    (
        ("data_generation_reference", "data generation reference"),
        ("data_fingerprint_sha256", "dataset fingerprint"),
        ("grouping_strategy", "grouping strategy"),
        ("split_seed", "split seed"),
        ("grouping_column", "grouping column"),
    ),
)
def test_missing_reproducibility_metadata_is_rejected(
    tmp_path,
    family,
    key,
    message,
):
    def mutate(data):
        data["training_metadata"].pop(key)

    path = _mutated_artifact(tmp_path, family, mutate)
    with pytest.raises(ModelArtifactValidationError, match=message):
        _load(family, path)


@pytest.mark.parametrize("family", FAMILIES)
def test_incomplete_split_metadata_is_rejected(tmp_path, family):
    def mutate(data):
        data["training_metadata"]["partition_rows"].pop("validation")

    path = _mutated_artifact(tmp_path, family, mutate)
    with pytest.raises(ModelArtifactValidationError, match="Incomplete partition_rows"):
        _load(family, path)


@pytest.mark.parametrize("family", FAMILIES)
def test_forbidden_identity_overlap_metadata_is_rejected(tmp_path, family):
    def mutate(data):
        data["training_metadata"]["identity_overlaps"]["customer_id"][
            "train_test"
        ] = 1

    path = _mutated_artifact(tmp_path, family, mutate)
    with pytest.raises(ModelArtifactValidationError, match="forbidden customer_id overlap"):
        _load(family, path)


@pytest.mark.parametrize("family", FAMILIES)
def test_unsupported_artifact_version_is_rejected(tmp_path, family):
    path = _mutated_artifact(
        tmp_path,
        family,
        lambda data: data.__setitem__("model_format_version", 999),
    )
    with pytest.raises(ModelArtifactValidationError, match="Unsupported artifact format"):
        _load(family, path)


@pytest.mark.parametrize("family", FAMILIES)
def test_wrong_model_kind_is_rejected(tmp_path, family):
    path = _mutated_artifact(
        tmp_path,
        family,
        lambda data: data.__setitem__("model_kind", "wrong_model_family"),
    )
    with pytest.raises(ModelArtifactValidationError, match="Wrong model kind"):
        _load(family, path)


@pytest.mark.parametrize("family", FAMILIES)
def test_feature_contract_mismatch_is_rejected(tmp_path, family):
    def mutate(data):
        data["model_features"] = data["model_features"][:-1]

    path = _mutated_artifact(tmp_path, family, mutate)
    with pytest.raises(ModelArtifactValidationError, match="feature contract mismatch"):
        _load(family, path)


@pytest.mark.parametrize("family", FAMILIES)
def test_malformed_training_metadata_is_rejected(tmp_path, family):
    path = _mutated_artifact(
        tmp_path,
        family,
        lambda data: data.__setitem__("training_metadata", []),
    )
    with pytest.raises(ModelArtifactValidationError, match="must be a dictionary"):
        _load(family, path)


@pytest.mark.parametrize("family", FAMILIES)
def test_non_dictionary_top_level_metadata_is_rejected(tmp_path, family):
    if family == "logistic":
        path = tmp_path / "logistic.joblib"
        joblib.dump([], path)
    else:
        path = tmp_path / "catboost.cbm"
        shutil.copy2(CATBOOST_ARTIFACT, path)
        path.with_suffix(".meta.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ModelArtifactValidationError, match="must be a dictionary"):
        _load(family, path)


@pytest.mark.parametrize(
    ("family", "field", "message"),
    (
        ("logistic", "pipeline", "Missing Logistic model metadata"),
        ("catboost", "iterations", "Missing CatBoost model metadata"),
    ),
)
def test_missing_model_family_metadata_is_rejected(tmp_path, family, field, message):
    path = _mutated_artifact(tmp_path, family, lambda data: data.pop(field))
    with pytest.raises(ModelArtifactValidationError, match=message):
        _load(family, path)


def test_catboost_categorical_metadata_mismatch_is_rejected(tmp_path):
    def mutate(data):
        data["categorical_features"] = data["categorical_features"][:-1]

    path = _mutated_artifact(tmp_path, "catboost", mutate)
    with pytest.raises(ModelArtifactValidationError, match="categorical feature"):
        CatBoostRecoveryModel.load(path)


@pytest.fixture(scope="module")
def selection_context():
    population = generate_synthetic_population(
        merchant_count=4,
        customers_per_merchant=25,
        seed=1200,
    )
    batch = generate_recovery_cases(
        population,
        25,
        seed=1201,
        reference_time=population.reference_time,
    )
    scenario = next(item for item in batch.scenarios if item.case.customer_id)
    recovery_case = scenario.case
    merchant = next(item for item in population.merchants if item.id == recovery_case.merchant_id)
    customer = next(item for item in population.customers if item.id == recovery_case.customer_id)
    diagnosis = diagnose_case(recovery_case)
    candidates = generate_candidate_actions(recovery_case, diagnosis)
    source = RecoverySourceContext(
        bank=scenario.payment.bank if scenario.payment else None,
        payment_attempt_number=(scenario.payment.attempt_number if scenario.payment else None),
        subscription_retry_count=(
            scenario.subscription.retry_count if scenario.subscription else None
        ),
        mandate_active=(scenario.subscription.mandate_active if scenario.subscription else None),
        invoice_days_overdue=scenario.invoice.days_overdue if scenario.invoice else None,
    )
    return {
        "recovery_case": recovery_case,
        "customer": customer,
        "diagnosis": diagnosis,
        "candidate_actions": candidates.actions,
        "merchant": Merchant(
            merchant_id=merchant.id,
            archetype=merchant.archetype.value,
            average_order_value=merchant.average_order_value,
        ),
        "merchant_policy": merchant.policy,
        "policy_context": PolicyContext(
            now=population.reference_time,
            customer_do_not_contact=customer.do_not_contact,
            action_history=[],
        ),
        "source_context": source,
    }


@pytest.mark.parametrize("family", FAMILIES)
def test_layer13_receives_calibrated_canonical_probability(family, selection_context):
    path = LOGISTIC_ARTIFACT if family == "logistic" else CATBOOST_ARTIFACT
    model = _load(family, path)

    class RecordingPredictor:
        def predict_recovery_probability(self, dataframe):
            self.action_types = list(dataframe["action_type"])
            self.raw = model.predict_raw_recovery_probability(dataframe)
            self.calibrated = model.predict_recovery_probability(dataframe)
            return self.calibrated

    predictor = RecordingPredictor()
    result = select_best_recovery_action(
        **selection_context,
        probability_model=predictor,
    )

    selected_type = result.selected_score.action_type.value
    selected_index = predictor.action_types.index(selected_type)
    assert result.selected_score.predicted_recovery_probability == pytest.approx(
        predictor.calibrated[selected_index]
    )
    assert result.selected_score.predicted_recovery_probability != pytest.approx(
        predictor.raw[selected_index]
    )
