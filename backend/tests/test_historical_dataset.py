import csv
import json
from datetime import (
    datetime,
    timezone,
)

import pytest

from app.domain.enums import (
    PolicyDecision,
)
from simulator.customers import (
    generate_synthetic_population,
)
from simulator.historical import (
    HistoricalRecoveryRecord,
    generate_historical_dataset,
    write_historical_dataset_csv,
)


REFERENCE_TIME = datetime(
    2026,
    8,
    23,
    0,
    0,
    tzinfo=timezone.utc,
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture(
    scope="module"
)
def population():

    return (
        generate_synthetic_population(
            merchant_count=8,

            customers_per_merchant=40,

            seed=700,

            reference_time=(
                REFERENCE_TIME
            ),
        )
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================


def test_historical_dataset_is_reproducible(
    population,
):

    first = (
        generate_historical_dataset(
            population,
            300,

            case_seed=701,

            environment_seed=702,
        )
    )

    second = (
        generate_historical_dataset(
            population,
            300,

            case_seed=701,

            environment_seed=702,
        )
    )

    assert (
        first
        == second
    )


def test_different_environment_seed_changes_outcomes(
    population,
):

    first = (
        generate_historical_dataset(
            population,
            300,

            case_seed=710,

            environment_seed=711,
        )
    )

    second = (
        generate_historical_dataset(
            population,
            300,

            case_seed=710,

            environment_seed=999,
        )
    )

    first_labels = [
        record.recovered
        for record
        in first.records
    ]

    second_labels = [
        record.recovered
        for record
        in second.records
    ]

    assert (
        first_labels
        != second_labels
    )


# ============================================================
# NO SIMULATOR LEAKAGE
# ============================================================


def test_hidden_simulator_ground_truth_is_not_in_record_schema():

    fields = set(
        HistoricalRecoveryRecord
        .model_fields
    )

    forbidden = {
        "expected_failure_class",
        "latent_recovery_probability",
        "random_draw",
    }

    assert (
        fields.isdisjoint(
            forbidden
        )
    )


# ============================================================
# IDS + COUNTS
# ============================================================


def test_each_written_record_has_stable_unique_ids(
    population,
):

    dataset = (
        generate_historical_dataset(
            population,
            500,

            case_seed=720,

            environment_seed=721,
        )
    )

    history_ids = [
        record.history_id
        for record
        in dataset.records
    ]

    case_ids = [
        record.case_id
        for record
        in dataset.records
    ]

    assert (
        len(history_ids)
        == len(
            set(history_ids)
        )
    )

    assert (
        len(case_ids)
        == len(
            set(case_ids)
        )
    )

    assert (
        dataset.record_count
        + dataset.skipped_case_count
        == 500
    )


# ============================================================
# POLICY
# ============================================================


def test_only_executable_or_validly_deferred_actions_become_history(
    population,
):

    dataset = (
        generate_historical_dataset(
            population,
            500,

            case_seed=730,

            environment_seed=731,
        )
    )

    assert (
        dataset.records
    )

    for record in (
        dataset.records
    ):

        assert (
            record
            .policy_decision_at_selection
            in {
                PolicyDecision.ALLOWED,
                PolicyDecision.DEFERRED,
            }
        )

        if (
            record
            .policy_decision_at_selection
            == PolicyDecision.ALLOWED
        ):

            assert (
                record.was_deferred
                is False
            )

            assert (
                record.action_delay_hours
                == pytest.approx(
                    0.0
                )
            )

        else:

            assert (
                record.was_deferred
                is True
            )

            assert (
                record.action_delay_hours
                >= 0.0
            )


# ============================================================
# LABEL CONSISTENCY
# ============================================================


def test_primary_binary_label_matches_recovered_amount(
    population,
):

    dataset = (
        generate_historical_dataset(
            population,
            600,

            case_seed=740,

            environment_seed=741,
        )
    )

    for record in (
        dataset.records
    ):

        if (
            record.recovered
            == 1
        ):

            assert (
                record.recovered_amount
                > 0
            )

            assert (
                record.recovered_fraction
                > 0
            )

        else:

            assert (
                record.recovered_amount
                == pytest.approx(
                    0.0
                )
            )

            assert (
                record.recovered_fraction
                == pytest.approx(
                    0.0
                )
            )

        if (
            record.fully_recovered
            == 1
        ):

            assert (
                record.recovered
                == 1
            )

            assert (
                record.recovered_fraction
                == pytest.approx(
                    1.0
                )
            )


# ============================================================
# ML DATA VARIETY
# ============================================================


def test_dataset_has_action_and_label_variety_for_ml(
    population,
):

    dataset = (
        generate_historical_dataset(
            population,
            1200,

            case_seed=750,

            environment_seed=751,
        )
    )

    actions = {
        record.action_type
        for record
        in dataset.records
    }

    labels = {
        record.recovered
        for record
        in dataset.records
    }

    assert (
        len(actions)
        >= 5
    )

    assert (
        labels
        == {
            0,
            1,
        }
    )


# ============================================================
# CSV EXPORT
# ============================================================


def test_csv_writer_creates_training_file_and_metadata(
    population,
    tmp_path,
):

    dataset = (
        generate_historical_dataset(
            population,
            100,

            case_seed=760,

            environment_seed=761,
        )
    )

    output_path = (
        tmp_path
        / "recovery_history.csv"
    )

    returned = (
        write_historical_dataset_csv(
            dataset,
            output_path,
        )
    )

    assert (
        returned
        == output_path
    )

    assert (
        output_path.exists()
    )

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        rows = list(
            csv.DictReader(
                handle
            )
        )

    assert (
        len(rows)
        == dataset.record_count
    )

    assert (
        "recovered"
        in rows[0]
    )

    assert (
        "action_type"
        in rows[0]
    )

    assert (
        "latent_recovery_probability"
        not in rows[0]
    )

    assert (
        "random_draw"
        not in rows[0]
    )

    assert (
        "expected_failure_class"
        not in rows[0]
    )

    # --------------------------------------------------------
    # Metadata sidecar
    # --------------------------------------------------------

    metadata_path = (
        output_path
        .with_suffix(
            ".meta.json"
        )
    )

    assert (
        metadata_path.exists()
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        metadata[
            "record_count"
        ]
        == dataset.record_count
    )

    assert (
        metadata[
            "primary_target"
        ]
        == "recovered"
    )

    assert (
        "latent_recovery_probability"
        in metadata[
            "leakage_excluded"
        ]
    )