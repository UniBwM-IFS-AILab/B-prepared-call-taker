from __future__ import annotations

from pathlib import Path

from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency
from experiments.nlu_eval.common.constants import NON_MEDICAL_FIELDS
from experiments.nlu_eval.common.models import PredictionRecord, RunConfig


def test_run_config_parses_paths() -> None:
    config = RunConfig.model_validate(
        {
            "experiment_id": "experiment1",
            "dataset_path": "experiments/nlu_eval/datasets/pilot_master.jsonl",
        }
    )

    assert isinstance(config.dataset_path, Path)


def test_non_medical_fields_are_derived_from_emergency_call_schema() -> None:
    assert NON_MEDICAL_FIELDS == tuple(
        sorted(set(EmergencyCall.model_fields) - set(MedicalEmergency.model_fields))
    )


def test_prediction_record_defaults_sparse_states_to_empty_dict() -> None:
    record = PredictionRecord(
        experiment_id="experiment2",
        system_id="full_agent",
        item_id="item-1",
        repeat_index=0,
        response_kind="state",
    )

    assert record.predicted_medical_state == {}
    assert record.predicted_non_medical_state == {}
