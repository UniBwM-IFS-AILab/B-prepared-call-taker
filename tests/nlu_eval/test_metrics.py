from __future__ import annotations

import pandas as pd
import pytest

from experiments.nlu_eval.common.constants import MEDICAL_FIELDS
from experiments.nlu_eval.common.metrics import experiment1_summary, experiment2_summary, slot_metrics
from experiments.nlu_eval.common.models import PredictionRecord


def test_experiment1_summary_treats_tied_majority_as_none() -> None:
    items = pd.DataFrame(
        [
            {
                "id": "item-1",
                "operator_question": "What has just happened acutely?",
                "caller_utterance": "She suddenly cannot speak properly.",
                "gt_medical_state": {"acute_speech_disorder": True},
                "gt_non_medical_state": {},
            }
        ]
    )
    records = pd.DataFrame(
        [
            PredictionRecord(
                experiment_id="experiment1",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_outcome=None,
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment1",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=1,
                response_kind="state",
                predicted_outcome="rd1",
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment1",
                system_id="outcome_baseline",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_outcome="rd1",
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment1",
                system_id="outcome_baseline",
                item_id="item-1",
                repeat_index=1,
                response_kind="state",
                predicted_outcome="rd1",
                latency_ms=1.0,
            ).model_dump(),
        ]
    )

    summary = experiment1_summary(records=records, items=items)

    assert summary["full_agent"]["outcome_subset_accuracy"] == 0.0
    assert summary["full_agent"]["final_outcome_accuracy"] == 0.0
    assert summary["full_agent"]["outcome_hamming_loss"] == 1 / 3
    assert summary["full_agent"]["outcome_jaccard"] == 0.0
    assert summary["outcome_baseline"]["outcome_subset_accuracy"] == 1.0
    assert summary["outcome_baseline"]["final_outcome_accuracy"] == 1.0
    assert summary["outcome_baseline"]["outcome_hamming_loss"] == 0.0
    assert summary["outcome_baseline"]["outcome_jaccard"] == 1.0
    assert "comparison" not in summary


def test_experiment1_summary_reports_subset_and_final_outcome_accuracy() -> None:
    items = pd.DataFrame(
        [
            {
                "id": "item-1",
                "operator_question": "What has just happened acutely?",
                "caller_utterance": "She suddenly cannot speak properly.",
                "gt_medical_state": {"acute_speech_disorder": True},
                "gt_non_medical_state": {},
            },
            {
                "id": "item-2",
                "operator_question": "What has just happened acutely?",
                "caller_utterance": "He is blue and deteriorating quickly.",
                "gt_medical_state": {"cyanosis": True},
                "gt_non_medical_state": {},
            },
            {
                "id": "item-3",
                "operator_question": "What has just happened acutely?",
                "caller_utterance": "He is in cardiac arrest.",
                "gt_medical_state": {"cardiac_arrest": True},
                "gt_non_medical_state": {},
            },
        ]
    )
    records = pd.DataFrame(
        [
            PredictionRecord(
                experiment_id="experiment1",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_outcome="rd1",
                raw_output={"rd1": True},
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment1",
                system_id="outcome_baseline",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_outcome="rd1",
                raw_output={"rd1": True},
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment1",
                system_id="full_agent",
                item_id="item-2",
                repeat_index=0,
                response_kind="state",
                predicted_outcome="rd2",
                raw_output={"rd2": True},
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment1",
                system_id="outcome_baseline",
                item_id="item-2",
                repeat_index=0,
                response_kind="state",
                predicted_outcome="rd1",
                raw_output={"rd1": True},
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment1",
                system_id="full_agent",
                item_id="item-3",
                repeat_index=0,
                response_kind="state",
                predicted_outcome="cpr",
                raw_output={"cpr_needed": True},
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment1",
                system_id="outcome_baseline",
                item_id="item-3",
                repeat_index=0,
                response_kind="state",
                predicted_outcome="cpr",
                raw_output={"rd1": True, "rd2": True, "cpr_needed": True},
                latency_ms=1.0,
            ).model_dump(),
        ]
    )

    summary = experiment1_summary(records=records, items=items)

    assert summary["full_agent"]["outcome_subset_accuracy"] == 1 / 3
    assert summary["full_agent"]["final_outcome_accuracy"] == 1.0
    assert summary["full_agent"]["outcome_hamming_loss"] == 1 / 3
    assert summary["full_agent"]["outcome_jaccard"] == pytest.approx(11 / 18)
    assert summary["outcome_baseline"]["outcome_subset_accuracy"] == 2 / 3
    assert summary["outcome_baseline"]["final_outcome_accuracy"] == 2 / 3
    assert summary["outcome_baseline"]["outcome_hamming_loss"] == pytest.approx(1 / 9)
    assert summary["outcome_baseline"]["outcome_jaccard"] == pytest.approx(5 / 6)


def test_slot_metrics_weight_slots_equally() -> None:
    items = pd.DataFrame(
        [
            {
                "id": "item-1",
                "operator_question": "What is happening?",
                "caller_utterance": "She cannot speak properly.",
                "gt_medical_state": {"acute_speech_disorder": True},
                "gt_non_medical_state": {},
            },
            {
                "id": "item-2",
                "operator_question": "How is he breathing?",
                "caller_utterance": "He has a harsh breathing noise.",
                "gt_medical_state": {"stridor": True},
                "gt_non_medical_state": {},
            },
        ]
    )
    records = pd.DataFrame(
        [
            PredictionRecord(
                experiment_id="experiment2",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_medical_state={"acute_speech_disorder": True},
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment2",
                system_id="full_agent",
                item_id="item-2",
                repeat_index=0,
                response_kind="state",
                predicted_medical_state={},
                latency_ms=1.0,
            ).model_dump(),
        ]
    )

    metrics = slot_metrics(records=records, items=items)

    assert metrics["macro_precision"] == 0.5
    assert metrics["macro_recall"] == 0.5
    assert metrics["macro_f1"] == 0.5
    expected_accuracy = ((len(MEDICAL_FIELDS) - 1) + len(MEDICAL_FIELDS)) / (2 * len(MEDICAL_FIELDS))
    assert metrics["medical_slot_accuracy"] == expected_accuracy


def test_experiment2_summary_ignores_false_for_medical_slot_set_metrics() -> None:
    items = pd.DataFrame(
        [
            {
                "id": "item-1",
                "operator_question": "What is happening?",
                "caller_utterance": "He says there is pressure but not pain.",
                "gt_medical_state": {"acute_chest_pain": False},
                "gt_non_medical_state": {},
            }
        ]
    )
    records = pd.DataFrame(
        [
            PredictionRecord(
                experiment_id="experiment2",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_medical_state={},
                latency_ms=1.0,
            ).model_dump()
        ]
    )

    summary = experiment2_summary(records=records, items=items)

    assert summary["medical_slot_accuracy"] == 1.0
    assert summary["medical_hamming_loss"] == 0.0
    assert summary["medical_subset_accuracy"] == 1.0
    assert summary["medical_slot_jaccard"] == 1.0


def test_experiment2_summary_reports_accuracy_by_slot_count() -> None:
    items = pd.DataFrame(
        [
            {
                "id": "item-1",
                "operator_question": "What is happening?",
                "caller_utterance": "He cannot speak and cannot move his arm.",
                "gt_medical_state": {
                    "acute_speech_disorder": True,
                    "acute_paralysis": True,
                },
                "gt_non_medical_state": {},
            },
            {
                "id": "item-2",
                "operator_question": "What is happening?",
                "caller_utterance": "She has severe chest pain.",
                "gt_medical_state": {"acute_chest_pain": True},
                "gt_non_medical_state": {},
            },
        ]
    )
    records = pd.DataFrame(
        [
            PredictionRecord(
                experiment_id="experiment2",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_medical_state={
                    "acute_speech_disorder": True,
                    "acute_paralysis": True,
                },
                predicted_outcome="rd1",
                latency_ms=1.0,
            ).model_dump(),
            PredictionRecord(
                experiment_id="experiment2",
                system_id="full_agent",
                item_id="item-2",
                repeat_index=0,
                response_kind="state",
                predicted_medical_state={},
                predicted_outcome=None,
                latency_ms=1.0,
            ).model_dump(),
        ]
    )

    summary = experiment2_summary(records=records, items=items)

    assert summary["medical_subset_accuracy"] == 0.5
    assert summary["per_slot_count"] == {
        "1": {
            "item_count": 1,
            "medical_slot_accuracy": (len(MEDICAL_FIELDS) - 1) / len(MEDICAL_FIELDS),
            "medical_hamming_loss": 1 / len(MEDICAL_FIELDS),
            "medical_subset_accuracy": 0.0,
            "medical_slot_jaccard": 0.0,
            "non_medical_exact_match_accuracy": None,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "followup_rate": 0.0,
            "parse_failure_rate": 0.0,
            "repeat_instability_rate": 0.0,
        },
        "2": {
            "item_count": 1,
            "medical_slot_accuracy": 1.0,
            "medical_hamming_loss": 0.0,
            "medical_subset_accuracy": 1.0,
            "medical_slot_jaccard": 1.0,
            "non_medical_exact_match_accuracy": None,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "followup_rate": 0.0,
            "parse_failure_rate": 0.0,
            "repeat_instability_rate": 0.0,
        },
    }


def test_experiment2_summary_reports_jaccard_as_partial_overlap() -> None:
    items = pd.DataFrame(
        [
            {
                "id": "item-1",
                "operator_question": "What is happening?",
                "caller_utterance": "He is not breathing and does not react.",
                "gt_medical_state": {"unconscious": True, "apnea": True},
                "gt_non_medical_state": {},
            }
        ]
    )
    records = pd.DataFrame(
        [
            PredictionRecord(
                experiment_id="experiment2",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_medical_state={"unconscious": True, "now_unresponsive": True},
                predicted_outcome="rd2",
                latency_ms=1.0,
            ).model_dump()
        ]
    )

    summary = experiment2_summary(records=records, items=items)

    assert summary["medical_subset_accuracy"] == 0.0
    assert summary["medical_hamming_loss"] == 2 / len(MEDICAL_FIELDS)
    assert summary["medical_slot_jaccard"] == 1 / 3


def test_experiment2_summary_reports_false_positive_and_negative_fields() -> None:
    items = pd.DataFrame(
        [
            {
                "id": "item-1",
                "operator_question": "What is happening?",
                "caller_utterance": "He is not breathing and does not react.",
                "gt_medical_state": {"unconscious": True, "apnea": True},
                "gt_non_medical_state": {},
            }
        ]
    )
    records = pd.DataFrame(
        [
            PredictionRecord(
                experiment_id="experiment2",
                system_id="full_agent",
                item_id="item-1",
                repeat_index=0,
                response_kind="state",
                predicted_medical_state={"unconscious": True, "now_unresponsive": True},
                predicted_outcome="rd2",
                latency_ms=1.0,
            ).model_dump()
        ]
    )

    summary = experiment2_summary(records=records, items=items)

    assert summary["error_analysis"]["item_pattern_counts"]["mixed_error_items"] == 1
    assert summary["error_analysis"]["false_positive_fields"] == [
        {"field": "now_unresponsive", "count": 1}
    ]
    assert summary["error_analysis"]["false_negative_fields"] == [
        {"field": "apnea", "count": 1}
    ]
    assert summary["error_analysis"]["confusion_pairs"] == [
        {"missing_field": "apnea", "extra_field": "now_unresponsive", "count": 1}
    ]
