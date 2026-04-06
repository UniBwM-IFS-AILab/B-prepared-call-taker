from __future__ import annotations

import pandas as pd
import pytest

from experiments.nlu_eval.common.constants import MEDICAL_FIELDS
from experiments.nlu_eval.common.metrics import slot_metrics
from experiments.nlu_eval.common.models import PredictionRecord
from experiments.nlu_eval.experiment1_metrics import experiment1_summary
from experiments.nlu_eval.experiment2_metrics import experiment2_summary


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

    assert summary["full_agent"]["subset_accuracy"] == 0.0
    assert summary["full_agent"]["final_accuracy"] == 0.0
    assert summary["full_agent"]["hamming_loss"] == 1 / 3
    assert summary["full_agent"]["jaccard"] == 0.0
    assert summary["full_agent"]["precision"] == 0.0
    assert summary["full_agent"]["recall"] == 0.0
    assert summary["full_agent"]["f1"] == 0.0
    assert summary["outcome_baseline"]["subset_accuracy"] == 1.0
    assert summary["outcome_baseline"]["final_accuracy"] == 1.0
    assert summary["outcome_baseline"]["hamming_loss"] == 0.0
    assert summary["outcome_baseline"]["jaccard"] == 1.0
    assert summary["outcome_baseline"]["precision"] == 1.0
    assert summary["outcome_baseline"]["recall"] == 1.0
    assert summary["outcome_baseline"]["f1"] == 1.0
    assert "comparison" not in summary


def test_experiment1_summary_reports_subset_and_final_accuracy() -> None:
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

    assert summary["full_agent"]["subset_accuracy"] == 1 / 3
    assert summary["full_agent"]["final_accuracy"] == 1.0
    assert summary["full_agent"]["hamming_loss"] == 1 / 3
    assert summary["full_agent"]["jaccard"] == pytest.approx(11 / 18)
    assert summary["outcome_baseline"]["subset_accuracy"] == 2 / 3
    assert summary["outcome_baseline"]["final_accuracy"] == 2 / 3
    assert summary["outcome_baseline"]["hamming_loss"] == pytest.approx(1 / 9)
    assert summary["outcome_baseline"]["jaccard"] == pytest.approx(5 / 6)

    for system_id in ("full_agent", "outcome_baseline"):
        for metric_name in ("precision", "recall", "f1"):
            assert 0.0 <= summary[system_id][metric_name] <= 1.0


def test_slot_metrics_report_prf_metrics() -> None:
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

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["hamming_loss"] == 1 / (2 * len(MEDICAL_FIELDS))
    assert metrics["jaccard"] == 0.5


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

    assert summary["accuracy"] == 1.0
    assert summary["hamming_loss"] == 0.0
    assert summary["jaccard"] == 1.0


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

    assert summary["accuracy"] == 0.5
    assert summary["per_slot_count"] == {
        "1": {
            "item_count": 1,
            "accuracy": 0.0,
            "hamming_loss": 1 / len(MEDICAL_FIELDS),
            "jaccard": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        "2": {
            "item_count": 1,
            "accuracy": 1.0,
            "hamming_loss": 0.0,
            "jaccard": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
    }
    assert "non_medical_exact_match_accuracy" not in summary["per_slot_count"]["1"]
    assert "followup_rate" not in summary["per_slot_count"]["1"]
    assert "parse_failure_rate" not in summary["per_slot_count"]["1"]
    assert "repeat_instability_rate" not in summary["per_slot_count"]["1"]


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

    assert summary["accuracy"] == 0.0
    assert summary["hamming_loss"] == 2 / len(MEDICAL_FIELDS)
    assert summary["jaccard"] == 1 / 3


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
