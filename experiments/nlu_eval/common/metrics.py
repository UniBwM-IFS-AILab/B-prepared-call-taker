"""Shared metric primitives for the NLU benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
)

from experiments.nlu_eval.common.constants import MEDICAL_FIELDS

MEDICAL_GT_COLUMNS: list[str] = [f"gt_{field}" for field in MEDICAL_FIELDS]
MEDICAL_PRED_COLUMNS: list[str] = [f"pred_{field}" for field in MEDICAL_FIELDS]
PRF_METRIC_NAMES: tuple[str, ...] = ("precision", "recall", "f1")


def zero_prf_metrics() -> dict[str, float]:
    return {metric: 0.0 for metric in PRF_METRIC_NAMES}


def prf_metrics(
    *, y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    if labels.size == 0:
        return zero_prf_metrics()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="samples",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def expand_medical_state(states: pd.Series, *, prefix: str = "") -> pd.DataFrame:
    return (
        pd.DataFrame.from_records(states, index=states.index)
        .reindex(columns=MEDICAL_FIELDS, fill_value=False)
        .eq(True)
        .astype(int)
        .rename(columns=lambda field: f"{prefix}{field}")
    )


def medical_label_frame(
    *,
    records: pd.DataFrame,
    items: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    items_with_ids = items.assign(item_id=items["id"])
    item_medical = expand_medical_state(items_with_ids["gt_medical_state"])
    support_by_field = item_medical.sum(axis=0).astype(int)
    item_columns = [column for column in items_with_ids.columns if column != "id"]
    merged = records.merge(
        items_with_ids[item_columns],
        on="item_id",
        how="left",
        validate="many_to_one",
    )
    return (
        pd.concat(
            [
                merged,
                expand_medical_state(merged["gt_medical_state"], prefix="gt_"),
                expand_medical_state(merged["predicted_medical_state"], prefix="pred_"),
            ],
            axis=1,
        ),
        support_by_field,
    )


def summarize_slot_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    support_by_field: pd.Series,
) -> dict[str, Any]:
    scored_indices = np.flatnonzero(
        (support_by_field.to_numpy() > 0) | (y_pred.sum(axis=0) > 0)
    )
    averaged_prf = prf_metrics(y_true=y_true, y_pred=y_pred, labels=scored_indices)

    active_fields = support_by_field.loc[support_by_field.gt(0)]
    active_indices = np.flatnonzero(support_by_field.to_numpy() > 0)
    if active_indices.size == 0:
        return {**averaged_prf, "per_slot": []}

    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=active_indices,
        average=None,
        zero_division=0,
    )

    per_slot = pd.DataFrame(
        {
            "field": active_fields.index,
            "support": active_fields.to_numpy(dtype=int),
            "precision": np.asarray(precisions, dtype=float),
            "recall": np.asarray(recalls, dtype=float),
            "f1": np.asarray(f1s, dtype=float),
        }
    )
    return {
        **averaged_prf,
        "per_slot": per_slot.to_dict(orient="records"),
    }


def slot_metrics(*, records: pd.DataFrame, items: pd.DataFrame) -> dict[str, Any]:
    label_frame, support_by_field = medical_label_frame(records=records, items=items)
    y_true = label_frame[MEDICAL_GT_COLUMNS].to_numpy(dtype=int)
    y_pred = label_frame[MEDICAL_PRED_COLUMNS].to_numpy(dtype=int)
    slot_summary = summarize_slot_metrics(
        y_true=y_true,
        y_pred=y_pred,
        support_by_field=support_by_field,
    )
    return {
        **slot_summary,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard": float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=1.0)
        ),
    }
