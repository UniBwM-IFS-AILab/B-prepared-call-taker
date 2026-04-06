"""Experiment 2 metric computation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score

from experiments.nlu_eval.common.constants import MEDICAL_FIELDS
from experiments.nlu_eval.common.metrics import (
    MEDICAL_GT_COLUMNS,
    MEDICAL_PRED_COLUMNS,
    expand_medical_state,
    medical_label_frame,
    summarize_slot_metrics,
    zero_prf_metrics,
)


def _core_slot_summary(
    *,
    y_true,
    y_pred,
    support_by_field: pd.Series,
) -> dict[str, float]:
    slot_summary = summarize_slot_metrics(
        y_true=y_true,
        y_pred=y_pred,
        support_by_field=support_by_field,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard": float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=1.0)
        ),
        **{key: slot_summary[key] for key in zero_prf_metrics()},
    }


def _overall_summary(
    *,
    group_frame: pd.DataFrame,
    support_by_field: pd.Series,
) -> dict[str, float | None]:
    y_true = group_frame[MEDICAL_GT_COLUMNS].to_numpy(dtype=int)
    y_pred = group_frame[MEDICAL_PRED_COLUMNS].to_numpy(dtype=int)
    summary: dict[str, float | None] = _core_slot_summary(
        y_true=y_true,
        y_pred=y_pred,
        support_by_field=support_by_field,
    )

    non_medical_means = (
        group_frame.loc[group_frame["has_non_medical"]]
        .groupby("item_id", sort=False)["non_medical_exact_match"]
        .mean()
    )
    return {
        **summary,
        "non_medical_exact_match_accuracy": (
            float(non_medical_means.mean()) if not non_medical_means.empty else None
        ),
        "followup_rate": float(group_frame["response_kind"].eq("followup").mean()),
        "parse_failure_rate": float(
            group_frame["response_kind"].eq("parse_error").mean()
        ),
        "repeat_instability_rate": float(
            group_frame.groupby("item_id", sort=False)["signature"]
            .nunique()
            .gt(1)
            .mean()
        ),
    }


def experiment2_summary(
    *,
    records: pd.DataFrame,
    items: pd.DataFrame,
) -> dict[str, Any]:
    if records.empty:
        raise ValueError("Experiment 2 requires prediction records")

    items_enriched = items.assign(
        item_id=items["id"],
        has_non_medical=items["gt_non_medical_state"].map(bool),
    )
    item_medical = expand_medical_state(items_enriched["gt_medical_state"])
    items_enriched = items_enriched.assign(
        gt_positive_slot_count=item_medical.sum(axis=1).astype(int),
    )
    slot_count_item_counts = (
        items_enriched["gt_positive_slot_count"].value_counts(sort=False).to_dict()
    )

    label_frame, support_by_field = medical_label_frame(
        records=records, items=items_enriched
    )
    label_frame = label_frame.assign(
        non_medical_exact_match=label_frame["predicted_non_medical_state"]
        .eq(label_frame["gt_non_medical_state"])
        .astype(int),
        signature=list(
            zip(
                label_frame["response_kind"],
                label_frame["predicted_medical_state"].map(
                    lambda state: tuple(sorted(state.items()))
                ),
                label_frame["predicted_non_medical_state"].map(
                    lambda state: tuple(sorted(state.items()))
                ),
                label_frame["predicted_outcome"],
                label_frame["followup_text"],
            )
        ),
    )

    summary_metrics = _overall_summary(
        group_frame=label_frame,
        support_by_field=support_by_field,
    )
    overall_slot_metrics = summarize_slot_metrics(
        y_true=label_frame[MEDICAL_GT_COLUMNS].to_numpy(dtype=int),
        y_pred=label_frame[MEDICAL_PRED_COLUMNS].to_numpy(dtype=int),
        support_by_field=support_by_field,
    )

    item_support_by_slot_count = item_medical.groupby(
        items_enriched["gt_positive_slot_count"],
        sort=True,
    ).sum()
    per_slot_count: dict[str, dict[str, float]] = {}
    for slot_count, support in item_support_by_slot_count.sort_index().iterrows():
        group = label_frame[label_frame["gt_positive_slot_count"] == slot_count]
        per_slot_count[str(int(slot_count))] = {
            "item_count": int(slot_count_item_counts[slot_count]),
            **_core_slot_summary(
                y_true=group[MEDICAL_GT_COLUMNS].to_numpy(dtype=int),
                y_pred=group[MEDICAL_PRED_COLUMNS].to_numpy(dtype=int),
                support_by_field=support.astype(int),
            ),
        }

    first_predictions = label_frame.drop_duplicates(
        subset=["item_id"], keep="first"
    ).set_index("item_id")
    gt_item = first_predictions[MEDICAL_GT_COLUMNS].set_axis(MEDICAL_FIELDS, axis=1)
    pred_item = first_predictions[MEDICAL_PRED_COLUMNS].set_axis(MEDICAL_FIELDS, axis=1)
    extra_matrix = pred_item.gt(gt_item)
    missing_matrix = gt_item.gt(pred_item)
    extra_counts = extra_matrix.sum(axis=1)
    missing_counts = missing_matrix.sum(axis=1)

    extra_matrix.columns.name = "extra_field"
    missing_matrix.columns.name = "missing_field"
    extra_fields = (
        extra_matrix.stack()
        .loc[lambda series: series]
        .rename("is_extra")
        .reset_index()[["item_id", "extra_field"]]
    )
    missing_fields = (
        missing_matrix.stack()
        .loc[lambda series: series]
        .rename("is_missing")
        .reset_index()[["item_id", "missing_field"]]
    )
    confusion_pairs = (
        missing_fields.merge(extra_fields, on="item_id", how="inner")
        .groupby(["missing_field", "extra_field"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(
            ["count", "missing_field", "extra_field"], ascending=[False, True, True]
        )
        .reset_index(drop=True)
    )

    return {
        **summary_metrics,
        "per_slot_count": per_slot_count,
        "per_slot": overall_slot_metrics["per_slot"],
        "error_analysis": {
            "item_pattern_counts": {
                "exact_match_items": int(
                    (extra_counts.eq(0) & missing_counts.eq(0)).sum()
                ),
                "extra_only_items": int(
                    (extra_counts.gt(0) & missing_counts.eq(0)).sum()
                ),
                "missing_only_items": int(
                    (missing_counts.gt(0) & extra_counts.eq(0)).sum()
                ),
                "mixed_error_items": int(
                    (extra_counts.gt(0) & missing_counts.gt(0)).sum()
                ),
            },
            "false_positive_fields": (
                extra_fields.groupby("extra_field", as_index=False)
                .size()
                .rename(columns={"extra_field": "field", "size": "count"})
                .sort_values(["count", "field"], ascending=[False, True])
                .reset_index(drop=True)
                .to_dict(orient="records")
            ),
            "false_negative_fields": (
                missing_fields.groupby("missing_field", as_index=False)
                .size()
                .rename(columns={"missing_field": "field", "size": "count"})
                .sort_values(["count", "field"], ascending=[False, True])
                .reset_index(drop=True)
                .to_dict(orient="records")
            ),
            "confusion_pairs": confusion_pairs.to_dict(orient="records"),
        },
    }
