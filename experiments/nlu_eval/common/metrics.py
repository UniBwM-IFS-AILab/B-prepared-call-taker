"""Metric computation for the NLU benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, precision_recall_fscore_support

from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency
from experiments.nlu_eval.common.constants import MEDICAL_FIELDS

MEDICAL_GT_COLUMNS: list[str] = [f"gt_{field}" for field in MEDICAL_FIELDS]
MEDICAL_PRED_COLUMNS: list[str] = [f"pred_{field}" for field in MEDICAL_FIELDS]


def expand_medical_state(states: pd.Series, *, prefix: str = "") -> pd.DataFrame:
    return (
        pd.DataFrame.from_records(
            states.map(lambda state: state if isinstance(state, dict) else {}),
            index=states.index,
        )
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
    items_with_ids = items if "item_id" in items.columns else items.assign(item_id=items["id"])
    item_medical = expand_medical_state(items_with_ids["gt_medical_state"])
    support_by_field = item_medical.sum(axis=0).astype(int)
    item_columns = [column for column in items_with_ids.columns if column not in MEDICAL_GT_COLUMNS]
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
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "medical_slot_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_slot": [],
        }

    medical_slot_accuracy = float(accuracy_score(y_true.ravel(), y_pred.ravel()))
    active_fields = support_by_field.loc[support_by_field.gt(0)]
    if active_fields.empty:
        return {
            "medical_slot_accuracy": medical_slot_accuracy,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_slot": [],
        }

    active_indices = np.flatnonzero(support_by_field.to_numpy() > 0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=active_indices,
        average="macro",
        zero_division=0.0,
    )
    precisions, recalls, f1s, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=active_indices,
        average=None,
        zero_division=0.0,
    )

    per_slot = pd.DataFrame(
        {
            "field": active_fields.index,
            "support": active_fields.to_numpy(dtype=int),
            "precision": precisions.astype(float),
            "recall": recalls.astype(float),
            "f1": f1s.astype(float),
        }
    )
    return {
        "medical_slot_accuracy": medical_slot_accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_slot": per_slot.to_dict(orient="records"),
    }


def slot_metrics(*, records: pd.DataFrame, items: pd.DataFrame) -> dict[str, Any]:
    if records.empty:
        return {
            "medical_slot_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_slot": [],
        }

    label_frame, support_by_field = medical_label_frame(records=records, items=items)
    return summarize_slot_metrics(
        y_true=label_frame[MEDICAL_GT_COLUMNS].to_numpy(dtype=int),
        y_pred=label_frame[MEDICAL_PRED_COLUMNS].to_numpy(dtype=int),
        support_by_field=support_by_field,
    )


def summarize_experiment2_group(
    *,
    group_frame: pd.DataFrame,
    support_by_field: pd.Series,
) -> dict[str, float | None]:
    if group_frame.empty:
        return {
            "medical_slot_accuracy": 0.0,
            "medical_hamming_loss": 0.0,
            "medical_subset_accuracy": 0.0,
            "medical_slot_jaccard": 0.0,
            "non_medical_exact_match_accuracy": None,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "followup_rate": 0.0,
            "parse_failure_rate": 0.0,
            "repeat_instability_rate": 0.0,
        }

    y_true = group_frame[MEDICAL_GT_COLUMNS].to_numpy(dtype=int)
    y_pred = group_frame[MEDICAL_PRED_COLUMNS].to_numpy(dtype=int)
    slot_summary = summarize_slot_metrics(
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
        "medical_slot_accuracy": slot_summary["medical_slot_accuracy"],
        "medical_hamming_loss": float(hamming_loss(y_true, y_pred)),
        "medical_subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "medical_slot_jaccard": float(jaccard_score(y_true, y_pred, average="samples", zero_division=1.0)),
        "non_medical_exact_match_accuracy": (
            float(non_medical_means.mean()) if not non_medical_means.empty else None
        ),
        "macro_precision": slot_summary["macro_precision"],
        "macro_recall": slot_summary["macro_recall"],
        "macro_f1": slot_summary["macro_f1"],
        "followup_rate": float(group_frame["response_kind"].eq("followup").mean()),
        "parse_failure_rate": float(group_frame["response_kind"].eq("parse_error").mean()),
        "repeat_instability_rate": float(
            group_frame.groupby("item_id", sort=False)["signature"].nunique().gt(1).mean()
        ),
    }


def experiment1_summary(
    *,
    records: pd.DataFrame,
    items: pd.DataFrame,
) -> dict[str, Any]:
    if records.empty:
        raise ValueError("Experiment 1 requires prediction records")

    items_with_outcomes = items.assign(
        item_id=items["id"],
        gt_final_outcome=items["gt_medical_state"].map(
            lambda state: MedicalEmergency(**state).get_outcome()
        ),
    ).assign(
        gt_rd1=lambda frame: frame["gt_final_outcome"].isin(["rd1", "rd2", "cpr"]).astype(int),
        gt_rd2=lambda frame: frame["gt_final_outcome"].isin(["rd2", "cpr"]).astype(int),
        gt_cpr=lambda frame: frame["gt_final_outcome"].eq("cpr").astype(int),
    )
    raw_output_frame = pd.DataFrame.from_records(
        records["raw_output"].map(lambda raw: raw if isinstance(raw, dict) else {}),
        index=records.index,
    )
    raw_outcome_flags = (
        raw_output_frame.reindex(columns=["rd1", "rd2", "cpr_needed"])
        .eq(True)
        .astype(int)
        .rename(columns={"cpr_needed": "cpr"})
    )
    fallback_outcome_flags = pd.DataFrame(
        {
            "rd1": records["predicted_outcome"].isin(["rd1", "rd2", "cpr"]).astype(int),
            "rd2": records["predicted_outcome"].isin(["rd2", "cpr"]).astype(int),
            "cpr": records["predicted_outcome"].eq("cpr").astype(int),
        },
        index=records.index,
    )
    has_raw_outcome_flags = raw_output_frame.reindex(columns=["rd1", "rd2", "cpr_needed"]).notna().any(axis=1)
    collapsed = (
        pd.concat(
            [
                records[["system_id", "item_id"]],
                pd.DataFrame(
                    np.where(
                        has_raw_outcome_flags.to_numpy()[:, None],
                        raw_outcome_flags.to_numpy(dtype=int),
                        fallback_outcome_flags.to_numpy(dtype=int),
                    ),
                    index=records.index,
                    columns=["rd1", "rd2", "cpr"],
                ),
            ],
            axis=1,
        )
        .groupby(["system_id", "item_id"], sort=False)[["rd1", "rd2", "cpr"]]
        .mean()
        .gt(0.5)
        .astype(int)
        .reset_index()
        .merge(
            items_with_outcomes[["item_id", "gt_final_outcome", "gt_rd1", "gt_rd2", "gt_cpr"]],
            on="item_id",
            how="left",
            validate="many_to_one",
        )
        .assign(
            outcome_subset_accuracy=lambda frame: (
                frame[["rd1", "rd2", "cpr"]].to_numpy(dtype=int)
                == frame[["gt_rd1", "gt_rd2", "gt_cpr"]].to_numpy(dtype=int)
            )
            .all(axis=1)
            .astype(float),
            final_predicted_outcome=lambda frame: pd.Series(
                np.select(
                    [
                        frame["cpr"].eq(1),
                        frame["rd2"].eq(1),
                        frame["rd1"].eq(1),
                    ],
                    ["cpr", "rd2", "rd1"],
                    default=None,
                ),
                index=frame.index,
            ),
        )
        .assign(
            final_outcome_accuracy=lambda frame: frame["final_predicted_outcome"]
            .eq(frame["gt_final_outcome"])
            .astype(float)
        )
    )

    system_order = records["system_id"].drop_duplicates().tolist()
    system_metrics = (
        collapsed.groupby("system_id", sort=False)[
            [
                "gt_rd1",
                "gt_rd2",
                "gt_cpr",
                "rd1",
                "rd2",
                "cpr",
                "outcome_subset_accuracy",
                "final_outcome_accuracy",
            ]
        ]
        .apply(
            lambda frame: pd.Series(
                {
                    "outcome_subset_accuracy": float(frame["outcome_subset_accuracy"].mean()),
                    "final_outcome_accuracy": float(frame["final_outcome_accuracy"].mean()),
                    "outcome_hamming_loss": float(
                        hamming_loss(
                            frame[["gt_rd1", "gt_rd2", "gt_cpr"]].to_numpy(dtype=int),
                            frame[["rd1", "rd2", "cpr"]].to_numpy(dtype=int),
                        )
                    ),
                    "outcome_jaccard": float(
                        jaccard_score(
                            frame[["gt_rd1", "gt_rd2", "gt_cpr"]].to_numpy(dtype=int),
                            frame[["rd1", "rd2", "cpr"]].to_numpy(dtype=int),
                            average="samples",
                            zero_division=1.0,
                        )
                    ),
                }
            )
        )
        .join(
            records.assign(
                followup_rate=records["response_kind"].eq("followup").astype(float),
                parse_failure_rate=records["response_kind"].eq("parse_error").astype(float),
            )
            .groupby("system_id", sort=False)[["followup_rate", "parse_failure_rate"]]
            .mean()
        )
        .reindex(system_order)
        .fillna(0.0)
        .to_dict(orient="index")
    )

    return system_metrics


def experiment2_summary(
    *,
    records: pd.DataFrame,
    items: pd.DataFrame,
) -> dict[str, Any]:
    empty_error_analysis = {
        "item_pattern_counts": {
            "exact_match_items": 0,
            "extra_only_items": 0,
            "missing_only_items": 0,
            "mixed_error_items": 0,
        },
        "false_positive_fields": [],
        "false_negative_fields": [],
        "confusion_pairs": [],
    }
    if records.empty:
        return {
            "medical_slot_accuracy": 0.0,
            "medical_hamming_loss": 0.0,
            "medical_subset_accuracy": 0.0,
            "medical_slot_jaccard": 0.0,
            "non_medical_exact_match_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "followup_rate": 0.0,
            "parse_failure_rate": 0.0,
            "repeat_instability_rate": 0.0,
            "per_slot_count": {},
            "per_slot": [],
            "error_analysis": empty_error_analysis,
        }

    items_enriched = items.assign(
        item_id=items["id"],
        has_non_medical=items["gt_non_medical_state"].map(bool),
    )
    item_medical = expand_medical_state(items_enriched["gt_medical_state"])
    items_enriched = items_enriched.assign(
        gt_positive_slot_count=item_medical.sum(axis=1).astype(int),
    )
    label_frame, support_by_field = medical_label_frame(records=records, items=items_enriched)
    label_frame = label_frame.assign(
        non_medical_exact_match=label_frame["predicted_non_medical_state"]
        .eq(label_frame["gt_non_medical_state"])
        .astype(int),
        signature=list(
            zip(
                label_frame["response_kind"],
                label_frame["predicted_medical_state"].map(lambda state: tuple(sorted(state.items()))),
                label_frame["predicted_non_medical_state"].map(lambda state: tuple(sorted(state.items()))),
                label_frame["predicted_outcome"],
                label_frame["followup_text"],
            )
        ),
    )

    summary_metrics = summarize_experiment2_group(
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

    per_slot_count_metrics = (
        label_frame.groupby("gt_positive_slot_count", sort=True)[
            [column for column in label_frame.columns if column != "gt_positive_slot_count"]
        ]
        .apply(
            lambda group: pd.Series(
                summarize_experiment2_group(
                    group_frame=group,
                    support_by_field=item_support_by_slot_count.loc[group.name].astype(int),
                )
            )
        )
        .reindex(items_enriched["gt_positive_slot_count"].drop_duplicates().sort_values())
    )
    per_slot_count_defaults = {
        "medical_slot_accuracy": 0.0,
        "medical_hamming_loss": 0.0,
        "medical_subset_accuracy": 0.0,
        "medical_slot_jaccard": 0.0,
        "non_medical_exact_match_accuracy": None,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "followup_rate": 0.0,
        "parse_failure_rate": 0.0,
        "repeat_instability_rate": 0.0,
    }
    for column, value in per_slot_count_defaults.items():
        if value is None:
            per_slot_count_metrics[column] = per_slot_count_metrics[column].astype(object).where(
                per_slot_count_metrics[column].notna(),
                None,
            )
            continue
        per_slot_count_metrics[column] = per_slot_count_metrics[column].fillna(value)

    per_slot_count = (
        items_enriched.groupby("gt_positive_slot_count", sort=True)
        .size()
        .rename("item_count")
        .to_frame()
        .join(per_slot_count_metrics)
        .assign(item_count=lambda frame: frame["item_count"].astype(int))
        .rename(index=lambda value: str(int(value)))
        .to_dict(orient="index")
    )

    first_predictions = label_frame.drop_duplicates(subset=["item_id"], keep="first").set_index("item_id")
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
        .sort_values(["count", "missing_field", "extra_field"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    return {
        **summary_metrics,
        "per_slot_count": per_slot_count,
        "per_slot": overall_slot_metrics["per_slot"],
        "error_analysis": {
            "item_pattern_counts": {
                "exact_match_items": int((extra_counts.eq(0) & missing_counts.eq(0)).sum()),
                "extra_only_items": int((extra_counts.gt(0) & missing_counts.eq(0)).sum()),
                "missing_only_items": int((missing_counts.gt(0) & extra_counts.eq(0)).sum()),
                "mixed_error_items": int((extra_counts.gt(0) & missing_counts.gt(0)).sum()),
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
