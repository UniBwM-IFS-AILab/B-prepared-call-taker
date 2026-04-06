"""Experiment 1 metric computation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss, jaccard_score

from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency
from experiments.nlu_eval.common.metrics import prf_metrics

OUTCOME_LABELS = np.array([0, 1, 2], dtype=int)


def _collapsed_outcome_flags(records: pd.DataFrame) -> pd.DataFrame:
    raw_output = records["raw_output"]
    raw_flags = pd.DataFrame(
        {
            "rd1": raw_output.map(lambda raw: raw.get("rd1") is True).astype(int),
            "rd2": raw_output.map(lambda raw: raw.get("rd2") is True).astype(int),
            "cpr": raw_output.map(lambda raw: raw.get("cpr_needed") is True).astype(
                int
            ),
        },
        index=records.index,
    )
    derived_flags = pd.DataFrame(
        {
            "rd1": records["predicted_outcome"].isin(["rd1", "rd2", "cpr"]).astype(int),
            "rd2": records["predicted_outcome"].isin(["rd2", "cpr"]).astype(int),
            "cpr": records["predicted_outcome"].eq("cpr").astype(int),
        },
        index=records.index,
    )
    has_explicit_raw_flags = raw_output.map(
        lambda raw: any(key in raw for key in ("rd1", "rd2", "cpr_needed"))
    )
    outcome_flags = pd.DataFrame(
        np.where(
            has_explicit_raw_flags.to_numpy()[:, None],
            raw_flags.to_numpy(dtype=int),
            derived_flags.to_numpy(dtype=int),
        ),
        index=records.index,
        columns=["rd1", "rd2", "cpr"],
    )
    return (
        pd.concat([records[["system_id", "item_id"]], outcome_flags], axis=1)
        .groupby(["system_id", "item_id"], sort=False)[["rd1", "rd2", "cpr"]]
        .mean()
        .gt(0.5)
        .astype(int)
        .reset_index()
    )


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
        gt_rd1=lambda frame: (
            frame["gt_final_outcome"].isin(["rd1", "rd2", "cpr"]).astype(int)
        ),
        gt_rd2=lambda frame: frame["gt_final_outcome"].isin(["rd2", "cpr"]).astype(int),
        gt_cpr=lambda frame: frame["gt_final_outcome"].eq("cpr").astype(int),
    )

    collapsed = (
        _collapsed_outcome_flags(records)
        .merge(
            items_with_outcomes[
                ["item_id", "gt_final_outcome", "gt_rd1", "gt_rd2", "gt_cpr"]
            ],
            on="item_id",
            how="left",
            validate="many_to_one",
        )
        .assign(
            subset_accuracy=lambda frame: (
                (
                    frame[["rd1", "rd2", "cpr"]].to_numpy(dtype=int)
                    == frame[["gt_rd1", "gt_rd2", "gt_cpr"]].to_numpy(dtype=int)
                )
                .all(axis=1)
                .astype(float)
            ),
            final_predicted_outcome=lambda frame: [
                "cpr"
                if cpr == 1
                else "rd2"
                if rd2 == 1
                else "rd1"
                if rd1 == 1
                else None
                for rd1, rd2, cpr in zip(
                    frame["rd1"], frame["rd2"], frame["cpr"], strict=False
                )
            ],
        )
        .assign(
            final_accuracy=lambda frame: (
                frame["final_predicted_outcome"]
                .eq(frame["gt_final_outcome"])
                .astype(float)
            )
        )
    )

    def summarize_experiment1_system(frame: pd.DataFrame) -> pd.Series:
        y_true = frame[["gt_rd1", "gt_rd2", "gt_cpr"]].to_numpy(dtype=int)
        y_pred = frame[["rd1", "rd2", "cpr"]].to_numpy(dtype=int)
        scored_labels = OUTCOME_LABELS[
            (y_true.sum(axis=0) > 0) | (y_pred.sum(axis=0) > 0)
        ]
        return pd.Series(
            {
                "subset_accuracy": float(frame["subset_accuracy"].mean()),
                "final_accuracy": float(frame["final_accuracy"].mean()),
                "hamming_loss": float(hamming_loss(y_true, y_pred)),
                "jaccard": float(
                    jaccard_score(
                        y_true,
                        y_pred,
                        average="samples",
                        zero_division=1.0,
                    )
                ),
                **prf_metrics(y_true=y_true, y_pred=y_pred, labels=scored_labels),
            }
        )

    system_order = records["system_id"].drop_duplicates().tolist()
    return (
        collapsed.groupby("system_id", sort=False)[
            [
                "gt_rd1",
                "gt_rd2",
                "gt_cpr",
                "rd1",
                "rd2",
                "cpr",
                "subset_accuracy",
                "final_accuracy",
            ]
        ]
        .apply(summarize_experiment1_system)
        .join(
            records.assign(
                followup_rate=records["response_kind"].eq("followup").astype(float),
                parse_failure_rate=records["response_kind"]
                .eq("parse_error")
                .astype(float),
            )
            .groupby("system_id", sort=False)[["followup_rate", "parse_failure_rate"]]
            .mean()
        )
        .reindex(system_order)
        .fillna(0.0)
        .to_dict(orient="index")
    )
