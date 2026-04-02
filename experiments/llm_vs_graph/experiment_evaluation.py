"""Marimo notebook for descriptive experiment evaluation."""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")

with app.setup:
    import json
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    from ems_prepared.adapters.gradio.survey import DEFAULT_BY_ID
    from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency

    UNKNOWN_LABEL = 0
    NO_LABEL = 1
    YES_LABEL = 2
    TRI_STATE_LABELS = [UNKNOWN_LABEL, NO_LABEL, YES_LABEL]
    TRI_STATE_NAMES = {
        UNKNOWN_LABEL: "unknown",
        NO_LABEL: "no",
        YES_LABEL: "yes",
    }


@app.cell
def _():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Experiment Context

    This notebook is scoped to the `llm_vs_graph` experiment directory.

    It uses:

    - `scenario_ground_truth/` for state comparison
    - `logs/` for dialogue and survey logs
    - the experiment directory itself as the export target for descriptive CSV summaries
    """)
    return


@app.cell
def _():
    experiment_dir = Path(__file__).resolve().parent
    logs_path = experiment_dir / "logs"
    gt_dir = experiment_dir / "scenario_ground_truth"
    export_dir = experiment_dir
    {
        "experiment_dir": experiment_dir,
        "logs_path": logs_path,
        "gt_dir": gt_dir,
        "export_dir": export_dir,
    }
    return experiment_dir, export_dir, gt_dir, logs_path


@app.function
def get_sample_paths(experiment_path: Path) -> list[Path]:
    if not experiment_path.exists():
        return []
    return [
        sample
        for session in experiment_path.iterdir()
        if session.is_dir()
        for sample in session.iterdir()
        if sample.is_dir()
    ]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Survey Results

    Survey items are summarized descriptively per policy.

    **Mean score**

    $$
    \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
    $$

    **Sample standard deviation**

    $$
    s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}
    $$

    **Interpretation**

    - The mean score describes the average user rating for a question.
    - The standard deviation describes how much respondents disagree.

    **Usefulness**

    - Useful for subjective impressions such as clarity, fluency, and perceived responsiveness.
    - Useful as a complement to task metrics when you want to know whether users noticed behavioral differences.

    **Limitations**

    - Likert scores are ordinal, so very small mean differences should not be over-interpreted.
    - A high mean does not imply task correctness.
    - A large standard deviation means users experienced the system less consistently.
    """)
    return


@app.cell
def _(logs_path):
    _survey_rows = []

    for _session_path in get_sample_paths(logs_path):
        try:
            with open(
                _session_path / "survey.json", "r", encoding="utf-8"
            ) as _survey_file:
                survey_data = json.load(_survey_file)
        except Exception:
            continue

        try:
            with open(
                _session_path / "deps.json", "r", encoding="utf-8"
            ) as _deps_file:
                _policy_name = json.load(_deps_file).get("policy_name") or "agent"
        except Exception:
            _policy_name = "agent"

        metadata = survey_data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)

        feedback = survey_data.get("feedback")
        if not isinstance(feedback, str):
            legacy_feedback = metadata.pop("feedback", None)
            feedback = legacy_feedback if isinstance(legacy_feedback, str) else None
        else:
            metadata.pop("feedback", None)

        responses = survey_data.get("responses", {})
        session_scores = {}
        if isinstance(responses, dict):
            for idx, (_, value) in enumerate(responses.items(), 1):
                session_scores[f"Q{idx}"] = value["score"]
        elif isinstance(responses, list):
            for idx, item in enumerate(responses, 1):
                session_scores[f"Q{idx}"] = item["score"]
        else:
            continue

        _scenario_name = metadata.get("scenario")
        if not isinstance(_scenario_name, str):
            _scenario_name = metadata.get("scenario_name")
        if isinstance(_scenario_name, str):
            metadata["scenario"] = _scenario_name.strip(".md")

        _survey_rows.append(
            {
                "source_path": str(_session_path),
                **metadata,
                "feedback": feedback,
                "policy": _policy_name,
                **session_scores,
            }
        )

    survey_results_df = pd.DataFrame(_survey_rows)
    if survey_results_df.empty:
        survey_results_df = pd.DataFrame(
            columns=["user_id", "session_id", "policy", "feedback"]
        ).set_index(["user_id", "session_id"])
    else:
        survey_results_df = survey_results_df.set_index(["user_id", "session_id"])
    survey_results_df
    return (survey_results_df,)


@app.cell
def _(survey_results_df):
    if survey_results_df.empty or "policy" not in survey_results_df.columns:
        pd.Series(dtype=float)
    else:
        survey_results_df["policy"].value_counts(normalize=True)
    return


@app.cell
def _(survey_results_df):
    _survey_rows = survey_results_df.reset_index()
    survey_question_cols = sorted(
        column for column in _survey_rows.columns if column.startswith("Q")
    )

    if survey_question_cols and "policy" in _survey_rows.columns:
        survey_long_df = (
            _survey_rows.melt(
                id_vars=["policy"],
                value_vars=survey_question_cols,
                var_name="metric",
                value_name="score",
            )
            .dropna(subset=["score"])
            .sort_values(["metric", "policy"], ignore_index=True)
        )
    else:
        survey_long_df = pd.DataFrame(columns=["policy", "metric", "score"])

    question_metadata_df = pd.DataFrame(
        [
            {
                "metric": metric,
                "question_id": int(metric[1:]),
                "question_label": DEFAULT_BY_ID[int(metric[1:])].label,
                "question_category": DEFAULT_BY_ID[int(metric[1:])].category,
                "question_text": DEFAULT_BY_ID[int(metric[1:])].text,
            }
            for metric in survey_question_cols
        ]
    )

    if survey_long_df.empty:
        likert_summary_df = pd.DataFrame(
            columns=[
                "policy",
                "metric",
                "question_id",
                "question_label",
                "question_category",
                "question_text",
                "count",
                "mean_score",
                "std_dev",
            ]
        )
    else:
        likert_summary_df = (
            survey_long_df.groupby(["policy", "metric"], as_index=False)
            .agg(
                count=("score", "count"),
                mean_score=("score", "mean"),
                std_dev=(
                    "score",
                    lambda series: float(series.std(ddof=1))
                    if len(series) > 1
                    else 0.0,
                ),
            )
            .merge(question_metadata_df, on="metric", how="left")
            [
                [
                    "policy",
                    "metric",
                    "question_id",
                    "question_label",
                    "question_category",
                    "question_text",
                    "count",
                    "mean_score",
                    "std_dev",
                ]
            ]
            .sort_values(["question_id", "policy"], ignore_index=True)
        )

    likert_summary_df
    return likert_summary_df, survey_question_cols


@app.cell
def _(likert_summary_df):
    if likert_summary_df.empty:
        survey_summary_table = pd.DataFrame()
    else:
        survey_summary_table = likert_summary_df.pivot(
            index="policy",
            columns="metric",
            values=["mean_score", "std_dev", "count"],
        ).sort_index(axis=1, level=1)
    survey_summary_table.round(3)
    return


@app.cell
def _(likert_summary_df, survey_question_cols):
    _fig, _ax = plt.subplots(figsize=(10, 5))
    if likert_summary_df.empty or not survey_question_cols:
        _ax.text(0.5, 0.5, "No survey data available", ha="center", va="center")
        _ax.axis("off")
    else:
        policies = list(pd.unique(likert_summary_df["policy"]))
        width = 0.8 / max(len(policies), 1)
        positions = np.arange(len(survey_question_cols), dtype=float)

        for index, policy in enumerate(policies):
            policy_df = (
                likert_summary_df[likert_summary_df["policy"] == policy]
                .set_index("metric")
                .reindex(survey_question_cols)
            )
            x = positions + (index - (len(policies) - 1) / 2) * width
            _ax.bar(x, policy_df["mean_score"], width=width, label=policy)

        _ax.set_xticks(positions)
        _ax.set_xticklabels(survey_question_cols)
        _ax.set_xlabel("Survey question")
        _ax.set_ylabel("Mean Likert score")
        _ax.set_ylim(1, 5)
        _ax.legend(
            ncols=max(len(policies), 1),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
        )
        _fig.tight_layout()
    _fig
    return


@app.cell
def _(survey_results_df):
    feedbacks = (
        survey_results_df["feedback"].dropna()
        if "feedback" in survey_results_df.columns
        else pd.Series(dtype=object)
    )
    feedbacks
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## State Analysis

    Each dialogue is treated as one independent observation.
    The notebook evaluates only the final state of the dialogue, not turn-by-turn state tracking.
    """)
    return


@app.cell
def _(logs_path):
    state_rows = []

    for _session_path in get_sample_paths(logs_path):
        try:
            messages_df = pd.read_csv(_session_path / "messages.tsv", sep="\t")
            turn_count = int((messages_df["speaker"] == "caller").sum())
        except Exception as exc:
            print(f"Error reading messages.tsv in {_session_path}: {exc}")
            continue

        try:
            with open(
                _session_path / "final_state.json", "r", encoding="utf-8"
            ) as state_file:
                state_data = json.load(state_file)
                if "call_state" in state_data:
                    state_data = state_data["call_state"]
                outcome = MedicalEmergency(**state_data).get_outcome()
        except Exception as exc:
            print(f"Error parsing final_state.json in {_session_path}: {exc}")
            continue

        try:
            with open(
                _session_path / "deps.json", "r", encoding="utf-8"
            ) as _deps_file:
                deps_data = json.load(_deps_file)
                _policy_name = deps_data.get("policy_name") or "agent"
                _scenario_name = deps_data.get("scenario_name")
        except FileNotFoundError:
            _policy_name = "agent"
            try:
                with open(
                    _session_path / "survey.json", "r", encoding="utf-8"
                ) as _survey_file:
                    _scenario_name = json.load(_survey_file).get("metadata", {}).get(
                        "scenario"
                    )
            except FileNotFoundError as exc:
                print(f"Error parsing survey.json in {_session_path}: {exc}")
                continue

        state_rows.append(
            {
                "scenario": _scenario_name.strip(".md")
                if isinstance(_scenario_name, str)
                else None,
                "outcome": outcome,
                "dialogue_turns": turn_count,
                "policy": _policy_name,
                **state_data,
            }
        )

    state_results_df = pd.DataFrame(state_rows)
    if state_results_df.empty:
        state_results_df = pd.DataFrame(
            columns=[
                "scenario",
                "outcome",
                "policy",
                "dialogue_turns",
                *MedicalEmergency.model_fields.keys(),
            ]
        )
    state_results_df = state_results_df.loc[
        :,
        state_results_df.columns.isin(
            [
                "scenario",
                "outcome",
                "policy",
                "dialogue_turns",
                *MedicalEmergency.model_fields.keys(),
            ]
        ),
    ]
    state_results_df
    return (state_results_df,)


@app.cell
def _(gt_dir):
    gt_rows = []
    if gt_dir.exists():
        for gt_path in sorted(gt_dir.rglob("*.json")):
            match = re.search(r"scenario_(\d+)", gt_path.stem, flags=re.IGNORECASE)
            if match is None:
                raise ValueError(f"Could not extract scenario id from {gt_path.name}")

            with open(gt_path, "r", encoding="utf-8") as gt_file:
                gt_data = json.load(gt_file)

            gt_rows.append(
                {
                    "scenario": f"Scenario_{match.group(1)}",
                    "outcome": MedicalEmergency(**gt_data).get_outcome(),
                    **gt_data,
                }
            )

    gt_df = pd.DataFrame(gt_rows)
    if gt_df.empty:
        gt_df = pd.DataFrame(
            columns=["scenario", "outcome", *MedicalEmergency.model_fields.keys()]
        )
    gt_df = gt_df.loc[
        :,
        gt_df.columns.isin(["scenario", "outcome", *MedicalEmergency.model_fields.keys()]),
    ]
    gt_df
    return (gt_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Metric Reading Guide

    Recommended reading order:

    1. `outcome_match`
    2. `slot_accuracy_any_valid` versus `slot_accuracy_predicted_outcome`
    3. `precision_yes_predicted_outcome`, `recall_yes_predicted_outcome`, and `f1_yes_predicted_outcome`
    4. `dialogue_turns`
    5. Survey means and standard deviations
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Metric: `outcome_match`

    **Formula**

    For predicted outcome $\hat{o}_i$ and a valid ground-truth outcome $o_i^*$ for dialogue $i$:

    $$
    \mathrm{OutcomeMatch}_i = \mathbb{1}[\hat{o}_i = o_i^*]
    $$

    **Interpretation**

    - `1` means the final dispatch outcome is correct.
    - `0` means the final dispatch outcome is incorrect.

    **Usefulness**

    - This is the clearest task-success metric.
    - It directly answers whether the policy reached the correct final decision.

    **Limitations**

    - It ignores whether the internal state justified the outcome.
    - A policy can score well here even if the final symptom state is poorly grounded.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Metrics: `slot_accuracy_any_valid` and `slot_accuracy_predicted_outcome`

    All final-state slot metrics operate on tri-state slot labels:

    - `Unknown`
    - `No`
    - `Yes`

    If a policy never touches a slot, the slot remains `Unknown`.

    **Per-slot accuracy against one reference state**

    For predicted slot vector $\hat{s}_i$ and ground-truth slot vector $s_i^*$:

    $$
    \mathrm{SlotAccuracy}(\hat{s}_i, s_i^*) =
    \frac{1}{K}\sum_{k=1}^{K}\mathbb{1}[\hat{s}_{ik} = s_{ik}^*]
    $$

    where $K$ is the number of evaluated slots.

    **`slot_accuracy_any_valid`**

    If a scenario has multiple valid ground-truth variants $V_i$:

    $$
    \mathrm{SlotAccuracyAnyValid}_i =
    \max_{s \in V_i} \mathrm{SlotAccuracy}(\hat{s}_i, s)
    $$

    **Interpretation**

    - Measures how close the final state is to at least one valid interpretation of the scenario.

    **Usefulness**

    - Useful when several end states are acceptable for the same scenario.
    - Good for checking whether the final state is broadly plausible.

    **Limitations**

    - Forgiving when multiple valid variants exist.
    - Does not require consistency with the chosen outcome.

    **`slot_accuracy_predicted_outcome`**

    Let $s_i^{(\hat{o}_i)}$ be the ground-truth variant whose outcome matches the prediction.

    $$
    \mathrm{SlotAccuracyPredictedOutcome}_i =
    \begin{cases}
    \mathrm{SlotAccuracy}(\hat{s}_i, s_i^{(\hat{o}_i)}) & \text{if such a variant exists} \\
    0 & \text{otherwise}
    \end{cases}
    $$

    **Interpretation**

    - Measures whether the final state is internally consistent with the outcome the system actually chose.

    **Usefulness**

    - This is the stricter and more operationally meaningful slot-accuracy metric.

    **Limitations**

    - Penalizes an otherwise-close state if the chosen outcome is wrong.
    - Will always be less than or equal to `slot_accuracy_any_valid`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Metric: `exact_match_predicted_outcome`

    **Formula**

    $$
    \mathrm{ExactMatchPredictedOutcome}_i =
    \mathbb{1}[\hat{s}_i = s_i^{(\hat{o}_i)}]
    $$

    **Interpretation**

    - `1` means every evaluated slot matches exactly.
    - `0` means at least one slot differs.

    **Usefulness**

    - Useful as a strict final-state correctness criterion.

    **Limitations**

    - Very brittle.
    - A single slot error collapses the whole metric to zero.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Multiclass PRF Metrics On `{Unknown, No, Yes}`

    The notebook uses standard multiclass precision, recall, and F1 on the predicted-outcome ground-truth variant.

    For class $c$:

    $$
    \mathrm{Precision}_c = \frac{TP_c}{TP_c + FP_c}
    \qquad
    \mathrm{Recall}_c = \frac{TP_c}{TP_c + FN_c}
    $$

    $$
    \mathrm{F1}_c = \frac{2 \cdot \mathrm{Precision}_c \cdot \mathrm{Recall}_c}{\mathrm{Precision}_c + \mathrm{Recall}_c}
    $$

    Macro averages are the unweighted class means:

    $$
    \mathrm{MacroPrecision} = \frac{1}{3}\sum_{c \in \{\mathrm{Unknown,No,Yes}\}} \mathrm{Precision}_c
    $$

    $$
    \mathrm{MacroRecall} = \frac{1}{3}\sum_{c \in \{\mathrm{Unknown,No,Yes}\}} \mathrm{Recall}_c
    $$

    $$
    \mathrm{MacroF1} = \frac{1}{3}\sum_{c \in \{\mathrm{Unknown,No,Yes}\}} \mathrm{F1}_c
    $$

    **Metrics for `Unknown`**

    - `precision_unknown_predicted_outcome`
    - `recall_unknown_predicted_outcome`
    - `f1_unknown_predicted_outcome`

    **Interpretation**

    - These measure whether the policy leaves unsupported state unresolved instead of inventing labels.

    **Usefulness**

    - Useful for checking over-assertion.

    **Limitations**

    - Because many slots may remain `Unknown`, these values can dominate macro summaries.

    **Metrics for `No`**

    - `precision_no_predicted_outcome`
    - `recall_no_predicted_outcome`
    - `f1_no_predicted_outcome`

    **Interpretation**

    - These measure recovery of explicit negative findings.

    **Usefulness**

    - Useful only when the data contains meaningful numbers of `No` labels.

    **Limitations**

    - If `No` labels are rare, these metrics become unstable or uninformative.

    **Metrics for `Yes`**

    - `precision_yes_predicted_outcome`
    - `recall_yes_predicted_outcome`
    - `f1_yes_predicted_outcome`

    **Interpretation**

    - `precision_yes_predicted_outcome`: when the notebook predicts a positive slot, how often it is correct
    - `recall_yes_predicted_outcome`: when a positive slot is present in ground truth, how often it is recovered
    - `f1_yes_predicted_outcome`: balanced summary of positive-slot recovery

    **Usefulness**

    - These are usually the most important PRF diagnostics when positive findings are sparse and operationally important.

    **Limitations**

    - They still ignore downstream turn cost.

    **Macro metrics**

    - `macro_precision_predicted_outcome`
    - `macro_recall_predicted_outcome`
    - `macro_f1_predicted_outcome`

    **Interpretation**

    - Compact summary across the three classes.

    **Usefulness**

    - Helpful for one-line comparison.

    **Limitations**

    - Can hide which class is actually driving the result.
    - Should never replace the classwise view entirely.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Metric Family: `dialogue_turns`

    `dialogue_turns` counts caller utterances.

    **Mean**

    $$
    \bar{t} = \frac{1}{n}\sum_{i=1}^{n} t_i
    $$

    **Sample standard deviation**

    $$
    s_t = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(t_i - \bar{t})^2}
    $$

    **Median and quartiles**

    - `median`: 50th percentile of the caller-turn distribution
    - `Q1`: 25th percentile of the caller-turn distribution
    - `Q3`: 75th percentile of the caller-turn distribution

    **Interquartile range**

    $$
    \mathrm{IQR} = Q3 - Q1
    $$

    **Interpretation**

    - Mean and standard deviation describe average burden and overall spread.
    - Median, quartiles, and IQR describe the typical case and robustness to skew.

    **Usefulness**

    - Direct efficiency metric for conversational burden.

    **Limitations**

    - Fewer turns are not automatically better if task correctness drops.
    - Mean and median should be read together when the distribution is skewed.
    """)
    return


@app.cell
def _(gt_df, state_results_df):
    label_cols = list(
        gt_df.columns.intersection(state_results_df.columns).difference(
            ["scenario", "outcome", "policy", "dialogue_turns"]
        )
    )
    if not label_cols:
        label_cols = list(
            state_results_df.columns.intersection(MedicalEmergency.model_fields.keys())
        )

    metric_types = {
        "outcome_match": "binary",
        "slot_accuracy_any_valid": "continuous",
        "slot_accuracy_predicted_outcome": "continuous",
        "exact_match_predicted_outcome": "binary",
        "macro_precision_predicted_outcome": "continuous",
        "macro_recall_predicted_outcome": "continuous",
        "macro_f1_predicted_outcome": "continuous",
        "precision_unknown_predicted_outcome": "continuous",
        "recall_unknown_predicted_outcome": "continuous",
        "f1_unknown_predicted_outcome": "continuous",
        "precision_no_predicted_outcome": "continuous",
        "recall_no_predicted_outcome": "continuous",
        "f1_no_predicted_outcome": "continuous",
        "precision_yes_predicted_outcome": "continuous",
        "recall_yes_predicted_outcome": "continuous",
        "f1_yes_predicted_outcome": "continuous",
    }

    metric_cols = list(metric_types.keys())
    dialogue_metric_cols = [
        "policy",
        "scenario",
        "outcome",
        "dialogue_turns",
        "matched_scenario_ground_truth",
        "matched_outcome_ground_truth",
        *metric_cols,
    ]

    if state_results_df.empty:
        dialogue_metrics_df = pd.DataFrame(columns=dialogue_metric_cols)
    else:
        base_cols = ["policy", "scenario", "outcome", "dialogue_turns"]
        state_eval_df = state_results_df[base_cols + label_cols].copy()
        gt_eval_df = gt_df[["scenario", "outcome", *label_cols]].drop_duplicates(
            ["scenario", "outcome"],
            keep="first",
        )

        state_encoded_df = state_eval_df.copy()
        gt_encoded_df = gt_eval_df.copy()
        if label_cols:
            state_encoded_df[label_cols] = (
                state_eval_df[label_cols].eq(True).astype(int) * YES_LABEL
                + state_eval_df[label_cols].eq(False).astype(int) * NO_LABEL
            )
            gt_encoded_df[label_cols] = (
                gt_eval_df[label_cols].eq(True).astype(int) * YES_LABEL
                + gt_eval_df[label_cols].eq(False).astype(int) * NO_LABEL
            )

        scenario_groups = {
            scenario: group.reset_index(drop=True)
            for scenario, group in gt_encoded_df.groupby("scenario", sort=False)
        }

        dialogue_metric_rows = []
        for _, pred_row in state_encoded_df.iterrows():
            metric_row = {
                "policy": pred_row["policy"],
                "scenario": pred_row["scenario"],
                "outcome": pred_row["outcome"],
                "dialogue_turns": pred_row["dialogue_turns"],
                "matched_scenario_ground_truth": False,
                "matched_outcome_ground_truth": False,
                **{metric: 0.0 for metric in metric_cols},
            }

            scenario_candidates = scenario_groups.get(pred_row["scenario"])
            if scenario_candidates is None or scenario_candidates.empty:
                dialogue_metric_rows.append(metric_row)
                continue

            metric_row["matched_scenario_ground_truth"] = True
            pred_vector = pred_row[label_cols].to_numpy(dtype=int)
            gt_matrix = scenario_candidates[label_cols].to_numpy(dtype=int)

            candidate_scores = (
                (gt_matrix == pred_vector).mean(axis=1) if label_cols else np.array([0.0])
            )
            metric_row["slot_accuracy_any_valid"] = (
                float(candidate_scores.max()) if candidate_scores.size else 0.0
            )

            outcome_candidates = scenario_candidates.loc[
                scenario_candidates["outcome"] == pred_row["outcome"]
            ]
            if outcome_candidates.empty:
                dialogue_metric_rows.append(metric_row)
                continue

            gt_vector = outcome_candidates.iloc[0][label_cols].to_numpy(dtype=int)
            metric_row["matched_outcome_ground_truth"] = True
            metric_row["outcome_match"] = 1.0
            metric_row["slot_accuracy_predicted_outcome"] = (
                float(accuracy_score(gt_vector, pred_vector)) if label_cols else 0.0
            )
            metric_row["exact_match_predicted_outcome"] = float(
                np.array_equal(gt_vector, pred_vector)
            )

            macro_precision, macro_recall, macro_f1, _ = (
                precision_recall_fscore_support(
                    gt_vector,
                    pred_vector,
                    labels=TRI_STATE_LABELS,
                    average="macro",
                    zero_division=0,
                )
                if label_cols
                else (0.0, 0.0, 0.0, None)
            )
            metric_row["macro_precision_predicted_outcome"] = float(macro_precision)
            metric_row["macro_recall_predicted_outcome"] = float(macro_recall)
            metric_row["macro_f1_predicted_outcome"] = float(macro_f1)

            class_precisions, class_recalls, class_f1_scores, _ = (
                precision_recall_fscore_support(
                    gt_vector,
                    pred_vector,
                    labels=TRI_STATE_LABELS,
                    average=None,
                    zero_division=0,
                )
                if label_cols
                else (np.zeros(3), np.zeros(3), np.zeros(3), None)
            )
            for label_index, label_name in TRI_STATE_NAMES.items():
                metric_row[f"precision_{label_name}_predicted_outcome"] = float(
                    class_precisions[label_index]
                )
                metric_row[f"recall_{label_name}_predicted_outcome"] = float(
                    class_recalls[label_index]
                )
                metric_row[f"f1_{label_name}_predicted_outcome"] = float(
                    class_f1_scores[label_index]
                )

            dialogue_metric_rows.append(metric_row)

        dialogue_metrics_df = pd.DataFrame(dialogue_metric_rows, columns=dialogue_metric_cols)

    dialogue_metrics_df
    return dialogue_metrics_df, metric_types


@app.cell
def _(dialogue_metrics_df):
    dialogue_metrics_df[
        [
            "policy",
            "scenario",
            "outcome",
            "outcome_match",
            "slot_accuracy_any_valid",
            "slot_accuracy_predicted_outcome",
            "exact_match_predicted_outcome",
            "precision_yes_predicted_outcome",
            "recall_yes_predicted_outcome",
            "f1_yes_predicted_outcome",
            "dialogue_turns",
        ]
    ]
    return


@app.cell
def _(dialogue_metrics_df, metric_types):
    metric_names = list(metric_types.keys())

    if dialogue_metrics_df.empty:
        metrics_summary_df = pd.DataFrame(
            columns=["policy", "metric", "metric_type", "count", "mean", "std_dev"]
        )
    else:
        metrics_summary_df = (
            dialogue_metrics_df.melt(
                id_vars=["policy"],
                value_vars=metric_names,
                var_name="metric",
                value_name="value",
            )
            .dropna(subset=["value"])
            .groupby(["policy", "metric"], as_index=False)
            .agg(
                count=("value", "count"),
                mean=("value", "mean"),
                std_dev=(
                    "value",
                    lambda series: float(series.std(ddof=1))
                    if len(series) > 1
                    else 0.0,
                ),
            )
        )
        metrics_summary_df["metric_type"] = metrics_summary_df["metric"].map(metric_types)
        metrics_summary_df = metrics_summary_df[
            ["policy", "metric", "metric_type", "count", "mean", "std_dev"]
        ].sort_values(["metric", "policy"], ignore_index=True)

    metrics_summary_df
    return (metrics_summary_df,)


@app.cell
def _(dialogue_metrics_df):
    if dialogue_metrics_df.empty:
        dialogue_turns_summary_df = pd.DataFrame(
            columns=["policy", "mean", "std_dev", "median", "q1", "q3", "count", "iqr"]
        )
    else:
        grouped_turns = dialogue_metrics_df.groupby("policy")["dialogue_turns"]
        dialogue_turns_summary_df = (
            grouped_turns.agg(["mean", "median", "count"])
            .reset_index()
            .sort_values("policy", ignore_index=True)
        )
        dialogue_turns_summary_df["std_dev"] = (
            grouped_turns.std(ddof=1).fillna(0.0).reindex(dialogue_turns_summary_df["policy"]).to_numpy()
        )
        dialogue_turns_summary_df["q1"] = (
            grouped_turns.quantile(0.25).reindex(dialogue_turns_summary_df["policy"]).to_numpy()
        )
        dialogue_turns_summary_df["q3"] = (
            grouped_turns.quantile(0.75).reindex(dialogue_turns_summary_df["policy"]).to_numpy()
        )
        dialogue_turns_summary_df["iqr"] = (
            dialogue_turns_summary_df["q3"] - dialogue_turns_summary_df["q1"]
        )
        dialogue_turns_summary_df = dialogue_turns_summary_df[
            ["policy", "mean", "std_dev", "median", "q1", "q3", "count", "iqr"]
        ]
    dialogue_turns_summary_df
    return (dialogue_turns_summary_df,)


@app.cell
def _(metrics_summary_df):
    outcome_summary_df = (
        metrics_summary_df.loc[metrics_summary_df["metric"] == "outcome_match"]
        .set_index("policy")
        .sort_index()
    )
    outcome_summary_df
    return (outcome_summary_df,)


@app.cell
def _(metrics_summary_df):
    slot_accuracy_summary_df = (
        metrics_summary_df.loc[
            metrics_summary_df["metric"].isin(
                ["slot_accuracy_any_valid", "slot_accuracy_predicted_outcome"]
            )
        ]
        .sort_values(["metric", "policy"], ignore_index=True)
    )
    slot_accuracy_summary_df
    return (slot_accuracy_summary_df,)


@app.cell
def _(metrics_summary_df):
    prf_summary_df = (
        metrics_summary_df.loc[
            metrics_summary_df["metric"].isin(
                [
                    "macro_precision_predicted_outcome",
                    "macro_recall_predicted_outcome",
                    "macro_f1_predicted_outcome",
                    "precision_unknown_predicted_outcome",
                    "recall_unknown_predicted_outcome",
                    "f1_unknown_predicted_outcome",
                    "precision_no_predicted_outcome",
                    "recall_no_predicted_outcome",
                    "f1_no_predicted_outcome",
                    "precision_yes_predicted_outcome",
                    "recall_yes_predicted_outcome",
                    "f1_yes_predicted_outcome",
                ]
            )
        ]
        .sort_values(["metric", "policy"], ignore_index=True)
    )
    prf_summary_df
    return


@app.cell
def _(outcome_summary_df):
    _fig, _ax = plt.subplots()
    if outcome_summary_df.empty:
        _ax.text(0.5, 0.5, "No outcome data available", ha="center", va="center")
        _ax.axis("off")
    else:
        outcome_plot_df = outcome_summary_df.reset_index()
        _ax.bar(outcome_plot_df["policy"], outcome_plot_df["mean"])
        _ax.set_xlabel("Policy")
        _ax.set_ylabel("Mean outcome accuracy")
        _ax.set_ylim(0.0, 1.0)
        _ax.set_yticks([x / 10 for x in range(0, 11)])
        _fig.tight_layout()
    _fig
    return


@app.cell
def _(slot_accuracy_summary_df):
    _fig, _ax = plt.subplots()
    if slot_accuracy_summary_df.empty:
        _ax.text(0.5, 0.5, "No slot-accuracy data available", ha="center", va="center")
        _ax.axis("off")
    else:
        (
            slot_accuracy_summary_df.pivot(
                index="policy",
                columns="metric",
                values="mean",
            )
            .sort_index()
            .plot(kind="bar", rot=0, ylim=(0.0, 1.0), ax=_ax)
        )
        _ax.set_xlabel("Policy")
        _ax.set_ylabel("Mean slot accuracy")
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Efficiency Analysis

    The notebook reports `dialogue_turns` separately because efficiency benefits from both
    moment-based summaries (`mean`, `std_dev`) and robust summaries (`median`, `Q1`, `Q3`, `IQR`).
    """)
    return


@app.cell
def _(dialogue_turns_summary_df):
    dialogue_turns_summary_df
    return


@app.cell
def _(dialogue_metrics_df):
    _fig, _ax = plt.subplots()
    if dialogue_metrics_df.empty:
        _ax.text(0.5, 0.5, "No dialogue-turn data available", ha="center", va="center")
        _ax.axis("off")
    else:
        sns.boxplot(
            data=dialogue_metrics_df,
            x="policy",
            y="dialogue_turns",
            ax=_ax,
        )
        _ax.set_xlabel("Policy")
        _ax.set_ylabel("Dialogue turns")
        _fig.tight_layout()
    _fig
    return


@app.cell
def _(dialogue_metrics_df):
    dialogue_turns_per_outcome = (
        dialogue_metrics_df[["policy", "dialogue_turns", "outcome_match"]]
        .groupby(["policy", "outcome_match"], as_index=False)
        .agg(mean_dialogue_turns=("dialogue_turns", "mean"))
    )
    dialogue_turns_per_outcome
    return (dialogue_turns_per_outcome,)


@app.cell
def _(dialogue_turns_per_outcome):
    _fig, _ax = plt.subplots()
    correct_outcome_turns_df = dialogue_turns_per_outcome.loc[
        dialogue_turns_per_outcome["outcome_match"] == 1.0
    ]
    if correct_outcome_turns_df.empty:
        _ax.text(
            0.5,
            0.5,
            "No correct-outcome dialogue-turn data available",
            ha="center",
            va="center",
        )
        _ax.axis("off")
    else:
        _ax.bar(
            correct_outcome_turns_df["policy"],
            correct_outcome_turns_df["mean_dialogue_turns"],
        )
        _ax.set_xlabel("Policy")
        _ax.set_ylabel("Mean dialogue turns for correct outcomes")
        _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## CSV Export

    The notebook writes three descriptive summaries to the selected experiment directory:

    - `metrics_summary.csv`
    - `dialogue_turns_summary.csv`
    - `likert_summary.csv`
    """)
    return


@app.cell
def _(
    dialogue_turns_summary_df,
    export_dir,
    likert_summary_df,
    metrics_summary_df,
):
    metrics_summary_path = export_dir / "metrics_summary.csv"
    dialogue_turns_summary_path = export_dir / "dialogue_turns_summary.csv"
    likert_summary_path = export_dir / "likert_summary.csv"

    metrics_summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_summary_df.loc[metrics_summary_df["metric"] != "dialogue_turns"].to_csv(
        metrics_summary_path,
        index=False,
    )
    dialogue_turns_summary_df.to_csv(dialogue_turns_summary_path, index=False)
    likert_summary_df.to_csv(likert_summary_path, index=False)

    {
        "metrics_summary": metrics_summary_path,
        "dialogue_turns_summary": dialogue_turns_summary_path,
        "likert_summary": likert_summary_path,
    }
    return


if __name__ == "__main__":
    app.run()
