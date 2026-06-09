"""Marimo notebook for descriptive experiment evaluation."""

import marimo

__generated_with = "0.22.0"
app = marimo.App()

with app.setup(hide_code=True):
    import json
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score,
        recall_score,
        precision_score,
        f1_score,
        hamming_loss,
        jaccard_score,
        precision_recall_fscore_support,
        balanced_accuracy_score,
    )

    from ems_prepared.adapters.gradio.survey import DEFAULT_BY_ID
    from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100


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


@app.cell(hide_code=True)
def _():
    experiment_dir = Path(__file__).resolve().parent
    # logs_path = experiment_dir / ".." / ".." / "logs" / "FLAIRS39_old"
    logs_path = experiment_dir / "logs"
    gt_dir = experiment_dir / "scenario_ground_truth"
    export_dir = experiment_dir
    {
        "experiment_dir": experiment_dir,
        "logs_path": logs_path,
        "gt_dir": gt_dir,
        "export_dir": export_dir,
    }
    return export_dir, gt_dir, logs_path


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
    """)
    return


@app.function(hide_code=True)
def load_survey_results(logs_path):
    survey_rows = []

    for session_path in get_sample_paths(logs_path):
        try:
            with open(
                session_path / "survey.json", "r", encoding="utf-8"
            ) as _survey_file:
                survey_data = json.load(_survey_file)
        except Exception:
            continue

        try:
            with open(
                session_path / "deps.json", "r", encoding="utf-8"
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

        survey_rows.append(
            {
                "source_path": str(session_path),
                **metadata,
                "feedback": feedback,
                "policy": _policy_name,
                **session_scores,
            }
        )
    return survey_rows


@app.cell
def _(logs_path):
    survey_results_df = pd.DataFrame(load_survey_results(logs_path=logs_path))
    survey_results_df = survey_results_df.set_index(["user_id", "session_id"])
    # survey_results_df=survey_results_df.query('scenario not in ["Scenario_05","Scenario_08","Scenario_11","Scenario_12"]')
    survey_results_df
    return (survey_results_df,)


@app.cell
def _(survey_results_df):
    survey_results_df["policy"].value_counts()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Metrics

    #### Mean score

    $$
    \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
    $$

    #### Sample standard deviation

    $$
    s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}
    $$


    - Interpretation
        - The mean score describes the average user rating for a question.
        - The standard deviation describes how much respondents disagree.
    - Usefulness
        - Useful for subjective impressions such as clarity, fluency, and perceived responsiveness.
        - Useful as a complement to task metrics when you want to know whether users noticed behavioral differences.
    - Limitations
        - Likert scores are ordinal, so very small mean differences should not be over-interpreted.
        - A high mean does not imply task correctness.
        - A large standard deviation means users experienced the system less consistently.
    """)
    return


@app.cell
def _(survey_results_df):
    survey_question_cols = sorted(
        column for column in survey_results_df.columns if column.startswith("Q")
    )
    survey_long_df = (
        survey_results_df.melt(
            id_vars=["policy"],
            value_vars=survey_question_cols,
            var_name="id",
            value_name="score",
        )
        .dropna(subset=["score"])
        .sort_values(["id", "policy"], ignore_index=True)
    )

    question_metadata_df = pd.DataFrame(
        [
            {
                "id": id,
                "question_category": DEFAULT_BY_ID[int(id[1:])].category,
                "question_text": DEFAULT_BY_ID[int(id[1:])].text,
            }
            for id in survey_question_cols
        ]
    )

    likert_summary_df = (
        survey_long_df.groupby(["policy", "id"], as_index=False)
        .agg(
            count=("score", "count"),
            score_mean=("score", "mean"),
            score_std=("score", "std"),
        )
        .merge(question_metadata_df, on="id", how="left")
        .sort_values(["id", "policy"], ignore_index=True)
    )

    likert_summary_df
    return likert_summary_df, survey_question_cols


@app.cell
def _(likert_summary_df):
    survey_summary_table = likert_summary_df.pivot(
        index="policy",
        columns="id",
        values=["score_mean", "score_std", "count"],
    ).sort_index(axis=1, level=1)
    survey_summary_table.round(3)
    return


@app.cell
def _(likert_summary_df, survey_question_cols):
    _fig, _ax = plt.subplots(figsize=(10, 5))

    policies = list(pd.unique(likert_summary_df["policy"]))
    width = 0.8 / max(len(policies), 1)
    positions = np.arange(len(survey_question_cols), dtype=float)

    for index, policy in enumerate(policies):
        policy_df = (
            likert_summary_df[likert_summary_df["policy"] == policy]
            .set_index("id")
            .reindex(survey_question_cols)
        )
        x = positions + (index - (len(policies) - 1) / 2) * width
        _ax.bar(x, policy_df["score_mean"], width=width, label=policy)

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
    survey_results_df["feedback"].dropna()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## NLU / Slot filling

    Each dialogue is treated as one independent observation.
    The notebook evaluates only the final state of the dialogue, not turn-by-turn state tracking.
    """)
    return


@app.function(hide_code=True)
def load_state_results(logs_path):
    state_rows = []

    for session_path in get_sample_paths(logs_path):
        try:
            messages_df = pd.read_csv(session_path / "messages.tsv", sep="\t")
            turn_count = int((messages_df["speaker"] == "caller").sum())
        except Exception as exc:
            print(f"Error reading messages.tsv in {session_path}: {exc}")
            continue

        try:
            with open(
                session_path / "final_state.json", "r", encoding="utf-8"
            ) as state_file:
                state_data = json.load(state_file)
                if "call_state" in state_data:
                    state_data = state_data["call_state"]
                outcome = MedicalEmergency(**state_data).get_outcome()
        except Exception as exc:
            print(f"Error parsing final_state.json in {session_path}: {exc}")
            continue

        try:
            with open(
                session_path / "deps.json", "r", encoding="utf-8"
            ) as _deps_file:
                deps_data = json.load(_deps_file)
                _policy_name = deps_data.get("policy_name") or "agent"
                _scenario_name = deps_data.get("scenario_name")
        except FileNotFoundError:
            _policy_name = "agent"
            try:
                with open(
                    session_path / "survey.json", "r", encoding="utf-8"
                ) as _survey_file:
                    _scenario_name = (
                        json.load(_survey_file).get("metadata", {}).get("scenario")
                    )
            except FileNotFoundError as exc:
                print(f"Error parsing survey.json in {session_path}: {exc}")
                continue

        user_id, session_id = session_path.parent.stem, session_path.stem

        state_rows.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "scenario": _scenario_name.strip(".md")
                if isinstance(_scenario_name, str)
                else None,
                "outcome": outcome,
                "dialogue_turns": turn_count,
                "policy": _policy_name,
                **state_data,
            }
        )
    return state_rows


@app.cell
def _(logs_path):
    t = pd.DataFrame(load_state_results(logs_path=logs_path))
    # t.query("not dialogue_turns ==0")
    t
    return


@app.cell
def _(logs_path):
    state_results_df = pd.DataFrame(load_state_results(logs_path=logs_path)).query("not dialogue_turns ==0")
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
    # state_results_df=state_results_df.query('scenario not in ["Scenario_05","Scenario_08","Scenario_11","Scenario_12"]')
    state_results_df['scenario'].unique()
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
    del gt_data
    return (gt_rows,)


@app.cell
def _(gt_rows):
    gt_df = pd.DataFrame(gt_rows)
    gt_df = gt_df.loc[
        :,
        gt_df.columns.isin(
            ["scenario", "outcome", *MedicalEmergency.model_fields.keys()]
        ),
    ]
    gt_df
    return (gt_df,)


@app.cell
def _(gt_df):
    gt_df[gt_df['scenario'].duplicated()]['scenario']
    return


@app.cell
def _(state_results_df):
    metrics_df = state_results_df[
        ["scenario", "outcome", "policy", "dialogue_turns"]
    ].copy()
    outcome_df = metrics_df.copy()
    return metrics_df, outcome_df


@app.cell
def _(metrics_df):
    metrics_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Outcome Accuracy

    #### Formula

    For predicted outcome $\hat{o}_i$ and a valid ground-truth outcome $o_i^*$ for dialogue $i$:

    $$
    \mathrm{OutcomeMatch}_i = \mathbb{1}[\hat{o}_i = o_i^*]
    $$

    #### Interpretation

    - `1` means the final dispatch outcome is correct.
    - `0` means the final dispatch outcome is incorrect.

    #### Usefulness

    - This is the clearest task-success metric.
    - It directly answers whether the policy reached the correct final decision.

    #### Limitations

    - It ignores whether the internal state justified the outcome.
    - A policy can score well here even if the final symptom state is poorly grounded.
    """)
    return


@app.cell
def _(gt_df):
    valid_outcomes_by_scenario = gt_df.groupby("scenario")["outcome"].agg(
        lambda values: set(values)
    )
    valid_outcomes_by_scenario
    return (valid_outcomes_by_scenario,)


@app.cell
def _(state_results_df, valid_outcomes_by_scenario):
    state_results_df.apply(
        lambda row: row["outcome"] in valid_outcomes_by_scenario.get(row["scenario"]),
        axis=1,
    ).value_counts(normalize=True)
    return


@app.cell
def _(outcome_df, state_results_df, valid_outcomes_by_scenario):
    outcome_df["outcome_match"] = state_results_df.apply(
        lambda row: row["outcome"] in valid_outcomes_by_scenario.get(row["scenario"]),
        axis=1,
    )
    outcome_summary_df = outcome_df.groupby("policy")["outcome_match"].agg(
        ["mean", "sum", "count"]
    )
    outcome_summary_df
    return (outcome_summary_df,)


@app.cell
def _(outcome_df, valid_outcomes_by_scenario):
    valid_outcome_values = []
    for values in valid_outcomes_by_scenario.tolist():
        if isinstance(values, set):
            valid_outcome_values.extend(
                [value for value in values if value is not None and not pd.isna(value)]
            )

    predicted_outcome_values = [
        value
        for value in outcome_df["outcome"].tolist()
        if value is not None and not pd.isna(value)
    ]

    outcome_labels = sorted(
        {str(value) for value in [*valid_outcome_values, *predicted_outcome_values]}
    )

    outcome_correctness_metrics_rows = []
    for policy_name, _policy_df in outcome_df.groupby("policy", sort=False):
        y_true = []
        y_pred = []
        skipped_no_ground_truth = 0

        for row in _policy_df.itertuples(index=False):
            valid_labels = valid_outcomes_by_scenario.get(row.scenario, set())
            if not isinstance(valid_labels, set):
                valid_labels = set()
            valid_labels = sorted(
                {
                    str(valid_label)
                    for valid_label in valid_labels
                    if valid_label is not None and not pd.isna(valid_label)
                }
            )

            if len(valid_labels) == 0:
                skipped_no_ground_truth += 1
                continue

            predicted_label = (
                None
                if row.outcome is None or pd.isna(row.outcome)
                else str(row.outcome)
            )
            true_label = (
                predicted_label
                if predicted_label is not None and predicted_label in valid_labels
                else valid_labels[0]
            )

            y_true.append(true_label)
            y_pred.append(
                predicted_label if predicted_label is not None else "__missing_outcome__"
            )

        scoring_labels = outcome_labels + sorted(
            {label for label in y_pred if label not in outcome_labels}
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=scoring_labels,
            average="macro",
            zero_division=0,
        )
        jaccard = jaccard_score(
            y_true,
            y_pred,
            labels=scoring_labels,
            average="macro",
            zero_division=0,
        )

        outcome_correctness_metrics_rows.append(
            {
                "policy": policy_name,
                "count": int(len(_policy_df)),
                "evaluated_count": int(len(y_true)),
                "skipped_no_ground_truth": int(skipped_no_ground_truth),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "jaccard": float(jaccard),
            }
        )

    outcome_correctness_metrics_df = pd.DataFrame(outcome_correctness_metrics_rows)
    outcome_correctness_metrics_df
    return (outcome_correctness_metrics_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Slot Filling

    ### Data prep
    """)
    return


@app.cell
def _(state_results_df):
    # get diff between medical_fields and state_results.columns
    medical_cols = set(MedicalEmergency.model_fields.keys())
    state_result_fields = set(state_results_df.columns)
    print(medical_cols - state_result_fields)
    # remove fields from medical_colums that are not in state_result_fields
    medical_cols = medical_cols.intersection(state_result_fields)
    medical_cols
    return (medical_cols,)


@app.cell
def _(medical_cols):
    pred_cols = [f"{column}_pred" for column in medical_cols]
    gt_cols = [f"{column}_gt" for column in medical_cols]
    return gt_cols, pred_cols


@app.cell
def _(gt_cols, medical_cols):
    def filter_gt_policy(policy: str, how, gt, state_df):
        matched_df = state_df[
            ["scenario", "outcome", "policy", "dialogue_turns", *medical_cols]
        ].merge(
            gt,
            on=how,
            how="left",
            suffixes=("_pred", "_gt"),
            sort=False,
        )
        has_gt_match = matched_df[gt_cols].notna().any(axis=1)
        return matched_df[has_gt_match & (matched_df["policy"] == policy)]

    return (filter_gt_policy,)


@app.cell
def _(gt_cols, pred_cols):
    def get_pred_true(filtered_df) -> [np.array, np.array]:
        y_pred = filtered_df[pred_cols].fillna(False).to_numpy(np.int8)
        y_true = filtered_df[gt_cols].fillna(False).to_numpy(np.int8)
        return (y_true, y_pred)

    return (get_pred_true,)


@app.cell(disabled=True, hide_code=True)
def _(gt_cols, gt_df, medical_cols, pred_cols, state_results_df):
    matched_df = state_results_df[["scenario", "outcome", "policy", *medical_cols]].merge(
        gt_df,
        on=["scenario", "outcome"],
        how="left",
        suffixes=("_pred", "_gt"),
        sort=False,
    )

    rowwise_match_matrix = (
        matched_df[pred_cols].fillna(False).to_numpy()
        == matched_df[gt_cols].fillna(False).to_numpy()
    )
    rowwise_match_matrix
    return (matched_df,)


@app.cell
def _(gt_cols, matched_df):
    # rows where got at least one value that is not python None or numpy.NAN in gt cols
    has_gt_match = matched_df[gt_cols].notna().any(axis=1)
    has_gt_match
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Outcome-matched Slot-Filling Metrics

    - GTs are matched based on Scenario and outcome
    - Unkown (None) values are assumed false to allow binary comparison
        - False / Unknown only matters to evaluate dialogue efficiency.
        - If a slot is False it has been "visited", no semantic interpretation beyond that.
    """)
    return


@app.cell
def _():
    from scipy.stats import iqr

    def get_metrics(y_true, y_pred) -> dict[str, float | None]:
        f1_per_row = np.array([
            f1_score(y_true[i], y_pred[i], zero_division=0)
            for i in range(len(y_true))
        ])

        result = dict(
            zip(
                (
                    "precision",
                    "recall",
                    "f1",
                    "support",
                    "f1_iqr",
                    "subset_accuracy",
                    "hamming_loss",
                    "jaccard",
                ),
                (
                    *precision_recall_fscore_support(
                        y_true, y_pred, average="samples", zero_division=0
                    ),
                    iqr(f1_per_row),
                    accuracy_score(y_true, y_pred),
                    hamming_loss(y_true, y_pred),
                    jaccard_score(y_true, y_pred, average="samples", zero_division=0),
                ),
            )
        )
        result.pop("support")
        return result

    return (get_metrics,)


@app.cell
def _(filter_gt_policy, get_metrics, get_pred_true, gt_df, state_results_df):
    filtered_df_graph = filter_gt_policy(
        "graph", ["scenario", "outcome"], gt_df, state_results_df
    )
    graph_results = get_metrics(*get_pred_true(filtered_df_graph))
    graph_results
    return filtered_df_graph, graph_results


@app.cell
def _(filter_gt_policy, get_metrics, get_pred_true, gt_df, state_results_df):
    filtered_df_agent = filter_gt_policy(
        "agent", ["scenario", "outcome"], gt_df, state_results_df
    )
    agent_results = get_metrics(*get_pred_true(filtered_df_agent))
    agent_results
    return agent_results, filtered_df_agent


@app.cell
def _(agent_results, filtered_df_agent, filtered_df_graph, graph_results):
    pd.DataFrame(
        [
            {
                "policy": "graph",
                "count": len(filtered_df_graph),
                **graph_results,
            },
            {
                "policy": "agent",
                "count": len(filtered_df_agent),
                **agent_results,
            },
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Outcome-Matched Turn Efficiency

    Turn-efficiency below is computed only on outcome-matched rows from the exact `(scenario, outcome)` alignment above.
    """)
    return


@app.cell
def _(filtered_df_agent, filtered_df_graph):
    outcome_matched_turns_df = pd.concat(
        [
            filtered_df_graph[["policy", "dialogue_turns"]],
            filtered_df_agent[["policy", "dialogue_turns"]],
        ],
        ignore_index=True,
    )

    outcome_matched_turns_summary_df = (
        outcome_matched_turns_df.groupby("policy", as_index=False)["dialogue_turns"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            q25=lambda turns: turns.quantile(0.25),
            q75=lambda turns: turns.quantile(0.75),
            max="max",
        )
        .sort_values("policy", ignore_index=True)
    )
    outcome_matched_turns_summary_df["iqr"] = (
        outcome_matched_turns_summary_df["q75"]
        - outcome_matched_turns_summary_df["q25"]
    )
    outcome_matched_turns_summary_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Forgiving Slot-Filling Metrics

    - GTs are matched on `scenario` only.
    - Each prediction is compared against every GT candidate for that scenario.
    - For each metric, the best score across available GT candidates is kept instead of collapsing GTs into one union row.
    """)
    return


@app.cell
def _(gt_df):
    scenario_gt_candidates = gt_df.drop(columns=["outcome"], errors="ignore").reset_index(
        drop=True
    )
    scenario_gt_candidates.insert(0, "gt_candidate_id", scenario_gt_candidates.index)

    def get_best_scenario_metrics(policy: str, state_df: pd.DataFrame):
        prediction_cols = [
            column
            for column in scenario_gt_candidates.columns
            if column not in {"gt_candidate_id", "scenario"}
        ]

        policy_df = (
            state_df.loc[
                state_df["policy"] == policy,
                ["scenario", "outcome", "policy", *prediction_cols],
            ]
            .reset_index(drop=True)
            .copy()
        )

        policy_df.insert(0, "prediction_id", policy_df.index)
        candidate_pairs = policy_df.merge(
            scenario_gt_candidates,
            on="scenario",
            how="inner",
            suffixes=("_pred", "_gt"),
            sort=False,
        )

        pred_cols = [f"{column}_pred" for column in prediction_cols]
        gt_cols = [f"{column}_gt" for column in prediction_cols]

        def score_candidate_pair(row: pd.Series) -> pd.Series:
            y_pred = row[pred_cols].fillna(False).to_numpy(np.int8)
            y_true = row[gt_cols].fillna(False).to_numpy(np.int8)
            return pd.Series(
                {
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "subset_accuracy": float(np.array_equal(y_true, y_pred)),
                    "hamming_loss": float(np.mean(y_true != y_pred)),
                    "jaccard": jaccard_score(y_true, y_pred, zero_division=0),
                }
            )

        candidate_scores = candidate_pairs.apply(score_candidate_pair, axis=1)
        candidate_pairs = pd.concat([candidate_pairs, candidate_scores], axis=1)

        best_scores = candidate_pairs.groupby("prediction_id", sort=False).agg(
            precision=("precision", "max"),
            recall=("recall", "max"),
            f1=("f1", "max"),
            subset_accuracy=("subset_accuracy", "max"),
            hamming_loss=("hamming_loss", "min"),
            jaccard=("jaccard", "max"),
        )

        f1_iqr = best_scores["f1"].quantile(0.75) - best_scores["f1"].quantile(0.25)
        return {
            "count": int(len(best_scores)),
            "precision": float(best_scores["precision"].mean()),
            "recall": float(best_scores["recall"].mean()),
            "f1": float(best_scores["f1"].mean()),
            "f1_iqr": float(f1_iqr),
            "subset_accuracy": float(best_scores["subset_accuracy"].mean()),
            "hamming_loss": float(best_scores["hamming_loss"].mean()),
            "jaccard": float(best_scores["jaccard"].mean()),
        }

    forgiving_gt_df = {
        "scenario_gt_candidates": scenario_gt_candidates,
        "get_best_scenario_metrics": get_best_scenario_metrics,
    }
    scenario_gt_candidates
    return (forgiving_gt_df,)


@app.cell
def _(forgiving_gt_df, state_results_df):
    forgiving_graph_results = forgiving_gt_df["get_best_scenario_metrics"](
        "graph", state_results_df
    )
    forgiving_graph_results
    return (forgiving_graph_results,)


@app.cell
def _(forgiving_gt_df, state_results_df):
    forgiving_agent_results = forgiving_gt_df["get_best_scenario_metrics"](
        "agent", state_results_df
    )
    forgiving_agent_results
    return (forgiving_agent_results,)


@app.cell
def _(forgiving_agent_results, forgiving_graph_results):
    pd.DataFrame(
        [
            {"policy": "graph", **forgiving_graph_results},
            {"policy": "agent", **forgiving_agent_results},
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Collapsed Slot-Filling Metric

    - GTs are matched on `scenario` only.
    - All GT variants for the same scenario are collapsed into one union row.
    - A slot is treated as `True` if any GT variant for that scenario marks it as `True`.
    """)
    return


@app.cell
def _(gt_df):
    collapsed_gt_df = (
        gt_df.drop(columns=["outcome"], errors="ignore")
        .groupby("scenario")
        .agg("sum")
        .astype(bool)
        .reset_index()
    )
    collapsed_gt_df
    return (collapsed_gt_df,)


@app.cell
def _(
    collapsed_gt_df,
    filter_gt_policy,
    get_metrics,
    get_pred_true,
    state_results_df,
):
    filtered_collapsed_graph_df = filter_gt_policy(
        "graph", "scenario", collapsed_gt_df, state_results_df
    )
    collapsed_graph_results = get_metrics(
        *get_pred_true(filtered_collapsed_graph_df)
    )
    collapsed_graph_results
    return collapsed_graph_results, filtered_collapsed_graph_df


@app.cell
def _(
    collapsed_gt_df,
    filter_gt_policy,
    get_metrics,
    get_pred_true,
    state_results_df,
):
    filtered_collapsed_agent_df = filter_gt_policy(
        "agent", "scenario", collapsed_gt_df, state_results_df
    )
    collapsed_agent_results = get_metrics(
        *get_pred_true(filtered_collapsed_agent_df)
    )
    collapsed_agent_results
    return collapsed_agent_results, filtered_collapsed_agent_df


@app.cell
def _(
    collapsed_agent_results,
    collapsed_graph_results,
    filtered_collapsed_agent_df,
    filtered_collapsed_graph_df,
):
    pd.DataFrame(
        [
            {
                "policy": "graph",
                "count": len(filtered_collapsed_graph_df),
                **collapsed_graph_results,
            },
            {
                "policy": "agent",
                "count": len(filtered_collapsed_agent_df),
                **collapsed_agent_results,
            },
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Metric: `subset_accuracy`

    #### Formula
    $$
    \mathrm{SubsetAccuracy}_i =
    \mathbb{1}[\hat{s}_i = s_i^{(\hat{o}_i)}]
    $$

    Where:
    - $\hat{s}_i$ is the predicted slot vector for dialogue $i$.
    - $s_i^{(\hat{o}_i)}$ is the aligned ground-truth slot vector for dialogue $i$ under predicted outcome $\hat{o}_i$.
    - $\mathbb{1}[\cdot]$ is an indicator that equals 1 when the condition is true, and 0 otherwise.

    In plain words, this formula gives score 1 only when every slot matches exactly; if even one slot differs, the score is 0.

    Across matched rows, sklearn computes:
    $$
    \mathrm{SubsetAccuracy} = \mathrm{accuracy\_score}(S^*, \hat{S})
    $$

    - `1` means every evaluated slot matches exactly.
    - `0` means at least one slot differs.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Metric: `jaccard_index`

    $$
    \mathrm{JaccardIndex}_i = \frac{|Y_i \cap \hat{Y}_i|}{|Y_i \cup \hat{Y}_i|}
    $$

    Where $Y_i$ is the set of positive slots in ground truth and $\hat{Y}_i$ is the set of positive slots in the prediction.

    This notebook computes a row-wise Jaccard index using sklearn and masks rows without a matched `(scenario, outcome)` ground-truth row to `0.0`.

    - `1.0` means perfect overlap on positive slots.
    - `0.0` means no overlap on positive slots.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Efficiency Analysis

    The notebook reports `dialogue_turns` separately because efficiency benefits from both
    moment-based summaries (`mean`, `std_dev`) and robust summaries (`median`, `Q1`, `Q3`, `IQR`).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Metric Family: `dialogue_turns`

    `dialogue_turns` counts caller utterances.

    ### Mean

    $$
    \bar{t} = \frac{1}{n}\sum_{i=1}^{n} t_i
    $$

    ### Sample standard deviation

    $$
    s_t = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(t_i - \bar{t})^2}
    $$

    ### Median and quartiles

    - `median`: 50th percentile of the caller-turn distribution
    - `Q1`: 25th percentile of the caller-turn distribution
    - `Q3`: 75th percentile of the caller-turn distribution

    ### Interquartile range

    $$
    \mathrm{IQR} = Q3 - Q1
    $$

    ### Interpretation

    - Mean and standard deviation describe average burden and overall spread.
    - Median, quartiles, and IQR describe the typical case and robustness to skew.

    ### Usefulness

    - Direct efficiency metric for conversational burden.

    ### Limitations

    - Fewer turns are not automatically better if task correctness drops.
    - Mean and median should be read together when the distribution is skewed.
    """)
    return


@app.cell
def _(grouped_turns):
    grouped_turns.agg(
        mean="mean",
        median="median",
        count="count",
        std_dev="std",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        iqr=lambda x: x.quantile(0.75) - x.quantile(0.25)
    )
    return


@app.cell
def _(state_results_df):
    grouped_turns = state_results_df.groupby("policy")["dialogue_turns"]
    dialogue_turns_summary_df= grouped_turns.agg(
        count="count",
        mean="mean",
        std_dev="std",
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        iqr=lambda x: x.quantile(0.75) - x.quantile(0.25)
    )
    dialogue_turns_summary_df
    return dialogue_turns_summary_df, grouped_turns


@app.cell
def _(state_results_df):
    _fig, _ax = plt.subplots()

    sns.boxplot(
        data=state_results_df,
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
def _(dialogue_turns_summary_df, export_dir, state_results_df):
    export_dir.mkdir(parents=True, exist_ok=True)

    summary_reference_df = dialogue_turns_summary_df.reset_index().sort_values(
        "policy", kind="stable", ignore_index=True
    ).assign(
        draw_position=lambda df: np.arange(1, len(df) + 1, dtype=int),
        lower_fence=lambda df: df["q1"] - 1.5 * df["iqr"],
        upper_fence=lambda df: df["q3"] + 1.5 * df["iqr"],
    )
    whisker_rows = []
    outlier_map = {}

    for summary_row in summary_reference_df.itertuples(index=False):
        policy_values = np.sort(
            pd.to_numeric(
                state_results_df.loc[
                    state_results_df["policy"] == summary_row.policy,
                    "dialogue_turns",
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        if policy_values.size == 0:
            raise ValueError(
                f"No dialogue-turn samples available for policy {summary_row.policy}."
            )

        inlier_values = policy_values[
            (policy_values >= summary_row.lower_fence)
            & (policy_values <= summary_row.upper_fence)
        ]
        if inlier_values.size == 0:
            raise ValueError(
                f"No inlier dialogue-turn samples available for policy {summary_row.policy}."
            )

        lower_whisker = float(inlier_values.min())
        upper_whisker = float(inlier_values.max())
        outlier_values = policy_values[
            (policy_values < lower_whisker) | (policy_values > upper_whisker)
        ]

        whisker_rows.append(
            {
                "policy": summary_row.policy,
                "lower_whisker": lower_whisker,
                "upper_whisker": upper_whisker,
            }
        )
        outlier_map[summary_row.policy] = [int(value) for value in outlier_values]

    dialogue_turns_boxplot_df = summary_reference_df.merge(
        pd.DataFrame(whisker_rows),
        on="policy",
        how="left",
        validate="one_to_one",
    ).sort_values(
        ["draw_position", "policy"], kind="stable", ignore_index=True
    )
    dialogue_turns_boxplot_df["outlier_count"] = dialogue_turns_boxplot_df[
        "policy"
    ].map(lambda policy: len(outlier_map.get(policy, [])))

    max_outlier_count = max(dialogue_turns_boxplot_df["outlier_count"], default=0)
    for outlier_idx in range(max_outlier_count):
        column_name = f"outlier_{outlier_idx + 1}"
        dialogue_turns_boxplot_df[column_name] = pd.array(
            dialogue_turns_boxplot_df["policy"].map(
                lambda policy: (
                    outlier_map.get(policy, [])[outlier_idx]
                    if outlier_idx < len(outlier_map.get(policy, []))
                    else None
                )
            ),
            dtype="Int64",
        )

    dialogue_turns_boxplot_df.to_csv(
        export_dir / "dialogue_turns_boxplot.csv", index=False
    )
    dialogue_turns_boxplot_df
    return


@app.cell
def _(outcome_df):
    dialogue_turns_per_outcome = (
        outcome_df[["policy", "dialogue_turns", "outcome_match"]]
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

    The notebook writes descriptive summaries and a prepared boxplot table to the selected experiment directory:

    - `metrics_summary.csv`
    - `dialogue_turns_summary.csv`
    - `dialogue_turns_boxplot.csv`
    - `likert_summary.csv`
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(outcome_correctness_metrics_df):
    metrics_summary_df = outcome_correctness_metrics_df.copy()
    metrics_summary_df
    return (metrics_summary_df,)


@app.cell
def _(
    dialogue_turns_summary_df,
    export_dir,
    likert_summary_df,
    metrics_summary_df,
    outcome_summary_df,
):
    export_dir.mkdir(parents=True, exist_ok=True)


    outcome_summary_df.reset_index().to_csv(export_dir / "outcome_summary.csv", index=False)
    metrics_summary_df.to_csv(export_dir / "metrics_summary.csv", index=False)
    dialogue_turns_summary_df.reset_index().to_csv(
        export_dir / "dialogue_turns_summary.csv", index=False
    )
    likert_summary_df.to_csv(export_dir / "likert_summary.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
