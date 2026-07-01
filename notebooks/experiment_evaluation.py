"""Marimo Notebook for evaluating experiments. This notebook loads survey results and state extraction results from experiment logs, compares them to ground truths, and visualizes the findings."""

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")

with app.setup:
    # Initialization code that runs before all other cells
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import accuracy_score

    from ems_prepared.dialogue_state.medical_symptoms_state import MedicalEmergency


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Select Experiment
    """)
    return


@app.cell
def _():
    # Create dropdown for experiment selection
    experiment_dropdown = mo.ui.dropdown(
        options=["FLAIRS_old", "flairs", "ale"],
        value="flairs",
        label="Select Experiment:",
    )
    experiment_dropdown
    return (experiment_dropdown,)


@app.function
def get_sample_paths(experiment_path: Path) -> list[Path]:
    """Retrieve all sample paths from the given experiment directory.

    Args:
        experiment_path (Path): The path to the experiment directory.

    Returns:
        list[Path]: A list of paths to sample directories within the experiment directory.

    """
    return [
        sample
        for session in experiment_path.iterdir()
        if session.is_dir()
        for sample in session.iterdir()
        if sample.is_dir()
    ]


@app.cell
def _():
    # Set display options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Set plotting style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 100
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Survey Results Analysis

    This notebook demonstrates how to analyze survey results from experiments.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Load Survey Results
    """)
    return


@app.cell
def _(experiment_dropdown):
    experiment_path = Path("./logs") / experiment_dropdown.value
    list(experiment_path.rglob("survey.json"))[:3]
    return (experiment_path,)


@app.function
def load_surveys(experiment_path: Path):
    """Load survey data from the given experiment path.

    Args:
        experiment_path (Path): The path to the experiment directory.

    Returns:
        list[dict]: A list of dictionaries containing survey data and metadata.

    """
    sessions = get_sample_paths(experiment_path)
    rows = []

    for session_path in sessions:
        try:
            with open(
                (session_path / "survey.json"), "r", encoding="utf-8"
            ) as survey_file:
                survey_data = json.load(survey_file)
        except Exception:
            # Skip files that can't be parsed
            continue

        try:
            with open((session_path / "deps.json"), "r", encoding="utf-8") as deps_file:
                policy_name = json.load(deps_file)["policy_name"]
        except Exception:
            policy_name = "agent"

        assert isinstance(survey_data, dict)

        # Extract metadata (common keys)
        metadata = survey_data["metadata"]
        assert isinstance(metadata, dict)

        # Find responses container
        responses = survey_data["responses"]
        assert isinstance(responses, dict)

        session_results = {}

        if isinstance(responses, dict):
            for idx, (key, value) in enumerate(responses.items(), 1):
                score = value["score"]
                session_results[f"Q{idx}"] = score

        elif isinstance(responses, list):
            for idx, item in enumerate(responses, 1):
                score = item["score"]
                session_results[f"Q{idx}"] = score

        else:
            raise ValueError("Unexpected format for responses in survey data.")

        metadata["scenario"] = metadata["scenario"].strip(".md")
        rows.append(
            {
                "source_path": str(session_path),
                **metadata,
                "policy": policy_name,
                **session_results,
            }
        )

    return rows


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Create Dataframes
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Surveys
    """)
    return


@app.cell
def _(experiment_path):
    survey_results_df = pd.DataFrame(load_surveys(experiment_path)).set_index(
        ["user_id", "session_id"]
    )
    survey_results_df
    return (survey_results_df,)


@app.cell
def _(survey_results_df):
    survey_results_df["policy"].value_counts(normalize=True)
    return


@app.cell
def _(survey_results_df):
    by_policy = survey_results_df.groupby("policy").mean(numeric_only=True)
    by_policy.round(2)
    return (by_policy,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    #### Feedbacks
    """)
    return


@app.cell
def _(survey_results_df):
    feedbacks = survey_results_df["feedback"].dropna()
    feedbacks
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Plotting
    """)
    return


@app.cell
def _(by_policy):
    by_policy.T.plot(
        kind="bar",
        width=0.9,
        figsize=(10, 5),
        # rot=25,
        # position=0,
        rot=0,
        xlabel="Survey Results",
        ylim=(1, 5),
        # yticks=[
        #     "Strongly Disagree",
        #     "Disagree",
        #     "Neutral",
        #     "Agree",
        #     "Strongly Agree",
        # ],
    ).legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.1))

    # matplot2tikz.save("test.tex")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## State Analysis
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Load extraction results
    """)
    return


@app.function
def load_state_results(experiment_path: Path):
    """Load state results from the given experiment path.

    Args:
        experiment_path (Path): The path to the experiment directory.

    Returns:
        list[dict]: A list of dictionaries containing state results and metadata.

    """
    rows = []

    sessions = get_sample_paths(experiment_path)

    for session_path in sessions:
        try:
            messages_df = pd.read_csv(session_path / "messages.tsv", sep="\t")
            turn_count = (messages_df["speaker"] == "caller").sum()
        except Exception as exc:
            print(f"Error reading messages.tsv in {session_path}: {exc}")
            continue

        try:
            with open(
                (session_path / "final_state.json"), "r", encoding="utf-8"
            ) as state_file:
                state_data = json.load(state_file)
                if "call_state" in state_data:
                    state_data = state_data["call_state"]
                outcome = MedicalEmergency(**state_data).get_outcome()
        except Exception as exc:
            print(f"Error parsing final_state.json in {session_path}: {exc}")
            continue

        try:
            with open((session_path / "deps.json"), "r", encoding="utf-8") as deps_file:
                deps_data = json.load(deps_file)
                policy_name = deps_data.get("policy_name")
                scenario_name = deps_data.get("scenario_name")
        except FileNotFoundError:
            policy_name = "agent"
            try:
                with open(
                    (session_path / "survey.json"), "r", encoding="utf-8"
                ) as survey_file:
                    scenario_name = (
                        json.load(survey_file).get("metadata", {}).get("scenario")
                    )
            except FileNotFoundError as exc:
                print(f"Error parsing survey.json in {session_path}: {exc}")
                continue

        assert isinstance(state_data, dict)

        rows.append(
            {
                "scenario": scenario_name.strip(".md"),
                "outcome": outcome,
                "dialogue_turns": turn_count,
                "policy": policy_name,
                **state_data,
            }
        )

    return rows


@app.cell
def _(experiment_path):
    state_results_df = pd.DataFrame(load_state_results(experiment_path))
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Load Ground Truths
    """)
    return


@app.cell
def _():
    gt_dir = Path(".") / "docs" / "scenarios" / "ScenarioGroundTruth"
    gt_dir.absolute()
    return (gt_dir,)


@app.function
def extract_scenario_name(path: Path):
    """Extract the scenario name from the given file path.

    Args:
        path (Path): The path to the file.

    Returns:
        str: The extracted scenario name.

    """
    file_name = path.stem
    splits = file_name.split("_")
    return splits[0] + "_" + splits[1]


@app.function
def load_ground_truths(dir_path: Path):
    """Load ground truth data from the given directory path.

    Args:
        dir_path (Path): The path to the directory containing ground truth files.

    Returns:
        list[dict]: A list of dictionaries containing scenario names, outcomes, and ground truth data.

    """
    rows = []

    paths = dir_path.rglob("Scenario_*.json")
    for gt_path in paths:
        with open(gt_path, "r", encoding="utf-8") as gt_file:
            gt_data = json.load(gt_file)
            outcome = MedicalEmergency(**gt_data).get_outcome()

        rows.append(
            {"scenario": extract_scenario_name(gt_path), "outcome": outcome, **gt_data}
        )
    return rows


@app.cell
def _(gt_dir):
    gt_df = pd.DataFrame(load_ground_truths(gt_dir))
    gt_df = gt_df.loc[
        :,
        gt_df.columns.isin(
            ["scenario", "outcome", *MedicalEmergency.model_fields.keys()]
        ),
    ]
    gt_df
    return (gt_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Compare Outcomes
    """)
    return


@app.cell
def _(gt_df):
    def get_row_match(row):
        scenario, outcome = row["scenario"], row["outcome"]
        row["match"] = (scenario, outcome) in gt_df.set_index(
            ["scenario", "outcome"]
        ).index
        return row

    return (get_row_match,)


@app.cell
def _(gt_df):
    gt_df[gt_df["scenario"] == "Scenario_05"]["severe_injury"]  # .dropna(axis=1)
    return


@app.cell
def _(get_row_match, state_results_df):
    state_results_df["match"] = state_results_df.apply(get_row_match, axis=1)["match"]
    outcome_match_df = (
        state_results_df[["match", "policy"]].groupby("policy").agg(["mean", "count"])
    )
    print(outcome_match_df.columns)
    outcome_match_df
    return (outcome_match_df,)


@app.cell
def _(outcome_match_df):
    outcome_match_df[("match", "mean")]
    return


@app.cell
def _(outcome_match_df):
    pd.DataFrame(outcome_match_df[("match", "mean")]).T.plot(
        kind="bar",
        rot=0,
        ylim=(0.0, 1.0),
        xlabel="Dialogue-level accuracy",
        xticks=[],
        yticks=[x / 10 for x in range(0, 100, 1)],
    )  # .legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    return


@app.cell(disabled=True, hide_code=True)
def _(gt_df, state_results_df):
    outcome_columns = ["scenario", "outcome"]
    outcomes_df = pd.merge(
        state_results_df[["policy", *outcome_columns]],
        # state_results_df[outcome_columns],
        gt_df[outcome_columns],
        how="outer",
        on="scenario",
        suffixes=("_result", "_gt"),
        indicator=True,
    )
    outcomes_df["match"] = outcomes_df["outcome_gt"] == outcomes_df["outcome_result"]
    outcomes_df
    return (outcomes_df,)


@app.cell(disabled=True)
def _(outcomes_df):
    outcomes_df.groupby("policy").mean(numeric_only=True)
    return


@app.cell(disabled=True)
def _(outcomes_df):
    outcomes_df.groupby("policy").mean(numeric_only=True).T.plot(
        kind="bar",
        rot=0,
        xticks=[],
        ylim=(0.0, 1.0),  # , xlabel="policy"
    ).legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    return


@app.cell(disabled=True, hide_code=True)
def _(outcomes_df):
    sns.barplot(outcomes_df, x="policy", y="match")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Field-level Comparison

    - Three classes in pred: True False Unknown
    - Ground Truth only contains True and Unknown since setting to False is deemed inefficient
    - Comparison penalises values any difference between GT and pred, due to aim of efficiency (setting smth to false does not affect outcome but turn number)
    """)
    return


@app.cell(disabled=True, hide_code=True)
def _(gt_results_granular, state_results_granular):
    merged = pd.merge(
        state_results_granular,
        gt_results_granular,
        how="left",
        on=["scenario", "outcome"],
        suffixes=("_result", "_gt"),
        indicator=True,
    )
    FP = merged[merged["_merge"] == "left_only"]
    FP
    return


@app.cell
def _(gt_df):
    row = gt_df[
        (gt_df["scenario"] == "Scenario_05") & (gt_df["outcome"] == "rd2")
    ].drop(["scenario", "outcome"], axis=1)
    row.columns
    return


@app.cell(disabled=True)
def _(gt_df):
    def get_row_acc(row):
        replace_dict = {True: 1, False: 0}
        scenario, outcome = row["scenario"], row["outcome"]
        if (scenario, outcome) in gt_df.set_index(["scenario", "outcome"]).index:
            is_scenario = gt_df["scenario"] == scenario
            is_outcome = gt_df["outcome"] == outcome
            gt_row = (
                gt_df[is_scenario & is_outcome]
                .drop(["scenario", "outcome"], axis=1)
                .iloc[0]
            )
            common_columns = list(set(gt_row.index).intersection(row.index))
            row["acc"] = accuracy_score(
                *(
                    series[common_columns].fillna(2).astype(int).replace(replace_dict)
                    for series in [gt_row, row]
                ),
            )

        else:
            row["acc"] = 0.0
        return row

    return (get_row_acc,)


@app.cell
def _(get_row_acc, state_results_df):
    state_results_df.apply(get_row_acc, axis=1)["acc"].mean(numeric_only=True)
    return


@app.cell
def _(gt_df, state_results_df):
    def field_level_accuracy(
        df, gt_df, missing_sentinel=2, column_keys=["scenario", "outcome"]
    ):
        assert missing_sentinel != 0 and missing_sentinel != 1, (
            "0 and 1 are used for True and False, choose a different representation for the third class!"
        )

        # Label columns = columns you want to score (everything except keys)
        label_cols = gt_df.columns.difference(column_keys).intersection(df.columns)

        # Ensure unique key in GT (your old code silently took .iloc[0])
        gt_unique = gt_df[column_keys + list(label_cols)].drop_duplicates(
            column_keys, keep="first"
        )

        # Align GT to df rows
        merged = df.merge(
            gt_unique,
            on=column_keys,
            how="left",
            suffixes=("", "_gt"),
            indicator=True,
        )

        pred = merged[list(label_cols)]
        gt = merged[[col + "_gt" for col in label_cols]].copy()
        gt.columns = label_cols  # make columns match for easy comparison

        pred_int = pred.astype("Int8").fillna(missing_sentinel)
        gt_int = gt.astype("Int8").fillna(missing_sentinel)

        merged["acc"] = pred_int.eq(gt_int).mean(axis=1)

        # no ground truth available, miss-classification
        merged.loc[merged["_merge"] != "both", "acc"] = 0.0

        # Optional: drop helper columns created by merge
        merged = merged.drop(columns=["_merge"] + [col + "_gt" for col in label_cols])

        return merged

    field_level_accuracy(state_results_df, gt_df)["acc"].mean()
    return (field_level_accuracy,)


@app.cell
def _(gt_df, state_results_df):
    # pred, gt are DataFrames with identical label columns (no scenario/outcome columns)
    gt_pos: pd.DataFrame = gt_df.eq(
        True
    )  # gt_neg does not exist, no false values in gt
    pred_pos: pd.DataFrame = state_results_df.eq(True)
    pred_changed: pd.DataFrame = state_results_df.notna()

    # Per-row counts:
    # tp: required fields correctly set True
    # fn: required fields not set True (either False or untouched/NaN)
    # fp: non-required fields that were touched unnecessarily
    # tn: non-required fields left untouched
    tp: int = (gt_pos & pred_pos).sum(axis=1)
    fn: int = (gt_pos & ~pred_pos).sum(axis=1)
    fp: int = (~gt_pos & pred_changed).sum(axis=1)
    tn: int = (~gt_pos & ~pred_changed).sum(axis=1)

    # Number of action-positives (all touched fields), not just True-valued predictions.
    pred_changed_count: int = pred_changed.sum(axis=1)

    # True when every required field was satisfied, regardless of extras.
    recall: float = tp / tp + fn

    # Among touched fields, how many were truly required and set correctly.
    # Uses clip(lower=1) to avoid division by zero when nothing was touched.
    precision = tp / pred_changed_count.clip(lower=1)

    # Fraction of touched fields that were unnecessary (lower is better).
    fdr = fp / pred_changed_count.clip(lower=1)

    # Standard per-row accuracy over fields under the same action-based confusion setup.
    accuracy = (tp + tn) / (tp + tn + fp + fn).clip(lower=1)
    # Strict pass condition: all required fields satisfied and no unnecessary touches.
    perfect = (fn == 0) & (fp == 0)
    return


@app.cell
def _(field_level_accuracy, gt_df, state_results_df):
    accuracy_df = (
        field_level_accuracy(state_results_df, gt_df)[["policy", "acc"]]
        .groupby("policy")
        .agg(["mean", "count"])
    )
    accuracy_df
    return (accuracy_df,)


@app.cell
def _(accuracy_df):
    pd.DataFrame(accuracy_df[("acc", "mean")]).T.plot(
        yticks=[x / 10 for x in range(0, 100, 1)],
        kind="bar",
        rot=0,
        xlabel="Slot-level accuracy",
        ylim=(0.0, 1.0),
        xticks=[],
    )  # .legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Efficiency Analysis
    """)
    return


@app.cell
def _(state_results_df):
    state_results_df.head(1)
    return


@app.cell
def _(state_results_df):
    dialogue_turns = (
        state_results_df[["policy", "dialogue_turns"]]
        .groupby("policy")
        .agg(["mean", "median", "count"])
    )
    dialogue_turns
    return (dialogue_turns,)


@app.cell
def _(dialogue_turns):
    pd.DataFrame(dialogue_turns[("dialogue_turns", "mean")].rename("turns")).T.plot(
        kind="bar",
        rot=0,
        xlabel="Mean dialogue turns",
        xticks=[],  # ylim=(0.0, 1.0)
    )
    return


@app.cell
def _(state_results_df):
    dialogue_turns_per_outcome = (
        state_results_df[["policy", "dialogue_turns", "match"]]
        .groupby(["policy", "match"])
        .mean()
    )
    dialogue_turns_per_outcome
    return (dialogue_turns_per_outcome,)


@app.cell
def _(dialogue_turns_per_outcome):
    dialogue_turns_per_outcome.index
    return


@app.cell
def _(dialogue_turns_per_outcome):
    rows = [True in row for row in dialogue_turns_per_outcome.index]
    dialogue_turns_per_outcome[rows].T.plot(
        kind="bar",
        rot=0,
        xlabel="Mean dialogue turns for true positives",
        xticks=[],  # ylim=(0.0, 1.0)
    )
    return


if __name__ == "__main__":
    app.run()
