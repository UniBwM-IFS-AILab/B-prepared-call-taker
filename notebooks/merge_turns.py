# /// script
# dependencies = [
#     "marimo",
#     "polars==1.40.1",
#     "ems-prepared",
#     "sqlalchemy==2.0.50",
# ]
# requires-python = ">=3.13"
# [tool.uv.sources]
# ems-prepared = { path = "..", editable = true }
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from ems_prepared.util.custom_deepmerge import ignore_empty_merger


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Merge Consecutive Turns

    This notebook reads `dialog_data_raw.jsonl` from `convert_ods.py`
    and merges consecutive turns spoken by the same participant.
    """)
    return


@app.cell
def _():
    source_path = Path(__file__).with_name("dialog_data_raw.jsonl")
    merged_jsonl_path = Path(__file__).with_name("dialog_data_merged.jsonl")
    merged_sqlite_path = Path(__file__).with_name("dialog_data_merged.sqlite")
    valid_speakers = ["CALLER", "DISPATCHER", "EXTRA", "PATIENT", "BYSTANDER"]
    return merged_jsonl_path, merged_sqlite_path, source_path, valid_speakers


@app.cell
def _(source_path):
    mo.stop(
        not source_path.exists(),
        mo.md(
            f"""
            `dialog_data_raw.jsonl` was not found at `{source_path}`.

            Run `convert_ods.py` first to generate the raw dataset for this notebook.
            """
        ),
    )
    return


@app.cell
def _(source_path):
    full_dataset = pl.read_ndjson(source_path).sort("dialog_id", "turn_index")
    full_dataset
    return (full_dataset,)


@app.cell
def _():
    def merge_states(series: pl.Series) -> pl.Series:
        merged_state: dict[str, object] = {}

        for state_json in series.to_list():
            if state_json is None:
                continue

            parsed_state = json.loads(state_json)
            if not parsed_state:
                continue

            merged_state = ignore_empty_merger.merge(merged_state, parsed_state)

        return pl.Series([json.dumps(merged_state, sort_keys=True)])

    return (merge_states,)


@app.cell
def _(full_dataset, merge_states):
    fds_clean = (
        full_dataset.with_columns(
            consecutive_count=(
                (
                    pl.col("speaker").ne(pl.col("speaker").shift(1)).fill_null(True)
                    | pl.col("dialog_id")
                    .ne(pl.col("dialog_id").shift(1))
                    .fill_null(True)
                )
                .cast(pl.UInt32)
                .cum_sum()
            )
        )
        .group_by("consecutive_count", "dialog_id", maintain_order=True)
        .agg(
            num_merged=pl.len(),
            turn_index=pl.col("turn_index"),
            speaker=pl.col("speaker").unique(maintain_order=True),
            text=pl.col("text"),
            source=pl.col("source").unique(maintain_order=True),
            state=pl.col("state").map_batches(merge_states, return_dtype=pl.String),
        )
        .with_columns(first_turn_index=pl.col("turn_index").list.first())
        .sort("dialog_id", "first_turn_index", descending=False, maintain_order=True)
        .drop("first_turn_index")
        .with_columns(turn_id=pl.cum_count("dialog_id").over("dialog_id"))
    )
    fds_clean
    return (fds_clean,)


@app.cell
def _(fds_clean, valid_speakers):
    violations = fds_clean.filter(
        (pl.col("speaker").list.unique().list.len() != 1)
        | (pl.col("source").list.unique().list.len() != 1)
        | (pl.col("state").list.unique().list.len() != 1)
        | (pl.col("speaker").list.set_difference(valid_speakers).list.len() != 0)
        | (pl.col("turn_index").list.len() != pl.col("text").list.len())
    )
    assert violations.height == 0, (
        f"\nFound {violations.height} violations:\n{violations}"
    )
    violations
    return


@app.cell
def _(fds_clean):
    fds_checked = fds_clean.select(
        dialog_id=pl.col("dialog_id"),
        turn_id=pl.cum_count("dialog_id").over("dialog_id"),
        speaker=pl.col("speaker").list.first(),
        text=pl.col("text").list.agg(pl.element().str.strip_chars().str.join(" ")),
        state=pl.col("state").list.first(),
        source=pl.col("source").list.first(),
    )
    fds_checked
    return (fds_checked,)


@app.cell
def _(fds_checked, merged_jsonl_path, merged_sqlite_path):
    fds_checked.write_ndjson(merged_jsonl_path)
    fds_checked.write_database(
        table_name="records",
        connection=f"sqlite:///{merged_sqlite_path.as_posix()}",
        if_table_exists="replace",
    )
    mo.md(
        f"""
        Wrote `{merged_jsonl_path.name}` and `{merged_sqlite_path.name}`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
