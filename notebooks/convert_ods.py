# /// script
# dependencies = [
#     "fastexcel==0.20.2",
#     "marimo",
#     "mcp==1.27.0",
#     "polars==1.40.1",
#     "ems-prepared",
#     "frictionless[ods]==5.19.0",
#     "pydantic==2.13.3",
#     "pandera==0.31.1",
#     "pointblank==0.24.0",
#     "altair==6.1.0",
#     "sqlalchemy==2.0.50",
#     "pyarrow>=8.0.0",
# ]
# requires-python = ">=3.13"
# [tool.uv.sources]
# ems-prepared = { path = "..", editable = true }
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import os
    import re
    from pathlib import Path
    from uuid import UUID, uuid5

    import fastexcel
    import marimo as mo
    import pandera.polars as pa
    import polars as pl
    from pandera.polars import PolarsData
    from pydantic import ValidationError

    from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall


@app.cell
def _():
    search_dir = Path(os.path.expanduser(r"~/Documents/DialogData/Medical_Transcripts_files"))
    audio_dir = Path(
        os.path.expanduser(
            r"~/Spaces/B-prepared/Datasets/Schulungsgespräche Notrufannahme/"
        )
    )

    pl_schema = {
        "dialog_id": pl.UInt32,
        "turn_id": pl.UInt32,
        "text": pl.String,
        "speaker": pl.String,
        "state": pl.String,
    }
    valid_speakers = ["CALLER", "DISPATCHER", "EXTRA", "PATIENT", "BYSTANDER"]
    pattern = r"^.*(\d{6}).*?(\d+)"

    filter_df = pl.DataFrame(data=None, schema=pl_schema)
    return audio_dir, filter_df, search_dir, valid_speakers


@app.cell
def _(search_dir):
    len(list(search_dir.rglob("[!~$]*.ods")))
    return


@app.cell(disabled=True)
def _(audio_dir, dir, search_dir, wav):
    import shutil

    # move wav files to corresponding transcript dir
    # /home/seapat/Spaces/B-prepared/Datasets/Schulungsgespräche Notrufannahme/
    for _wav in audio_dir.rglob("*.wav"):
        if _wav.is_file():
            for transcript in search_dir.rglob("[!~$]*.ods"):
                _dir = transcript.parent
                if dir.stem.startswith(wav.stem):
                    shutil.copy2(_wav, Path(_dir) / _wav.name)
    return


@app.cell
def _(valid_speakers):
    def state_is_valid(value: str | None) -> bool:
        if value is None:
            return True

        try:
            EmergencyCall.model_validate_json(value)
            return True
        except ValidationError:
            return False

    def speaker_is_valid(value: str | None) -> bool:
        return value in valid_speakers

    def non_empty(value: str | None) -> bool:
        return value is not None

    return non_empty, speaker_is_valid, state_is_valid


@app.cell
def _(non_empty, speaker_is_valid, state_is_valid):
    pa_import_schema = pa.DataFrameSchema(
        {
            "start": pa.Column(str, nullable=True),
            "end": pa.Column(str, nullable=True),
            "speaker": pa.Column(
                str,
                checks=pa.Check(speaker_is_valid, element_wise=True),
                nullable=False,
            ),
            "text": pa.Column(
                str,
                checks=pa.Check(non_empty, element_wise=True),
                nullable=False,
            ),
            "State": pa.Column(
                str,
                checks=pa.Check(state_is_valid, element_wise=True),
                nullable=True,
            ),
            "approx_line_number": pa.Column(int),
        },
        strict=False,
        coerce=True,
    )
    return (pa_import_schema,)


@app.cell
def _():
    pl_report_schema = pl.Schema(
        {
            "source_file": pl.String,
            "approx_line_number": pl.Int32,
            "origin": pl.String,
            "exception_class": pl.String,
            "description": pl.String,
            "cell_value": pl.String,
        }
    )
    return (pl_report_schema,)


@app.cell
def _(pa_import_schema, pl_report_schema, search_dir):
    correct_paths = []
    errors = []

    for _path in search_dir.rglob("[!~$]*.ods"):
        contains_error: bool = False
        file_id = f"{_path.parent.name}/{_path.name}"

        _ods_df = pl.read_ods(_path).with_row_index("approx_line_number", offset=2)

        body_df = _ods_df.slice(0, _ods_df.height - 1)
        final_row = _ods_df.tail(1)

        try:
            pa_import_schema.validate(body_df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            contains_error = True
            errors.append(
                exc.failure_cases.with_columns(
                    pl.lit(file_id).alias("source_file"),
                    (pl.col("index") + 2).alias("approx_line_number"),
                    pl.lit("transcript").alias("origin"),
                    pl.lit("SchemaErrors").alias("exception_class"),
                    pl.concat_str(
                        [
                            pl.lit("column: "),
                            pl.col("column"),
                            pl.lit(" | check: "),
                            pl.col("check"),
                        ],
                        ignore_nulls=True,
                    ).alias("description"),
                    pl.col("failure_case").alias("cell_value"),
                ).select(
                    "source_file",
                    "approx_line_number",
                    "origin",
                    "exception_class",
                    "description",
                    "cell_value",
                )
            )
            # continue
        except pl.exceptions.ColumnNotFoundError as exc:
            contains_error = True
            errors.append(
                pl.DataFrame(
                    {
                        "source_file": [file_id],
                        "approx_line_number": [1],
                        "origin": "transcript",
                        "exception_class": "ColumnNotFoundError",
                        "description": [str(exc)],
                        "cell_value": [None],
                    },
                    schema=pl_report_schema,
                )
            )
            continue

        final_state = final_row["State"][0]
        try:
            EmergencyCall.model_validate_json(final_state)
        except ValidationError as exc:
            contains_error = True
            final_line_number = final_row["approx_line_number"][0]
            errors.append(
                pl.DataFrame(
                    {
                        "source_file": [file_id],
                        "origin": "final_row",
                        "approx_line_number": [final_line_number],
                        "exception_class": "ValidationError",
                        "description": [str(exc)],
                        "cell_value": [
                            final_state or f"last line:{_ods_df.height + 2}"
                        ],
                    },
                    schema=pl_report_schema,
                )
            )
        if not contains_error:
            correct_paths.append(_path)

    errors = (
        pl.concat(errors, how="diagonal")
        if errors
        else pl.DataFrame(schema=pl_report_schema)
    )

    errors.write_csv(os.path.expanduser(r"~/Documents/DialogData/validation_errors.csv"))
    errors
    return (correct_paths,)


@app.cell
def _(correct_paths):
    pl_fds_schema = pl.Schema(
        {
            "source": pl.String,
            "dialog_id": pl.UInt64,
            "turn_index": pl.UInt32,
            "speaker": pl.String,  # pl.Enum(valid_speakers),
            "text": pl.String,
            "state": pl.String,
            # "state": pl.Struct(fields={name: pl.String for name in EmergencyCall.model_fields.keys()}), #Struct(fields={name: pl.Bool | pl.String for name in EmergencyCall.model_fields.keys()})
            "path": pl.String,
        
        }
    )
    full_dataset = pl.DataFrame(schema=pl_fds_schema)
    # full_dataset = pl.DataFrame()
    for _path in correct_paths:
        _ods_df = pl.read_ods(
            _path, 
            drop_empty_rows=True, 
            drop_empty_cols=True, 
            # schema_overrides={"State": pl.Struct(fields={name: pl.String for name in EmergencyCall.model_fields.keys()})}
        )#.col('State').fill_null(pl.lit("{}"))

        _body_df = _ods_df.slice(0, _ods_df.height - 1)
        _final_row = _ods_df.tail(1)

        file_hash = _body_df.hash_rows(seed=42).sum()
        filter_ods_df = (
            _body_df.rename({"State": "state"})
            .with_row_index(name="turn_index")
            .with_columns(
                dialog_id=pl.lit(file_hash, dtype=pl.UInt64),
                source=pl.lit(f"{_path.parent.name}/{_path.name}".strip(".ods")),
                path=pl.lit(str(_path)),
                # audio=pl.lit()
            )
            .select(
                "source", "dialog_id", "turn_index", "speaker", "text", "state", "path"
            )
        )
        full_dataset.vstack(filter_ods_df, in_place=True).rechunk()

    # filter_ods_df#.to_dicts
    full_dataset
    # _ods_df.hash_rows(seed=42).sum()
    return (full_dataset,)


@app.cell
def _(full_dataset_state):
    full_dataset_state

    return


@app.cell
def state(full_dataset):
    # Ensure 'state' column exists and fill missing values for non‑caller turns
    full_dataset_state = full_dataset.with_columns(
        state=pl.when(pl.col("turn_index") != 0 & pl.col("state").is_null())
        .then(
            pl.coalesce(pl.col("state").fill_null(strategy="forward"), pl.lit("{}"))
        )  # copy previous row or '{}'
        .otherwise(
            pl.col("state").struct.json_encode(), # keep caller’s own state unchanged
        )
        .fill_null(pl.lit("{}"))
    )

    # Show the updated DataFrame so you can verify the changes
    # full_dataset_state

    full_dataset_state = full_dataset_state.filter(pl.col("speaker").eq("EXTRA").not_())
    full_dataset_state.write_ndjson("dialog_data.jsonl") 
    # full_dataset_state.match_to_schema(
    #     {"state": pl.Struct(fields={name: pl.String for name in EmergencyCall.model_fields.keys()})},
    #     extra_columns="ignore",
    #     missing_struct_fields="insert",
    #     extra_struct_fields="ignore"
    # )
    full_dataset_state.write_database(
        table_name="records",
        connection="sqlite:///data.sqlite",
        if_table_exists="replace",
    )
    return (full_dataset_state,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Add audio paths
    """)
    return


@app.cell(disabled=True)
def _(audio_dir):
    audio_df = pl.DataFrame(
        {"audio_path": audio_dir.rglob("*.wav")},
    ).with_columns(
        pl.col("audio_path")
        .map_elements(lambda p: p.stem, return_dtype=pl.String)
        .alias("wav_stem")
    )
    audio_df
    return (audio_df,)


@app.cell(disabled=True)
def _(audio_dir, search_dir):
    wav = list(audio_dir.rglob("*.wav"))[0]
    path = list(search_dir.rglob("[!~$]*.ods"))[0]
    dir = path.parent.stem
    print(dir)
    wav.stem
    return dir, wav


@app.cell(disabled=True)
def _(full_dataset):
    full_dataset.with_columns(
        path_stem=pl.col("path").map_elements(
            lambda p: str(Path(p).parent.stem), return_dtype=pl.String
        )
    )
    return


@app.cell
def _(audio_df, full_dataset):
    matches = full_dataset.with_columns(
        path_stem=pl.col("path").map_elements(
            lambda p: str(Path(p).parent.stem), return_dtype=pl.String
        )
    ).join(audio_df, left_on="path_stem", right_on="wav_stem", how="left")
    matches
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Merge cells
    """)
    return


@app.cell
def _():
    import json

    from ems_prepared.util.custom_deepmerge import ignore_empty_merger

    def merge_states(series: pl.Series) -> str:
        """Merge a series of state JSON strings using ignore_empty_merger."""
        merged: dict = {}
        for val in series.to_list():
            if val is None:
                continue
            parsed = json.loads(val)
            if not parsed:
                continue
            merged = ignore_empty_merger.merge(merged, parsed)
        return pl.Series([json.dumps(merged, sort_keys=True)])

    return (merge_states,)


@app.cell
def _(full_dataset, merge_states):
    fds_clean = (
        (  # give rows with the same speaker from the same call the same consecutive_count to allow merging
            full_dataset.with_columns(
                (
                    (pl.col("speaker") != pl.col("speaker").shift(1))
                    & (pl.col("dialog_id") == pl.col("dialog_id").shift(1))
                    & (pl.col("turn_index") > pl.col("turn_index").shift(1))
                )
                .cum_sum()
                .over("dialog_id")
                .alias("consecutive_count")
            ).group_by("consecutive_count", "dialog_id", maintain_order=True)
        )
        .agg(
            pl.col("speaker").len().alias("num_merged"),
            turn_index=pl.col("turn_index"),
            speaker=pl.col("speaker").unique(),
            text=pl.col("text"),
            source=pl.col("source").unique(),
            state=pl.col("state").map_batches(merge_states, pl.String),
        )
        .sort("dialog_id", "turn_index", descending=False, maintain_order=True)
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
        | (pl.col("speaker").list.set_difference(valid_speakers).list.len == 0)
        | (pl.col("turn_index").list.len == pl.col("text").list.len)
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
        source=pl.col("source").list.agg(
            pl.element().str.strip_chars().str.join(" ")
        ),  # .str.join(" "),
    )
    fds_checked
    return


@app.cell(disabled=True, hide_code=True)
def _(valid_speakers):
    pl_aggregate_schema = pl.Schema(
        {
            "speaker": pl.Enum(valid_speakers),
            "text": pl.String,
            "state": pl.String,
            "dialog_id": pl.UInt64,
            "turn_id": pl.UInt64,
            "source": pl.List(pl.String),
        }
    )
    return


@app.cell(disabled=True, hide_code=True)
def _(non_empty, speaker_is_valid, state_is_valid):
    pa_aggregate_schema = pa.DataFrameSchema(
        {
            "speaker": pa.Column(
                str,
                checks=pa.Check(speaker_is_valid, element_wise=True),
                nullable=False,
            ),
            "text": pa.Column(
                str,
                checks=pa.Check(non_empty, element_wise=True),
                nullable=False,
            ),
            "state": pa.Column(
                str,
                checks=pa.Check(state_is_valid, element_wise=True),
                nullable=True,
            ),
            "approx_line_number": pa.Column(int),
        },
        strict=False,
        coerce=True,
    )
    return


@app.cell
def _():
    # 2. produce json output from
    return


@app.cell
def _():
    # Schema.validate(filter_df)  # .collect()
    return


@app.cell
def _(filter_df):
    filter_df.write_ndjson()
    return


if __name__ == "__main__":
    app.run()
