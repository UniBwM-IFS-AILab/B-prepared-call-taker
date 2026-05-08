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
# ]
# requires-python = ">=3.13"
# [tool.uv.sources]
# ems-prepared = { path = "..", editable = true }
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import re
    from pathlib import Path
    import os
    from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
    import polars as pl
    import fastexcel
    from pydantic import ValidationError
    import pandera.polars as pa
    from pandera.polars import PolarsData
    from uuid import uuid5, UUID


@app.cell
def _():
    search_dir = Path(os.path.expanduser(r"~/Documents/DialogData/"))
    pl_schema = {
        "dialog_id": pl.UInt32,
        "turn_id": pl.UInt32,
        "text": pl.String,
        "speaker": pl.String,
        "state": pl.String,
    }
    valid_speakers = ["CALLER", "DISPATCHER", "EXTRA", "PATIENT"]
    pattern = r"^.*(\d{6}).*?(\d+)"

    filter_df = pl.DataFrame(data=None, schema=pl_schema)
    return filter_df, search_dir, valid_speakers


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
    pa_schema = pa.DataFrameSchema(
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
    return (pa_schema,)


@app.cell
def _():
    report_schema = {
        "source_file": pl.String,
        "approx_line_number": pl.Int32,
        "origin": pl.String,
        "exception_class": pl.String,
        "description": pl.String,
        "cell_value": pl.String,
    }
    return (report_schema,)


@app.cell
def _(pa_schema, report_schema, search_dir):
    correct_paths = []
    errors = []

    for _path in search_dir.rglob("[!~$]*.ods"):
        contains_error: bool = False
        file_id = f"{_path.parent.name}/{_path.name}"

        _ods_df = pl.read_ods(_path).with_row_index("approx_line_number", offset=2)

        body_df = _ods_df.slice(0, _ods_df.height - 1)
        final_row = _ods_df.tail(1)

        try:
            pa_schema.validate(body_df, lazy=True)
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
                    schema=report_schema,
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
                        "cell_value": [final_state or f"last line:{_ods_df.height + 2}"],
                    },
                    schema=report_schema,
                )
            )
        if not contains_error:
            correct_paths.append(_path)

    errors = (
        pl.concat(errors, how="diagonal")
        if errors
        else pl.DataFrame(schema=report_schema)
    )

    errors.write_csv("validation_errors.csv")
    errors
    return (correct_paths,)


@app.cell
def _():
    #     mmyyyy, id = re.findall(pattern, str(path))[0]
    #     # print(mmyyyy, id)
    #     # print(uuid5(UUID(int=0), path.stem))
    #     # print(path)


    return


@app.cell
def _(correct_paths):
    correct_paths
    return


@app.cell
def _():
    EmergencyCall.model_fields["emergency_type"] #.keys()
    return


@app.cell
def _(correct_paths):
    fds_schema = pl.Schema({
        "source": pl.String,
        "dialog_id": pl.UInt64,
        "turn_index": pl.UInt32,
        "speaker": pl.String,
        "text": pl.String,
        "state": pl.String,
        # "state": pl.Struct(fields={name: pl.Binary | pl.String for name in EmergencyCall.model_fields.keys()})  #Struct(fields={name: pl.Bool | pl.String for name in EmergencyCall.model_fields.keys()})
    })
    full_dataset = pl.DataFrame(schema=fds_schema)
    for _path in correct_paths:
        _ods_df = pl.read_ods(_path)
        file_hash = _ods_df.hash_rows(seed=42).sum()
        filter_ods_df = (
            _ods_df.rename({"State": "state"})
            .with_row_index(name="turn_index")
            .with_columns(
                [
                    pl.Series(
                        "dialog_id",
                        [
                            file_hash
                            for _ in range(_ods_df.height)
                        ],
                        dtype=pl.UInt64
                    ),
                    pl.Series(
                        "source",
                        [
                            f"{_path.parent.name}/{_path.name}"
                            for _ in range(_ods_df.height)
                        ]
                    )
                ]
            )
            .select("source", "dialog_id", "turn_index", "speaker", "text", "state")
        )
        full_dataset.vstack(filter_ods_df, in_place=True).rechunk()

    # filter_ods_df#.to_dicts
    full_dataset
    # _ods_df.hash_rows(seed=42).sum()
    return (full_dataset,)


@app.cell
def _(full_dataset):
    full_dataset.columns
    return


@app.cell
def _(full_dataset):
    ( #https://stackoverflow.com/questions/73222000/polars-conditional-merge-of-rows
        # FIXME: some rows are missing
        full_dataset.with_columns(
            (
                (pl.col('dialog_id') == pl.col('dialog_id').shift(-1))
                &
                (pl.col("speaker")
                != pl.col("speaker").shift(-1))
            ).shift(1, fill_value=False)
            .cum_sum()
            .alias('consecutive_count')
        )
        .group_by('consecutive_count')
        .agg(
            source=pl.col("source").first(),
            dialog_id=pl.col("dialog_id").first(),
            text=pl.col("text"),
            turn_index=pl.col("turn_index").first(),
            speaker=pl.col('speaker')
            # pl.col('text').().alias('text'),
            # pl.col('state').sum().alias('state'),
            # pl.col('speaker').first().alias('speaker'),
        )
    ).sort(["dialog_id", "turn_index"], descending=False, maintain_order=True)
    return


@app.cell
def _():
    # 2. produce json output from 
    return


@app.cell
def _(Schema, filter_df):
    Schema.validate(filter_df)  # .collect()
    return


@app.cell
def _(filter_df):
    filter_df.write_ndjson()
    return


@app.cell
def _(base, filter_df):
    filter_df.write_parquet(base / "test.parquet")
    return


@app.cell
def _():
    import marimo as mo

    return


if __name__ == "__main__":
    app.run()
