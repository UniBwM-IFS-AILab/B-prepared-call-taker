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
# ems-prepared = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import re
    from pathlib import Path
    from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
    import polars as pl
    import fastexcel
    from pydantic import ValidationError
    import pandera.polars as pa
    from pandera.polars import PolarsData
    from uuid import uuid5, UUID


@app.cell
def _():
    search_dir = Path(r"F:\ODS")
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
def _(search_dir, valid_speakers):
    def state_is_valid(value: str | None) -> bool:
        if value is None or str(value).strip() == "":
            return True

        try:
            EmergencyCall.model_validate_json(value)
            return True
        except ValidationError:
            return False


    def speaker_is_valid(value: str | None) -> bool:
        return value in valid_speakers


    def non_empty(value: str | None) -> bool:
        return value is not None and str(value).strip() != ""


    pa_schema = pa.DataFrameSchema(
        {
            "start": pa.Column(float, nullable=True),
            "end": pa.Column(float, nullable=True),
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
            "line_number": pa.Column(int),
        },
        strict=False,
        coerce=True,
    )


    report_schema = {
        "source_file": pl.String,
        "line_number": pl.Int32,
        "description": pl.String,
        "cell_value": pl.String,
    }


    errors = []

    for path in search_dir.rglob("[!~$]*.ods"):
        file_id = f"{path.parent.name}/{path.name}"

        ods_df = pl.read_ods(path).with_row_index("line_number", offset=2)

        body_df = ods_df.slice(0, ods_df.height - 1)
        final_row = ods_df.tail(1)

        try:
            pa_schema.validate(body_df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            errors.append(
                exc.failure_cases.with_columns(
                    pl.lit(file_id).alias("source_file"),
                    pl.when(pl.col("index").is_null())
                    .then(1)
                    .otherwise(pl.col("index") + 2)
                    .alias("line_number"),
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
                    "line_number",
                    "description",
                    "cell_value",
                )
            )
        except pl.exceptions.ColumnNotFoundError as exc:
            errors.append(
                pl.DataFrame(
                    {
                        "source_file": [file_id],
                        "line_number": [1],
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
            final_line_number = final_row["line_number"][0]
            errors.append(
                pl.DataFrame(
                    {
                        "source_file": [file_id],
                        "line_number": [final_line_number],
                        "description": [str(exc)],
                        "cell_value": [final_row["State"][0]],
                    },
                    schema=report_schema,
                )
            )

    errors = (
        pl.concat(errors, how="diagonal")
        if errors
        else pl.DataFrame(schema=report_schema)
    )

    errors.write_csv("validation_errors.csv")
    errors
    return (ods_df,)


@app.cell(disabled=True, hide_code=True)
def _():
    def _state_is_valid(value) -> bool:
        if value is None or value == "":
            return True

        try:
            if isinstance(value, str):
                EmergencyCall.model_validate_json(value)
            else:
                EmergencyCall.model_validate(value)

            return True

        except (ValidationError, TypeError):
            return False

    return


@app.cell
def _():
    # REPORT_SCHEMA = {
    #     "source_file": pl.String,
    #     "line_number": pl.Int32,
    #     "failure_case": pl.String,
    #     "schema_context": pl.String,
    #     "column": pl.String,
    #     "check": pl.String,
    #     "check_number": pl.Int32,
    #     "index": pl.Int32,
    # }


    # def pandera_to_report(failure_cases: pl.DataFrame, file_id: str) -> pl.DataFrame:
    #     return (
    #         failure_cases.with_columns(
    #             pl.lit(file_id).alias("source_file"),
    #             # Pandera index is dataframe row index: 0 = first data row.
    #             # Excel row 1 is header, so first data row is line 2.
    #             pl.when(pl.col("index").is_null())
    #             .then(pl.lit(1))
    #             .otherwise(pl.col("index").cast(pl.Int32, strict=False) + 2)
    #             .cast(pl.Int32)
    #             .alias("line_number"),
    #         )
    #         .with_columns(
    #             pl.col("source_file").cast(pl.String),
    #             pl.col("failure_case").cast(pl.String),
    #             pl.col("schema_context").cast(pl.String),
    #             pl.col("column").cast(pl.String),
    #             pl.col("check").cast(pl.String),
    #             pl.col("check_number").cast(pl.Int32, strict=False),
    #             pl.col("index").cast(pl.Int32, strict=False),
    #         )
    #         .select(list(REPORT_SCHEMA))
    #     )


    # def column_error(exc: Exception, file_id: str) -> pl.DataFrame:
    #     return pl.DataFrame(
    #         {
    #             "source_file": [file_id],
    #             "line_number": [1],
    #             "failure_case": [str(exc)],
    #             "schema_context": ["Column"],
    #             "column": ["see failure_case"],
    #             "check": ["column_missing"],
    #             "check_number": [0],
    #             "index": [0],
    #         },
    #         schema=REPORT_SCHEMA,
    #     )


    # def speaker_missing_error(
    #     exc: Exception, file_id: str, ods_df: pl.DataFrame
    # ) -> pl.DataFrame:
    #     return pl.DataFrame(
    #         {
    #             "source_file": [file_id],
    #             "line_number": [ods_df.height + 1],
    #             "failure_case": [str(exc)],
    #             "schema_context": ["Column"],
    #             "column": ["see failure_case"],
    #             "check": ["final_state_problem"],
    #             "check_number": [0],
    #             "index": [ods_df.height],
    #         },
    #         schema=REPORT_SCHEMA,
    #     )


    # issue_frames: list[pl.DataFrame] = []

    # for dialog_id, path in enumerate(search_dir.rglob("[!~$]*.ods")):
    #     file_id = f"{path.parent.name}/{path.name}"

    #     try:
    #         ods_df = pl.read_ods(path)

    #         pa_schema.validate(ods_df, lazy=True)

    #     except pa.errors.SchemaErrors as exc:
    #         issue_frames.append(pandera_to_report(exc.failure_cases, file_id))

    #     except pl.exceptions.ColumnNotFoundError as exc:
    #         issue_frames.append(column_error(exc, file_id))

    #     # try:
    #     #     assert ods_df.get_column("speaker").null_count() == 1, (
    #     #         "missing speakers? only last row (final_state) should contain none"
    #     #     )
    #     # except AssertionError as exc:
    #     #     issue_frames.append(speaker_missing_error(exc, file_id, ods_df))


    # errors = (
    #     pl.concat(issue_frames, how="vertical")
    #     if issue_frames
    #     else pl.DataFrame(schema=REPORT_SCHEMA)
    # )

    # errors  # .select("source_file", "line_number", "failure_case", "schema_context")
    return


@app.cell
def _():
    #     mmyyyy, id = re.findall(pattern, str(path))[0]
    #     # print(mmyyyy, id)
    #     # print(uuid5(UUID(int=0), path.stem))
    #     # print(path)

    #     try:
    #         ods_df = pl.read_ods(
    #             path,
    #             columns=["start", "end", "speaker", "text", "State"],
    #             schema_overrides={
    #                 "text": pl.String,
    #                 "speaker": pl.String,
    #                 "State": pl.String,
    #             },
    #         )
    #     except fastexcel.ColumnNotFoundError as ex:
    #         errors = errors.vstack(
    #             pl.DataFrame(
    #                 {
    #                     "filename": [file_id],
    #                     "line_number": [1],
    #                     "incorrect_value": [None],
    #                     "issue": [f"schema error: {ex}"],
    #                 },
    #                 schema=errors.schema,
    #             )
    #         )
    #         continue
    return


@app.cell
def _(ods_df):
    ods_df.columns = [col.lower() for col in ods_df.columns]
    ods_df
    return


@app.cell
def _(filter_df, ods_df):
    # TODO: avoid extend somehow when cell is executed again
    filter_df.extend(ods_df.select("file_id", "turn_id", "text", "speaker", "state"))
    filter_df
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


if __name__ == "__main__":
    app.run()
