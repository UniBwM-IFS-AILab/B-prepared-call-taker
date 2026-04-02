"""Minimal benchmark models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ems_prepared.dialogue_state.type_defs import KnownBoolean

type OutcomeLabel = Literal["rd1", "rd2", "cpr"] | None
type ResponseKind = Literal["state", "followup", "parse_error"]
type SparseMedicalState = dict[str, KnownBoolean]
type SparseNonMedicalState = dict[str, str]


class RunConfig(BaseModel):
    """Config for one benchmark execution."""

    experiment_id: Literal["experiment1", "experiment2"]
    dataset_path: Path
    results_root: Path = Path("experiments/nlu_eval/results")
    repeats: int = 5
    run_id: str | None = None


class PredictionRecord(BaseModel):
    """Stored output for one item/system/repeat."""

    experiment_id: Literal["experiment1", "experiment2"]
    system_id: str
    item_id: str
    repeat_index: int
    response_kind: ResponseKind
    followup_text: str | None = None
    predicted_medical_state: SparseMedicalState = Field(default_factory=dict)
    predicted_non_medical_state: SparseNonMedicalState = Field(default_factory=dict)
    predicted_outcome: OutcomeLabel = None
    raw_output: dict[str, Any] | str | None = None
    latency_ms: float = 0.0
    token_usage: dict[str, Any] | None = None
    scoring_notes: list[str] = Field(default_factory=list)
