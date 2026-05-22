"""Shared Gradio state dataclasses for guided and standard UIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Phase = Literal["consent", "selection", "brief", "starting", "call", "complete", "survey"]


@dataclass
class SessionRef:
    """Serializable session reference shared by Gradio modes."""

    user_id: str
    session_id: str
    policy_name: str
    scenario_name: str | None
    experiment_name: str


@dataclass
class StartedSession:
    """Result returned after one backend session start."""

    session: SessionRef
    events: list
    user_id: str
    is_complete: bool


@dataclass
class CompletedScenario:
    """One completed guided scenario entry."""

    scenario_name: str


@dataclass
class GuidedState:
    """Guided-mode run state."""

    scenario_names: tuple[str, ...]
    user_id: str = ""
    run_session_id: str | None = None
    active_index: int = 0
    phase: Phase = "brief"
    survey_submitted: bool = False
    consent_accepted: bool = False
    active_session: SessionRef | None = None
    completed_sessions: list[CompletedScenario] = field(default_factory=list)
